#pragma once
// The persistent accelerator trusts Windows current-user isolation, not record
// writers. Same-user code execution and administrators are outside this boundary.
#include <windows.h>
#include <wincrypt.h>
#include <bcrypt.h>
#include <sddl.h>
#include <aclapi.h>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <span>
#include <stdexcept>
#include <vector>
#include <functional>
#include "strix/model_content_hash.hpp"

namespace strix::glm::rtxpf::producer {
using HashKey=std::array<std::byte,32>;
#ifdef STRIX_HASH_AUTHORITY_TESTING
inline std::function<void(const char*)> authority_fault;
inline void authority_stage(const char* stage){if(authority_fault)authority_fault(stage);}
#else
inline void authority_stage(const char*){}
#endif
inline void authority_need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct LocalMemory {void* p=nullptr;~LocalMemory(){if(p)LocalFree(p);}};
inline void protect_hash_directory(const std::filesystem::path& directory){
    authority_stage("directory");
    std::filesystem::create_directories(directory);
    auto attrs=GetFileAttributesW(directory.c_str());
    authority_need(attrs!=INVALID_FILE_ATTRIBUTES&&!(attrs&FILE_ATTRIBUTE_REPARSE_POINT),"hash directory reparse point");
    HANDLE token=nullptr;authority_need(OpenProcessToken(GetCurrentProcess(),TOKEN_QUERY,&token)!=0,"hash user token");
    DWORD n=0;GetTokenInformation(token,TokenUser,nullptr,0,&n);std::vector<std::byte> buffer(n);
    auto ok=GetTokenInformation(token,TokenUser,buffer.data(),n,&n);CloseHandle(token);authority_need(ok!=0,"hash user SID");
    LocalMemory sid,descriptor;
    authority_need(ConvertSidToStringSidW(reinterpret_cast<TOKEN_USER*>(buffer.data())->User.Sid,reinterpret_cast<LPWSTR*>(&sid.p))!=0,"hash SID string");
    std::wstring sddl=L"D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;"+std::wstring(static_cast<wchar_t*>(sid.p))+L")";
    authority_need(ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl.c_str(),SDDL_REVISION_1,&descriptor.p,nullptr)!=0,"hash directory descriptor");
    PACL acl=nullptr;BOOL present=FALSE,defaulted=FALSE;
    authority_need(GetSecurityDescriptorDacl(descriptor.p,&present,&acl,&defaulted)&&present,"hash directory DACL");
    authority_stage("ACL");
    authority_need(SetNamedSecurityInfoW(const_cast<wchar_t*>(directory.c_str()),SE_FILE_OBJECT,
        DACL_SECURITY_INFORMATION|PROTECTED_DACL_SECURITY_INFORMATION,nullptr,nullptr,acl,nullptr)==ERROR_SUCCESS,"protect hash directory ACL");
}
inline HashKey hash_authority(const std::filesystem::path& directory,bool* usable=nullptr){
    if(usable)*usable=false;
    static std::mutex mutex;std::lock_guard lock(mutex);
    protect_hash_directory(directory);auto path=directory/L"user-key.dpapi";
    // The persistent lock file is never deleted. Its exclusive handle serializes
    // creation AND invalid-key repair across processes, including abandoned owners.
    struct RepairLock {HANDLE h=INVALID_HANDLE_VALUE;~RepairLock(){if(h!=INVALID_HANDLE_VALUE)CloseHandle(h);}} repair;
    const auto until=GetTickCount64()+5000;
    do{
        repair.h=CreateFileW((directory/L"authority.lock").c_str(),GENERIC_READ|GENERIC_WRITE,0,nullptr,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(repair.h!=INVALID_HANDLE_VALUE)break;
        authority_need(GetLastError()==ERROR_SHARING_VIOLATION&&GetTickCount64()<until,"hash authority repair lock");Sleep(10);
    }while(true);
    // Re-check only after acquiring the interprocess lock. An invalid existing
    // key can be repaired, but this attempt must still cold-hash without publishing.
    authority_stage("load");
    std::error_code exists_error;bool existed=std::filesystem::exists(path,exists_error);
    authority_need(!exists_error,"hash authority status");
    auto load=[&](HashKey& key){
        std::ifstream input(path,std::ios::binary);if(!input)return false;
        input.seekg(0,std::ios::end);auto n=input.tellg();if(n<=0||n>16384)return false;input.seekg(0);
        std::vector<BYTE> encrypted(static_cast<std::size_t>(n));input.read(reinterpret_cast<char*>(encrypted.data()),encrypted.size());if(!input)return false;
        DATA_BLOB in{DWORD(encrypted.size()),encrypted.data()},out{};LocalMemory plain;
        if(!CryptUnprotectData(&in,nullptr,nullptr,nullptr,nullptr,CRYPTPROTECT_UI_FORBIDDEN,&out))return false;
        plain.p=out.pbData;bool valid=out.cbData==key.size();if(valid)std::memcpy(key.data(),out.pbData,key.size());
        SecureZeroMemory(out.pbData,out.cbData);return valid;
    };
    HashKey key{};if(load(key)){if(usable)*usable=true;return key;}
    authority_stage("RNG");
    authority_need(BCryptGenRandom(nullptr,reinterpret_cast<PUCHAR>(key.data()),DWORD(key.size()),BCRYPT_USE_SYSTEM_PREFERRED_RNG)==0,"hash key random");
    DATA_BLOB in{DWORD(key.size()),reinterpret_cast<BYTE*>(key.data())},out{};LocalMemory encrypted;
    authority_stage("DPAPI");
    authority_need(CryptProtectData(&in,L"GLM verified hash key v2",nullptr,nullptr,nullptr,CRYPTPROTECT_UI_FORBIDDEN,&out)!=0,"protect hash key DPAPI");encrypted.p=out.pbData;
    auto temporary=path;temporary+=L"."+std::to_wstring(GetCurrentProcessId())+L".partial";
    struct Cleanup {std::filesystem::path path;~Cleanup(){std::error_code ec;std::filesystem::remove(path,ec);}} cleanup{temporary};
    authority_stage("persist");
    {std::ofstream file(temporary,std::ios::binary|std::ios::trunc);file.write(reinterpret_cast<const char*>(out.pbData),out.cbData);file.close();authority_need(bool(file),"persist protected hash key");}
    authority_stage("publish");
    authority_need(MoveFileExW(temporary.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)!=0,"publish protected hash key");
    if(usable)*usable=!existed;
    return key;
}
inline std::string hash_record_mac(const HashKey& key,const std::string& message){
    std::array<std::byte,64> inner{},outer{};
    for(std::size_t i=0;i<64;++i){auto k=i<key.size()?key[i]:std::byte{0};inner[i]=k^std::byte{0x36};outer[i]=k^std::byte{0x5c};}
    strix::Sha256Fast a;a.update(inner);a.update(std::as_bytes(std::span(message)));auto digest=a.finalize();
    strix::Sha256Fast b;b.update(outer);b.update(std::as_bytes(std::span(digest.bytes)));return strix::model_content_identity_hex(b.finalize());
}
inline bool equal_record_mac(const std::string& a,const std::string& b){
    if(a.size()!=64||b.size()!=64)return false;unsigned different=0;for(unsigned i=0;i<64;++i)different|=unsigned(a[i]^b[i]);return different==0;
}
}
