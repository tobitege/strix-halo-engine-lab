#define STRIX_HASH_AUTHORITY_TESTING
#include "cuda/models/glm/producer/verified_manifest.hpp"
#include "cuda/models/glm/producer/shards.hpp"
#include <iostream>
#include <memory>
#include <future>
#include <winioctl.h>
using namespace strix::glm::rtxpf::producer;
void check(bool b){if(!b)throw std::runtime_error("verified IO test failed");}
void junction(const std::filesystem::path& link,const std::filesystem::path& target){
    std::filesystem::create_directory(link);
    // Mount-point reparse points need no symlink privilege. Replace only the
    // alias; the actual manifest handle continues to lock its original target.
    struct Reparse {DWORD tag;USHORT length,reserved,sub_offset,sub_length,print_offset,print_length;wchar_t paths[4096];} data{};
    auto native=std::filesystem::absolute(target).make_preferred().wstring();
    auto substitute=L"\\??\\"+native;auto print=native;
    data.tag=IO_REPARSE_TAG_MOUNT_POINT;data.sub_length=USHORT(substitute.size()*2);data.print_offset=data.sub_length+2;data.print_length=USHORT(print.size()*2);
    data.length=USHORT(8+data.print_offset+data.print_length+2);check(data.length+8<=sizeof data);
    std::memcpy(data.paths,substitute.c_str(),data.sub_length+2);std::memcpy(reinterpret_cast<char*>(data.paths)+data.print_offset,print.c_str(),data.print_length+2);
    WinHandle handle;handle.h=CreateFileW(link.c_str(),GENERIC_WRITE,0,nullptr,OPEN_EXISTING,FILE_FLAG_OPEN_REPARSE_POINT|FILE_FLAG_BACKUP_SEMANTICS,nullptr);
    DWORD bytes=0;check(handle.h!=INVALID_HANDLE_VALUE&&DeviceIoControl(handle.h,FSCTL_SET_REPARSE_POINT,&data,data.length+8,nullptr,0,&bytes,nullptr));
}
namespace strix::glm::rtxpf::producer {
// Only this CPU test can construct a schema-sized-down Shards. It exercises
// the production inventory, config/header/marker readers, packing and streams.
struct ShardsFixture {
    static void write(const std::filesystem::path& root,bool forged){
        std::filesystem::create_directories(root);
        std::ofstream(root/"config.json")<<(forged?"{\"source\":\"swapped!\"}":"{\"source\":\"original\"}");
        for(unsigned file=0;file<2;++file){
            auto header=Json::object();std::vector<std::byte> data;
            auto add=[&](const std::string& name,const char* dtype,Json shape,std::size_t n,std::uint32_t word){
                auto t=Json::object(),offsets=Json::array();offsets.push(std::uint64_t(data.size()));offsets.push(std::uint64_t(data.size()+n));
                t.set("dtype",dtype);t.set("shape",shape);t.set("data_offsets",offsets);header.set(name,t);
                auto start=data.size();data.resize(start+n);for(std::size_t i=0;i<n;++i)data[start+i]=std::byte((word>>((i%4)*8))&255);
            };
            auto shape=[](std::initializer_list<unsigned> dims){auto a=Json::array();for(auto d:dims)a.push(std::uint64_t(d));return a;};
            const std::string p="model.language_model.";
            if(!file){
                add(p+"probe.mul1","I32",shape({}),4,forged?0:0x83dcd12d);
                add(p+"probe.trellis","I16",shape({1,1,48}),96,forged?0x12341234:0x87658765);
                add(p+"probe.suh","F16",shape({16}),32,forged?0x34003400:0x3c003c00);
                add(p+(forged?"pr0be.svh":"probe.svh"),"F16",shape({16}),32,forged?0x38003800:0x40004000);
                add(p+"a.weight","BF16",shape({6}),12,forged?0x3f003f00:0x3f803f80);
                add(p+"layers.3.mlp.experts.0.z","F32",shape({3}),12,forged?0x3f000000:0x3f800000);
            }else{
                // More than two IO chunks, unaligned starts, sector tail and
                // a tensor crossing the 16 MiB reader window boundary.
                add(p+"b.weight","I32",shape({unsigned((2*io_chunk+124)/4)}),2*io_chunk+124,forged?0x77777777:0x31415926);
                add(p+"layers.3.mlp.experts.0.a","I16",shape({7}),14,forged?0x11111111:0x98765432);
                add(p+"layers.4.mlp.experts.0.a","F16",shape({9}),18,forged?0x38003800:0x3c003c00);
            }
            auto raw=header.dump();std::uint64_t n=raw.size();
            std::ofstream out(root/(file?"b.safetensors":"a.safetensors"),std::ios::binary);
            out.write(reinterpret_cast<const char*>(&n),8);out.write(raw.data(),raw.size());out.write(reinterpret_cast<const char*>(data.data()),data.size());check(bool(out));
        }
    }
    static std::map<std::string,std::vector<std::byte>> legacy(const Shards& source,const std::filesystem::path& root){
        // Old pathname/seek/read algorithm, independently computing grouping,
        // sorted placement and 256-byte zero padding (not using pieces/stream).
        std::map<std::string,std::vector<std::byte>> blobs;
        for(auto& [name,t]:source.selected){
            auto marker=name.find(".mlp.experts.");std::string blob="base.bin";
            if(marker!=std::string::npos){auto begin=name.find(".layers.")+8;auto layer=std::stoul(name.substr(begin));blob="experts-"+std::string(layer<10?"0":"")+std::to_string(layer)+".bin";}
            auto& bytes=blobs[blob];auto offset=(bytes.size()+255)&~std::size_t(255);bytes.resize(offset+t.bytes);
            std::ifstream f(root/t.file,std::ios::binary);f.seekg(t.offset);f.read(reinterpret_cast<char*>(bytes.data()+offset),t.bytes);check(bool(f));
        }return blobs;
    }
    static void run(const std::filesystem::path& root){
        auto original=root/"shards-original",forged=root/"shards-forged",alias=root/"shards-alias",cache=root/"shard-hashes";
        write(original,false);write(forged,true);junction(alias,original);
        // Deterministic authenticated recall even without a real test-volume
        // journal. The existing tests separately check real USN/fallback policy.
        journal_test_identity="shard-stream-fixture 1";
        {
            Shards cold(alias.string(),Shards::InventoryOnly{});cold.sources_->records(true,cache);check(cold.sources_->verified_hash_hits==0);
        }
        {
            Shards held(alias.string(),Shards::InventoryOnly{});
            auto records=held.sources_->records(true,cache);check(held.sources_->verified_hash_hits==2);
            for(auto& f:held.sources_->files)check(f.held->bytes_read()==0);
            // Swap the ancestor after recall, before any metadata/payload use.
            check(RemoveDirectoryW(alias.c_str())!=0);junction(alias,forged);
            auto config=held.config();check(member(config,"source").as_string()=="original");
            held.read_headers();check(held.tensors.size()==9);held.matrix("probe",16,16);
            held.selected=held.tensors;held.manifest=shard_content_manifest(config,records);held.pack();
            const auto identity=held.manifest.dump();auto expected=legacy(held,original);auto replacement=legacy(held,forged);
            check(expected.size()==3);
            for(auto& [blob,bytes]:expected){
                check(bytes!=replacement.at(blob));check(natural(member(member(member(held.manifest,"blobs"),blob),"bytes"))==bytes.size());
                for(std::size_t capacity:{256u,65540u,8u<<20}){
                    std::vector<std::byte> staging(capacity),actual;held.stream(blob,staging.data(),capacity,[&](std::uint64_t off,const void* p,std::size_t n){check(off==actual.size());auto b=static_cast<const std::byte*>(p);actual.insert(actual.end(),b,b+n);});check(actual==bytes);
                }
            }
            std::cout<<"PASS Shards config/header/marker/stream after authenticated recall and ancestor junction swap; synthetic legacy packed blobs byte-identical\n";
            for(unsigned request=0;request<3;++request){
                // A fresh call each request is the exact CPU file-tier reader
                // used by runtime read_file/read-ahead and resident loads.
                check(RemoveDirectoryW(alias.c_str())!=0);junction(alias,request%2?original:forged);
                std::vector<std::future<void>> reads;
                for(auto& [blob,bytes]:expected)reads.push_back(std::async(std::launch::async,[&,blob]{
                    auto& reference=expected.at(blob);std::vector<std::byte> actual(reference.size(),std::byte{0xff});held.read_into(blob,actual.data());check(actual==reference);
                }));
                for(auto& read:reads)read.get();check(held.manifest.dump()==identity);
            }
            std::cout<<"PASS repeated asynchronous file-tier reader requests across ancestor junction swaps; retained bytes and identity\n";
            // Deferred source authentication (early budget plan): an unhashed
            // inventory has the same blob sizes; authenticate_sources() then yields
            // the identical manifest and a re-packed (not duplicated) stream layout.
            Shards deferred(original.string(),Shards::InventoryOnly{});deferred.defer_markers_=true;deferred.read_headers();deferred.matrix("probe",16,16);deferred.selected=deferred.tensors;
            check(deferred.deferred_marker_count()==1); // no payload read before authentication
            deferred.manifest=shard_content_manifest(deferred.config(),deferred.sources_->records(false));deferred.pack();
            const auto unhashed_blobs=member(deferred.manifest,"blobs").dump();check(deferred.manifest.dump()!=identity);
            deferred.authenticate_sources(cache);check(deferred.manifest.dump()==identity&&member(deferred.manifest,"blobs").dump()==unhashed_blobs&&deferred.deferred_marker_count()==0);
            for(auto& [blob,bytes]:expected){std::vector<std::byte> staging(65540),actual;
                deferred.stream(blob,staging.data(),staging.size(),[&](std::uint64_t off,const void* p,std::size_t n){check(off==actual.size());auto b=static_cast<const std::byte*>(p);actual.insert(actual.end(),b,b+n);});check(actual==bytes);}
            std::cout<<"PASS deferred source authentication: identical manifest, same blob sizes before hashing, stream layout not duplicated\n";
        }
        journal_test_identity.reset();check(RemoveDirectoryW(alias.c_str())!=0);
    }
};
}
int main(int argc,char** argv){try{
    if(argc==4&&std::string(argv[1])=="--repair-child"){
        WinHandle gate;gate.h=OpenEventA(SYNCHRONIZE,FALSE,argv[3]);check(gate.h&&WaitForSingleObject(gate.h,10000)==WAIT_OBJECT_0);
        auto key=hash_authority(argv[2]);Sleep(100);check(hash_authority(argv[2])==key);
        std::ofstream out(std::filesystem::path(argv[2])/(std::to_string(GetCurrentProcessId())+".key"),std::ios::binary);
        out.write(reinterpret_cast<const char*>(key.data()),key.size());return out?0:1;
    }
    check(argc==2);auto root=std::filesystem::path(argv[1])/("verified-io-tests-"+std::to_string(GetCurrentProcessId()));
    std::filesystem::create_directories(root);auto path=root/"input.bin",cache=root/"hashes";
    HashKey known{};for(unsigned i=0;i<known.size();++i)known[i]=std::byte(i);
    check(hash_record_mac(known,"canonical record\nexpected digest\n")=="6d7b6d8045889c8cd44e03907387c9ad88eaa98754c3d317e5f96e49aecd3605");
    // >2 chunks, non-sector-aligned EOF, deterministic data, bounded artifacts.
    std::vector<std::byte> input(2*io_chunk+123);for(std::size_t i=0;i<input.size();++i)input[i]=std::byte((i*37+i/8191)%256);
    auto write=[&]{std::ofstream f(path,std::ios::binary|std::ios::trunc);f.write(reinterpret_cast<const char*>(input.data()),input.size());check(bool(f));};
    write();auto digest=strix::model_content_identity_hex(strix::sha256_bytes_fast(input));
    bool journal=false;DWORD journal_error=0;
    {WinHandle h;h.h=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_OVERLAPPED,nullptr);journal=file_journal_identity(h.h,&journal_error).has_value();}
    if(!journal){
        // Access denial or a malformed query is a test failure, not an excuse
        // to skip the replay test on an otherwise journal-capable test volume.
        if(journal_error!=ERROR_INVALID_FUNCTION&&journal_error!=ERROR_JOURNAL_NOT_ACTIVE&&journal_error!=ERROR_JOURNAL_DELETE_IN_PROGRESS)
            throw std::runtime_error("cannot test journal replay: Windows error "+std::to_string(journal_error));
        std::cout<<"SKIP warm-record timestamp replay: test volume has no active/supported NTFS journal (Windows error "<<journal_error<<"); full-hash fallback tested\n";
    }
    {VerifiedFile f(path,input.size());check(!f.verify(digest,cache));check(f.verify(digest,cache)==journal);
        AlignedMemory out(input.size());f.read_into(out.p);check(!std::memcmp(out.p,input.data(),input.size()));
        AlignedMemory chunk(io_chunk);f.read_at(2*io_chunk,chunk.p,123);check(!std::memcmp(chunk.p,input.data()+2*io_chunk,123));
        VerifiedFile::Reader reader(f,2*io_chunk);std::vector<std::byte> range(io_chunk+39);
        reader.read_at(io_chunk-7,range.data(),range.size());check(!std::memcmp(range.data(),input.data()+io_chunk-7,range.size()));
        reader.read_at(3,range.data(),13);check(!std::memcmp(range.data(),input.data()+3,13));
        reader.read_at(input.size(),range.data(),0);bool outside=false;try{reader.read_at(input.size(),range.data(),1);}catch(...){outside=true;}check(outside);
        HANDLE writer=CreateFileW(path.c_str(),GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,0,nullptr);
        check(writer==INVALID_HANDLE_VALUE&&GetLastError()==ERROR_SHARING_VIOLATION);
        check(!DeleteFileW(path.c_str())&&GetLastError()==ERROR_SHARING_VIOLATION);
        bool rejected=false;try{f.verify(std::string(64,'0'),cache);}catch(...){rejected=true;}check(rejected);
    }
    {VerifiedFile restart(path,input.size());check(restart.verify(digest,cache)==journal);}
    {VerifiedFile restart(path,input.size());auto result=restart.compute_or_recall(cache);
        check(result.sha256==digest&&result.recalled==journal);check(restart.bytes_read()==(journal?0:input.size()));}
    {journal_unavailable=true;auto unavailable=root/"no-journal";VerifiedFile f(path,input.size());
        check(!f.verify(digest,cache));check(!f.verify(digest,unavailable));check(!std::filesystem::exists(unavailable));
        bool rejected=false;try{f.verify(std::string(64,'0'),cache);}catch(...){rejected=true;}check(rejected);journal_unavailable=false;}
    // No USN journal, opted in (unit file): an mtime-stamped record bound to the
    // pin. Recall only with the same stamp, pin, MAC and expected digest.
    {journal_unavailable=true;auto stamped=root/"no-usn-records";const std::string pin_a(64,'a'),pin_b(64,'b');
        {VerifiedFile f(path,input.size());auto d=f.verify_digest(digest,stamped,RecordOptions{true,pin_a});check(!d.recalled&&d.basis=="hashed"&&f.bytes_read()==input.size());}
        {VerifiedFile f(path,input.size());auto d=f.verify_digest(digest,stamped,RecordOptions{true,pin_a});check(d.recalled&&d.basis=="no-usn record (mtime-stamped)"&&f.bytes_read()==0);}
        {VerifiedFile f(path,input.size());check(!f.verify(digest,stamped));}  // no opt-in: never recalled without a journal
        {VerifiedFile f(path,input.size());auto d=f.verify_digest(digest,stamped,RecordOptions{true,pin_b});check(!d.recalled&&f.bytes_read()==input.size());} // other pin
        {VerifiedFile f(path,input.size());auto d=f.verify_digest(digest,stamped,RecordOptions{true,""});check(!d.recalled);}           // no pin: no record
        {VerifiedFile f(path,input.size());bool rejected=false;try{f.verify_digest(std::string(64,'0'),stamped,RecordOptions{true,pin_a});}catch(...){rejected=true;}check(rejected);}
        // A changed last-write time misses (the record key is the handle stamp).
        {WinHandle w;w.h=CreateFileW(path.c_str(),FILE_WRITE_ATTRIBUTES,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,0,nullptr);check(w.h!=INVALID_HANDLE_VALUE);
            FILETIME c{},a{},m{};check(GetFileTime(w.h,&c,&a,&m)!=0);ULARGE_INTEGER u;u.LowPart=m.dwLowDateTime;u.HighPart=m.dwHighDateTime;u.QuadPart+=10000000ULL;
            m.dwLowDateTime=u.LowPart;m.dwHighDateTime=u.HighPart;check(SetFileTime(w.h,nullptr,nullptr,&m)!=0);}
        {VerifiedFile f(path,input.size());auto d=f.verify_digest(digest,stamped,RecordOptions{true,pin_a});check(!d.recalled&&f.bytes_read()==input.size());}
        {VerifiedFile f(path,input.size());auto d=f.verify_digest(digest,stamped,RecordOptions{true,pin_a});check(d.recalled&&f.bytes_read()==0);}
        // Engine stamp v3 excludes ChangeTime: a hard link made and removed (metadata only) still recalls.
        {auto link=root/"input-link.bin";check(CreateHardLinkW(link.c_str(),path.c_str(),nullptr)!=0);check(DeleteFileW(link.c_str())!=0);
            VerifiedFile f(path,input.size());auto d=f.verify_digest(digest,stamped,RecordOptions{true,pin_a});check(d.recalled&&f.bytes_read()==0);}
        // A forged MAC on the matching record misses, then the record is repaired.
        for(auto& entry:std::filesystem::directory_iterator(stamped))if(entry.path().extension()==".sha256"){
            std::ifstream saved(entry.path());std::string metadata,sha;std::getline(saved,metadata);std::getline(saved,sha);saved.close();
            if(metadata.find(" no-usn pin="+pin_a)==std::string::npos)continue;
            std::ofstream forged(entry.path(),std::ios::trunc);forged<<metadata<<'\n'<<sha<<'\n'<<std::string(64,'0')<<'\n';}
        {VerifiedFile f(path,input.size());auto d=f.verify_digest(digest,stamped,RecordOptions{true,pin_a});check(!d.recalled&&f.bytes_read()==input.size());}
        {VerifiedFile f(path,input.size());check(f.verify_digest(digest,stamped,RecordOptions{true,pin_a}).recalled);}
        journal_unavailable=false;
        std::cout<<"PASS no-USN mtime-stamped record: recall on equal stamp+pin+MAC only; pin, timestamp, forged MAC and opt-out all re-hash\n";}
    // Check the installed DACL, not just the requested SDDL text.
    hash_authority(cache);
    {LocalMemory descriptor;PACL acl=nullptr;check(GetNamedSecurityInfoW(const_cast<wchar_t*>(cache.c_str()),SE_FILE_OBJECT,DACL_SECURITY_INFORMATION,nullptr,nullptr,&acl,nullptr,&descriptor.p)==ERROR_SUCCESS);
        SECURITY_DESCRIPTOR_CONTROL control{};DWORD revision=0;check(GetSecurityDescriptorControl(descriptor.p,&control,&revision)&&bool(control&SE_DACL_PROTECTED));
        check(acl&&acl->AceCount==2);
        WinHandle token;check(OpenProcessToken(GetCurrentProcess(),TOKEN_QUERY,&token.h)!=0);DWORD n=0;GetTokenInformation(token.h,TokenUser,nullptr,0,&n);std::vector<std::byte> user(n);check(GetTokenInformation(token.h,TokenUser,user.data(),n,&n)!=0);
        BYTE system[SECURITY_MAX_SID_SIZE];DWORD system_size=sizeof system;check(CreateWellKnownSid(WinLocalSystemSid,nullptr,system,&system_size)!=0);
        bool found_user=false,found_system=false;for(unsigned i=0;i<acl->AceCount;++i){void* raw=nullptr;check(GetAce(acl,i,&raw)!=0);auto* ace=static_cast<ACCESS_ALLOWED_ACE*>(raw);check(ace->Header.AceType==ACCESS_ALLOWED_ACE_TYPE&&ace->Mask==FILE_ALL_ACCESS);auto* sid=&ace->SidStart;
            bool is_user=EqualSid(sid,reinterpret_cast<TOKEN_USER*>(user.data())->User.Sid)!=0,is_system=EqualSid(sid,system)!=0;check(is_user||is_system);found_user|=is_user;found_system|=is_system;}
        check(found_user&&found_system);
    }
    // Creation and last-write changes each invalidate an otherwise valid hit.
    for(bool creation:{true,false}){
        HANDLE h=CreateFileW(path.c_str(),FILE_READ_ATTRIBUTES|FILE_WRITE_ATTRIBUTES,FILE_SHARE_READ,nullptr,OPEN_EXISTING,0,nullptr);check(h!=INVALID_HANDLE_VALUE);
        FILETIME created{},written{};check(GetFileTime(h,&created,nullptr,&written)!=0);
        auto advance=[](FILETIME& time){ULARGE_INTEGER value{};value.LowPart=time.dwLowDateTime;value.HighPart=time.dwHighDateTime;value.QuadPart+=10000000;time.dwLowDateTime=value.LowPart;time.dwHighDateTime=value.HighPart;};
        if(creation)advance(created);else advance(written);
        check(SetFileTime(h,&created,nullptr,&written)!=0);CloseHandle(h);
        VerifiedFile changed_time(path,input.size());check(!changed_time.verify(digest,cache));check(changed_time.verify(digest,cache)==journal);
    }
    // A valid-format forgery preserves the exact metadata key and expected
    // digest. Both a missing MAC and a wrong 64-hex MAC MUST hash, then repair.
    for(bool missing:{true,false}){
        for(auto& entry:std::filesystem::directory_iterator(cache))if(entry.path().extension()==".sha256"){
            std::ifstream saved(entry.path());std::string metadata,sha;std::getline(saved,metadata);std::getline(saved,sha);saved.close();
            std::ofstream forged(entry.path(),std::ios::trunc);forged<<metadata<<'\n'<<sha<<'\n';if(!missing)forged<<std::string(64,'0')<<'\n';
        }
        VerifiedFile forged_record(path,input.size());check(!forged_record.verify(digest,cache));check(forged_record.verify(digest,cache)==journal);
    }
    // Lost/invalid DPAPI authority rotates the key; even valid old MACs miss.
    {std::ofstream corrupt(cache/"user-key.dpapi",std::ios::trunc);corrupt<<"invalid protected key";}
    {VerifiedFile rotated(path,input.size());check(!rotated.verify(digest,cache));check(!rotated.verify(digest,cache));check(rotated.verify(digest,cache)==journal);}
    // A different, valid current-user DPAPI key also invalidates all old MACs.
    auto other_cache=root/"other-authority";hash_authority(other_cache);
    std::filesystem::copy_file(other_cache/"user-key.dpapi",cache/"user-key.dpapi",std::filesystem::copy_options::overwrite_existing);
    {VerifiedFile mismatched_key(path,input.size());check(!mismatched_key.verify(digest,cache));check(mismatched_key.verify(digest,cache)==journal);}
    // Preserve the ORIGINAL authenticated record. Rewrite bytes and restore ALL
    // FILE_BASIC_INFO timestamps, including ChangeTime, on the writer handle.
    std::vector<std::pair<std::filesystem::path,std::string>> records;
    for(auto& entry:std::filesystem::directory_iterator(cache))if(entry.path().extension()==".sha256"){
        std::ifstream f(entry.path(),std::ios::binary);records.emplace_back(entry.path(),std::string(std::istreambuf_iterator<char>(f),{}));}
    if(journal)check(!records.empty());
    FILE_BASIC_INFO original_times{};
    {WinHandle h;h.h=CreateFileW(path.c_str(),GENERIC_WRITE|FILE_READ_ATTRIBUTES|FILE_WRITE_ATTRIBUTES,FILE_SHARE_READ,nullptr,OPEN_EXISTING,0,nullptr);
        check(h.h!=INVALID_HANDLE_VALUE&&GetFileInformationByHandleEx(h.h,FileBasicInfo,&original_times,sizeof original_times));
        input[0]^=std::byte{1};DWORD wrote=0;check(WriteFile(h.h,input.data(),1,&wrote,nullptr)&&wrote==1&&FlushFileBuffers(h.h));
        check(SetFileInformationByHandle(h.h,FileBasicInfo,&original_times,sizeof original_times)!=0);}
    {WinHandle h;h.h=CreateFileW(path.c_str(),FILE_READ_ATTRIBUTES,FILE_SHARE_READ,nullptr,OPEN_EXISTING,0,nullptr);FILE_BASIC_INFO restored{};
        check(GetFileInformationByHandleEx(h.h,FileBasicInfo,&restored,sizeof restored));
        check(original_times.CreationTime.QuadPart==restored.CreationTime.QuadPart&&original_times.LastAccessTime.QuadPart==restored.LastAccessTime.QuadPart&&
              original_times.LastWriteTime.QuadPart==restored.LastWriteTime.QuadPart&&original_times.ChangeTime.QuadPart==restored.ChangeTime.QuadPart);}
    for(auto& [record,bytes]:records){std::ifstream f(record,std::ios::binary);check(std::string(std::istreambuf_iterator<char>(f),{})==bytes);}
    {VerifiedFile changed(path,input.size());auto result=changed.compute_or_recall(cache);
        check(!result.recalled&&result.sha256!=digest&&changed.bytes_read()==input.size());
        bool rejected=false;try{changed.verify(digest,cache);}catch(...){rejected=true;}check(rejected);
        digest=strix::model_content_identity_hex(strix::sha256_bytes_fast(input));check(result.sha256==digest);check(changed.verify(digest,cache)==journal);}
    if(journal)std::cout<<"PASS retained authenticated record, changed bytes, all four basic timestamps restored: USN forces re-hash\n";
    // Every authority operation can fail independently. Existing authenticated
    // records must be ignored; correct content succeeds, wrong content rejects,
    // and no record is published (fault injection is absent in production).
    // Isolate authority policy from journal availability, so every failure path
    // is exercised even on a volume where the real replay gate must be skipped.
    journal_test_identity="authority-fixture-journal 1";
    for(const char* stage:{"directory","ACL","load","RNG","DPAPI","persist","publish"}){
        auto failure=root/(std::string("failure-")+stage);hash_authority(failure);
        {VerifiedFile f(path,input.size());check(!f.verify(digest,failure));check(f.verify(digest,failure));}
        std::vector<std::pair<std::filesystem::path,std::string>> saved;
        for(auto& entry:std::filesystem::directory_iterator(failure))if(entry.path().extension()==".sha256"){
            std::ifstream f(entry.path(),std::ios::binary);saved.emplace_back(entry.path(),std::string(std::istreambuf_iterator<char>(f),{}));}
        if(std::string(stage)=="RNG"||std::string(stage)=="DPAPI"||std::string(stage)=="persist"||std::string(stage)=="publish")std::filesystem::remove(failure/"user-key.dpapi");
        authority_fault=[&](const char* at){if(std::string(at)==stage)throw std::runtime_error("injected authority failure");};
        bool failed=false;try{hash_authority(failure);}catch(...){failed=true;}check(failed);
        {VerifiedFile f(path,input.size());auto result=f.compute_or_recall(failure);check(!result.recalled&&result.sha256==digest&&f.bytes_read()==input.size());
            check(!f.verify(digest,failure));bool rejected=false;try{f.verify(std::string(64,'0'),failure);}catch(...){rejected=true;}check(rejected);}
        authority_fault={};
        unsigned count=0;for(auto& entry:std::filesystem::directory_iterator(failure))if(entry.path().extension()==".sha256")++count;check(count==saved.size());
        for(auto& [record,bytes]:saved){std::ifstream f(record,std::ios::binary);check(std::string(std::istreambuf_iterator<char>(f),{})==bytes);}
    }
    // Actual filesystem failures, including an unusable directory and locked key.
    {auto blocked=root/"not-a-directory";std::ofstream(blocked)<<"blocked";VerifiedFile f(path,input.size());check(!f.verify(digest,blocked));}
    {WinHandle locked;locked.h=CreateFileW((cache/"user-key.dpapi").c_str(),GENERIC_READ,0,nullptr,OPEN_EXISTING,0,nullptr);check(locked.h!=INVALID_HANDLE_VALUE);
        VerifiedFile f(path,input.size());check(!f.verify(digest,cache));}
    journal_test_identity.reset();
    // All concurrent repairers must observe the same authority, including on
    // their second load. A later replacement would invalidate a child's MACs.
    for(unsigned round=0;round<4;++round){
        auto repair=root/("repair-"+std::to_string(round));hash_authority(repair);std::ofstream(repair/"user-key.dpapi",std::ios::trunc)<<"corrupt";
        auto event_name="Local\\glm-authority-test-"+std::to_string(GetCurrentProcessId())+"-"+std::to_string(round);
        WinHandle gate;gate.h=CreateEventA(nullptr,TRUE,FALSE,event_name.c_str());check(gate.h!=nullptr);
        std::vector<HANDLE> children;
        for(unsigned i=0;i<4;++i){
            std::string command="\""+std::filesystem::absolute(argv[0]).string()+"\" --repair-child \""+repair.string()+"\" "+event_name;
            STARTUPINFOA si{};si.cb=sizeof si;PROCESS_INFORMATION pi{};
            check(CreateProcessA(nullptr,command.data(),nullptr,nullptr,FALSE,IDLE_PRIORITY_CLASS|CREATE_NO_WINDOW,nullptr,nullptr,&si,&pi)!=0);
            CloseHandle(pi.hThread);children.push_back(pi.hProcess);
        }
        check(SetEvent(gate.h)!=0);
        for(auto child:children){check(WaitForSingleObject(child,15000)==WAIT_OBJECT_0);DWORD code=1;check(GetExitCodeProcess(child,&code)&&code==0);CloseHandle(child);}
        auto stable=hash_authority(repair);unsigned count=0;
        for(auto& entry:std::filesystem::directory_iterator(repair))if(entry.path().extension()==".key"){
            HashKey observed{};std::ifstream f(entry.path(),std::ios::binary);f.read(reinterpret_cast<char*>(observed.data()),observed.size());check(bool(f)&&observed==stable);++count;}
        check(count==4);
    }
    // Changing the file ID (replacement with same size/content) is a cold miss.
    std::filesystem::rename(path,root/"old.bin");write();
    {VerifiedFile replaced(path,input.size());check(!replaced.verify(digest,cache));}
    std::vector<std::unique_ptr<VerifiedFile>> files;
    std::vector<std::filesystem::path> paths;
    for(unsigned i=0;i<45;++i){auto fixture_path=root/("small-"+std::to_string(i));std::ofstream f(fixture_path,std::ios::binary);f.write("hello",5);f.close();paths.push_back(fixture_path);files.push_back(std::make_unique<VerifiedFile>(fixture_path,5));}
    const auto hello=strix::model_content_identity_hex(strix::sha256_bytes_fast(std::as_bytes(std::span("hello",5))));
    parallel_hash_files(files.size(),[&](std::size_t j){check(!files[j]->verify(hello,cache));});
    parallel_hash_files(files.size(),[&](std::size_t j){check(files[j]->verify(hello,cache)==journal);});files.clear();
    std::atomic<unsigned> jobs=0;parallel_hash_files(45,[&](std::size_t){++jobs;});check(jobs==45);
    bool propagated=false;try{parallel_hash_files(45,[](std::size_t){throw std::runtime_error("worker");});}catch(...){propagated=true;}check(propagated);
    // Metadata changes, truncation, bad records, and exact EOF all fail closed.
    std::filesystem::resize_file(path,0);
    bool rejected=false;try{VerifiedFile truncated(path,input.size());}catch(...){rejected=true;}check(rejected);
    {VerifiedFile empty(path,0);auto sha=strix::model_content_identity_hex(strix::sha256_bytes_fast({}));check(!empty.verify(sha,cache));check(empty.verify(sha,cache)==journal);}
    // Replace an ancestor pathname after verification. Runtime and integrity
    // consume this same parsed object, so an unpinned replacement is irrelevant.
    auto original=root/"manifest-parent",replacement=root/"manifest-replacement",alias=root/"manifest-alias";std::filesystem::create_directory(original);std::filesystem::create_directory(replacement);
    const std::string manifest="{\"runtime_marker\":\"verified\"}";
    {std::ofstream out(original/"manifest.json");out<<manifest;}
    {
        junction(alias,original);
        VerifiedManifest held(alias/"manifest.json",strix::model_content_identity_hex(strix::sha256_bytes_fast(std::as_bytes(std::span(manifest)))));
        check(RemoveDirectoryW(alias.c_str())!=0);junction(alias,replacement);
        {std::ofstream out(replacement/"manifest.json");out<<"{\"runtime_marker\":\"forged\"}";}
        check(held.parsed.find("runtime_marker")->as_string()=="verified");
        bool rejected=false;try{VerifiedManifest swapped(alias/"manifest.json",strix::model_content_identity_hex(strix::sha256_bytes_fast(std::as_bytes(std::span(manifest)))));}catch(...){rejected=true;}check(rejected);
        check(RemoveDirectoryW(alias.c_str())!=0);
    }
    ShardsFixture::run(root);
    std::filesystem::remove_all(root);
    std::cout<<"PASS aligned overlapped reads/tail, SHA integrity, restart hits, rewrite/replacement/size invalidation, deny-write/delete, worker failures\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
