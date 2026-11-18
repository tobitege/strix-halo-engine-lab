#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "glm/rtxpf/import/glm_import_socket.hpp"
#include <future>
#include <deque>

namespace strix::glm::rtxpf::producer {
namespace wire = strix::glm::import;
// Wire v1 is shared verbatim with the node. No compression negotiation exists.
class Sender {
    wire::SocketOwner socket_;
    wire::Bytes key_, nonce_;
    std::uint64_t sequence_=0, raw_=0, wire_=32;
    std::vector<wire::Digest> digests_;
    std::vector<wire::Component> layout_;
    std::size_t component_=0;
    std::uint64_t row_=0;
    bool sealed_=false;
public:
    Sender(const std::string& endpoint,const wire::Bytes& key,const wire::Begin& begin):key_(key),nonce_(32),layout_(wire::layout(begin)) {
        wire::require(begin.frontier+1==begin.total,"sender requires P=N-1");
        wire::require(begin.producer=="glm-pf-rtx-native-v1"||begin.producer=="glm-pf-node-replay-v1","wire v1 producer identity unsupported");
        const auto colon=endpoint.rfind(':');wire::require(colon!=std::string::npos,"numeric IPv4:port required");
        addrinfo hints{};hints.ai_family=AF_INET;hints.ai_socktype=SOCK_STREAM;hints.ai_flags=AI_NUMERICHOST|AI_NUMERICSERV;
        addrinfo* addr=nullptr;
        wire::require(getaddrinfo(endpoint.substr(0,colon).c_str(),endpoint.substr(colon+1).c_str(),&hints,&addr)==0,"numeric endpoint required");
        std::unique_ptr<addrinfo,decltype(&freeaddrinfo)> guard(addr,freeaddrinfo);
        socket_.fd=::socket(AF_INET,SOCK_STREAM,0);wire::require(socket_.fd!=wire::bad_socket,"sender socket");
#ifdef _WIN32
        DWORD timeout=120000;
#else
        timeval timeout{120,0};
#endif
        wire::require(setsockopt(socket_.fd,SOL_SOCKET,SO_RCVTIMEO,reinterpret_cast<const char*>(&timeout),sizeof timeout)==0&&setsockopt(socket_.fd,SOL_SOCKET,SO_SNDTIMEO,reinterpret_cast<const char*>(&timeout),sizeof timeout)==0,"sender timeout");
        // Numeric connect; receiver challenge/authentication remains mandatory.
        wire::require(connect(socket_.fd,addr->ai_addr,static_cast<int>(addr->ai_addrlen))==0,"sender connect");
        wire::io(socket_.fd,nonce_,false);packet(begin.encode(),0);
    }
    void packet(const wire::Bytes& body,unsigned expected) {
        wire::require(body.size()>=4&&body.size()<=8388608,"packet extent");
        wire::Bytes size;wire::number(size,body.size(),4);
        auto authenticated=nonce_;wire::number(authenticated,sequence_++);wire::append(authenticated,size);wire::append(authenticated,body);
        wire::Bytes packet=size;wire::append(packet,body);wire::digest(packet,wire::hmac(key_,authenticated));
        wire::io(socket_.fd,packet,true);wire::Bytes ack(1);wire::io(socket_.fd,ack,false);
        wire_+=packet.size()+1;wire::require(std::to_integer<unsigned>(ack[0])==expected,"sender unexpected ACK");
    }
    void frame(unsigned layer,unsigned id,std::span<const std::byte> bytes) {
        wire::require(!sealed_&&component_<layout_.size(),"extra component");
        auto c=layout_[component_];wire::require(c.layer==layer&&c.id==id&&bytes.size()==c.rows*c.stride,"component order/extent");
        for(row_=0;row_<c.rows;) {
            auto rows=std::min(c.rows-row_,std::max<std::uint64_t>(1,1048576/c.stride));
            auto payload=bytes.subspan(static_cast<std::size_t>(row_*c.stride),static_cast<std::size_t>(rows*c.stride));
            wire::Bytes b;wire::number(b,2,4);wire::number(b,layer,4);wire::number(b,id,4);wire::number(b,row_);wire::number(b,rows);wire::number(b,payload.size());wire::digest(b,wire::hash(payload));wire::append(b,payload);
            wire::require(digests_.size()<32768,"too many frames");packet(b,0);digests_.push_back(wire::hash(b));raw_+=payload.size();row_+=rows;
        }
        ++component_;
    }
    void commit() {
        wire::require(!sealed_&&component_==layout_.size(),"incomplete sender manifest");
        wire::Bytes b;wire::number(b,3,4);wire::number(b,digests_.size());wire::number(b,raw_);for(auto d:digests_)wire::digest(b,d);packet(b,1);sealed_=true;
    }
    void abort() noexcept {
        if(sealed_||socket_.fd==wire::bad_socket)return;
        try {wire::Bytes b;wire::number(b,4,4);packet(b,0);}catch(...){}
        wire::close_socket(socket_.fd);socket_.fd=wire::bad_socket;
    }
    ~Sender(){abort();}
    std::uint64_t raw_bytes()const{return raw_;}
    std::uint64_t wire_bytes()const{return wire_;}
};
struct SocketRuntime {
#ifdef _WIN32
    SocketRuntime(){WSADATA w{};wire::require(WSAStartup(MAKEWORD(2,2),&w)==0,"WSAStartup");}
    ~SocketRuntime(){WSACleanup();}
#endif
};
}
