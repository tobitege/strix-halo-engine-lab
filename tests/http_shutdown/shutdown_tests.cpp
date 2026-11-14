#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif
#include "strix/serve/http_server.hpp"
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <condition_variable>
using namespace strix::serve;
using Clock = std::chrono::steady_clock;
#define CHECK(x) do { if (!(x)) throw std::runtime_error("http shutdown check: " #x); } while(false)
struct Peer {
#ifdef _WIN32
    SOCKET fd = INVALID_SOCKET;
    void close_fd() { if(fd != INVALID_SOCKET)closesocket(fd);fd=INVALID_SOCKET; }
#else
    int fd = -1;
    void close_fd() { if(fd >= 0)close(fd);fd=-1; }
#endif
    ~Peer(){close_fd();}
    Peer(const Peer&)=delete;Peer& operator=(const Peer&)=delete;
    explicit Peer(unsigned port, bool small_buffer=false) {
        try {
        fd=socket(AF_INET,SOCK_STREAM,0);
        if(small_buffer){int bytes=4096;CHECK(setsockopt(fd,SOL_SOCKET,SO_RCVBUF,reinterpret_cast<const char*>(&bytes),sizeof(bytes))==0);}
        sockaddr_in a{};a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(INADDR_LOOPBACK);a.sin_port=htons(static_cast<unsigned short>(port));
        CHECK(connect(fd,reinterpret_cast<sockaddr*>(&a),sizeof(a))==0);
        }catch(...){close_fd();throw;}
    }
    void send_bytes(std::string_view s) { CHECK(::send(fd,s.data(),static_cast<int>(s.size()),0)==static_cast<int>(s.size())); }
    std::string response(std::string_view tail="ok") {
        std::string out;char b[1024];
        while(out.find("\r\n\r\n")==std::string::npos || !out.ends_with(tail)) {
            auto n=recv(fd,b,sizeof(b),0);CHECK(n>0);out.append(b,static_cast<std::size_t>(n));
        }
        return out;
    }
};
template<class F> void until(F f) {
    auto end=Clock::now()+std::chrono::seconds(3);
    while(!f() && Clock::now()<end) std::this_thread::yield();
    CHECK(f());
}
int main(int argc,char** argv) try {
    CHECK(argc==2);std::string mode=argv[1];
    if(mode=="--check-sentinel") {CHECK(false);}
    HttpServerConfig c;c.port=0; // All production read/write deadlines retained.
    if(mode=="deadline")c.request_timeout_seconds=1; // Separate deadline coverage, never default-reader evidence.
    if(mode=="refusal") c.max_connections=1;
    std::promise<void> writing;auto write_phase=writing.get_future();
    std::atomic<unsigned> handlers{0},returned{0},writes{0};
    auto server=std::make_unique<HttpServer>(c,[&](const auto&,auto& w) {
        ++handlers;
        if(mode=="write") {
            CHECK(w.begin_stream(200,"text/plain"));
            writing.set_value();
            const std::string block(1U<<20U,'x');
            // No voluntary peer reads; this exceeds socket buffering.
            for(unsigned i=0;i<256;++i){++writes;if(!w.write(block))break;}
        } else if(mode=="sse") {CHECK(w.begin_stream(200,"text/event-stream"));CHECK(w.write("data: ok\n\n"));w.end_stream();}
        else w.send(200,"text/plain","ok");
        ++returned;
    });
    server->listen();auto port=server->port();
    auto accepting=std::async(std::launch::async,[&] {server->serve();std::cerr<<"phase accept-exit\n";});
    struct StopOnExit { std::unique_ptr<HttpServer>& s; ~StopOnExit(){if(s)s->stop();} } cleanup{server};
    Peer p(port,mode=="write");
    until([&]{return server->active_connections()==1;});
    std::cerr<<"phase accepted mode="<<mode<<" deadlines="<<c.idle_timeout_seconds<<"/"<<c.request_timeout_seconds<<"/"<<c.write_timeout_seconds<<"\n";
    if(mode=="deadline") {
        auto began=Clock::now();p.send_bytes("GET / HTTP/1.1\r\nHost:");
        auto bytes=p.response("}}");CHECK(bytes.starts_with("HTTP/1.1 408 Request Timeout"));
        CHECK(Clock::now()-began>=std::chrono::seconds(1));
    } else if(mode=="sse") {
        p.send_bytes("GET / HTTP/1.1\r\nHost: localhost\r\n\r\n");
        CHECK(p.response("0\r\n\r\n")=="HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: keep-alive\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\nX-Accel-Buffering: no\r\n\r\na\r\ndata: ok\n\n\r\n0\r\n\r\n");
    } else if(mode=="header") p.send_bytes("GET / HTTP/1.1\r\nHost:");
    else if(mode=="body") p.send_bytes("POST / HTTP/1.1\r\nContent-Length: 50\r\n\r\na");
    else if(mode=="keepalive" || mode=="bytes") {
        p.send_bytes("GET / HTTP/1.1\r\nHost: localhost\r\n\r\n");
        const auto bytes=p.response();
        CHECK(bytes=="HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: keep-alive\r\nContent-Length: 2\r\n\r\nok");
        until([&]{return returned==1;});std::cerr<<"phase response-received\n";
    } else if(mode=="write") {
        p.send_bytes("GET / HTTP/1.1\r\nHost: localhost\r\n\r\n");
        CHECK(write_phase.wait_for(std::chrono::seconds(3))==std::future_status::ready);
        until([&]{return writes>=2;});CHECK(returned==0);
        std::cerr<<"phase writing peer-not-reading\n";
    } else if(mode=="refusal") {
        Peer refused(port);char data[256];auto n=recv(refused.fd,data,sizeof(data),0);CHECK(n>0);
        CHECK(std::string(data,static_cast<std::size_t>(n)).starts_with("HTTP/1.1 503"));
    }
    // The peer remains alive through stop, accept join, drain and destruction.
    auto begin=Clock::now();std::cerr<<"phase stop-enter active="<<server->active_connections()<<'\n';
    if(mode=="accept") {
        auto arriving=std::async(std::launch::async,[&]{for(unsigned i=0;i<100;++i){try{Peer race(port);}catch(...){break;}}});
        server->stop();arriving.get();
    } else if(mode=="repeat") {
        auto a=std::async(std::launch::async,[&]{for(unsigned i=0;i<100;++i)server->stop();});
        for(unsigned i=0;i<100;++i)server->stop();a.get();
    } else server->stop();
    if(mode=="stop-refusal") {
        bool refused=false;try{Peer after(port);}catch(const std::runtime_error&){refused=true;}
        CHECK(refused);
    }
    std::cerr<<"phase stop-return active="<<server->active_connections()<<'\n';
    CHECK(accepting.wait_for(std::chrono::seconds(3))==std::future_status::ready);accepting.get();
    until([&]{return server->active_connections()==0;});
    CHECK(handlers==returned);server.reset();
    CHECK(Clock::now()-begin<std::chrono::seconds(3));
    std::cerr<<"phase destroyed peer-still-open active=0 handlers="<<handlers<<'\n';return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
