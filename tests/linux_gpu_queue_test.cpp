#include "strix/linux_gpu_queue.hpp"
#include "strix/gpu_queue_session.hpp"
#include "gpu_queue_driver_bridge/bridge.h"
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>
#include <poll.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace strix::transport;
namespace {
template<class Function> void rejects(Function fn) { bool threw=false;try{fn();}catch(const std::exception&){threw=true;}assert(threw); }
// STRIX_TEST_GPU_QUEUE_LEAN=1 runs every case on the lean hot path (GpuQueueExpected::lean).
bool lean_mode() { const char* v=std::getenv("STRIX_TEST_GPU_QUEUE_LEAN");return v&&v[0]=='1'; }
struct Fixture {
    std::shared_ptr<void> driver;
    bool gpu_idle=true,change_identity=false,auto_complete=false;
    int mutate_event=0,submit_again=0,event_error=0;unsigned long mutate_cleanup=0;
    unsigned ioctls=0,owned_ioctls=0,events=0,identity_checks=0; bool corrupt_reset=false;
    std::unique_ptr<LinuxGpuQueue> queue;
    Fixture(unsigned depth=8,unsigned frames=32,bool reverse=false,bool lean=lean_mode()) {
        driver={queue_fixture_create(frames),queue_fixture_destroy};
        GpuQueueExpected expected;expected.local.fill(0x11);expected.peer.fill(0x22);expected.depth=depth;expected.ring_frames=frames;
        if(reverse){queue_fixture_reverse_endpoints(driver.get());std::swap(expected.local,expected.peer);}
        expected.lean=lean;expected.identity_interval=std::chrono::hours(1); // cases choose their own re-check interval
        const auto call=[this](unsigned long command,void* argument){
            ++ioctls;events+=command==SGU_IOC_EVENT;
            if(command==SGU_IOC_EVENT&&event_error){const int error=event_error;event_error=0;return GpuQueueSyscall{-1,error};}
            if(command==SGU_IOC_SUBMIT&&submit_again){--submit_again;return GpuQueueSyscall{-1,EAGAIN};}
            if(command==SGU_IOC_EVENT&&auto_complete)queue_fixture_complete_pending(driver.get());
            int result=queue_fixture_ioctl(driver.get(),command,argument);
            if(!result&&command==SGU_IOC_EVENT&&mutate_event){
                auto& e=*static_cast<sgu_event*>(argument);
                switch(mutate_event){case 1:++e.sequence;break;case 2:++e.frames[0].actual_bytes;break;
                    case 3:++e.frames[e.count-1].pool_offset;break;case 4:++e.fence.import_generation;break;
                    case 5:e.status=SGU_EVENT_FAILED;break;case 6:++e.first_frame;break;
                    case 7:e.frames[e.count].capacity=1;break;case 8:++e.handle;break;default:assert(false);}
                mutate_event=0;
            }
            if(!result&&command==SGU_IOC_RESET_GROUP&&corrupt_reset)static_cast<sgu_reset_group*>(argument)->members[0].fence.import_generation++;
            if(!result&&command==mutate_cleanup)static_cast<sgu_action*>(argument)->result=99;
            return GpuQueueSyscall{result<0?-1:result,result<0?-result:0};
        };
        GpuQueueIo io{driver,call,[this]{++identity_checks;if(change_identity)throw std::runtime_error("injected owner replacement");},
            [this]{return queue_fixture_fd(driver.get());},[this,call](unsigned long command,void* argument){++owned_ioctls;return call(command,argument);}};
        queue=std::make_unique<LinuxGpuQueue>(std::move(io),expected);
    }
    GpuQueuePool pool() {
        GpuQueuePool p;queue_fixture_pool(driver.get(),&p.tx,&p.rx);p.lifetime=driver;
        p.require_current=[this]{++identity_checks;if(change_identity)throw std::runtime_error("injected pool replacement");};
        p.gpu_quiescent=[this]{return gpu_idle;};return p;
    }
    void start(std::uint64_t epoch=1){queue->import_pool(pool(),epoch,101+epoch);queue->arm();}
    GpuQueueCompletion finish(const GpuQueueOperation& op){
        queue_fixture_complete(driver.get(),op.direction,op.slot);
        for(unsigned n=0;n<100;++n)if(auto e=queue->poll()){assert(e->operation==op);return *e;}
        assert(false);return {};
    }
    void consume(const GpuQueueOperation& op){if(op.direction)queue->gpu_consumed(op);else queue->peer_consumed(op);queue->retire(op);}
};
void concurrent() {
    Fixture f;f.start();std::vector<GpuQueueOperation> ops;
    for(unsigned dir=0;dir<2;++dir)for(unsigned slot=0;slot<8;++slot){
        auto op=f.queue->prepare(dir,slot,slot+1,4097,32);if(!dir)f.queue->gpu_produced(op);assert(f.queue->submit(op));ops.push_back(op);
        assert(f.queue->terminal_offset(slot)==(24+slot)*4096);
    }
    assert(queue_fixture_pending(f.driver.get())==48);
    // Complete a reversed/interleaved order. Lookup must use exact handle, not
    // the most recently submitted operation or one shared event per direction.
    for(unsigned n=0;n<8;++n)for(unsigned dir=0;dir<2;++dir){auto& op=ops[dir*8+7-n];f.finish(op);f.consume(op);}
    assert(queue_fixture_pending(f.driver.get())==0);
    const auto stats=f.queue->stats();assert(stats.submissions[0]==8&&stats.submissions[1]==8);
    assert(stats.payload_bytes[0]==8*4097&&stats.payload_bytes[1]==8*4097);
    assert(f.queue->release());
}
void reuse() {
    Fixture f;f.start();std::uint64_t seq=0;
    for(unsigned batch=0;batch<256;++batch){std::vector<GpuQueueOperation> ops;
        for(unsigned slot=0;slot<8;++slot){auto op=f.queue->prepare(1,slot,++seq,1+slot*2,32);assert(f.queue->submit(op));ops.push_back(op);}
        for(auto& op:ops){f.finish(op);f.consume(op);}
    }
    assert(f.queue->stats().completions[1]==2048);assert(f.queue->release());
}
void geometry() {
    for(unsigned depth=2;depth<=8;++depth){Fixture f(depth,256);f.start();
        assert(f.queue->slot_capacity()==((256-depth)/depth)*4096ULL);
        rejects([&]{f.queue->prepare(0,depth,1,1,32);});
        rejects([&]{f.queue->prepare(0,0,1,f.queue->slot_capacity()+1,32);});
        rejects([&]{f.queue->prepare(2,0,1,1,32);});
        auto op=f.queue->prepare(0,0,1,f.queue->slot_capacity(),4096);f.queue->gpu_produced(op);assert(f.queue->submit(op));f.finish(op);f.consume(op);
        rejects([&]{f.queue->prepare(0,0,1,1,32);});assert(f.queue->release());
    }
}
void partial_events() {
    Fixture f(2,256);f.start();auto op=f.queue->prepare(1,1,1,80*4096-1,32);assert(f.queue->submit(op));
    queue_fixture_complete(f.driver.get(),1,1);assert(!f.queue->poll());
    rejects([&]{f.queue->gpu_consumed(op);});auto e=f.queue->poll();assert(e&&e->operation==op);f.consume(op);assert(f.queue->release());
}
void corruption(int kind) {
    Fixture f;f.start();auto op=f.queue->prepare(1,0,1,4097,32);assert(f.queue->submit(op));
    queue_fixture_complete(f.driver.get(),1,0);f.mutate_event=kind;rejects([&]{static_cast<void>(f.queue->poll());});
    assert(f.queue->failed());rejects([&]{f.queue->gpu_consumed(op);});assert(f.queue->release());
}
void credits() {
    Fixture f;f.start();auto tx=f.queue->prepare(0,0,1,4097,32);auto rx=f.queue->prepare(1,0,1,4097,32);
    rejects([&]{static_cast<void>(f.queue->submit(tx));});f.queue->gpu_produced(tx);
    assert(f.queue->submit(tx)&&f.queue->submit(rx));
    rejects([&]{f.queue->retire(tx);});rejects([&]{f.queue->gpu_consumed(rx);});
    f.finish(tx);f.finish(rx);rejects([&]{f.queue->retire(rx);});rejects([&]{f.queue->peer_consumed(rx);});
    rejects([&]{f.queue->gpu_consumed(tx);});f.queue->gpu_consumed(rx);rejects([&]{f.queue->gpu_consumed(rx);});
    f.queue->retire(rx);rejects([&]{f.queue->retire(rx);});f.consume(tx);assert(f.queue->release());
}
void ordering() {
    Fixture f;f.start();auto a=f.queue->prepare(0,0,1,17,32),b=f.queue->prepare(0,1,2,17,32);
    f.queue->gpu_produced(a);f.queue->gpu_produced(b);
    f.submit_again=1;assert(!f.queue->submit(a));rejects([&]{static_cast<void>(f.queue->submit(b));});
    assert(f.queue->submit(a));assert(f.queue->submit(b));rejects([&]{static_cast<void>(f.queue->submit(a));});
    f.finish(a);f.finish(b);f.consume(a);f.consume(b);assert(f.queue->release());
}
void failed_dependency() {
    Fixture f;f.start();auto op=f.queue->prepare(0,0,1,17,32);f.queue->gpu_produced(op);queue_fixture_dependency(f.driver.get(),0,-EIO);
    rejects([&]{static_cast<void>(f.queue->submit(op));});assert(f.queue->failed());assert(!queue_fixture_pending(f.driver.get()));assert(f.queue->release());
}
void recovery() {
    Fixture f;f.start();auto old=f.queue->prepare(1,0,1,4097,32);assert(f.queue->submit(old));
    f.queue->invalidate();f.gpu_idle=false;assert(!f.queue->recover());f.gpu_idle=true;assert(f.queue->recover());
    rejects([&]{f.queue->import_pool(f.pool(),1,103);});f.start(2);
    auto next=f.queue->prepare(1,0,1,4097,32);assert(next.handle>old.handle);assert(next.import_generation>old.import_generation);
    rejects([&]{static_cast<void>(f.queue->submit(old));});assert(f.queue->submit(next));f.finish(next);f.consume(next);assert(f.queue->release());
}
void stop_retention() {
    Fixture f;f.start();auto op=f.queue->prepare(1,0,1,4097,32);assert(f.queue->submit(op));
    queue_fixture_inject_stop_failure(1);assert(!f.queue->release());queue_fixture_inject_stop_failure(0);
    f.gpu_idle=false;assert(!f.queue->release());f.gpu_idle=true;assert(f.queue->release());
}
void identity() {
    Fixture f;f.start();f.change_identity=true;rejects([&]{f.queue->prepare(0,0,1,1,32);});assert(!f.queue->release());
    f.change_identity=false;assert(f.queue->release());
}
// Lean: no FD/pool metadata check and no EVENT between session transitions; the next transition
// (here RELEASE) still sees a replaced owner, and operation identity is still enforced.
void lean() {
    Fixture f(8,32,false,true);f.start();
    const auto checks=f.identity_checks,ioctls=f.ioctls;
    for(unsigned n=0;n<4;++n)assert(!f.queue->poll());
    assert(f.ioctls==ioctls&&f.events==0);
    auto op=f.queue->prepare(1,0,1,4097,32);assert(f.queue->submit(op));
    assert(!f.queue->poll()&&f.events==1);
    const auto e=f.finish(op);assert(e.operation==op);
    assert(!f.queue->poll()&&f.events==2); // the completion EVENT only: complete, not yet retired
    f.consume(op);assert(!f.queue->poll()&&f.events==2);
    assert(f.identity_checks==checks&&f.owned_ioctls==f.ioctls-ioctls);
    auto foreign=op;++foreign.handle;rejects([&]{f.queue->gpu_consumed(foreign);});
    f.change_identity=true;assert(!f.queue->poll()); // nothing in flight: no check, no EVENT
    rejects([&]{static_cast<void>(f.queue->stats());});
    assert(!f.queue->release()&&f.queue->failed());f.change_identity=false;assert(f.queue->release());
}
// What one exchange costs around its ioctls: four empty polls, then prepare / submit / poll / completion / poll /
// consume / poll. Printed for both modes (an identity check is one fstat and two fcntl on the real descriptor).
void counts() {
    unsigned full_checks=0;
    for(const bool lean_path:{false,true}) {
        Fixture f(8,32,false,lean_path);f.start();
        const auto checks=f.identity_checks,ioctls=f.ioctls,events=f.events;
        for(unsigned n=0;n<4;++n)assert(!f.queue->poll());
        auto op=f.queue->prepare(1,0,1,4097,32);assert(f.queue->submit(op));
        static_cast<void>(f.queue->poll());const auto e=f.finish(op);assert(e.operation==op);
        static_cast<void>(f.queue->poll());f.consume(op);static_cast<void>(f.queue->poll());
        std::cout<<"COUNT lean="<<lean_path<<" identity_checks="<<f.identity_checks-checks<<" ioctls="<<f.ioctls-ioctls
                 <<" events="<<f.events-events<<std::endl;
        if(lean_path)assert(f.identity_checks==checks&&full_checks>0);else full_checks=f.identity_checks-checks;
        assert(f.queue->release());
    }
}
// Expired intervals check ownership; malformed successful replies still fail validation.
void lean_recheck() {
    {
        Fixture f(8,32,false,true);
        GpuQueueExpected expected;expected.local.fill(0x11);expected.peer.fill(0x22);expected.depth=8;expected.ring_frames=32;
        expected.lean=true;expected.identity_interval=std::chrono::milliseconds(0);
        const auto call=[&f](unsigned long command,void* argument){
            ++f.ioctls;int result=queue_fixture_ioctl(f.driver.get(),command,argument);
            return GpuQueueSyscall{result<0?-1:result,result<0?-result:0};
        };
        GpuQueueIo io{f.driver,call,[&f]{++f.identity_checks;if(f.change_identity)throw std::runtime_error("injected owner replacement");},
            [&f]{return queue_fixture_fd(f.driver.get());},call};
        f.queue=std::make_unique<LinuxGpuQueue>(std::move(io),expected);f.start();
        const auto checks=f.identity_checks;
        auto op=f.queue->prepare(1,0,1,4097,32);assert(f.identity_checks>checks); // interval 0: every hot operation
        f.change_identity=true;rejects([&]{static_cast<void>(f.queue->submit(op));});
        f.change_identity=false;assert(f.queue->release());
    }
    {
        Fixture f(8,32,false,true);f.start(); // interval one hour: successful malformed replies do not re-check
        auto op=f.queue->prepare(1,0,1,4097,32);assert(f.queue->submit(op));
        const auto checks=f.identity_checks;
        f.mutate_event=8;queue_fixture_complete(f.driver.get(),op.direction,op.slot); // a foreign reply: no ioctl error, no re-check
        rejects([&]{static_cast<void>(f.queue->poll());});assert(f.identity_checks==checks&&f.queue->failed());
        static_cast<void>(f.queue->release());
    }
}
void lean_ioctl_errors() {
    for(const bool replaced:{false,true}) {
        Fixture f(8,32,false,true);f.start(); // one hour: exercise only the error-triggered check
        auto op=f.queue->prepare(1,0,1,4097,32);assert(f.queue->submit(op));
        const auto checks=f.identity_checks,events=f.events;
        f.change_identity=replaced;f.event_error=EBADF;
        bool caught=false;
        try { static_cast<void>(f.queue->poll()); }
        catch(const std::system_error& error) {
            assert(!replaced);
            assert(error.code()==std::error_code(EBADF,std::generic_category()));
            caught=true;
        }
        catch(const std::runtime_error& error) {
            assert(replaced&&std::string(error.what())=="injected owner replacement");
            caught=true;
        }
        assert(caught&&f.identity_checks>checks&&f.queue->failed());
        assert(f.events==events+1&&f.event_error==0);
        f.change_identity=false;assert(f.queue->release());
    }
    for(const int error:{EAGAIN,EINTR}) {
        Fixture f(8,32,false,true);f.start();
        auto op=f.queue->prepare(1,0,1,4097,32);assert(f.queue->submit(op));
        const auto checks=f.identity_checks,events=f.events;
        f.change_identity=true;f.event_error=error;
        assert(!f.queue->poll());
        assert(f.identity_checks==checks&&!f.queue->failed());
        assert(f.events==events+1&&f.event_error==0);
        f.change_identity=false;f.finish(op);f.consume(op);assert(f.queue->release());
    }
}
void wire() {
    GpuQueuePeerFrame a;a.campaign.fill(1);a.local.fill(2);a.peer.fill(3);a.device_instance.fill(4);
    a.epoch=1;a.import_generation=2;a.pool_identity=3;a.first_sequence=1;a.payload_bytes=4097;
    a.depth=8;a.ring_frames=32;a.kind=gpu_queue_hello;a.directions=3;
    const auto bytes=encode_gpu_queue_peer_frame(a);assert(bytes.size()==160);assert(decode_gpu_queue_peer_frame(bytes)==a);
    auto b=a;std::swap(b.local,b.peer);b.device_instance.fill(5);b.pool_identity=8;b.import_generation=9;
    require_gpu_queue_peer(a,b);auto bad=b;bad.epoch++;rejects([&]{require_gpu_queue_peer(a,bad);});
    bad=b;bad.device_instance[0]++;rejects([&]{require_gpu_queue_peer(a,bad,&b);});
    auto encoded=bytes;encoded.back()=std::byte{1};rejects([&]{static_cast<void>(decode_gpu_queue_peer_frame(encoded));});
    for(std::size_t size=0;size<bytes.size();++size)rejects([&]{static_cast<void>(decode_gpu_queue_peer_frame(std::span(bytes).first(size)));});
    a.kind=b.kind=gpu_queue_consumed;b.completed_mask=255;require_gpu_queue_peer(a,b);
    b.completed_mask=127;rejects([&]{require_gpu_queue_peer(a,b);});
}
void session(const std::string& kind) {
    Fixture f;f.start();f.auto_complete=true;
    GpuQueueSessionPolicy p;p.campaign.fill(1);p.payload_bytes=4097;p.windows=3;p.directions=3;
    if(kind=="session_recovery")p.recovery_every_windows=1;
    if(kind=="session_rx")p.directions=2;
    if(kind=="session_tx")p.directions=1;
    std::vector<unsigned> phases;GpuQueuePeerFrame sent;unsigned produced=0,consumed=0;
    GpuQueueSessionIo io{true,[&](auto bytes){sent=decode_gpu_queue_peer_frame(bytes);phases.push_back(sent.kind);},[&]{
        auto peer=sent;std::swap(peer.local,peer.peer);peer.device_instance.fill(7);peer.import_generation+=7;peer.pool_identity+=7;
        peer.directions=((p.directions&1U)<<1)|((p.directions&2U)>>1);
        if(peer.kind==gpu_queue_consumed)peer.completed_mask=(p.directions&1)?255:0;
        if(kind=="session_bad_credit"&&peer.kind==gpu_queue_consumed)peer.completed_mask=0;
        return encode_gpu_queue_peer_frame(peer);
    }};
    GpuQueueSessionGpu gpu{[&](const auto&){++produced;},[&](const auto&){++consumed;return kind!="session_bad_gpu"||consumed!=2;}};
    const auto report=run_gpu_queue_session(*f.queue,f.pool(),p,io,gpu);
    assert(report.cleanup_released);
    if(kind=="session_bad_credit"||kind=="session_bad_gpu"){
        assert(!report.error.empty()&&report.completed_windows==0);
        if(kind=="session_bad_gpu")assert(std::find(phases.begin(),phases.end(),gpu_queue_consumed)==phases.end());
    }else{
        assert(report.error.empty()&&report.completed_windows==3);
        assert(report.completed_tx==((p.directions&1)?24:0));assert(report.completed_rx==((p.directions&2)?24:0));
        assert(produced==report.completed_tx&&consumed==report.completed_rx);
        assert(report.recoveries==(kind=="session_recovery"?2:0));assert(report.final_epoch==(kind=="session_recovery"?3:1));
    }
}

void ambiguous_cleanup(const std::string& name) {
    Fixture f;f.start();f.mutate_cleanup=name=="ambiguous_release"?SGU_IOC_RELEASE:SGU_IOC_RECOVER;
    assert(!(name=="ambiguous_release"?f.queue->release():f.queue->recover()));
    const auto before=f.ioctls;
    assert(!f.queue->release()&&!f.queue->recover());f.queue.reset();assert(f.ioctls==before);
    // Intentional lifetime quarantine: the process exits without pretending
    // an ambiguous release was reclaimed. This is not a leak-freedom test.
    std::cout<<"PASS "<<name<<" (intentional retention)\n"<<std::flush;std::_Exit(0);
}

void paired(const std::string& name) {
    int sockets[2];assert(socketpair(AF_UNIX,SOCK_SEQPACKET|SOCK_CLOEXEC,0,sockets)==0);
    const auto child=fork();assert(child>=0);const bool reverse=child==0;
    const int socket=sockets[reverse?1:0];close(sockets[reverse?0:1]);
    {
        Fixture f(8,32,reverse);f.start();f.auto_complete=true;
        GpuQueueSessionPolicy policy;policy.campaign.fill(1);policy.payload_bytes=4097;policy.windows=3;policy.directions=3;
        if(name=="paired_oneway")policy.directions=reverse?2:1;
        if(name=="paired_recovery"||name=="paired_controller"||name=="paired_network")policy.recovery_every_windows=1;
        if(name=="paired_controller"||name=="paired_network")policy.recovery_flags=gpu_queue_reset_controller;
        if(name=="paired_network"){policy.recovery_flags|=gpu_queue_reset_network;policy.independent_control=true;}
        const auto ready=[&](short events){struct pollfd fd{socket,events,0};if(poll(&fd,1,2000)!=1||!(fd.revents&events))throw std::runtime_error("bounded paired test socket timeout");};
        GpuQueueSessionIo io{!reverse,[&](auto bytes){ready(POLLOUT);assert(send(socket,bytes.data(),bytes.size(),MSG_NOSIGNAL)==static_cast<ssize_t>(bytes.size()));},[&]{
            ready(POLLIN);std::vector<std::byte> bytes(161);const auto count=recv(socket,bytes.data(),bytes.size(),0);
            if(count<=0)throw std::runtime_error("paired test peer closed");
            bytes.resize(static_cast<std::size_t>(count));return bytes;
        }};
        const auto result=run_gpu_queue_session(*f.queue,f.pool(),policy,io,{[](const auto&){},[](const auto&){return true;}});
        assert(result.cleanup_released&&result.error.empty()&&result.completed_windows==3);
        assert(result.recoveries==((name=="paired_recovery"||name=="paired_controller"||name=="paired_network")?2:0));
        assert(result.controller_resets==((name=="paired_controller"||name=="paired_network")?2:0));
    }
    close(socket);
    if(reverse)std::_Exit(0);
    int status=0;assert(waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==0);
}

void group_reset(const std::string& name) {
    Fixture f;f.start();std::array<LinuxGpuQueue*,1> group{f.queue.get()};
    if(name=="group_admission") {
        auto before=f.ioctls;auto result=LinuxGpuQueue::reset_controller_group(group);
        assert(!result.success&&!result.may_have_reset&&result.error==EBUSY&&before==f.ioctls);
        assert(f.queue->stop());before=f.ioctls;
        f.gpu_idle=false;result=LinuxGpuQueue::reset_controller_group(group);
        assert(!result.success&&!result.may_have_reset&&before==f.ioctls);f.gpu_idle=true;
        std::array<LinuxGpuQueue*,2> duplicate{f.queue.get(),f.queue.get()};
        result=LinuxGpuQueue::reset_controller_group(duplicate);
        assert(!result.success&&!result.may_have_reset&&result.error==EINVAL);
        result=LinuxGpuQueue::reset_controller_group({});assert(result.error==EINVAL);
    } else {
        assert(f.queue->stop());
        if(name=="group_failure")queue_fixture_inject_reset_failure(-EIO);
        if(name=="group_bad_reply")f.corrupt_reset=true;
        const auto result=LinuxGpuQueue::reset_controller_group(group,name=="group_network");
        assert(result.may_have_reset&&f.queue->failed());
        assert(result.success==(name!="group_failure"&&name!="group_bad_reply"));
        rejects([&]{f.queue->arm();});
        queue_fixture_inject_reset_failure(0);f.corrupt_reset=false;
        assert(f.queue->recover());rejects([&]{f.queue->import_pool(f.pool(),1,777);});f.start(2);
    }
    assert(f.queue->release());
}
void reset_wire() {
    GpuQueuePeerFrame a;a.campaign.fill(1);a.local.fill(2);a.peer.fill(3);a.device_instance.fill(4);
    a.epoch=1;a.import_generation=2;a.pool_identity=3;a.first_sequence=1;a.payload_bytes=4097;
    a.depth=8;a.ring_frames=32;a.kind=gpu_queue_hello;a.directions=3;a.recovery_flags=gpu_queue_reset_controller;
    assert(decode_gpu_queue_peer_frame(encode_gpu_queue_peer_frame(a))==a);
    auto b=a;std::swap(b.local,b.peer);require_gpu_queue_peer(a,b);
    b.recovery_flags=0;rejects([&]{require_gpu_queue_peer(a,b);});
    a.recovery_flags=gpu_queue_reset_network;rejects([&]{encode_gpu_queue_peer_frame(a);});
    a.recovery_flags=gpu_queue_reset_controller;a.kind=gpu_queue_reset_done;b=a;std::swap(b.local,b.peer);
    require_gpu_queue_peer(a,b);b.kind=gpu_queue_reset_failed;rejects([&]{require_gpu_queue_peer(a,b);});
    a.recovery_flags=0;rejects([&]{encode_gpu_queue_peer_frame(a);});
}

}
int main(int argc,char** argv) {
    assert(argc==2);std::string name=argv[1];
    if(name.starts_with("group_"))group_reset(name);else if(name=="reset_wire")reset_wire();
    else if(name=="concurrent")concurrent();else if(name=="reuse")reuse();else if(name=="geometry")geometry();
    else if(name=="partial_events")partial_events();else if(name=="credits")credits();else if(name=="ordering")ordering();
    else if(name=="failed_dependency")failed_dependency();else if(name=="recovery")recovery();else if(name=="stop_retention")stop_retention();
    else if(name.starts_with("ambiguous"))ambiguous_cleanup(name);
    else if(name.starts_with("paired"))paired(name);
    else if(name=="lean_ioctl_errors")lean_ioctl_errors();
    else if(name=="identity")identity();else if(name=="lean")lean();else if(name=="lean_recheck")lean_recheck();else if(name=="counts")counts();else if(name=="wire")wire();else if(name.starts_with("session"))session(name);
    else if(name.starts_with("corruption"))corruption(std::stoi(name.substr(10)));else assert(false);
    std::cout<<"PASS "<<name<<'\n';
}
