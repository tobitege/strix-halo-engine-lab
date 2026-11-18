#include "../../src/weight/native_runtime.hpp"
#include <atomic>
#include <cstring>
#include <cstdlib>
namespace {
using namespace strix::weight;
struct Api {
    using Error=int;
    struct Properties { int warpSize=32; std::size_t sharedMemPerMultiprocessor=65536; };
    struct EventData { void* stream=nullptr; };
    using Event=EventData*; using Stream=void*;
    static constexpr auto backend=Backend::cuda;
    static constexpr int Success=0,OutOfMemory=1,EventDisableTiming=1,MemcpyHostToDevice=2;
    inline static std::atomic<int> allocations=0,frees=0,events=0;
    inline static std::atomic<bool> complete_a=true,complete_b=false,fail_b=false,drain_b=false;
    inline static int a=0,b=0;
    inline static void* upload=nullptr;
    inline static std::vector<Byte> staged;
    inline static std::mutex mutex;
    inline static std::condition_variable wake;
    static std::string arch(const Properties&) { return "fake-arch"; }
    static int SetDevice(int) { return 0; }
    static int GetDeviceProperties(Properties* p,int) { *p={}; return 0; }
    static int MemGetInfo(std::size_t* free,std::size_t* total) { *free=*total=1024; return 0; }
    static int Malloc(void** p,std::size_t bytes) { *p=std::malloc(bytes); ++allocations; return *p?0:1; }
    static int Free(void* p) { std::free(p); ++frees; wake.notify_all(); return 0; }
    static int EventCreateWithFlags(Event* e,int) { *e=new EventData; ++events; return 0; }
    static int EventRecord(Event e,Stream s) { e->stream=s; return s==&b && fail_b ? 2 : 0; }
    static int StreamQuery(Stream s) { return s==&b ? (complete_b?0:2) : (complete_a?0:2); }
    static int EventQuery(Event e) { return StreamQuery(e->stream); }
    static int EventDestroy(Event e) { delete e; --events; wake.notify_all(); return 0; }
    static int StreamWaitEvent(Stream,Event,int) { return 0; }
    // Simulate a pageable copy that returns after staging, before device DMA.
    static int Memcpy(void* dst,const void* src,std::size_t bytes,int) {
        upload=dst; const auto* first=static_cast<const Byte*>(src); staged.assign(first,first+bytes); return 0;
    }
    static int StreamSynchronize(Stream s) {
        if (s==&b && drain_b) complete_b=true;
        if (!s && upload) { std::memcpy(upload,staged.data(),staged.size()); upload=nullptr; }
        return StreamQuery(s);
    }
    static int EventSynchronize(Event e) { return StreamSynchronize(e->stream); }
    static const char* GetErrorString(int) { return "fake device error"; }
    static bool reclaimed(int count) {
        std::unique_lock lock(mutex);
        return wake.wait_for(lock,std::chrono::seconds(2),[&] { return frees==count && events==0; });
    }
};
struct Memory final : Source {
    std::array<Byte,16> bytes{Byte{42}};
    Status read(ByteRange r,std::span<Byte> out) const override {
        if (r.offset>bytes.size() || r.bytes>bytes.size()-r.offset || out.size()!=r.bytes) return {Error::malformed,"range"};
        std::memcpy(out.data(),bytes.data()+r.offset,out.size()); return {Error::ok,{}};
    }
    std::uint64_t size() const noexcept override { return bytes.size(); }
    Digest256 identity() const override { return digest_bytes(bytes); }
};
}
bool wfmt_runtime_test() {
    using Runtime=detail::NativeRuntime<Api>;
    ExecutionContext ca; Hardware hardware;
    if (!Runtime::bind(&Api::a,0,ca,hardware) || ca.backend!=Backend::cuda || hardware.arch!="fake-arch") return false;
    if (hardware.compiler_id.empty() || hardware.memory_bytes_per_second!=0) return false;
    auto cb=ca; cb.stream=&Api::b;
    auto source=std::make_shared<Memory>();
    WeightView view{}; view.tensor=1; view.storage={"plain.f32",1}; view.shape={{1,4,0,0},2}; view.payload={source,{0,16}};
    Plan plan{}; plan.selected.storage=view.storage; plan.selected.arithmetic={"cpu-q8a8-tree32",1};
    plan.selected.image={"image.raw",1}; plan.selected.backend=Backend::cuda; plan.selected.strategy=Strategy::native;
    plan.selected.resources.image_bytes=16;
    ImageKey key{}; if (!make_image_key(view,plan,"fake-arch","test@1",key)) return false;
    {
        ImageCache cache(16); ImageLease a,b;
        if (!acquire_image(cache,key,plan,view,ca,a)) return false;
        // Readiness on nonblocking A must follow the actual default-stream upload.
        if (Api::upload || std::memcmp(a.data(),source->bytes.data(),16)!=0) return false;
        if (!acquire_image(cache,key,plan,view,cb,b)) return false;
        Api::fail_b=true;
        if (record_last_use(b,cb)) return false;
    }
    if (Api::frees!=0) return false; // A is complete, B is still reading.
    Api::complete_b=true;
    if (!Api::reclaimed(1)) return false; // No allocation, retirement or manual collection.
    Api::fail_b=false; Api::complete_b=false;
    {
        ImageCache cache(16); ImageLease b;
        if (!acquire_image(cache,key,plan,view,cb,b)) return false;
    }
    if (Api::frees!=1) return false;
    Api::complete_b=true;
    if (!Api::reclaimed(2)) return false; // Also covers ordinary final-use events.
    Api::complete_b=false;
    {
        ImageCache cache(16); ImageLease b;
        if (!acquire_image(cache,key,plan,view,cb,b)) return false;
    }
    Api::drain_b=true; // Only the explicit shutdown drain can complete this event.
    Runtime::shutdown();
    return Api::allocations==3 && Api::frees==3 && Api::events==0;
}
