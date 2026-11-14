#include "strix/session_state_cache.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/unbuffered_file.hpp"
#include "strix/session/checkpoint_io.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <system_error>
#include <sys/stat.h>
#ifndef _WIN32
extern char** environ;
#else
extern char** _environ;
#endif

namespace strix {
namespace {
using Clock = std::chrono::steady_clock;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
struct Writer {
    std::vector<std::byte> b;
    void u64(std::uint64_t x) { for (unsigned s=0;s<64;s+=8) b.push_back(static_cast<std::byte>((x>>s)&255)); }
    void raw(std::span<const std::byte> x) { b.insert(b.end(),x.begin(),x.end()); }
    void hash(const ModelContentIdentity& h) { raw(std::as_bytes(std::span(h.bytes))); }
    void str(std::string_view s) { u64(s.size()); raw(std::as_bytes(std::span(s.data(),s.size()))); }
};
struct Reader {
    std::span<const std::byte> b;
    std::size_t at=0;
    std::span<const std::byte> take(std::uint64_t n) {
        require(n<=b.size()-at,"session checkpoint truncated or out of bounds");
        return session::checkpoint_io::take(b,at,n);
    }
    std::uint64_t u64() { auto v=take(8); std::uint64_t x=0; for(unsigned i=0;i<8;++i) x|=std::uint64_t(std::to_integer<unsigned char>(v[i]))<<(8*i); return x; }
    ModelContentIdentity hash() { ModelContentIdentity h; auto s=take(32); std::memcpy(h.bytes.data(),s.data(),32); return h; }
    std::string str() { auto s=take(u64()); return {reinterpret_cast<const char*>(s.data()),s.size()}; }
};
void compat(Writer& w,const SessionCompatibility& c) {
    require(c.family=="qwen" && c.format_version==1 && c.record.empty(),"session schema mismatch");
    require(c.model.valid() && c.executable.valid(),"session requires full model and executable hashes");
    w.str("strix-session-v1"); w.u64(SessionCompatibility::schema); w.hash(c.model); w.hash(c.executable);
    w.str(c.numerical_config); w.u64(c.mtp?1:0);
}
void frontier_valid(const SessionFrontier& f) {
    require(f.sampling=="greedy","session sampling state unsupported (requires greedy)");
    require(f.tokens.size()==f.consumed && f.accepted==f.consumed,"session frontier/history differs");
    require(f.predictor<=f.consumed,"session predictor beyond committed frontier");
}
double elapsed(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
}
void validate_session_envelope(const SessionCheckpoint& c) {
    require(!c.compatibility.family.empty() && c.compatibility.format_version!=0,"session schema mismatch");
    require(c.world!=0 && c.rank<c.world,"session unsupported rank geometry");
    require(c.frontier.tokens.size()==c.frontier.consumed && c.frontier.accepted==c.frontier.consumed,
        "session frontier/history differs");
    require(c.frontier.predictor<=c.frontier.consumed && c.frontier.drafter<=c.frontier.consumed,
        "session predictor beyond committed frontier");
    std::set<std::string> names;
    for(const auto& s:c.sections) {
        require(!s.name.empty() && names.insert(s.name).second,"session duplicate/empty section");
        require(!s.rank || *s.rank==c.rank,"session unsupported rank geometry");
        require(s.payload.bytes<=SIZE_MAX && (!s.payload.bytes || s.payload.data),"session null or oversized section");
        require(s.bytes.empty() || (!s.payload.data && !s.payload.bytes),"session duplicate section storage");
    }
}
ModelContentIdentity SessionCompatibility::digest() const {
    Writer w;
    if(family=="qwen") compat(w,*this); // unchanged legacy Qwen identity bytes
    else { w.str("strix-session-family-compatibility"); w.str(family); w.u64(format_version); w.str(record);
        w.hash(model); w.hash(executable); w.str(numerical_config); w.u64(mtp?1:0); }
    return sha256_bytes(w.b);
}
ModelContentIdentity session_cached_file_hash(const std::filesystem::path& path, std::optional<ModelContentIdentity> known_full_digest) {
    // Revalidate identity before and after the FULL content pass; never sample weights.
    static std::mutex mutex;
    static std::map<std::string,ModelContentIdentity> cache;
    std::lock_guard lock(mutex);
    const auto absolute=std::filesystem::canonical(path).string();
    auto identity=[&]() {
        struct stat st{};
        require(::stat(absolute.c_str(),&st)==0,"session cannot stat content file");
        Writer w; w.str(absolute); w.u64(static_cast<std::uint64_t>(st.st_size));
        w.u64(static_cast<std::uint64_t>(st.st_ino)); w.u64(static_cast<std::uint64_t>(st.st_dev));
        w.u64(static_cast<std::uint64_t>(st.st_mtime)); w.u64(static_cast<std::uint64_t>(st.st_ctime));
        w.u64(static_cast<std::uint64_t>(std::filesystem::last_write_time(absolute).time_since_epoch().count()));
#ifndef _WIN32
        w.u64(static_cast<std::uint64_t>(st.st_mtim.tv_nsec)); w.u64(static_cast<std::uint64_t>(st.st_ctim.tv_nsec));
#endif
        return model_content_identity_hex(sha256_bytes(w.b));
    };
    const auto key=identity();
    if(auto it=cache.find(key);it!=cache.end()) return it->second;
    auto h=known_full_digest ? *known_full_digest : sha256_file_content(absolute);
    require(identity()==key,"session content file changed during hashing");
    cache.emplace(key,h); return h;
}
static std::string numerical_config(std::string_view cli, const RotaryIdentity& rotary, bool family,
    std::span<const std::string> replace = {}) {
    std::vector<std::string> env;
#ifdef _WIN32
    auto p=_environ;
#else
    auto p=environ;
#endif
    for(;p && *p;++p) {
        const std::string_view value(*p);
        // Policy/provenance is metadata; applied arithmetic switches remain bound.
        if (value.starts_with("STRIX_QWEN_PROFILE_ALLOW_PENDING=") ||
            value.starts_with("STRIX_QWEN_PROFILE_OWNED_")) continue;
        // Model-description admission is a refusal-only metadata check, never arithmetic.
        if (value.starts_with("STRIX_QWEN_MODEL_DESCRIPTION")) continue;
        // Operational, per-process values: the GPU-turn wrapper's ids and the hash-cache directory.
        if (value.starts_with("STRIX_QPF2K_T16K=") || value.starts_with("STRIX_QPF2K_INDEX_BATCH=") ||
            value.starts_with("STRIX_EAGER_CODE_LOAD=") || value.starts_with("HIP_ENABLE_DEFERRED_LOADING=") ||
            value.starts_with("STRIX_QPF2K_EARLY_SETUP=") || value.starts_with("STRIX_QPF2K_HC_LOWBAND=") ||
            value.starts_with("STRIX_QPF2K_COMPACT_IMAGES=")) continue;
        // The last-layer cache-only QSA step writes the same persistent state as the full step.
        if (value.starts_with("STRIX_QSA_CACHE_ONLY=")) continue;
        // The final-block cone skips rows nothing reads; the host wait policy only changes how the host waits.
        if (value.starts_with("STRIX_PREFILL_FINAL_CONE=") || value.starts_with("STRIX_HOST_WAIT=")) continue;
        if (value.starts_with("GPU_TURN_") || value.starts_with("GPU_WINDOW=") || value.starts_with("STRIX_HASH_CACHE=")) continue;
        // Startup provenance of injected defaults (qpf2k::pre_injected_env); the values themselves stay bound.
        if (value.starts_with("STRIX_QPF2K_PRE_INJECTED=")) continue;
        // This observer does not alter the single tiled-MTP execution/flush order.
        // PROMPT_STATE, PREFIX_CACHE, SERIAL_LOGITS and FINAL_STATE remain in the match:
        // their consumers can flush pending windows or synchronize a device readback.
        if (family && value.starts_with("STRIX_PREFILL_LOGITS_DIGEST=")) continue;
        if (value.starts_with("STRIX_PREFIX_CACHE_INFLIGHT=") || value.starts_with("STRIX_PREFIX_CACHE_DISK=") ||
            value.starts_with("STRIX_PREFIX_CACHE_DURABLE=") ||
            value.starts_with("STRIX_PREFIX_FAIR_RECLAIM_IDLE=")) continue;
        // State-file I/O mode (page cache, read-ahead, streamed verification): same bytes.
        if (value.starts_with("STRIX_SESSION_DIRECT_IO=") || value.starts_with("STRIX_SESSION_READAHEAD=") ||
            value.starts_with("STRIX_SESSION_CHUNKED=")) continue;
        for (const auto prefix : {"STRIX_", "HIP_", "HSA_", "AMD_", "ROCR_", "GPU_"})
            if (value.starts_with(prefix)) {
                const auto it = std::find_if(replace.begin(), replace.end(), [&](const std::string& r) {
                    const auto equal = r.find('=');
                    return equal != std::string::npos && value.starts_with(std::string_view(r).substr(0, equal + 1U));
                });
                if (it != replace.end()) env.emplace_back(*it); else env.emplace_back(*p);
                break;
            }
    }
    std::sort(env.begin(),env.end());
    Writer w; w.str("qwen-effective-execution-v3");
    const auto& descriptor=rotary.descriptor();
    w.u64(descriptor.mode()==RotaryMode::yarn ? 1U : 0U);
    w.u64(descriptor.factor()); w.u64(descriptor.original_max); w.u64(descriptor.effective_max());
    const auto identity=rotary.canonical_bytes(); w.u64(identity.size()); w.raw(identity);
    w.str(cli); w.u64(env.size()); for(const auto& v:env) w.str(v);
    return {reinterpret_cast<const char*>(w.b.data()),w.b.size()};
}
std::string session_numerical_config(std::string_view cli, const RotaryIdentity& rotary) {
    return numerical_config(cli,rotary,false);
}
std::string session_numerical_config(std::string_view cli, const RotaryIdentity& rotary,
    std::span<const std::string> replace) {
    return numerical_config(cli,rotary,false,replace);
}
std::string session_family_numerical_config(std::string_view cli, const RotaryIdentity& rotary) {
    return numerical_config(cli,rotary,true);
}
std::string session_family_numerical_config(std::string_view cli, const RotaryIdentity& rotary,
    std::span<const std::string> replace) {
    return numerical_config(cli,rotary,true,replace);
}
PrefixCacheBuffer encode_session_checkpoint(const SessionCheckpoint& c,const PrefixCacheAllocator& alloc) {
    validate_session_envelope(c);
    require(c.compatibility.family=="qwen" && c.compatibility.format_version==1 && c.compatibility.record.empty() &&
        c.rank==0 && c.world==1 && c.id.empty() && c.generation.empty(),"session unsupported legacy schema");
    require(c.frontier.drafter==0 && c.frontier.control==0 && c.frontier.selection==0 && c.frontier.rng_draws==0 &&
        !c.frontier.drafter_ready && c.frontier.lineage.empty(),"session unsupported legacy frontier");
    frontier_valid(c.frontier);
    require(c.compatibility.mtp || c.frontier.predictor==0,"target-only session has a predictor frontier");
    Writer w; compat(w,c.compatibility); w.hash(c.compatibility.digest());
    const auto& f=c.frontier;
    w.u64(f.consumed); w.u64(f.emitted); w.u64(f.accepted); w.u64(f.predictor);
    w.u64(f.pending?*f.pending:std::uint64_t{UINT32_MAX}+1); w.str(f.sampling);
    w.u64(f.tokens.size()); for(auto t:f.tokens) w.u64(t);
    w.u64(c.sections.size());
    std::set<std::string> names;
    for(const auto& s:c.sections) {
        require(s.bytes.empty() && s.replicated && (!s.rank || *s.rank==0),"session unsupported rank geometry");
        require(!s.name.empty() && names.insert(s.name).second,"session duplicate/empty section");
        require(s.payload.bytes<=std::numeric_limits<std::size_t>::max() &&
            (!s.payload.bytes || s.payload.data!=nullptr),"session null or oversized section");
        w.str(s.name); w.u64(0); w.u64(0); // rank 0 / replicated
        w.u64(s.payload.bytes); w.hash(sha256_bytes_fast(s.payload.span()));
        w.u64(w.b.size() + 8U); // explicit canonical payload offset from file start
        w.raw(s.payload.span());
    }
    w.hash(sha256_bytes_fast(w.b)); // manifest root commits all metadata, section order and bytes
    auto out=alloc(w.b.size()); require(out.bytes>=w.b.size() && out.data,"session allocation failed");
    out.bytes=w.b.size(); std::memcpy(out.data.get(),w.b.data(),w.b.size()); return out;
}
namespace {
// streamed_root: SHA-256 of the body computed while the bytes were read into this blob.
SessionCheckpoint decode_checkpoint(const PrefixCacheBuffer& blob,const SessionCompatibility& expected,
    const ModelContentIdentity* streamed_root) {
    require(blob.bytes>=32 && blob.bytes<=std::numeric_limits<std::size_t>::max() && blob.data,"session envelope truncated");
    auto bytes=blob.span(); auto body=bytes.first(bytes.size()-32);
    Reader footer{bytes.last(32)};
    require((streamed_root ? *streamed_root : sha256_bytes_fast(body))==footer.hash(),"session manifest root mismatch");
    Reader r{body}; SessionCheckpoint c;
    require(r.str()=="strix-session-v1" && r.u64()==SessionCompatibility::schema,"session schema mismatch");
    c.compatibility.model=r.hash(); c.compatibility.executable=r.hash();
    c.compatibility.numerical_config=r.str(); const auto mtp=r.u64(); require(mtp<=1,"session invalid MTP flag"); c.compatibility.mtp=mtp!=0;
    require(expected.family=="qwen" && expected.format_version==1 && expected.record.empty(),"session schema mismatch");
    require(r.hash()==c.compatibility.digest() && c.compatibility.digest()==expected.digest(),"session compatibility mismatch");
    auto& f=c.frontier; f.consumed=r.u64(); f.emitted=r.u64(); f.accepted=r.u64(); f.predictor=r.u64();
    auto pending=r.u64(); require(pending<=std::uint64_t{UINT32_MAX}+1,"session invalid pending token");
    if(pending<=UINT32_MAX) f.pending=static_cast<std::uint32_t>(pending);
    f.sampling=r.str(); const auto tokens=r.u64(); require(tokens<=(body.size()-r.at)/8,"session truncated token history");
    for(std::uint64_t i=0;i<tokens;++i) { auto t=r.u64(); require(t<=UINT32_MAX,"session invalid token"); f.tokens.push_back(static_cast<std::uint32_t>(t)); }
    frontier_valid(f);
    require(c.compatibility.mtp || f.predictor==0,"target-only session has a predictor frontier");
    const auto count=r.u64(); require(count<=(body.size()-r.at)/64,"session invalid section count");
    std::set<std::string> names;
    for(std::uint64_t i=0;i<count;++i) {
        auto name=r.str(); require(!name.empty() && names.insert(name).second,"session duplicate/empty section");
        require(r.u64()==0 && r.u64()==0,"session unsupported rank geometry");
        const auto n=r.u64(); const auto digest=r.hash(); const auto offset_record=r.u64();
        require(offset_record==r.at,"session section offset differs"); auto payload=r.take(n);
        require(sha256_bytes_fast(payload)==digest,"session section SHA-256 mismatch");
        const auto offset=static_cast<std::size_t>(payload.data()-bytes.data());
        c.sections.push_back({std::move(name),{std::shared_ptr<std::byte>(blob.data,blob.data.get()+offset),n}});
    }
    require(r.at==body.size(),"session trailing data"); validate_session_envelope(c); return c;
}
}
SessionCheckpoint decode_session_checkpoint(const PrefixCacheBuffer& blob,const SessionCompatibility& expected) {
    return decode_checkpoint(blob,expected,nullptr);
}
std::optional<SessionCheckpoint> lookup_session_checkpoint(PrefixStateCache& store, const std::string& id,
    const SessionCompatibility& expected) {
    auto blob = store.resume_session(id);
    if (!blob.data) return std::nullopt;
    try { return decode_session_checkpoint(blob, expected); }
    catch (const std::exception&) { return std::nullopt; }
}
void publish_session_checkpoint(const std::filesystem::path& dir,const PrefixCacheBuffer& blob,SessionIoStats* stats) {
    const auto start=Clock::now();
    require(blob.data && blob.bytes>=32 && blob.bytes<=SIZE_MAX,"session publication requires a complete envelope");
    session::checkpoint_io::atomic_write(dir/"session.bin",blob.span(),
        {true,session::checkpoint_io::DirectIo::automatic,0600,".session-"});
    if(stats) { stats->bytes=blob.bytes; stats->write_fsync_ms=elapsed(start); }
}
PrefixCacheBuffer read_session_checkpoint(const std::filesystem::path& dir,const SessionCompatibility& expected,
    const PrefixCacheAllocator& alloc,std::uint64_t max_bytes,SessionIoStats* stats,SessionCheckpoint* decoded) {
    const auto start=Clock::now(); PrefixCacheBuffer blob;
    std::optional<Sha256Fast> root;
    if(state_file_io::unbuffered_io_enabled() && state_file_io::switch_enabled("STRIX_SESSION_CHUNKED")) root.emplace();
    session::checkpoint_io::read(dir/"session.bin",32,max_bytes,[&](std::uint64_t n) {
        blob=alloc(n); require(blob.data && blob.bytes>=n,"session allocation failed");
        blob.bytes=n; return blob.span();
    },session::checkpoint_io::DirectIo::automatic,[&](std::span<const std::byte> chunk,std::uint64_t at) {
        const auto body=blob.bytes-32;
        if(root && at<body) root->update(chunk.first(static_cast<std::size_t>(std::min<std::uint64_t>(chunk.size(),body-at))));
    });
    std::optional<ModelContentIdentity> digest; if(root) digest=root->finalize();
    auto checkpoint=decode_checkpoint(blob,expected,digest ? &*digest : nullptr);
    if(decoded) *decoded=std::move(checkpoint);
    if(stats) { stats->bytes=blob.bytes; stats->read_verify_ms=elapsed(start); } return blob;
}

void preflight_session_checkpoint(const std::filesystem::path& dir,const ModelContentIdentity& executable,
    std::string_view numerical_prefix,std::uint64_t max_bytes) {
    max_bytes = session_checkpoint_envelope_limit(max_bytes);
    // Read only the bounded compatibility header first: wrong executable/settings avoid payload I/O.
    std::ifstream in(dir/"session.bin",std::ios::binary|std::ios::ate);
    require(bool(in),"session checkpoint open failed");
    const auto size=in.tellg();
    require(size>=32 && static_cast<std::uint64_t>(size)<=max_bytes,"session checkpoint size outside admission budget");
    in.seekg(0);
    auto u64=[&]() { std::array<unsigned char,8> b{}; in.read(reinterpret_cast<char*>(b.data()),8);
        require(bool(in),"session checkpoint truncated"); std::uint64_t v=0;
        for(unsigned i=0;i<8;++i) v|=std::uint64_t(b[i])<<(8*i);
        return v; };
    auto str=[&]() { const auto n=u64(); require(n<=16U<<20 && n<=static_cast<std::uint64_t>(size),"session header out of bounds");
        std::string v(static_cast<std::size_t>(n),'\0'); in.read(v.data(),static_cast<std::streamsize>(n));
        require(bool(in),"session checkpoint truncated"); return v; };
    require(str()=="strix-session-v1" && u64()==SessionCompatibility::schema,"session schema mismatch");
    SessionCompatibility expected;
    in.read(reinterpret_cast<char*>(expected.model.bytes.data()),32);
    in.read(reinterpret_cast<char*>(expected.executable.bytes.data()),32);
    expected.numerical_config=str(); const auto mtp=u64();
    require(mtp<=1,"session invalid MTP flag"); expected.mtp=mtp!=0;
    require(expected.executable==executable && expected.numerical_config.starts_with(numerical_prefix),"session compatibility mismatch");
    const auto heap=[](std::uint64_t n) -> PrefixCacheBuffer {
        require(n<=std::numeric_limits<std::size_t>::max(),"session allocation overflow");
        return {std::shared_ptr<std::byte>(new std::byte[static_cast<std::size_t>(n)],std::default_delete<std::byte[]>()),n}; };
    (void)read_session_checkpoint(dir,expected,heap,max_bytes);
}
} // namespace strix
