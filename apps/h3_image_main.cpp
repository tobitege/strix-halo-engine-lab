// MiniMax H3 image runner. The transformer, text encoder, attention and sampler are shared with H3 video.
#include "h3/block/image_policy.hpp"
#include "h3/block/torch_rng.hpp"
#include "h3/common/png.hpp"
#include "h3/vae/image.hpp"
#include "h3/dist/image_parallel.hpp"
#include "h3/common/alongside.hpp"
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <span>
#include <exception>
#include <future>
#ifdef STRIX_H3_IMAGE_PNG
#include <png.h>
#endif
#ifdef STRIX_H3_IMAGE_HIP
#include "h3/block/device_forward.hpp"
#include "h3/dist/device_dist.hpp"
#include "h3/dist/pair_link.hpp"
#include "h3/te/te_gpu.hpp"
#include "h3/vae/vae_decoder.hpp"
#include "h3/vae/enc_gpu.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/dist/gpu_direct.hpp"
#endif
namespace b=strix::h3::block;
namespace v=strix::h3::vae;
namespace d=strix::h3::dist;
namespace {
// --pair-split-tail auto: measured per regime (1024: 20 %, 2048: 30 %; one-message schedule = 0)
constexpr std::uint32_t kSplitAuto=0xffffffffU;
[[maybe_unused]] std::uint32_t split_tail_for(std::uint32_t requested,std::uint32_t packed_rows) {
    return requested!=kSplitAuto?requested:(packed_rows<=2048U?20U:30U);
}
struct Args {
    std::string decode_latent,model,prompt,te_pack,output="image.png",vae,dit,tokenizer,pdd;
    std::string transport="usb4gpu";
    std::uint32_t width=1024,height=1024,seed=42,world=1,rank=0,requests=1;
    bool inspect=false,fast_vae=false,fast_sparse=false,dist=false,attn_2p15=false,latent_only=false;
    bool eager_prepare=true,vae_parallel=true;
    // sequence pair: tail row share of the split K/V schedule (0: one message per block), per-block exposure
    // receipts; TE before (not alongside) DiT residency; VAE loaded after (not during) sampling
    std::uint32_t split_tail=kSplitAuto;
    bool pair_timing=false,serial_te=false,serial_vae=false;
    d::ImageParallel parallel=d::ImageParallel::automatic;
    d::ImageWork work;
    std::vector<std::string> refs;
};
std::uint32_t integer(const std::string& s) {
    std::size_t n=0; const auto v=std::stoull(s,&n);
    if(n!=s.size() || v>0xffffffffULL || s.empty() || s[0]=='-') throw std::invalid_argument("invalid unsigned integer");
    return static_cast<std::uint32_t>(v);
}
Args parse(int argc,char** argv) {
    Args a;
    for(int i=1;i<argc;++i) {
        const std::string k=argv[i];
        auto val=[&]() { if(i+1>=argc) throw std::invalid_argument("missing option value"); return std::string(argv[++i]); };
        if(k=="--model") a.model=val(); else if(k=="--prompt") a.prompt=val();
        else if(k=="--dit") a.dit=val(); else if(k=="--tokenizer") a.tokenizer=val();
        else if(k=="--pdd-fused-heads") a.pdd=val();
        else if(k=="--prompt-file") { std::ifstream f(val()); if(!f) throw std::runtime_error("missing prompt file"); a.prompt.assign(std::istreambuf_iterator<char>(f),{}); }
        else if(k=="--te-pack") a.te_pack=val(); else if(k=="--vae") a.vae=val();
        else if(k=="--output") a.output=val(); else if(k=="--ref") a.refs.push_back(val());
        else if(k=="--width") a.width=integer(val()); else if(k=="--height") a.height=integer(val());
        else if(k=="--seed") a.seed=integer(val()); else if(k=="--world") a.world=integer(val());
        else if(k=="--rank") a.rank=integer(val());
        else if(k=="--requests") a.requests=integer(val());
        else if(k=="--parallel") a.parallel=d::image_parallel(val());
        else if(k=="--vae-parallel") { const auto p=val(); if(p!="auto" && p!="off") throw std::invalid_argument("vae parallel must be auto or off"); a.vae_parallel=p=="auto"; }
        else if(k=="--lazy-prepare") a.eager_prepare=false;
        else if(k=="--pair-split-tail") { const auto v=val(); if(v=="auto") a.split_tail=kSplitAuto; else { a.split_tail=integer(v); if(a.split_tail>=100) throw std::invalid_argument("--pair-split-tail must be auto or 0..99"); } }
        else if(k=="--pair-timing") a.pair_timing=true; else if(k=="--serial-te") a.serial_te=true; else if(k=="--serial-vae") a.serial_vae=true;
        else if(k=="--transport") { a.transport=val(); if(a.transport!="usb4auto" && a.transport!="usb4gpu" && a.transport!="usb4-host") throw std::invalid_argument("image transport must be usb4auto, usb4gpu or usb4-host"); }
        else if(k=="--print-config") a.inspect=true; else if(k=="--decode-latent") a.decode_latent=val(); else if(k=="--latent-only") a.latent_only=true;
        else if(k=="--fast-vae-fp16") a.fast_vae=true; else if(k=="--fast-sparse") a.fast_sparse=true;
        else if(k=="--dist") a.dist=true; else if(k=="--attn-score-2p15") a.attn_2p15=true;
        else if(k=="--help") { std::cout<<"strix-h3-image --model MODEL_DIR_OR_FILE [--dit FILE] --prompt TEXT --te-pack TE_PACK [--tokenizer DIR] [--pdd-fused-heads FILE] --width W --height H --seed N [--ref PNG] [--vae FILE] [--output PNG] [--world 2 --rank R --transport usb4gpu|usb4auto|usb4-host] [--parallel auto|sequence|request] [--requests N (same prompt; seed+index)] [--vae-parallel auto|off] [--pair-split-tail auto|PCT (0..99; auto: 20 up to 2048 packed rows, else 30; 0 = one K/V message per block)] [--pair-timing] [--serial-te] [--serial-vae] [--lazy-prepare] [--dist] [--fast-vae-fp16] [--fast-sparse] [--attn-score-2p15] [--print-config] [--latent-only] [--decode-latent LATENT.f32 (host VAE only)]\n"; std::exit(0); }
        else throw std::invalid_argument("unknown image option");
    }
    a.work=d::image_work(a.parallel,a.requests,a.seed,a.world,a.rank);
    if(a.model.empty() || a.refs.size()>16 || (a.work.sequence && a.fast_sparse) || (a.dist && a.fast_sparse) ||
       (a.dist && a.world==2 && !a.work.sequence) || (!a.decode_latent.empty() && (a.world!=1 || a.requests!=1)))
        throw std::invalid_argument("invalid model, rank/world, reference count or unsupported FAST pair route");
    if(a.width>16384 || a.height>16384) throw std::invalid_argument("image side exceeds runner memory admission bound");
    (void)b::ImageConfig::layout(1,a.width,a.height);
    if(std::filesystem::path(a.model).extension()==".safetensors") {
        a.dit=a.model;auto parent=std::filesystem::path(a.model).parent_path();
        a.model=(parent.filename()=="diffusion_models"?parent.parent_path():parent).string();
    }
    if(a.dit.empty()) {
        const std::string base=a.model+"/diffusion_models/minimax_h3_image_pruned_";
        const bool i8=std::filesystem::exists(base+"int8_convrot.safetensors"),w4=std::filesystem::exists(base+"w4a8.safetensors");
        if(i8 && w4) throw std::invalid_argument("image: multiple checkpoints; select --dit FILE");
        a.dit=i8?base+"int8_convrot.safetensors":w4?base+"w4a8.safetensors":a.model+"/transformer";
    }
    if(a.tokenizer.empty()) { const auto* env=std::getenv("H3_TOKENIZER_DIR");a.tokenizer=env?env:a.model+"/tokenizer"; }
    if(a.pdd.empty() && std::filesystem::exists(a.model+"/pdd_fused_heads.safetensors")) a.pdd=a.model+"/pdd_fused_heads.safetensors";
    if(a.vae.empty()) {
        a.vae=a.model+"/vae/minimax_h3_image_vae_fp16.safetensors";
        if(!std::filesystem::exists(a.vae)) a.vae=a.model+"/vae";
    }
    return a;
}
struct Pixels { int h=0,w=0; std::vector<float> rgb,chw; };
Pixels read_png(const std::string& file) {
#ifdef STRIX_H3_IMAGE_PNG
    png_image image{}; image.version=PNG_IMAGE_VERSION;
    if(!png_image_begin_read_from_file(&image,file.c_str())) throw std::runtime_error("reference PNG header failed");
    struct Guard { png_image* p; ~Guard(){png_image_free(p);} } guard{&image};
    if(image.width<32 || image.height<32 || image.width>16384 || image.height>16384 || image.width%32 || image.height%32)
        throw std::invalid_argument("reference PNG sides must be multiples of 32 in [32,16384]; pre-resize explicitly");
    image.format=PNG_FORMAT_RGB;
    std::vector<png_byte> raw(PNG_IMAGE_SIZE(image));
    if(!png_image_finish_read(&image,nullptr,raw.data(),0,nullptr)) throw std::runtime_error("reference PNG decode failed");
    Pixels p; p.h=int(image.height);p.w=int(image.width);p.rgb.resize(raw.size());p.chw.resize(raw.size());
    for(std::size_t i=0;i<raw.size();++i) { p.rgb[i]=float(raw[i])/255; p.chw[(i%3)*std::size_t(p.h)*p.w+i/3]=p.rgb[i]; }
    return p;
#else
    (void)file; throw std::runtime_error("PNG reference input requires a build with libpng");
#endif
}
template<class T> std::uint64_t hash(const std::vector<T>& x) {
    const auto* bytes=reinterpret_cast<const unsigned char*>(x.data()); std::uint64_t h=14695981039346656037ULL;
    for(std::size_t i=0;i<x.size()*sizeof(T);++i) h=(h^bytes[i])*1099511628211ULL;
    return h;
}
void write_f32(const std::string& name,const std::vector<float>& values) {
    std::ofstream f(name,std::ios::binary); f.write(reinterpret_cast<const char*>(values.data()),std::streamsize(values.size()*4));
    if(!f) throw std::runtime_error("image receipt write failed");
}
std::string request_output(const Args& a,std::uint32_t index) {
    if(a.requests==1) return a.output;
    const std::filesystem::path p(a.output);
    return (p.parent_path()/(p.stem().string()+".request-"+std::to_string(index)+p.extension().string())).string();
}
#ifdef STRIX_H3_IMAGE_HIP
namespace te=strix::h3::te;
std::vector<std::uint16_t> narrow(const std::vector<float>& x) {
    std::vector<std::uint16_t> y(x.size()); for(std::size_t i=0;i<x.size();++i) y[i]=b::tob(x[i]);return y;
}
std::vector<float> widen(const std::vector<std::uint16_t>& x) {
    std::vector<float> y(x.size());for(std::size_t i=0;i<x.size();++i)y[i]=b::bf(x[i]);return y;
}
int run(const Args& a,const b::ImageConfig& cfg) {
    if(a.work.requests.empty()) { std::cout<<"{\"image_rank_idle\":true}\n"; return 0; }
    if(a.te_pack.empty() || a.prompt.empty()) throw std::invalid_argument("execution needs --te-pack and --prompt");
    const auto started=std::chrono::steady_clock::now();
    auto stage=[&](const char* name,std::chrono::steady_clock::time_point begin) {
        std::cout<<"{\"image_stage\":\""<<name<<"\",\"rank\":"<<a.rank<<",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<<"}\n";
    };
    auto stage_start=started;
    // Sequence pair: USB4 pairing (stream discovery, cable measurement) runs alongside TE and DiT residency; the
    // input/policy agreement and direct setup still happen after both, before any collective.
    const auto link_opt=d::pair_link_options(a.rank);
    d::PairLinkJob link;
    if(a.work.sequence) link=d::start_pair_link(link_opt);
    std::vector<Pixels> refs; std::vector<b::RefBlock> shapes;
    for(const auto& file:a.refs) { refs.push_back(read_png(file)); const auto& p=refs.back();shapes.push_back({b::RefBlock::image,1,unsigned(p.h/16),unsigned(p.w/16),0}); }
    std::vector<float> conditions;
    if(!refs.empty()) {
        v::ImageVae vae(a.vae);
        std::unique_ptr<v::enc::GpuEncoder> fast;
        if(a.fast_vae) fast=std::make_unique<v::enc::GpuEncoder>(a.vae);
        v::ImageVae::TileFn fn;
        if(fast) fn=[&](const auto& tile,int h,int w) { const auto m=fast->moments(v::enc::normalize_pixels(tile,h,w)); std::vector<float> f(m.v.size());for(std::size_t i=0;i<f.size();++i)f[i]=v::h2f(m.v[i]);return f; };
        for(const auto& p:refs) {
            const auto z=vae.encode(p.chw,p.h,p.w,fn);
            // Each reference restarts the conditioning augmentation RNG, as the shared payload policy specifies.
            b::TorchCpuGenerator rng(0); const auto noise=rng.randn(std::size_t(p.h/32)*(p.w/32)*96);
            const auto rows=b::cond_video_rows(z.data(),1,unsigned(p.h/16),unsigned(p.w/16),b::kVisualCondT,noise.data());
            conditions.insert(conditions.end(),rows.begin(),rows.end());
        }
    }
    stage("reference_vae",stage_start);stage_start=std::chrono::steady_clock::now();
    std::vector<std::uint16_t> text; std::vector<std::int64_t> tags;
    auto run_te=[&] {
        auto tok=te::Tokenizer::load(a.tokenizer);
        std::vector<te::RefItem> items(refs.size());
        const auto prompt=refs.empty() || a.prompt.rfind("Edit <Picture",0)==0 ? a.prompt : "Edit <Picture 1>. "+a.prompt;
        const auto pres=te::present(*tok,prompt,0,items);
        auto encoder=te::TextEncoder::load(a.te_pack,50);
        std::unique_ptr<te::VisionTower> vision;
        std::vector<te::Frames> frames;std::vector<te::Patches> patches;
        if(!refs.empty()) vision=te::VisionTower::load(a.te_pack);
        const auto processor_path=a.model+"/processor/preprocessor_config.json";
        const auto processor=std::filesystem::exists(processor_path)?b::read_json_file(processor_path):
            strix::h3::fixture::JsonParser(R"({"size":{"shortest_edge":65536,"longest_edge":16777216},"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]})").parse();
        const auto min_pixels=processor.at("size").at("shortest_edge").num();
        const auto max_pixels=processor.at("size").at("longest_edge").num();
        if(processor.at("patch_size").num()!=16 || processor.at("temporal_patch_size").num()!=2 || processor.at("merge_size").num()!=2 ||
           !std::isfinite(min_pixels) || !std::isfinite(max_pixels) || min_pixels<1024 || max_pixels<min_pixels || max_pixels>16777216 ||
           std::floor(min_pixels)!=min_pixels || std::floor(max_pixels)!=max_pixels)
            throw std::invalid_argument("image: unsupported vision processor geometry");
        for(const auto* key:{"image_mean","image_std"}) {
            const auto& values=processor.at(key).arr();
            if(values.size()!=3 || std::any_of(values.begin(),values.end(),[](const auto& value){return value.num()!=0.5;}))
                throw std::invalid_argument("image: unsupported vision normalization");
        }
        for(const auto& p:refs) { frames.push_back({1,p.h,p.w,p.rgb,int(min_pixels),int(max_pixels)});patches.push_back(te::make_patches(frames.back())); }
        const auto hidden=te::encode_presented(*encoder,vision.get(),pres,frames);
        text=narrow(hidden);
        const auto layout=te::expand_layout(pres.ids,patches);
        tags.assign(layout.tags.begin(),layout.tags.end());
    }; // TE and vision residency end inside run_te.
    // Sequence pair and eager single node: TE runs on its own thread while the DiT checkpoint is read and made resident
    // (the same TE and DiT arithmetic; peak memory holds both). The admission check keeps the free memory seen before
    // both. The single-node block weights do not depend on the row count, which needs the TE's text length.
    const bool single_eager=!a.work.sequence && !a.dist && a.eager_prepare;
    const bool te_alongside=(a.work.sequence || single_eager) && !a.serial_te;
    std::size_t free_before=0,total_before=0;
    std::future<strix::h3::Timed<void>> te_job;
    if(te_alongside) {
        if(hipMemGetInfo(&free_before,&total_before)!=hipSuccess) throw std::runtime_error("image: memory inspection failed");
        te_job=strix::h3::start_alongside([&run_te] { run_te(); });
    } else {
        run_te();
        refs.clear(); refs.shrink_to_fit();
        stage("text_encoder",stage_start);stage_start=std::chrono::steady_clock::now();
    }
    b::SafeTensors st(a.dit,b::CheckpointNames::image_dit);
    std::cout<<"{\"image_checkpoint\":\""<<b::image_operand_identity(st)<<"\",\"timestep\":\""
             <<(st.find("adaln_t_table")?"checkpoint-curve-finite":"raw-mlp-finite")<<"\"}\n";
    auto g=b::load_global(st,b::ImageConfig::weights_policy(st));
    if(!a.pdd.empty()) { b::SafeTensors pdd(a.pdd); b::load_image_pdd(g,pdd,cfg); }
    else {
        if(g.pdd_steps!=6) throw std::invalid_argument("image: checkpoint needs six complete heads or --pdd-fused-heads FILE");
        g.pdd_fused=true;
    }
    b::ImageBlockSource source(st,g,cfg);
    std::unique_ptr<d::DeviceModel> early_model;
    std::unique_ptr<b::DeviceForward> forward_owner;  // single node: driver and resident linear weights
    std::unique_ptr<b::GemmDeviceLinear> linear_owner;
    if(te_alongside) {
        const auto residency_start=std::chrono::steady_clock::now();
        if(a.work.sequence) early_model=std::make_unique<d::DeviceModel>(g,source,50,true);
        else {
            forward_owner=std::make_unique<b::DeviceForward>(g);
            linear_owner=std::make_unique<b::GemmDeviceLinear>(0); // rows set once the text length is known
            forward_owner->prepare(source,*linear_owner);
        }
        stage("dit_residency_alongside_te",residency_start);
        const auto te_seconds=te_job.get().seconds;
        refs.clear(); refs.shrink_to_fit();
        std::cout<<"{\"image_stage\":\"text_encoder\",\"rank\":"<<a.rank<<",\"seconds\":"<<te_seconds<<",\"alongside_dit\":true}\n";
        stage("te_and_dit_residency",stage_start);
    }
    b::FlashDeviceAttention attention;
    std::vector<std::vector<float>> latents;
    std::unique_ptr<strix::dist::Communicator> comm;
    struct AbortPair {
        std::unique_ptr<strix::dist::Communicator>& comm;
        ~AbortPair() { if(std::uncaught_exceptions() && comm) comm->cancel_pending("image: request failed"); }
    } abort_pair{comm};
    std::uint64_t control_sequence=0;
    auto agree=[&](const auto& mine,std::uint32_t tag) {
        if(!comm) return;
        auto peer=mine;
        comm->exchange_host(std::as_bytes(std::span(mine)),std::as_writable_bytes(std::span(peer)),
            strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control),tag,control_sequence++});
        if(mine!=peer) throw std::runtime_error("image: pair input, policy or output identity mismatch");
    };
    const auto text_len=static_cast<std::uint32_t>(text.size()/5120);
    const auto L=b::ImageConfig::layout(text_len,a.width,a.height,shapes);
    const std::uint32_t split_tail=split_tail_for(a.split_tail,L.seq_len);
    std::size_t free_bytes=free_before,total_bytes=total_before;
    if(!te_alongside && hipMemGetInfo(&free_bytes,&total_bytes)!=hipSuccess) throw std::runtime_error("image: memory inspection failed");
    const std::uint64_t latent_hold=a.latent_only?0:std::uint64_t(a.work.requests.size())*24*(a.height/16)*(a.width/16)*sizeof(float);
    if(L.seq_len>(1U<<20) || b::ImageConfig::conservative_bytes(L.seq_len)+latent_hold>free_bytes)
        throw std::runtime_error("image: packed sequence exceeds conservative memory/attention admission; reduce reference area explicitly");
    // VAE (host head + FAST decoder) loads on a thread during sampling; decoding waits for it (--serial-vae: after
    // the DiT is released, as before). Off in the sequence pair until measured there.
    struct VaeSet { std::unique_ptr<v::ImageVae> host; std::unique_ptr<v::GpuDecoder> fast; };
    auto load_vae=[&a] { VaeSet s;s.host=std::make_unique<v::ImageVae>(a.vae);
        if(a.fast_vae) s.fast=std::make_unique<v::GpuDecoder>(a.vae,24,true,true);
        return s; };
    const bool vae_alongside=!a.latent_only && !a.work.sequence && !a.serial_vae;
    std::future<strix::h3::Timed<VaeSet>> vae_job;
    {
        if(!forward_owner) forward_owner=std::make_unique<b::DeviceForward>(g);
        auto& forward=*forward_owner;
        if(a.attn_2p15) forward.set_attn_score_limit(b::kAttnScoreLimitExtended);
        text=forward.refine_text(text.data(),text_len,attention);
        std::unique_ptr<b::SparseAccel> sparse;
        if(a.fast_sparse) { sparse=std::make_unique<b::SparseAccel>(0.75F,3,2,"39-42"); b::AccelOptions opt; opt.sparse=sparse.get();forward.set_accel(opt);forward.reset_accel_run(); }
        // fused fc1+SwiGLU where the operand allows (int8_convrot), else fc1+K5
        if(linear_owner) linear_owner->set_max_rows(L.seq_len); else linear_owner=std::make_unique<b::GemmDeviceLinear>(L.seq_len);
        auto& linear=*linear_owner;
        std::unique_ptr<d::DeviceModel> model;
        std::unique_ptr<d::DeviceDistRank> rank;
        const auto prepare_start=std::chrono::steady_clock::now();
        if(a.work.sequence || a.dist) {
            const auto split=d::Split::of(L.seq_len);
            // Residency before the pair agreement (the pairing itself started at entry, on its own thread).
            model=early_model?std::move(early_model):std::make_unique<d::DeviceModel>(g,source,50,true);
            if(a.work.sequence) {
                const auto link_start=std::chrono::steady_clock::now();
                auto paired=std::move(link.get().value);
                stage("pair_link_wait",link_start);
                const auto& opt=link_opt;
                const auto& topology=paired.topology;
                auto transport=std::move(paired.transport);
                // Agree BEFORE direct setup or size-dependent collectives. usb4auto is the
                // GPU-preferred alias; usb4-host is an explicit diagnostic off switch.
                const std::array<std::uint64_t,21> setup={0x4833495331370002ULL,split_tail,a.width,a.height,a.seed,L.seq_len,
                    hash(text),hash(conditions),hash(cfg.sigmas),std::uint64_t(a.fast_vae),std::uint64_t(a.attn_2p15),
                    a.requests,std::uint64_t(a.vae_parallel),std::uint64_t(a.latent_only),hash(g.inv_freq),
                    hash(g.video_out.w),hash(g.video_out.b),hash(L.position_ids),hash(tags),
                    std::uint64_t(a.transport!="usb4-host"),sizeof(float)};
                auto peer_setup=setup;
                std::unique_ptr<strix::dist::DeviceExchange> direct;
                std::string receipt=topology.receipt();
                try {
                    transport->exchange(std::as_bytes(std::span(setup)),std::as_writable_bytes(std::span(peer_setup)));
                    if(setup!=peer_setup) throw std::runtime_error("image pair input/policy identity mismatch");
                    if(a.transport!="usb4-host") {
                        std::string why, gpu_receipt;
                        if(strix::dist::usb4_gpu_direct_streams(opt,*transport,&why)) {
                            direct=strix::dist::usb4_gpu_direct(opt,topology,*transport,{},&gpu_receipt);
                            receipt+="; "+gpu_receipt;
                        } else receipt+="; gpu-direct unavailable ("+why+"): host path only";
                    } else receipt+="; explicit host path";
                } catch(...) { transport->abort("image pair setup failed"); throw; }
                std::cerr<<"image transport: "<<receipt<<'\n';
                const auto ty=v::split_tiles(int(a.height)),tx=v::split_tiles(int(a.width));
                std::vector<std::size_t> sizes;
                for(auto h:ty.len) for(auto w:tx.len) sizes.push_back(std::size_t(3)*h*w);
                const auto vae_bytes=d::ImageTileExchange(sizes).elements()*sizeof(float);
                comm=std::make_unique<strix::dist::Communicator>(strix::dist::Group{a.rank,2,0x483349U,0x4833494d47ULL},std::move(transport),std::move(direct),std::max({split.message_bytes(),vae_bytes,std::size_t(1024)}));
                comm->enable_direct_chunk_overlap();
                std::cerr<<"image effective transport: "<<comm->transport_name()<<"; first direct chunk overlap on\n";
                agree(std::array<std::uint64_t,17>{a.width,a.height,a.seed,L.seq_len,hash(text),hash(conditions),hash(cfg.sigmas),
                    std::uint64_t(a.fast_vae),std::uint64_t(a.attn_2p15),a.requests,std::uint64_t(a.vae_parallel),std::uint64_t(a.latent_only),
                    hash(g.inv_freq),hash(g.video_out.w),hash(g.video_out.b),hash(L.position_ids),hash(tags)},0x49U);
            }
            rank=std::make_unique<d::DeviceDistRank>(*model,split,d::Role{a.work.sequence?a.rank:0U,a.work.sequence?2U:1U,false},comm.get());
        } else if(a.eager_prepare) {
            forward.prepare(source,linear); // no-op for blocks already made resident alongside the TE
        }
        stage("dit_residency_and_pair",prepare_start);
        if(vae_alongside) vae_job=strix::h3::start_alongside(load_vae);
        d::DeviceRunOptions run_options;
        run_options.hashes=false; // retain mandatory device digests/admission; avoid per-block host residual downloads
        run_options.score_bound=a.attn_2p15?d::kScoreBoundExtended:d::kScoreBound;
        run_options.pair_split_tail_pct=a.work.sequence?split_tail:0U; // handshaken in the pair setup above
        run_options.timing=a.work.sequence && a.pair_timing;
        b::StreamInputs input;input.T=1;input.H=a.height/16;input.W=a.width/16;input.text=text;input.cond_video=conditions;
        std::cout<<"{\"image_parallel\":\""<<(a.work.sequence?"sequence":"request")<<"\",\"rank\":"<<a.rank
                 <<",\"packed_tokens\":"<<L.seq_len<<",\"local_requests\":"<<a.work.requests.size()<<"}\n";
        for(const auto request:a.work.requests) {
            if(sparse) forward.reset_accel_run();
            const auto output=request_output(a,request);
            b::TorchCpuGenerator rng(a.seed+request);
            const auto noise=rng.randn(std::size_t(24)*input.H*input.W);
            agree(std::array<std::uint64_t,2>{request,hash(noise)},0x4AU); // also wake the link before embedding/QKV
            write_f32(output+".noise.f32",noise);
            int step=0;
            const auto result=b::sample_image(cfg,noise,[&](const b::AvState& x,float sigma) {
                const auto begin=std::chrono::steady_clock::now();input.video_x=narrow(x.video);
                write_f32(output+".state"+std::to_string(step)+".f32",x.video);
                const auto plan=b::ForwardPlan::build(L,sigma,tags,b::ImageConfig::plan_policy());
                const b::PddContext pc{cfg.sigmas,sigma,1,1};b::StreamOutputs out;
                if(rank) {
                    auto f=rank->forward(L,plan,input,d::ForwardTables::build(g,L,plan),50,run_options,nullptr,&pc);
                    if(comm) std::cerr<<"image pair request "<<request<<" step "<<step<<" rank "<<a.rank<<": "<<f.run.comm_stats<<'\n';
                    if(comm && run_options.timing) {
                        double blk=0,loc=0,ex=0,fin=0;
                        for(const auto& t:f.run.timing) { blk+=t.block_ms;loc+=t.local_ms;ex+=t.exposed_ms;fin+=t.finish_host_ms; }
                        std::cerr<<"image pair timing step "<<step<<" rank "<<a.rank<<": blocks "<<blk<<" ms, K4-end->local-end "<<loc
                                 <<" ms, local-end->rx-ready (exposed) "<<ex<<" ms, host finish "<<fin<<" ms, split tail "<<run_options.pair_split_tail_pct<<"%\n";
                    }
                    out=std::move(f.out);
                }
                else out=forward.run(source,L,plan,input,linear,attention,nullptr,&pc);
                const auto velocity=widen(out.video_neg_v);
                agree(std::array<std::uint64_t,4>{request,std::uint64_t(step),hash(x.video),hash(velocity)},0x4BU);
                write_f32(output+".step"+std::to_string(step)+".f32",velocity);
                std::cout<<"{\"image_step\":"<<step<<",\"request\":"<<request<<",\"rank\":"<<a.rank<<",\"sigma\":"<<sigma
                         <<",\"state_hash\":"<<hash(x.video)<<",\"velocity_hash\":"<<hash(velocity)
                         <<",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<<"}\n";
                ++step;return b::AvState{velocity,{}};
            });
            agree(std::array<std::uint64_t,2>{request,hash(result.video)},0x4CU);
            write_f32(output+".latent.f32",result.video);
            if(!a.latent_only) latents.push_back(result.video);
        }
    }
    linear_owner.reset();forward_owner.reset(); // DiT residency ends before decoding (and before VAE loading with --serial-vae or in the pair)
    stage("dit_load_prepare_refiner_sample",stage_start);stage_start=std::chrono::steady_clock::now();
    if(a.latent_only) { std::cout<<"{\"image_complete\":\"latent-only\",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<"}"<<std::endl; return 0; }
    auto vae_set=(vae_job.valid()?std::move(vae_job):strix::h3::start_alongside(load_vae)).get();
    v::ImageVae& vae=*vae_set.value.host;
    std::unique_ptr<v::GpuDecoder> fast=std::move(vae_set.value.fast);
    // nested in vae_and_png (same start); decode+PNG = vae_and_png - vae_load. Alongside sampling: vae_load is the
    // load itself (hidden) and vae_load_wait the part left on the critical path.
    if(vae_alongside) { std::cout<<"{\"image_stage\":\"vae_load\",\"rank\":"<<a.rank<<",\"seconds\":"<<vae_set.seconds<<",\"alongside_dit\":true}\n";
        stage("vae_load_wait",stage_start); }
    else stage("vae_load",stage_start);
    v::ImageVae::TileFn fn;
    if(fast) fn=[&](const auto& pq,int h,int w) { v::H16 x(pq.size());for(std::size_t i=0;i<x.size();++i)x[i]=v::f2h(pq[i]);const auto p=fast->decode_call(x,1,h,w);std::vector<float> f(p.size());for(std::size_t i=0;i<f.size();++i)f[i]=v::h2f(p[i]);return f; };
    v::ImageVae::TileDispatch dispatch;
    if(comm && a.vae_parallel) dispatch=[&](const auto& sizes,const auto& eval) {
        const d::ImageTileExchange layout(sizes);
        std::vector<std::vector<float>> tiles(sizes.size());
        for(auto i=layout.first(a.rank);i<layout.last(a.rank);++i) tiles[i]=eval(i);
        const auto mine=layout.pack(a.rank,tiles);std::vector<float> peer(mine.size());
        comm->exchange_host(std::as_bytes(std::span(mine)),std::as_writable_bytes(std::span(peer)),
            strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::raw_slots),0x56U,control_sequence++});
        return a.rank==0?layout.merge(mine,peer):layout.merge(peer,mine);
    };
    for(std::size_t j=0;j<latents.size();++j) {
        const auto request=a.work.requests[j];const auto output=request_output(a,request);
        const auto pixels=vae.decode(latents[j],int(a.height),int(a.width),fn,dispatch);
        agree(std::array<std::uint64_t,2>{request,hash(pixels)},0x4DU);
        write_f32(output+".rgb.f32",pixels);
        std::vector<std::uint8_t> rgb(pixels.size());const auto plane=std::size_t(a.height)*a.width;
        for(std::size_t i=0;i<rgb.size();++i) rgb[i]=static_cast<std::uint8_t>(std::clamp(std::nearbyint(pixels[(i%3)*plane+i/3]*255),0.0F,255.0F));
        strix::h3::io::write_png(output,rgb.data(),int(a.width),int(a.height));
    }
    stage("vae_and_png",stage_start);
    std::cout<<"{\"image_complete\":true,\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<"}\n";
    return 0;
}
#endif
}
int main(int argc,char** argv) try {
    const auto a=parse(argc,argv);const auto cfg=b::ImageConfig::load(a.model);
    if(a.inspect) {
        std::string operand="pending-checkpoint",timestep="pending-checkpoint";
        bool available=std::filesystem::is_regular_file(a.dit);
        if(std::filesystem::is_directory(a.dit)) for(const auto& e:std::filesystem::directory_iterator(a.dit))
            available=available || e.path().extension()==".safetensors";
        if(available) { b::SafeTensors st(a.dit,b::CheckpointNames::image_dit);operand=b::image_operand_identity(st);
            timestep=st.find("adaln_t_table")?"checkpoint-curve-finite":"raw-mlp-finite"; }
        const auto L=b::ImageConfig::layout(0,a.width,a.height);
        std::cout<<"{\"mode\":\"image\",\"audio_enabled\":false,\"image_tokens\":"<<L.seq_len
                 <<",\"reference_count\":"<<a.refs.size()<<",\"packed_tokens\":null,\"steps\":6,\"cfg\":false,\"world\":"<<a.world
                 <<",\"rank\":"<<a.rank<<",\"requests\":"<<a.requests<<",\"local_requests\":"<<a.work.requests.size()
                 <<",\"parallel\":\""<<(a.work.sequence?"sequence":"request")<<"\",\"eager_prepare\":"<<(a.eager_prepare || a.work.sequence || a.dist?"true":"false")
                 <<",\"transport\":\""<<(a.work.sequence?a.transport:"none")<<"\",\"transport_effective\":\""<<(a.work.sequence?"pending-pairing":"none")<<"\",\"dist_reference\":"<<(a.dist && !a.work.sequence?"true":"false")
                 <<",\"vae_parallel\":"<<(a.work.sequence && a.vae_parallel && !a.latent_only?"true":"false")
                 <<",\"dit\":\""<<operand<<"\",\"timestep\":\""<<timestep<<"\",\"attention\":\""<<(a.fast_sparse?"FAST-h3-attn-sparse-thr-v1":"h3-flash-dense")
                 <<"\",\"attn_score_limit\":"<<(a.attn_2p15?"\"2^15-extended\"":"\"2^14\"")<<",\"vae\":\""<<(a.fast_vae?"FAST-image-vae-fp16-v1":"image-vae-fp32-host-v1")
                 <<"\",\"qualification\":\"pending-native-and-fidelity\"}\n";
        return 0;
    }
    if(!a.decode_latent.empty()) {  // host-only diagnostic: decode a saved normalized latent with the FP32 host VAE
        std::ifstream f(a.decode_latent,std::ios::binary); if(!f) throw std::runtime_error("missing latent file");
        std::vector<float> z(std::size_t(24)*(a.height/16)*(a.width/16));
        f.read(reinterpret_cast<char*>(z.data()),std::streamsize(z.size()*4)); if(!f || f.peek()!=EOF) throw std::runtime_error("latent size mismatch");
        const auto t0=std::chrono::steady_clock::now();
        v::ImageVae vae(a.vae); const auto pixels=vae.decode(z,int(a.height),int(a.width));write_f32(a.output+".rgb.f32",pixels);
        std::vector<std::uint8_t> rgb(pixels.size());const auto plane=std::size_t(a.height)*a.width;
        for(std::size_t i=0;i<rgb.size();++i) rgb[i]=static_cast<std::uint8_t>(std::clamp(std::nearbyint(pixels[(i%3)*plane+i/3]*255),0.0F,255.0F));
        strix::h3::io::write_png(a.output,rgb.data(),int(a.width),int(a.height));
        std::cout<<"{\"image_stage\":\"host_vae_decode\",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()<<"}"<<std::endl;
        return 0;
    }
#ifdef STRIX_H3_IMAGE_HIP
    return run(a,cfg);
#else
    throw std::runtime_error("execution requires an H3 HIP build; host build supports --print-config");
#endif
} catch(const std::exception& e) { std::cerr<<"image: "<<e.what()<<'\n';return 1; }
