#include "../dsa_semantics.hpp"
#include "../paired_index_model.hpp"
#include <array>
#include <bit>
#include <cstdio>
#include <limits>
#include <random>
using namespace strix::glm::rtxpf::dsa_host;
void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
// Radix threshold and lower-ID compaction emulator; independent sort oracle is
// in dsa_semantics. No device execution or claims of CUDA numerical fidelity.
std::vector<unsigned> radix(const std::vector<float>& s,unsigned t){
    unsigned P=(t+1)/4,k=std::min(P,512u),prefix=0,mask=0,remaining=k;std::vector<unsigned> keys;
    for(unsigned p=0;p<P;++p){auto b=std::bit_cast<unsigned>(s[p]);if((b&0x7fffffff)==0)b=0;keys.push_back(b&0x80000000?~b:b|0x80000000);}
    if(k)for(int shift=24;shift>=0;shift-=8){std::array<unsigned,256> hist{};for(auto key:keys)if((key&mask)==prefix)++hist[(key>>shift)&255];
        for(int d=255;d>=0;--d){if(hist[d]>=remaining){prefix|=unsigned(d)<<shift;break;}remaining-=hist[d];}mask|=255u<<shift;}
    std::vector<unsigned> out;unsigned equal=0;for(unsigned p=0;p<P;++p)if(keys[p]>prefix||(keys[p]==prefix&&equal++<remaining))for(unsigned j=0;j<4;++j)out.push_back(4*p+j);
    for(unsigned j=4*P;j<=t;++j)out.push_back(j);return out;
}
// Independent full sort is the oracle. This CPU port exercises the bounded
// streaming threshold and stable compaction, including ties spanning chunks.
std::vector<unsigned> streaming(const std::vector<float>& scores,unsigned t){
    auto key=[](float f){auto b=std::bit_cast<unsigned>(f);if((b&0x7fffffff)==0)b=0;return b&0x80000000?~b:b|0x80000000;};
    unsigned P=(t+1)/4;std::vector<unsigned> ids;
    for(unsigned base=0;base<P;base+=512){
        for(unsigned p=base;p<std::min(P,base+512);++p)ids.push_back(p);
        if(ids.size()<=512)continue;
        unsigned prefix=0,mask=0,remaining=512;
        for(int shift=24;shift>=0;shift-=8){std::array<unsigned,256> hist{};
            for(auto p:ids)if((key(scores[p])&mask)==prefix)++hist[(key(scores[p])>>shift)&255];
            for(int d=255;d>=0;--d){if(hist[d]>=remaining){prefix|=unsigned(d)<<shift;break;}remaining-=hist[d];}mask|=255u<<shift;
        }
        std::vector<unsigned> kept;unsigned equal=0;
        for(auto p:ids){auto k=key(scores[p]);if(k>prefix||(k==prefix&&equal++<remaining))kept.push_back(p);}
        need(kept.size()==512&&std::is_sorted(kept.begin(),kept.end()),"streaming stable cardinality");ids=std::move(kept);
    }
    std::vector<unsigned> out;for(auto p:ids)for(unsigned j=0;j<4;++j)out.push_back(4*p+j);
    for(unsigned j=4*P;j<=t;++j)out.push_back(j);return out;
}
float bf_round(float x){auto b=std::bit_cast<unsigned>(x);b+=0x7fff+((b>>16)&1);return std::bit_cast<float>(b&0xffff0000u);}
void split_products(){
    std::mt19937 rng(37);
    for(unsigned trial=0;trial<256;++trial){double exact=0,bound=0;float sum=0;
        for(unsigned k=0;k<128;++k){float a=float(int(rng()%200001)-100000)/713.f,b=float(int(rng()%200001)-100000)/997.f;
            float ah=bf_round(a),bh=bf_round(b),al=bf_round(a-ah),bl=bf_round(b-bh);
            sum=std::fma(al,bh,sum);sum=std::fma(ah,bl,sum);sum=std::fma(ah,bh,sum);
            exact+=double(a)*b;bound+=std::abs(double(a)*b);
        }
        need(std::abs(double(sum)-exact)<=3.e-5*bound,"BF16 three-product error envelope");
    }
}
// PTX m16n8k16 BF16 row/col operand layout and output ownership. Rebuild
// dense A/B from every lane's packed pairs; check two K16 PV steps against
// an independent dense product, and all heads/channels exactly once.
void mma_layout(){
    std::array<unsigned,64*512> owners{};
    for(unsigned warp=0;warp<16;++warp)for(unsigned lane=0;lane<32;++lane){
        unsigned hg=warp/4,dc=warp%4*128,g=lane/4,t=lane%4;
        for(unsigned col=0;col<16;++col)for(unsigned row=0;row<2;++row)for(unsigned v=0;v<2;++v)
            ++owners[(hg*16+g+8*row)*512+dc+8*col+2*t+v];
    }
    for(auto count:owners)need(count==1,"MMA output ownership overlap/hole");
    float a[16][40],b[32][520],ar[16][32]{},br[32][8]{};
    for(unsigned h=0;h<16;++h)for(unsigned k=0;k<32;++k)a[h][k]=float(int((h*3+k)%11)-5);
    for(unsigned k=0;k<32;++k)for(unsigned d=0;d<8;++d)b[k][d]=float(int((k+d*5)%13)-6);
    for(unsigned lane=0;lane<32;++lane)for(unsigned k0:{0u,16u}){
        unsigned g=lane/4,t=lane%4;
        for(unsigned half=0;half<2;++half)for(unsigned hi=0;hi<2;++hi)for(unsigned v=0;v<2;++v)
            ar[g+8*half][k0+2*t+8*hi+v]=a[g+8*half][k0+2*t+8*hi+v];
        for(unsigned hi=0;hi<2;++hi)for(unsigned v=0;v<2;++v)
            br[k0+2*t+8*hi+v][g]=b[k0+2*t+8*hi+v][g];
    }
    for(unsigned lane=0;lane<32;++lane)for(unsigned j=0;j<4;++j){
        unsigned h=lane/4+8*(j/2),d=2*(lane%4)+j%2;float expected=0,actual=0;
        for(unsigned k=0;k<32;++k){expected=std::fma(a[h][k],b[k][d],expected);actual=std::fma(ar[h][k],br[k][d],actual);}
        need(expected==actual,"MMA BF16 fragment mapping");
    }
}
void split_combine(){
    for(unsigned n:{1u,15u,16u,31u,32u,33u,255u,256u,257u,511u,512u,513u,1024u,2048u,2049u,2051u}){
        std::vector<double> s(n),x(n);for(unsigned j=0;j<n;++j){s[j]=double(int(j%23)-11)*7;x[j]=double(int(j%17)-8);}
        double mx=*std::max_element(s.begin(),s.end()),den=0,num=0;
        for(unsigned j=0;j<n;++j){double p=std::exp(s[j]-mx);den+=p;num+=p*x[j];}
        double m=-std::numeric_limits<double>::infinity(),d=0,v=0;
        for(unsigned base=0;base<n;base+=256){unsigned end=std::min(base+256,n);
            double sm=*std::max_element(s.begin()+base,s.begin()+end),sd=0,sv=0;
            for(unsigned j=base;j<end;++j){double p=std::exp(s[j]-sm);sd+=p;sv+=p*x[j];}
            double next=std::max(m,sm),alpha=std::exp(m-next),beta=std::exp(sm-next);
            v=std::fma(sv,beta,v*alpha);d=std::fma(sd,beta,d*alpha);m=next;
        }
        need(std::abs(v/d-num/den)<1.e-12,"fixed-order split combine");
    }
}
int main(){try{
    using strix::glm::rtxpf::paired_half_model;
    need(paired_half_model(-1,1,0x1p-12f,0x1p-13f,1)==0x1p-25f,"paired dot cancellation");
    need(paired_half_model(65504,65504,0x1p-24f,0x1p-24f,-float(double(65504)*65504))==float(double(65504)*65504-double(float(double(65504)*65504))+0x1p-48),"paired dot large exponent gap");
    need(paired_half_model(1,1,0x1p-12f,0x1p-12f,0)==1.f,"paired dot ties to even");
    need(paired_half_model(1,1,0x1p-12f,0x1.8p-11f,0)==1.f+0x1p-22f,"paired dot upper even tie");
    mma_layout();split_combine();split_products();
    std::mt19937 rng(20261027);unsigned cases=0;
    for(unsigned t:{0u,1u,2u,3u,4u,7u,31u,63u,64u,1023u,2046u,2047u,2048u,2050u,2051u,2052u,4095u,4096u,4100u,8191u,32763u,32767u}){
        unsigned P=(t+1)/4;
        for(unsigned fixture=0;fixture<5;++fixture){std::vector<float> scores(P+4,1.e20f);for(unsigned p=0;p<P;++p)scores[p]=fixture==0?0:fixture==1?-0.f:fixture==2?float(p%7):fixture==3?float(int(P)-int(p)):float(int(rng()%20001)-10000)/1024;
            auto expected=select(scores,t),actual=radix(scores,t);need(actual==expected,"radix/sort disagreement");need(streaming(scores,t)==expected,"streaming/sort disagreement");need(actual.size()==std::min(P,512u)*4+(t+1)%4,"cardinality");
            for(unsigned j=0;j<actual.size();++j){need(actual[j]<=t,"causality");if(j)need(actual[j]>actual[j-1],"canonical order");}
            for(unsigned j=4*P;j<=t;++j)need(std::binary_search(actual.begin(),actual.end(),j),"unfinished pool missing");
            if(fixture<2&&P>512)for(unsigned j=0;j<2048;++j)need(actual[j]==j,"lower-ID equal/signed-zero ties");
            // Appending arbitrary future pool scores cannot change this prefix.
            scores.resize(P+128,1.e30f);need(select(scores,t)==expected,"future scores affect prefix");++cases;
        }
    }
    bool threw=false;try{select({std::numeric_limits<float>::infinity()},3);}catch(const std::invalid_argument&){threw=true;}need(threw,"nonfinite score accepted");
    need(near_tie(1.001f,1)&&!near_tie(1.1f,1),"near-tie boundary");need(near_tie(-2,-2)&&!near_tie(-1,-2),"negative cutoff");
    // Per-channel gated four-token pooling (constant keys, extreme logits) and
    // FP32 online softmax reference: same output across every split boundary.
    std::array<double,4> key={2,2,2,2},g={1000,-1000,999,998};double den=0,num=0;for(unsigned j=0;j<4;++j){double p=std::exp(g[j]-1000);den+=p;num+=p*key[j];}need(std::abs(num/den-2)<1.e-14,"stable pool");
    std::vector<double> score(2051),values(2051);for(unsigned j=0;j<score.size();++j){score[j]=(int(j%31)-15)*17.;values[j]=int(j%13)-6;}
    double max=*std::max_element(score.begin(),score.end()),sum=0,weighted=0;for(unsigned j=0;j<score.size();++j){double p=std::exp(score[j]-max);sum+=p;weighted+=p*values[j];}
    for(unsigned block:{1u,4u,16u,31u,64u,2051u}){double m=-std::numeric_limits<double>::infinity(),d=0,v=0;for(unsigned base=0;base<score.size();base+=block){unsigned end=std::min(unsigned(score.size()),base+block);double next=std::max(m,*std::max_element(score.begin()+base,score.begin()+end)),alpha=std::exp(m-next);d*=alpha;v*=alpha;for(unsigned j=base;j<end;++j){double p=std::exp(score[j]-next);d+=p;v+=p*values[j];}m=next;}need(std::abs(v/d-weighted/sum)<1.e-12,"online softmax partition");}
    // Node wire buffers contain complete rows, including unfinished key/gate,
    // and only floor(P/4) FP32 pools; no padded latent rows exported.
    for(unsigned P:{1u,3u,4u,5u,2049u,32768u})need(2ull*(P*512ull+2*P*128ull)+4ull*(P/4)*128==1536ull*P+512ull*(P/4),"wire layout");
    std::printf("PASS DSA host semantics: %u radix/sort/causal/unfinished/tie fixtures; paired-half cancellation/exponent-gap/RNE fixtures, streaming radix/ties/partial chunks, BF16 three-product error envelope, MMA fragment/output ownership, split boundaries, stable pooling, online softmax partitions, wire extents, nonfinite and near-tie policy. CUDA/MTP device fidelity NOT RUN.\n",cases);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
