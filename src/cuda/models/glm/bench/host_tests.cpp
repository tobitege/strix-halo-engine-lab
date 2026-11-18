// No CUDA runtime API calls or cudart linkage. Runs on the CPU only.
#include "../trellis.cuh"
#include "../c1_layout.cuh"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
using namespace strix::glm::rtxpf;
void need(bool b,const char* s) { if(!b) { std::fprintf(stderr,"FAIL %s\n",s); std::exit(1); } }
// Emulation of the node's packed u16 multiply/MAD and SAD instructions,
// followed by its half FMA. Deliberately independent of book's numerator.
U16 node_book(U32 s) {
    constexpr U32 adjusted=0x83dcd12dU-1,upper=(adjusted&0xffff0000U)|1,lower=adjusted&65535;
    U32 high=(s&65535) | (((s*(upper>>16))&65535)<<16);
    U32 p=(high&65535)*lower+high;
    unsigned sad=0; for(unsigned j=0;j<4;++j) sad+=(p>>(8*j))&255;
    double exact=double(hf(U16(0x6400+sad)))*double(hf(0x1eee))+double(hf(0xc931));
    return fh(float(exact));
}
U32 scalar_state(const U16* p,unsigned pos,unsigned bits) {
    U32 s=0;
    for(unsigned b=0;b<16;++b) {
        unsigned k=((pos+1)*bits+256*bits-16+b)%(256*bits),w=k/32;
        U32 word=U32(p[2*w])|U32(p[2*w+1])<<16;
        s=(s<<1)|((word>>(31-k%32))&1);
    }
    return s;
}
// Direct port of the node's LaneGeom/TileStates register extraction.
U32 node_state(const U16* p,unsigned pos,unsigned bits) {
    unsigned lane=pos/8,j=pos%8,count=bits*8;
    int start=int((lane*8+1)*bits+256*bits-16),index=start/32-int(count);
    unsigned shift=start&31;
    auto word=[&](int n) { n=(n+int(count))%int(count); return U32(p[2*n])|U32(p[2*n+1])<<16; };
    U32 a=word(index),b=word(index+1),c=word(index+2);
    U32 high=(a<<shift)|(b>>(32-shift)),low=(b<<shift)|(c>>(32-shift)),tail=c<<shift;
    unsigned off=j*bits;
    if(off<=16)return(high>>(16-off))&65535;
    if(off<32)return((high<<(off-16))|(low>>(48-off)))&65535;
    if(off<=48)return(low>>(48-off))&65535;
    return((low<<(off-48))|(tail>>(80-off)))&65535;
}
// Exercise the same specialized word extraction as the CUDA decoder using
// host copies of the cooperative uint4 staging (without a CUDA runtime).
U32 candidate_state(const U16* p,unsigned pos,unsigned bits) {
    alignas(16) U32 words[48]{};
    std::memcpy(words,p,32*bits);
    switch(bits) {
        case 3:return c1_state<3>(words,pos);
        case 4:return c1_state<4>(words,pos);
        case 5:return c1_state<5>(words,pos);
        default:return c1_state<6>(words,pos);
    }
}
void node_h(float* p) {
    float v[32][4]; for(unsigned l=0;l<32;++l)for(unsigned j=0;j<4;++j)v[l][j]=p[l+32*j];
    for(unsigned s=1;s<32;s*=2) { float old[32][4]; std::memcpy(old,v,sizeof v); for(unsigned l=0;l<32;++l)for(unsigned j=0;j<4;++j)v[l][j]=(l&s)?old[l^s][j]-old[l][j]:old[l][j]+old[l^s][j]; }
    for(unsigned s=1;s<4;s*=2)for(unsigned l=0;l<32;++l)for(unsigned j=0;j<4;++j)if(!(j&s)){float a=v[l][j],b=v[l][j+s];v[l][j]=a+b;v[l][j+s]=a-b;}
    for(unsigned l=0;l<32;++l)for(unsigned j=0;j<4;++j)p[l+32*j]=v[l][j]*0x1.6a09e6p-4f;
}
int main() {
    for(unsigned s=0;s<65536;++s) need(book(s)==node_book(s),"mul1 all-state half equality");
    std::mt19937 rng(20261027);
    for(unsigned bits:{3,4,5,6}) {
        std::vector<U16> packed(16*bits);
        std::array<bool,65536> seen{};
        // Force every possible state at every supported rate, including wrap.
        for(unsigned s=0;s<65536;++s) {
            for(auto& w:packed)w=U16(rng());
            unsigned pos=s%256;
            for(unsigned b=0;b<16;++b) {
                unsigned k=((pos+1)*bits+256*bits-16+b)%(256*bits),wi=k/32,shift=31-k%32;
                U32 w=U32(packed[2*wi])|U32(packed[2*wi+1])<<16;
                w=(w&~(1U<<shift))|(((s>>(15-b))&1)<<shift);
                packed[2*wi]=U16(w);packed[2*wi+1]=U16(w>>16);
            }
            need(state(packed.data(),pos,bits)==s,"forced state");
            need(node_state(packed.data(),pos,bits)==s,"node state");
            need(candidate_state(packed.data(),pos,bits)==s,"specialized all-state extraction");
            need(book(candidate_state(packed.data(),pos,bits))==node_book(s),"specialized all-state FP16");
            need(book(state(packed.data(),pos,bits))==node_book(s),"rate all-state decode");seen[s]=true;
            if(s<1024) {
                bool coords[256]{};
                for(unsigned p=0;p<256;++p) {
                    unsigned a=state(packed.data(),p,bits),b=scalar_state(packed.data(),p,bits);
                    need(a==b && a==node_state(packed.data(),p,bits),"tile/circular/node reader");
                    need(a==candidate_state(packed.data(),p,bits),"specialized full tile reader");
                    need(c1_position(qi(p),qo(p))==p,"coalesced inverse permutation");
                    unsigned c=qi(p)*16+qo(p);need(c<256 && !coords[c],"fragment bijection");coords[c]=true;
                }
            }
        }
        for(bool b:seen)need(b,"all states covered");
        std::printf("PASS K%u: 65536 node decoded FP16 values; 1024 full tiles\n",bits);
    }
    for(unsigned trial=0;trial<1024;++trial) {
        float a[128],b[128];
        for(unsigned j=0;j<128;++j) {
            float x=float(int(rng()%2001)-1000)/1024;
            U16 scale=U16(rng()%0x6000); if(rng()&1)scale|=0x8000;
            a[j]=b[j]=hf(fh(x*hf(scale)));
        }
        hadamard(a);node_h(b);
        for(unsigned j=0;j<128;++j) need(fh(a[j])==fh(b[j]),"input transforms/scales half equality");
        // Output transform retains FP32, then applies svh (no half cast).
        for(unsigned j=0;j<128;++j) {float scale=hf(U16(0x1400+rng()%0x4000));a[j]*=scale;b[j]*=scale;}
        need(std::memcmp(a,b,sizeof a)==0,"output transforms/scales FP32 equality");
    }
    // Model the residency grid, warp tile stride, vector staging and store
    // enumeration. Rectangular shapes and uneven grid strides catch tails.
    for(unsigned grid:{1u,2u,7u,19u})for(unsigned output:{128u,256u})for(unsigned bits:{3u,4u,5u,6u}) {
        unsigned input=128,tiles=input/16*(output/16);
        std::vector<U16> packed(std::size_t(input)*output*bits/16),b(std::size_t(input)*output);
        std::vector<bool> written(b.size());
        for(auto& v:packed)v=U16(rng());
        for(unsigned cta=0;cta<grid;++cta)for(unsigned warp=0;warp<8;++warp) {
            unsigned nt=output/16,kt=(cta*8+warp)/nt,no=(cta*8+warp)%nt;
            for(unsigned t=cta*8+warp;t<tiles;t+=grid*8) {
                need(kt==t/nt && no==t%nt,"incremental tile coordinates");
                alignas(16) U16 staged[96]{};
                for(unsigned v=0;v<2*bits;++v)
                    std::memcpy(staged+8*v,packed.data()+std::size_t(t)*16*bits+8*v,16);
                for(unsigned lane=0;lane<32;++lane)for(unsigned j=0;j<8;++j) {
                    unsigned k=lane/16+2*j,n=lane%16,pos=c1_position(k,n);
                    auto at=std::size_t(kt*16+k)*output+no*16+n;
                    need(at==c1_b_index(t,pos,output),"coalesced B address");
                    need(at<b.size()&&!written[at],"C1 B decode unique/in bounds");written[at]=true;
                    b[at]=book(candidate_state(staged,pos,bits));
                }
                kt+=(grid*8)/nt;no+=(grid*8)%nt;
                if(no>=nt){no-=nt;++kt;}
            }
        }
        for(unsigned k=0;k<input;++k)for(unsigned n=0;n<output;++n) {
            unsigned tile=k/16*(output/16)+n/16,pos=0;
            while(qi(pos)!=k%16||qo(pos)!=n%16)++pos;
            need(written[std::size_t(k)*output+n]&&b[std::size_t(k)*output+n]==
                node_book(scalar_state(packed.data()+std::size_t(tile)*16*bits,pos,bits)),"C1 B logical coordinates");
        }
    }
    // Splitting the C1 epilogue and down transform must preserve both half
    // rounding boundaries, including signed zero and subnormal values.
    for(unsigned trial=0;trial<1024;++trial) {
        float gate[128],up[128],expected[128],split[128];U16 down_scale[128],rounded[128];
        for(unsigned j=0;j<128;++j){gate[j]=float(int(rng()%2001)-1000)/64;up[j]=float(int(rng()%2001)-1000)/64;down_scale[j]=U16(rng()%0x4000);}
        hadamard(gate);hadamard(up);
        for(unsigned j=0;j<128;++j) {
            float g=gate[j]*hf(U16(0x3000)),u=up[j]*hf(U16(0xb000));
            float activation=swiglu(g,u);
            expected[j]=hf(fh(activation*hf(down_scale[j])));
            rounded[j]=fh(swiglu(g,u)*hf(down_scale[j]));split[j]=hf(rounded[j]);
        }
        hadamard(expected);node_h(split);
        for(unsigned j=0;j<128;++j)need(fh(expected[j])==fh(split[j]),"C1 fused down-input rounding");
    }
    std::puts("PASS C1 resident warp/vector B layout K3..K6, rectangular strides, fused down rounding");
    std::puts("PASS H128 input/output scale rounding and permutation; host only");
}
