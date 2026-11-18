// Independent double-precision proof fixtures for the chunk algebra. This
// executable includes no CUDA headers/libraries and cannot initialize a GPU.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>
using V=std::vector<double>;
constexpr unsigned D=7,W=5,B=32;
void need(bool x,const char* s){if(!x)throw std::runtime_error(s);}
struct Fixture {unsigned n;V q,k,v,g,beta,state;};
struct Result {V state,out;};
Result sequential(const Fixture& f) {
    Result r{f.state,V(f.n*W)};
    for(unsigned t=0;t<f.n;++t){for(unsigned k=0;k<D;++k)for(unsigned v=0;v<W;++v)r.state[k*W+v]*=std::exp(f.g[t*D+k]);
        for(unsigned v=0;v<W;++v){double pred=0;for(unsigned k=0;k<D;++k)pred+=f.k[t*D+k]*r.state[k*W+v];double d=(f.v[t*W+v]-pred)*f.beta[t];
            for(unsigned k=0;k<D;++k){r.state[k*W+v]+=f.k[t*D+k]*d;r.out[t*W+v]+=f.q[t*D+k]*r.state[k*W+v];}}}
    return r;
}
Result blocked(const Fixture& f) {
    Result r{f.state,V(f.n*W)};
    for(unsigned first=0;first<f.n;first+=B){unsigned n=std::min(B,f.n-first);V pre(n*D),ck(n*n),cq(n*n),bk(n*W),bq(n*W),delta(n*W);
        for(unsigned k=0;k<D;++k){double p=0;for(unsigned t=0;t<n;++t){p+=f.g[(first+t)*D+k];pre[t*D+k]=p;}}
        for(unsigned t=0;t<n;++t){for(unsigned s=0;s<=t;++s)for(unsigned k=0;k<D;++k){double kd=f.k[(first+s)*D+k]*std::exp(pre[t*D+k]-pre[s*D+k]);
                    if(s<t)ck[t*n+s]+=f.k[(first+t)*D+k]*kd;cq[t*n+s]+=f.q[(first+t)*D+k]*kd;}
            for(unsigned v=0;v<W;++v)for(unsigned k=0;k<D;++k){double st=std::exp(pre[t*D+k])*r.state[k*W+v];bk[t*W+v]+=f.k[(first+t)*D+k]*st;bq[t*W+v]+=f.q[(first+t)*D+k]*st;}}
        for(unsigned t=0;t<n;++t)for(unsigned v=0;v<W;++v){double pred=bk[t*W+v];for(unsigned s=0;s<t;++s)pred+=ck[t*n+s]*delta[s*W+v];delta[t*W+v]=(f.v[(first+t)*W+v]-pred)*f.beta[first+t];}
        for(unsigned t=0;t<n;++t)for(unsigned v=0;v<W;++v){double o=bq[t*W+v];for(unsigned s=0;s<=t;++s)o+=cq[t*n+s]*delta[s*W+v];r.out[(first+t)*W+v]=o;}
        for(unsigned k=0;k<D;++k)for(unsigned v=0;v<W;++v){double st=std::exp(pre[(n-1)*D+k])*r.state[k*W+v];for(unsigned s=0;s<n;++s)st+=f.k[(first+s)*D+k]*std::exp(pre[(n-1)*D+k]-pre[s*D+k])*delta[s*W+v];r.state[k*W+v]=st;}
    }return r;
}
// Rows [b,e) of f with initial state st (a later call of the same prompt).
Fixture slice(const Fixture& f,unsigned b,unsigned e,const V& st){auto cut=[&](const V& x,unsigned w){return V(x.begin()+std::ptrdiff_t(b*w),x.begin()+std::ptrdiff_t(e*w));};
    return {e-b,cut(f.q,D),cut(f.k,D),cut(f.v,W),cut(f.g,D),cut(f.beta,1),st};}
// glm-kda-b32+chunkinv-attn-state-v1/b32/s0: a call cut at c commits state at floor(c/32)*32 and the next call replays
// the open block from that checkpoint, so every output row and the final state equal the whole run BITWISE (same
// block partition). Closing the tail at c (negative control) is another arithmetic for some cut.
bool chunk_contract_fixture(const Fixture& f){
    const auto whole=blocked(f);bool closing_differs=false;
    for(unsigned c:{1u,15u,31u,33u,47u,64u,100u}){if(c>=f.n)continue;
        const unsigned commit=c/B*B;auto first=blocked(slice(f,0,c,f.state));
        auto committed=commit?blocked(slice(f,0,commit,f.state)).state:f.state;auto rest=blocked(slice(f,commit,f.n,committed));
        for(unsigned j=0;j<commit*W;++j)need(first.out[j]==whole.out[j],"committed rows of the cut call");
        for(unsigned j=0;j<(f.n-commit)*W;++j)need(rest.out[j]==whole.out[commit*W+j],"replayed open block and later rows");
        need(rest.state==whole.state,"final state after replay");
        auto closed=blocked(slice(f,c,f.n,first.state));
        for(unsigned j=0;j<(f.n-c)*W&&!closing_differs;++j)closing_differs=closed.out[j]!=whole.out[c*W+j];
    }return closing_differs;}
void close(const V& a,const V& b){need(a.size()==b.size(),"size");for(unsigned j=0;j<a.size();++j)need(std::isfinite(a[j])&&std::isfinite(b[j])&&std::abs(a[j]-b[j])<=2.e-11*(1+std::abs(a[j])),"blocked recurrence algebra");}
V conv_parallel(const V& x,unsigned n,const V& w,V& carry){V out(n);auto old=carry;for(unsigned t=0;t<n;++t)for(unsigned j=0;j<4;++j){int p=int(t)+int(j)-3;out[t]+=w[j]*(p<0?old[3+p]:x[p]);}
    for(unsigned j=0;j<3;++j){int p=int(n)+int(j)-3;carry[j]=p<0?old[3+p]:x[p];}return out;}
int main(){try{std::mt19937 rng(20261027);bool closing_any=false;auto random=[&](unsigned n,double scale){V r(n);for(auto& x:r)x=(int(rng()%2001)-1000)*scale/1000;return r;};
    for(unsigned n:{1u,2u,3u,4u,31u,32u,33u,63u,64u,65u,129u})for(unsigned mode=0;mode<5;++mode){Fixture f{n,random(n*D,.2),random(n*D,.2),random(n*W,1),V(n*D),V(n),random(D*W,.1)};
        for(auto& g:f.g)g=mode==0?0:mode==1?-5:mode==2?-1.e-9:-double(rng()%5001)/1000;
        for(auto& beta:f.beta)beta=mode==3?0:double(rng()%1001)/1000;
        auto a=sequential(f),b=blocked(f);close(a.out,b.out);close(a.state,b.state);
        if(n==129)closing_any|=chunk_contract_fixture(f);
        // Complete output prefixes cannot depend on future q/k/v/g/beta.
        if(n>1){auto future=f;for(unsigned t=n/2;t<n;++t){future.beta[t]=1;for(unsigned k=0;k<D;++k){future.k[t*D+k]=.3;future.g[t*D+k]=-4;}}
            auto alt=blocked(future);for(unsigned j=0;j<(n/2)*W;++j)need(alt.out[j]==b.out[j],"causal chunk coefficients");}
        auto x=random(n,1),w=random(4,1),carry=random(3,1),st=carry;V expected(n);
        for(unsigned t=0;t<n;++t){for(unsigned j=0;j<3;++j)expected[t]+=w[j]*st[j];expected[t]+=w[3]*x[t];st={st[1],st[2],x[t]};}
        auto actual=conv_parallel(x,n,w,carry);close(expected,actual);close(st,carry);
        // Arbitrary partial-prefix calls preserve the convolution history.
        for(unsigned split=1;split<n;++split){auto cv=V{.1,.2,.3},cvall=cv;auto all=conv_parallel(x,n,w,cvall);V left(x.begin(),x.begin()+split),right(x.begin()+split,x.end());auto lo=conv_parallel(left,split,w,cv);auto hi=conv_parallel(right,n-split,w,cv);lo.insert(lo.end(),hi.begin(),hi.end());close(all,lo);close(cvall,cv);}
    }
    need(closing_any,"negative control: closing the tail per call never changed a bit");
    std::puts("PASS glm-kda-b32+chunkinv-attn-state-v1: cuts 1/15/31/33/47/64/100 commit at B32 and replay the open block, rows and state bitwise equal to the whole run; closing per call differs");
    std::puts("PASS KDA vector decay/state-dependent delta: double sequential vs 32-block triangular solve, nonzero initial state, tails 1..129, zero beta, gates 0/-5/tiny/random, causal outputs");
    std::puts("PASS convolution row-parallel vs sequential, nonzero carry, all split boundaries including incomplete chunks; host only");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
