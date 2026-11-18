#pragma once
// POST /v1/generate contract (greedy only), validated completely before any
// session mutation. Authentication is the generate body's own "key" (64 hex
// digits of the startup key), independent of prefill handoff fields.
#include "session.hpp"
#include "strix/serve/json.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
namespace strix::glm::rtxpf::producer {
struct RequestError:std::runtime_error {int code;RequestError(int c,const std::string& m):std::runtime_error(m),code(c){}};
struct GenerateRequest {std::vector<std::uint32_t> tokens,stop;unsigned max_new=0;std::string key_hex;};
inline constexpr const char* sampling_fields[]={"temperature","top_p","top_k","min_p","typical_p","seed","do_sample","repetition_penalty","presence_penalty","frequency_penalty","num_beams","logprobs","top_logprobs","n","best_of","logit_bias","sampling"};
inline constexpr unsigned max_stop_tokens=64;
inline std::uint32_t request_token(const strix::serve::Json& v,const char* what){
    if(!v.is_number())throw RequestError(400,std::string(what)+" must be integers");auto d=v.as_double();
    if(!(d>=0&&d<vocab_size&&std::floor(d)==d))throw RequestError(400,std::string(what)+" outside vocabulary");return std::uint32_t(d);
}
inline GenerateRequest parse_generate(const strix::serve::Json& body,unsigned capacity){
    if(!body.is_object())throw RequestError(400,"expected JSON object");
    for(auto& [key,value]:body.as_object()){(void)value;
        if(std::find_if(std::begin(sampling_fields),std::end(sampling_fields),[&](const char* f){return key==f;})!=std::end(sampling_fields))throw RequestError(400,"unsupported generation field: "+key+" (greedy only)");
        if(key!="tokens"&&key!="max_new_tokens"&&key!="identity"&&key!="stop_token_ids"&&key!="key")throw RequestError(400,"unsupported request field: "+key);}
    GenerateRequest r;
    auto* key=body.find("key");if(!key||!key->is_string()||key->as_string().size()!=64||key->as_string().find_first_not_of("0123456789abcdefABCDEF")!=std::string::npos)throw RequestError(401,"generate requires key (64 hex digits)");r.key_hex=key->as_string();
    if(auto* id=body.find("identity"))if(!id->is_string()||id->as_string()!=decode_identity)throw RequestError(400,std::string("identity must be ")+decode_identity);
    auto* tokens=body.find("tokens");if(!tokens||!tokens->is_array())throw RequestError(400,"tokens must be an array of token IDs");
    for(auto& t:tokens->as_array())r.tokens.push_back(request_token(t,"tokens"));
    if(r.tokens.size()<2||r.tokens.size()>capacity)throw RequestError(400,"prompt length must be 2..capacity");
    auto* n=body.find("max_new_tokens");if(!n||!n->is_number())throw RequestError(400,"max_new_tokens required");auto d=n->as_double();
    if(!(d>=1&&std::floor(d)==d&&d<=capacity))throw RequestError(400,"max_new_tokens must be a positive integer");r.max_new=unsigned(d);
    if(r.tokens.size()+r.max_new>capacity)throw RequestError(400,"prompt_length + max_new_tokens exceeds capacity");
    if(auto* s=body.find("stop_token_ids")){if(!s->is_array()||s->as_array().size()>max_stop_tokens)throw RequestError(400,"stop_token_ids must be an array of at most 64 IDs");for(auto& t:s->as_array())r.stop.push_back(request_token(t,"stop_token_ids"));}
    return r;
}
}
