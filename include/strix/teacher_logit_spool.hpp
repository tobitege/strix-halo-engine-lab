#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix {
// Diagnostic only. A complete file has the TMLEOF01 footer. Integers and IEEE
// binary32 logits are little endian; rows preserve ascending vocabulary order.
// Header: TMLLOG01, u64 vocab, u64 rows, u64 input_count, u32 input IDs.
// Each row: u32 input, u64 position, vocab*f32 logits. Footer: TMLEOF01,u64 rows.
// Exclusive creation preserves prior results; an interrupted file remains
// incomplete and must be rejected by consumers. No generation timing claim.
// Row selection (quality kits: logits at a few spread prompt positions without
// a prompt-sized file): with a nonempty strictly ascending `selected` list of
// row positions (< input_count - 1) the header is TMLLOG02, u64 vocab,
// u64 selected_count, u64 input_count, u32 input IDs, u64 selected positions;
// every row is still appended and checked in order, only the selected ones are
// written (same row layout), and the footer carries selected_count. Without a
// selection the file is the TMLLOG01 layout above, byte for byte.
class TeacherLogitSpool {
public:
    TeacherLogitSpool(const std::string& path,std::uint64_t vocabulary,
        std::span<const std::uint32_t> teacher_inputs,
        std::span<const std::uint64_t> selected = {})
        : vocabulary_(vocabulary) {
        static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559 &&
            std::endian::native==std::endian::little);
        constexpr std::uint64_t maximum_bytes=16ULL*1024U*1024U*1024U;
        if(path.empty() || path.find('\0')!=std::string::npos || !vocabulary ||
            vocabulary>1048576U ||
            teacher_inputs.size()<2U || teacher_inputs.size()>262144U)
            throw std::invalid_argument("invalid bounded teacher logit spool");
        for(const auto id:teacher_inputs)if(id>=vocabulary_)
            throw std::invalid_argument("teacher logit spool input outside vocabulary");
        const std::uint64_t all_rows=teacher_inputs.size()-1U;
        for(std::size_t i=0;i<selected.size();++i)
            if(selected[i]>=all_rows || (i && selected[i]<=selected[i-1U]))
                throw std::invalid_argument("teacher logit spool selection must be ascending rows inside the sequence");
        const std::uint64_t rows=selected.empty()?all_rows:selected.size();
        const std::uint64_t overhead=48U+teacher_inputs.size()*4U+selected.size()*8U+rows*12U;
        if(overhead>maximum_bytes || rows>(maximum_bytes-overhead)/(vocabulary_*4U))
            throw std::invalid_argument("teacher logit spool exceeds 16 GiB bound");
        inputs_.assign(teacher_inputs.begin(),teacher_inputs.end());
        selected_.assign(selected.begin(),selected.end());
#ifdef _MSC_VER
        if(fopen_s(&file_,path.c_str(),"wbx")!=0)file_=nullptr;
#else
        file_=std::fopen(path.c_str(),"wbx");
#endif
        if(!file_)throw std::runtime_error("cannot exclusively create teacher logit spool");
        try {
            write(selected_.empty()?"TMLLOG01":"TMLLOG02",8U);u64(vocabulary_);u64(rows);u64(inputs_.size());
            write(inputs_.data(),inputs_.size()*sizeof(std::uint32_t));
            if(!selected_.empty())write(selected_.data(),selected_.size()*sizeof(std::uint64_t));
        } catch(...) {std::fclose(file_);file_=nullptr;throw;}
    }
    TeacherLogitSpool(const TeacherLogitSpool&)=delete;
    TeacherLogitSpool& operator=(const TeacherLogitSpool&)=delete;
    ~TeacherLogitSpool(){if(file_)std::fclose(file_);}

    void append(std::uint32_t input,std::uint64_t position,std::span<const float> logits) {
        if(!file_ || failed_ || finished_ || rows_>=inputs_.size()-1U || position!=rows_ ||
            input!=inputs_[static_cast<std::size_t>(rows_)] || logits.size()!=vocabulary_)
            throw std::invalid_argument("teacher logit spool row differs from retained sequence");
        for(const float value:logits)if(!std::isfinite(value))
            throw std::invalid_argument("teacher logit spool requires finite full vocabulary");
        const bool keep=selected_.empty() ||
            (written_<selected_.size() && selected_[static_cast<std::size_t>(written_)]==position);
        try {
            if(keep){write(&input,sizeof(input));u64(position);write(logits.data(),logits.size_bytes());++written_;}
            ++rows_;
        }
        catch(...){failed_=true;throw;}
    }
    void finish() {
        if(!file_ || failed_ || finished_ || rows_!=inputs_.size()-1U ||
            written_!=(selected_.empty()?rows_:selected_.size()))
            throw std::invalid_argument("teacher logit spool is incomplete");
        try {
            write("TMLEOF01",8U);u64(written_);
            if(std::fflush(file_)!=0)throw std::runtime_error("teacher logit spool flush failed");
            auto* completed=file_;file_=nullptr;
            if(std::fclose(completed)!=0)throw std::runtime_error("teacher logit spool close failed");
            finished_=true;
        }catch(...){failed_=true;throw;}
    }
private:
    void write(const void* data,std::size_t bytes) {
        if(std::fwrite(data,1U,bytes,file_)!=bytes)
            throw std::runtime_error("teacher logit spool write failed");
    }
    void u64(std::uint64_t value){write(&value,sizeof(value));}
    std::uint64_t vocabulary_=0,rows_=0,written_=0;
    std::vector<std::uint32_t> inputs_;
    std::vector<std::uint64_t> selected_;
    std::FILE* file_=nullptr;
    bool failed_=false,finished_=false;
};
}
