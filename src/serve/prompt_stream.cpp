#include "strix/serve/prompt_stream.hpp"

#include "../core/host_lane.hpp"

#include <stdexcept>
#include <utility>

namespace strix::serve {

std::shared_ptr<PromptStream> PromptStream::start(std::shared_ptr<const ApiTokenizer> tokenizer, std::string text,
                                                  bool parse_special, std::size_t slice_bytes, core::HostLane* lane) {
    if (!tokenizer) throw std::invalid_argument("prompt stream: tokenizer is required");
    std::shared_ptr<PromptStream> self(new PromptStream());
    std::shared_ptr<IncrementalEncoding> encoding = tokenizer->encode_incremental(std::move(text), parse_special, slice_bytes);
    run(self, encoding, tokenizer, lane ? lane : &core::shared_host_lane());
    return self;
}

// One job encodes a bounded number of slices and then queues the next job as its last action. The chain keeps
// the slices of one encoder in order, lets the latency workers interleave the prompts of other requests, and
// lets independent prompts encode on different cores at the same time.
void PromptStream::run(const std::shared_ptr<PromptStream>& self, const std::shared_ptr<IncrementalEncoding>& encoding,
                       const std::shared_ptr<const ApiTokenizer>& tokenizer, core::HostLane* lane) {
    (void)lane->submit(core::HostPriority::latency, [self, encoding, tokenizer, lane] {
        constexpr int slices_per_job = 16;
        std::vector<std::uint32_t> ids;
        try {
            for (int slice = 0; slice < slices_per_job; ++slice) {
                ids.clear();
                if (!encoding->next(ids)) {
                    self->publish(ids, true, nullptr);
                    return;
                }
                self->publish(ids, false, nullptr);
            }
            run(self, encoding, tokenizer, lane);
        } catch (...) {
            self->publish({}, true, std::current_exception());
        }
    });
}

void PromptStream::publish(const std::vector<std::uint32_t>& ids, bool done, std::exception_ptr error) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ids_.insert(ids_.end(), ids.begin(), ids.end());
        done_ = done_ || done;
        if (error && !error_) error_ = error;
    }
    changed_.notify_all();
}

bool PromptStream::take(std::vector<std::uint32_t>& prompt, bool wait) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (prompt.size() > ids_.size()) throw std::logic_error("prompt stream: the prompt is ahead of the stream");
    if (wait) changed_.wait(lock, [&] { return done_ || ids_.size() > prompt.size(); });
    if (error_) std::rethrow_exception(error_);
    prompt.insert(prompt.end(), ids_.begin() + static_cast<std::ptrdiff_t>(prompt.size()), ids_.end());
    return done_;
}

std::size_t PromptStream::wait_for(std::size_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [&] { return done_ || ids_.size() >= count; });
    if (error_) std::rethrow_exception(error_);
    return ids_.size();
}

std::size_t PromptStream::final_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ids_.size();
}

bool PromptStream::complete() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return done_;
}

} // namespace strix::serve
