#pragma once

#include "strix/serve/engine.hpp"
#include "strix/serve/tokenizer.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strix::core {
class HostLane;
}

namespace strix::serve {

// A prompt being tokenized on the host lane while its first ids are already in use. The ids it publishes are final
// (IncrementalEncoding), so a consumer may start prefill on the first chunk while the rest is still being encoded;
// the complete sequence is exactly the one-shot encoding of the text.
class PromptStream final : public PromptTail {
public:
    // Starts encoding with critical priority on `lane` (null: the process-wide lane). The tokenizer is kept alive.
    static std::shared_ptr<PromptStream> start(std::shared_ptr<const ApiTokenizer> tokenizer, std::string text,
                                               bool parse_special = true, std::size_t slice_bytes = 16384,
                                               core::HostLane* lane = nullptr);

    // `prompt` must hold exactly the first prompt.size() ids of this stream. Appends every further id that is final.
    // wait: blocks until at least one more id is final or the encoding is complete. Returns true when the encoding
    // is complete and `prompt` now holds all of it. Rethrows a tokenizer failure.
    bool take(std::vector<std::uint32_t>& prompt, bool wait) override;
    // Blocks until `count` ids are final or the encoding is complete; returns the number of final ids.
    std::size_t wait_for(std::size_t count);
    std::size_t final_count() const;
    bool complete() const;

private:
    PromptStream() = default;
    void publish(const std::vector<std::uint32_t>& ids, bool done, std::exception_ptr error);
    static void run(const std::shared_ptr<PromptStream>& self, const std::shared_ptr<IncrementalEncoding>& encoding,
                    const std::shared_ptr<const ApiTokenizer>& tokenizer, core::HostLane* lane);

    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::vector<std::uint32_t> ids_;
    bool done_ = false;
    std::exception_ptr error_;
};

} // namespace strix::serve
