#include "strix/file_snapshot.hpp"
#include "strix/ds41_spec_tail.hpp"
#include "strix/hip_check.hpp"

#include "strix/dist/communicator.hpp"
#include "strix/ds41_dspark_files.hpp"
#include "strix/ds41_dspark_vocab.hpp"
#include "strix/hip_ds41_dspark.hpp"

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <optional>
#include <string>

namespace strix::ds41::dspark {
namespace {
void check(hipError_t e, const char* what) {
    ::strix::hip_check_prefixed("spec tail drafter: ", e, what);
}
}  // namespace

struct SpecTailDrafter::Impl {
    std::unique_ptr<DeviceDrafter> drafter;
    std::string vocab;
    std::optional<VocabularyChoice> fixed_vocabulary;
    float* staging = nullptr;  // [128][main_width]
    std::uint32_t staged = 0;
    std::uint64_t request = 0;  // the split head's exchanges (same sequence on both ranks)
};

SpecTailDrafter::SpecTailDrafter(const std::string& model, const std::string& draft, std::string vocabulary) : impl_(std::make_unique<Impl>()) {
    auto w = load_weights(model, draft);
    impl_->drafter = std::make_unique<DeviceDrafter>(w->weights);
    impl_->vocab = std::move(vocabulary);
    if (strix::LoaderCapture::active() && impl_->vocab.starts_with("files:"))
        impl_->fixed_vocabulary = choose_vocabulary(impl_->vocab, {});
    check(hipMalloc(reinterpret_cast<void**>(&impl_->staging), std::size_t{128} * main_width * 4U), "staging");
}

SpecTailDrafter::~SpecTailDrafter() {
    if (impl_ && impl_->staging) (void)hipFree(impl_->staging);
}

void SpecTailDrafter::set_split(std::uint32_t rank, strix::dist::Communicator& comm) {
    auto* m = impl_.get();
    m->drafter->set_rank(rank, 2U, [m, &comm](void* stream, const void* local, void* remote, std::size_t bytes) {
        comm.exchange(static_cast<hipStream_t>(stream), local, remote, bytes,
                      strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 1000U, m->request++});
    });
}

ContextSnapshot SpecTailDrafter::export_context() const {
    if (impl_->staged != 0U) throw std::logic_error("DSpark checkpoint has unflushed hidden rows");
    return impl_->drafter->export_context();
}
void SpecTailDrafter::import_context(const ContextSnapshot& state, std::uint64_t request) {
    if (impl_->staged != 0U) throw std::logic_error("DSpark import into nonempty staging");
    impl_->drafter->import_context(state);
    impl_->request = request;
}
std::uint64_t SpecTailDrafter::exchange_request() const noexcept { return impl_->request; }
void SpecTailDrafter::reset() {
    impl_->drafter->reset();
    impl_->staged = 0;
}

void SpecTailDrafter::flush(void* stream) {
    auto& m = *impl_;
    if (m.staged == 0U) return;
    m.drafter->append_hidden_device(m.staging, m.staged, stream);
    m.staged = 0;
}

void SpecTailDrafter::stage(const std::array<const float*, 3>& captures, std::uint32_t rows, void* stream) {
    auto& m = *impl_;
    if (rows == 0U || rows > 8U) throw std::invalid_argument("spec tail drafter: 1..8 rows per step");
    if (m.staged + rows > 128U) flush(stream);
    enqueue_main_hidden(captures[0], captures[1], captures[2], rows, m.staging + std::size_t{m.staged} * main_width, stream);
    m.staged += rows;
}

std::string SpecTailDrafter::begin(std::span<const std::uint32_t> context, std::uint32_t first) {
    auto& m = *impl_;
    auto choice = m.fixed_vocabulary ? *m.fixed_vocabulary : choose_vocabulary(m.vocab, context);
    if (m.fixed_vocabulary) {
        const auto added = add_context(choice.vocabulary, context);
        choice.label = m.vocab + " + " + std::to_string(added) + " context rows = " +
                       std::to_string(choice.vocabulary.size()) + " rows";
    }
    m.drafter->set_vocabulary(choice.vocabulary);
    (void)m.drafter->add_vocabulary(context);
    const std::uint32_t f[1] = {first};
    (void)m.drafter->add_vocabulary(f);
    return choice.label;
}

void SpecTailDrafter::add_vocabulary(std::span<const std::uint32_t> ids) { (void)impl_->drafter->add_vocabulary(ids); }

Draft SpecTailDrafter::draft(std::uint32_t token, void* stream) {
    flush(stream);
    return impl_->drafter->draft(token, false);
}

std::uint64_t SpecTailDrafter::next_position() const noexcept { return impl_->drafter->next(); }

}  // namespace strix::ds41::dspark
