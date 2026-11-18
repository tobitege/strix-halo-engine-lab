#include "../apps/qwen_effective_config.hpp"
#include "teacher_rows_test_directory.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

using namespace strix;
namespace {
void require(bool good, const char* message) { if (!good) throw std::runtime_error(message); }
void set(const char* key, const char* value) {
#ifdef _WIN32
    require(_putenv_s(key, value) == 0, "set test environment");
#else
    require(setenv(key, value, 1) == 0, "set test environment");
#endif
}
template<class F> void rejects(F f) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("expected validation failure");
}
}
int main() try {
    using namespace core::iu4_rows;
    set("STRIX_IU4_ROW_REDUCTION", "canonical");
    for (const char* policy : {"0", "force"}) {
        set("STRIX_IU4_R8_SINGLE_DECODE", policy);
        const auto captured = decode_cli::requested_prefill_environment();
        const auto resolved = onfly_i4::resolve({}, [&](const char* name) -> const char* {
            const auto it = captured.find(name);
            return it == captured.end() ? nullptr : it->second.c_str();
        }, true);
        require(resolved.row_decode.effective == parse_single_decode_policy(policy), "startup single-decode policy");
        require(resolved.row_reduction.effective == RowReduction::canonical, "startup reduction policy");
        require(resolved.identity().find("iu4-g64g96-gc32-v1") != std::string::npos, "startup canonical identity");
        configure_execution_policy({resolved.row_decode.effective, resolved.row_reduction.effective});
        set("STRIX_IU4_R8_SINGLE_DECODE", "auto");
        envcfg::record("STRIX_IU4_R8_SINGLE_DECODE", "auto", "untouched");
        for (unsigned i = 0; i < 100U; ++i) {
            require(execution_policy().single_decode == resolved.row_decode.effective, "immutable dispatch policy");
            require(execution_policy().reduction == resolved.row_reduction.effective, "immutable dispatch reduction");
        }
        for (const auto& entry : envcfg::registered())
            if (entry.name == "STRIX_IU4_R8_SINGLE_DECODE") require(entry.value == "untouched", "dispatch registry write");
    }
    std::filesystem::path retired;
    {
        const TeacherRowsDirectory first, second;
        retired = first.path;
        require(first.path != second.path, "unique concurrent fixture directories");
        { std::ofstream(first.path / "rows.txt") << "0 1"; std::ofstream(second.path / "rows.txt") << "1 1"; }
        std::ifstream good(first.path / "rows.txt"), bad(second.path / "rows.txt");
        require(decode_cli::read_teacher_logits_rows(good) == std::vector<std::uint64_t>{0U, 1U}, "isolated valid rows");
        rejects([&] { (void)decode_cli::read_teacher_logits_rows(bad); });
    }
    require(!std::filesystem::exists(retired), "scoped fixture cleanup");
    std::istringstream long_word(std::string(100000U, '0'));
    rejects([&] { (void)decode_cli::read_teacher_logits_rows(long_word); });
    require(long_word.tellg() <= std::streampos(20), "bounded decimal ingestion");
    std::ostringstream words;
    for (unsigned i = 0; i < 65536U; ++i) words << i << ' ';
    const auto text = words.str();
    std::istringstream many(text + "65536 invalid-tail"), exact(text);
    rejects([&] { (void)decode_cli::read_teacher_logits_rows(many); });
    require(many.tellg() == static_cast<std::streamoff>(text.size()), "bounded row count before next word");
    require(decode_cli::read_teacher_logits_rows(exact).size() == 65536U, "inclusive row limit");
    std::istringstream maximum("18446744073709551615");
    require(decode_cli::read_teacher_logits_rows(maximum).back() == UINT64_MAX, "uint64 maximum");
    std::cout << "startup policy, registry, fixture isolation and bounded reader PASS\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
