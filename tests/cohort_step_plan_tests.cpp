#include "strix/cohort_step_plan.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
void require(bool pass) { if (!pass) throw std::runtime_error("cohort step plan mismatch"); }
static_assert(strix::cohort_row_split(24U)[1].offset == 16U);
static_assert(strix::cohort_row_split(15U)[0].rows == 16U);
static_assert(strix::cohort_row_split(31U)[0].rows == 32U);
void check_row_splits() {
    constexpr std::array<std::array<std::size_t, 3>, 23> expected{{
        {8,1,0}, {8,2,0}, {8,3,0}, {8,4,0}, {8,5,0}, {8,6,0}, {16,0,0}, // 9..15
        {16,0,0}, {16,1,0}, {16,2,0}, {16,3,0}, {16,4,0}, {16,5,0}, {16,6,0}, {16,7,0}, {16,8,0}, // 16..24
        {16,8,1}, {16,8,2}, {16,8,3}, {16,8,4}, {16,8,5}, {16,8,6}, {32,0,0} // 25..31
    }};
    for (std::size_t count = 0; count <= 32U; ++count) {
        const auto plan = strix::cohort_row_split(count);
        std::size_t total = 0;
        bool unused = false;
        for (std::size_t i = 0; i < plan.size(); ++i) {
            const auto& launch = plan[i];
            if (count >= 9U && count <= 31U) require(launch.rows == expected[count - 9U][i]);
            else require(launch.rows == (i == 0U ? count : 0U));
            if (launch.rows == 0U) {
                unused = true;
                require(launch.offset == 0U);
                continue;
            }
            require(!unused && launch.offset == total);
            require(launch.rows <= 8U || launch.rows == 16U || launch.rows == 32U);
            total += launch.rows;
        }
        require(total == count + (count == 15U || count == 31U ? 1U : 0U));
    }
    bool rejected = false;
    try { static_cast<void>(strix::cohort_row_split(33U)); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
}
void check(std::span<const std::uint32_t> rows, std::size_t old_steps, std::size_t new_steps) {
    const auto old_plan = strix::cohort_step_plan(rows, 8U);
    const auto new_plan = strix::cohort_step_plan(rows, 16U);
    require(old_plan.size() == old_steps && new_plan.size() == new_steps);
    // Row identity is (session, window offset); preserve every complete session
    // window in order, including its padding rows, across both execution plans.
    std::vector<std::pair<std::size_t, std::uint32_t>> reference;
    for (std::size_t session = 0; session < rows.size(); ++session)
        for (std::uint32_t offset = 0; offset < rows[session]; ++offset) reference.emplace_back(session, offset);
    for (const auto* plan : {&old_plan, &new_plan}) {
        std::vector<std::pair<std::size_t, std::uint32_t>> actual;
        const std::uint32_t cap = plan == &old_plan ? 8U : 16U;
        for (const auto& chunk : *plan) {
            std::uint32_t total = 0;
            for (const auto session : chunk) {
                total += rows[session];
                for (std::uint32_t offset = 0; offset < rows[session]; ++offset) actual.emplace_back(session, offset);
            }
            require(total <= cap);
            require(chunk.size() < 2U || total == 4U || total == 8U || total == 16U);
        }
        require(actual == reference);
    }
}
}
int main(int argc, char* argv[]) {
    check_row_splits();
    if (argc > 1) require(strix::cohort_step_rows() == (std::string_view(argv[1]) == "16" ? 16U : 8U));
    check(std::array<std::uint32_t, 8>{2,2,2,2,2,2,2,2}, 2, 1);
    check(std::array<std::uint32_t, 6>{4,4,2,2,2,2}, 2, 1);
    check(std::array<std::uint32_t, 2>{8,8}, 2, 1);
    check(std::array<std::uint32_t, 3>{4,4,4}, 2, 2);
    check(std::array<std::uint32_t, 3>{2,2,2}, 2, 2);
    check(std::array<std::uint32_t, 0>{}, 0, 0);
    check(std::array<std::uint32_t, 1>{8}, 1, 1);
    require(strix::cohort_step_rows() == 8U || strix::cohort_step_rows() == 16U);
    std::cout << "PASS: row order, session windows, padding and step counts\n";
}
