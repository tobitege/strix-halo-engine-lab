#include "strix/session/parallel_job.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <utility>

using namespace strix::session;
namespace {
void check(bool ok) { if (!ok) throw std::runtime_error("parallel job contract"); }
template<class F> void reject(F&& f) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("expected invalid job");
}
ParallelJob example() {
    ParallelJob job;
    job.request_id = "request";
    job.points = {{"point", {"answer", 0, 16, {}}, {}}};
    job.merge.point_order = {"point"};
    job.policy = {32, 1};
    validate_job(job);
    return job;
}
}
int main() {
    try {
        auto job = example();
        std::vector<std::uint32_t> construction{1, 2, 3, 4};
        auto* alias = construction.data();
        job.prefix = {TokenBuffer(construction), 1, 2};
        job.points.push_back({"tail", {"finish", 1, 12, {"evidence"}}, {}});
        job.dependencies = {{"point", "tail"}};
        job.digests = {{"evidence", "point", {TokenBuffer(construction), 0, 4}}};
        job.merge.point_order.push_back("tail");
        validate_job(job);
        const auto frozen = job;
        alias[1] = 99; construction.clear(); construction.shrink_to_fit();
        job.prefix.buffer = {};
        check(frozen.prefix.tokens()[0] == 2 && frozen.digests[0].evidence.tokens()[1] == 2);
        validate_job(frozen);
        // Every case is checked in Release, too. Structural validation is pure.
        for (int n = 0; n < 24; ++n) {
            auto bad = frozen;
            switch (n) {
            case 0: bad.request_id.clear(); break;
            case 1: bad.points[1].id = "point"; break;
            case 2: bad.points[0].id.clear(); break;
            case 3: bad.dependencies[0].before = "missing"; break;
            case 4: bad.dependencies[0].after = "missing"; break;
            case 5: bad.dependencies.push_back({"tail", "point"}); break;
            case 6: bad.dependencies.push_back(bad.dependencies[0]); break;
            case 7: bad.dependencies.push_back({"tail", "tail"}); break;
            case 8: bad.policy.token_budget = 0; break;
            case 9: bad.policy.max_lanes = 0; break;
            case 10: bad.points[0].contract.max_tokens = 0; break;
            case 11: bad.points[0].contract.min_tokens = 17; break;
            case 12: bad.points[0].contract.max_tokens = 33; break;
            case 13: bad.points[0].contract.min_tokens = 16;
                     bad.points[1].contract.min_tokens = bad.points[1].contract.max_tokens = 17; break;
            case 14: bad.prefix.offset = UINT64_MAX; break;
            case 15: bad.digests[0].evidence.length = UINT64_MAX; break;
            case 16: bad.points[0].suffix = {{}, 1, 0}; break;
            case 17: bad.merge.point_order.pop_back(); break;
            case 18: bad.merge.point_order[1] = "point"; break;
            case 19: std::swap(bad.merge.point_order[0], bad.merge.point_order[1]); break;
            case 20: bad.digests[0].producer = "missing"; break;
            case 21: bad.digests.push_back(bad.digests[0]); break;
            case 22: bad.points[1].contract.evidence_ids.push_back("missing"); break;
            case 23: bad.dependencies.clear(); break;
            }
            reject([&] { validate_job(bad); });
            validate_job(frozen);
        }
        auto bad = frozen;
        bad.points[0].contract.evidence_ids = {"evidence"};
        reject([&] { validate_job(bad); }); // producer cannot consume its own digest
        bad = frozen; bad.points[1].contract.evidence_ids.push_back("evidence");
        reject([&] { validate_job(bad); });
        auto transitive = frozen;
        transitive.points.insert(transitive.points.begin(), {"middle", {}, {}});
        transitive.dependencies = {{"point", "middle"}, {"middle", "tail"}};
        transitive.merge.point_order = {"point", "middle", "tail"};
        validate_job(transitive);
        // A receipt is one owning request-wide value, independent of job lifetime.
        JobReceipt receipt{frozen.request_id, frozen.policy.token_budget, 0, {}};
        check(receipt.request_id == "request" && receipt.budget_tokens == 32);
        std::cout << "parallel job contracts passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
