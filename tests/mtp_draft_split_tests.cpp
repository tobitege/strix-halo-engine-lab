// strix::mtp::DraftSplit (include/strix/mtp_draft_split.hpp): first-segment choice of a split draft chain.
#include "strix/mtp_draft_split.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}
}  // namespace

int main() {
    using strix::mtp::DraftSplit;
    {
        DraftSplit off(0);
        check(off.first_segment(7) == 7, "mode 0 keeps the whole chain");
        DraftSplit fixed(3);
        check(fixed.first_segment(7) == 3 && fixed.first_segment(2) == 2, "fixed N, capped by depth");
        DraftSplit any;
        check(any.first_segment(1) == 1 && any.first_segment(0) == 0, "depth < 2 is never split");
        check(any.first_segment(7) == 2, "auto without samples starts at 2");
    }
    {
        // prose-like: 54 % keep 1, 26 % keep 2, 14 % keep 3, 6 % keep 7 -> a = 2 (expected steps 2 + .2 * 5.25 = 3.05)
        DraftSplit s;
        for (int i = 0; i < 54; ++i) s.note(1);
        for (int i = 0; i < 26; ++i) s.note(2);
        for (int i = 0; i < 14; ++i) s.note(3);
        for (int i = 0; i < 6; ++i) s.note(7);
        check(s.first_segment(7) == 2, "prose-like histogram picks 2");
    }
    {
        // every chain keeps the full depth: splitting only adds a host turn -> no split
        DraftSplit s;
        for (int i = 0; i < 20; ++i) s.note(7);
        check(s.first_segment(7) == 7, "full-depth chains are not split");
        check(s.kept(7) == 20 && s.kept(99) == 0, "histogram counts");
    }
    {
        // kept counts beyond the bins land in the last bin
        DraftSplit s;
        s.note(1000);
        check(s.kept(DraftSplit::kBins - 1) == 1, "overflow bin");
    }
    std::printf("%s mtp_draft_split_tests\n", failures ? "FAIL" : "PASS");
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
