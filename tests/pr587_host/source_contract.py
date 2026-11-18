"""Check receipt hooks in the HIP server that cannot be linked into a host test.

The C++ tests exercise their shared implementations and the real row planner.
These checks additionally catch a missing adapter call on any admission path.
"""
import pathlib
import sys


def check(source, case):
    def region(start, end):
        return source.split(start, 1)[1].split(end, 1)[0]

    def require(text, fragment):
        if fragment not in text:
            raise AssertionError(f"{case}: missing server hook: {fragment}")

    if case == "stop":
        require(region("const auto publish =", "const auto unpark_oldest"), "serve::retained_output_end(")
        require(region("const auto run_commit =", "const auto publish ="),
                "r.run_tokens.commit(run_window, r.seq.slot, r.seq.generated, r.published, job.max_tokens, job.stop_tokens)")
        require(region("const auto batch_step =", "if (++prof.steps"), "emitted += run_commit(*r)")
    elif case == "restored":
        direct = region("if (const auto m = parked.empty() || image ? -1", "if (sched.mode ==")
        require(direct, "run_window.request(now)")
        require(direct, "run_window.restored(now, run_commit(*running.back()))")
        if direct.index("run_window.request(now)") > direct.index("run_window.restored(now"):
            raise AssertionError("restored: request must open before output accounting")
        require(region("const auto take_parked =", "// one batch step"),
                "run_window.restored(run_ns(Clock::now()), run_commit(r))")
    elif case == "rows":
        batch = region("const auto batch_step =", "if (++prof.steps")
        require(batch, "serve::run_step_work(plan.owner, [&](std::uint32_t i) { return seqs[i]->reading_prompt(); })")
        require(batch, "run_window.planned_step(run_ns(t0), run_ns(last_round), run_work, emitted)")
        if batch.index("serve::run_step_work") > batch.index("fwd.step(plan.rows)"):
            raise AssertionError("rows: classification must precede execution")
    elif case == "prefill":
        for start, end, commit in [
            ("device_prefill(r, r.seq.prompt.size() >= sched.min_rows);", "if (step_trace)", "run_commit(r)"),
            ("run_unit(strix::deepseek41::sched::row_budget", "if (!active) publish()", "run_commit(*unit_of)"),
            ("device_prefill(*r, false);", "prefilled = true;", "run_commit(*r)"),
        ]:
            hook = region(start, end)
            require(hook, "if (run_on)")
            require(hook, "run_window.admission(")
            require(hook, commit)
    else:
        raise ValueError(case)


if __name__ == "__main__":
    try:
        check(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"), sys.argv[2])
    except (AssertionError, IndexError, ValueError) as exc:
        sys.exit(f"receipt adapter contract failed: {exc}")
