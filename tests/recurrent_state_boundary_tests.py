"""Wiring regressions for GPU-only entry points; arithmetic is tested in C++.

These check fail-closed admission stays before setup. They are not GPU execution
or a substitute for full logits/state comparisons when typed routes are enabled.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RecurrentStateBoundaries(unittest.TestCase):
    def test_cuda_application_rejects_before_device_or_cache_setup(self):
        source = (ROOT / "src/cuda/models/qwen/qwen_prefill_driver.cu").read_text()
        start = source.index("Prefill::Prefill(const Config& config)")
        startup = source[start:source.index("m.state_layer_floats = recurrent_state_bytes(", start)]
        self.assertIn("require_f32_recurrent_state(m.state_type,", startup)
        self.assertIn("schedule-independent prefill and prefix reuse unqualified", startup)

    def test_sender_rejects_before_cached_layout_or_http(self):
        source = (ROOT / "src/cuda/models/qwen/qwen_handoff.cpp").read_text()
        start = source.index("const Layout& Sender::layout()")
        admission = source[start:source.index("if (impl_->have_layout)", start)]
        self.assertIn("require_f32_recurrent_state(recurrent_state_policy().type,", admission)

    def test_policy_uses_governed_environment_reader(self):
        source = (ROOT / "include/strix/recurrent_state_policy.hpp").read_text()
        self.assertNotIn("std::getenv", source)
        self.assertIn("envcfg::text(kRecurrentStateEnv)", source)


if __name__ == "__main__":
    unittest.main()
