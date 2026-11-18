"""Compile the driver's real whole-layer planning route without CUDA or a model."""
from pathlib import Path
import sys

root, out = map(Path, sys.argv[1:])
source = (root / 'src/cuda/models/ds41/ds41_prefill_driver.cu').read_text(encoding='utf-8')
start = source.index('std::vector<ExpertTier> default_tiers(')
end = source.index('\n}\n', start) + 3
(out / 'default_tiers.inc').write_text(source[start:end], encoding='utf-8')
start = source.index('    const auto host_pin_fallback =')
end = source.index('\n    };', start) + len('\n    };')
(out / 'host_pin_fallback.inc').write_text(source[start:end], encoding='utf-8')
start = source.index('        if (!o.expert_residency) {', end)
end = source.index('            require(m.tiers.size()', start)
(out / 'whole_layer_route.inc').write_text(source[start:end] + '\n}\n', encoding='utf-8')
