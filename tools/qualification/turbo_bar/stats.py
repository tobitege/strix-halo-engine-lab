"""Paired statistics: exact conservative binary bounds; t/bootstrap bounds for continuous endpoints."""
import math

import numpy as np

from tools.qualification.option_qual.stats import t_quantile


def ci(d, level=0.95, B=10000, seed=0):
    """Two-sided CI of the mean: union (wider envelope) of the t interval and a percentile bootstrap."""
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    mean = float(d.mean()) if n else float("nan")
    if n < 2:
        return mean, float("-inf"), float("inf")
    se = float(d.std(ddof=1)) / math.sqrt(n)
    t = t_quantile(1 - (1 - level) / 2, n - 1)
    lo, hi = mean - t * se, mean + t * se
    rng = np.random.default_rng(seed)
    means = np.empty(B)
    step = max(1, 4_000_000 // n)
    for s in range(0, B, step):
        k = min(step, B - s)
        means[s:s + k] = d[rng.integers(0, n, size=(k, n))].mean(axis=1)
    a = (1 - level) / 2
    blo, bhi = np.quantile(means, [a, 1 - a])
    return mean, float(min(lo, blo)), float(max(hi, bhi))


def binary_ci(d, level=0.95, B=10000, seed=0):
    """Bound p(a-only) - p(b-only) using simultaneous Clopper-Pearson bounds.

    Four one-sided binomial bounds each spend (1-level)/4 error probability.
    The union bound requires no independence between the discordant counts and
    covers unseen discordances, including an entirely concordant sample.
    B and seed are accepted for compatibility with the continuous CI interface.
    """
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    if not 0 < level < 1:
        raise ValueError("confidence level must be between zero and one")
    if n == 0:
        return float("nan"), -1.0, 1.0
    alpha = (1 - level) / 4

    def upper(k):
        if k == n:
            return 1.0
        if k == 0:
            return -math.expm1(math.log(alpha) / n)
        j = np.arange(k + 1)
        log_choose = np.array([math.lgamma(n + 1) - math.lgamma(int(i) + 1)
                               - math.lgamma(n - int(i) + 1) for i in j])
        lo, hi = 0.0, 1.0
        for _ in range(60):
            p = (lo + hi) / 2
            log_cdf = np.logaddexp.reduce(log_choose + j * math.log(p) + (n - j) * math.log1p(-p))
            if log_cdf > math.log(alpha):
                lo = p
            else:
                hi = p
        return hi

    plus, minus = int((d == 1).sum()), int((d == -1).sum())
    return float(d.mean()), 1 - upper(n - plus) - upper(minus), upper(plus) + upper(n - minus) - 1


def verdict(d, limit=0.0, level=0.95, B=10000, seed=0, *, binary=False):
    """PASS iff CI_hi < limit; FAIL iff CI_lo > limit; otherwise INCONCLUSIVE (never PASS when the CI crosses)."""
    d = np.asarray(d, dtype=np.float64)
    n = int(d.size)
    out = {"n": n, "limit": limit}
    if n < 3 or not np.isfinite(d).all():
        out.update(mean=float(d.mean()) if n else None, lo=None, hi=None, verdict="INCONCLUSIVE",
                   n_needed="unbounded", message="enlarge sample")
        return out
    mean, lo, hi = (binary_ci if binary else ci)(d, level, B, seed)
    out.update(mean=mean, lo=lo, hi=hi)
    if hi < limit:
        out.update(verdict="PASS", n_needed=None, message="CI upper bound below limit")
    elif lo > limit:
        out.update(verdict="FAIL", n_needed=None, message="CI lower bound above limit")
    else:
        gap = abs(limit - mean)
        need = "unbounded" if gap == 0 else max(n + 1, math.ceil(n * (((hi - lo) / 2) / gap) ** 2))
        out.update(verdict="INCONCLUSIVE", n_needed=need, message="enlarge sample")
    return out


def paired_binary(a, b, limit=0.0, **kw):
    """a, b: 0/1 success per item; d = a - b. Reports rates and discordant counts."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired arrays differ in length")
    if not (np.isin(a, [0, 1]).all() and np.isin(b, [0, 1]).all()):
        raise ValueError("paired outcomes must be binary")
    out = verdict(a - b, limit, binary=True, **kw)
    out.update(rate_a=float(a.mean()) if a.size else None, rate_b=float(b.mean()) if b.size else None,
               a_only=int(((a == 1) & (b == 0)).sum()), b_only=int(((a == 0) & (b == 1)).sum()),
               both=int(((a == 1) & (b == 1)).sum()), neither=int(((a == 0) & (b == 0)).sum()))
    return out


def t_ci(samples, level=0.95):
    x = np.asarray(samples, dtype=np.float64)
    n = x.size
    m = float(x.mean())
    if n < 2:
        return m, float("-inf"), float("inf")
    h = t_quantile(1 - (1 - level) / 2, n - 1) * float(x.std(ddof=1)) / math.sqrt(n)
    return m, m - h, m + h
