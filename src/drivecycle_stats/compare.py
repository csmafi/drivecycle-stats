"""Two-sample comparison: energy distance and TOST equivalence testing.

Energy distance
----------------
Reference: Szekely, G.J. and Rizzo, M.L. (2013). "Energy statistics: A
class of statistics based on distances." Journal of Statistical Planning
and Inference, 143(8), 1249-1272.

The standard-form energy distance between two samples X and Y is::

    E(X, Y) = 2 * mean|x - y| - mean|x - x'| - mean|y - y'|

where the first term averages pairwise distances between X and Y, and
the second and third terms average pairwise distances within X and
within Y respectively (excluding a point paired with itself). E is
never negative and is zero only when X and Y are drawn from the same
distribution.

E is sensitive to differences across the WHOLE distribution -- shape
and tails included -- not only to a difference in means. Two samples
with the same average can still separate clearly on this statistic.
This is the reason to prefer it over a t-test when the question is
"are these two samples the same distribution" rather than "do these two
samples have the same mean".

E is NOT scale-invariant: multiplying one column by 10 changes E. How
the input columns are standardised therefore changes the answer, and
that choice is left to the caller (see the ``standardize`` note on each
function below).

TOST (two one-sided tests) for equivalence
-------------------------------------------
Reference: Schuirmann, D.J. (1987). "A comparison of the two one-sided
tests procedure and the power approach for assessing the equivalence
of average bioavailability." Journal of Pharmacokinetics and
Biopharmaceutics, 15(6), 657-680.

A conventional two-sample test asks "is there a significant
difference?" and a non-significant result is not evidence that two
samples are equivalent -- it may simply mean the test had too little
power to detect a real difference. TOST instead asks "can we reject
the hypothesis that the difference exceeds a pre-specified bound?" by
running two one-sided tests, one against each side of the bound, and
requires both to be significant to declare equivalence.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _pairwise_abs_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a_i - b_j| for all i, j, shape (len(a), len(b))."""
    return np.abs(a.reshape(-1, 1) - b.reshape(1, -1))


def _standardize(x: np.ndarray, y: np.ndarray, method: str | None):
    """Apply the requested standardisation, fit on the pooled sample."""
    if method is None:
        return x, y
    if method == "pooled_std":
        pooled = np.concatenate([x, y])
        scale = np.std(pooled, ddof=1)
        if scale == 0:
            raise ValueError("pooled standard deviation is zero; cannot standardize")
        return x / scale, y / scale
    raise ValueError(f"unknown standardize method: {method!r}")


def energy_distance(X, Y, standardize: str | None = None) -> float:
    """Standard-form energy distance between two 1-D samples.

    Parameters
    ----------
    X, Y : array-like
        One-dimensional samples. Need not be the same length.
    standardize : {None, "pooled_std"}, optional
        If ``"pooled_std"``, both samples are divided by the standard
        deviation of the pooled sample before computing the distance.
        Energy distance is not scale-invariant, so this choice affects
        the result; pass already-standardised inputs and leave this as
        ``None`` if you prefer to control scaling yourself.

    Returns
    -------
    float
        The energy distance E(X, Y) >= 0. E == 0 only when X and Y are
        drawn from the same distribution (in the population limit;
        with finite samples E is a small positive number even when the
        samples come from the same distribution).
    """
    X = np.asarray(X, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    if X.size == 0 or Y.size == 0:
        raise ValueError("X and Y must be non-empty")

    X, Y = _standardize(X, Y, standardize)

    dXY = _pairwise_abs_matrix(X, Y).mean()

    dXX = _pairwise_abs_matrix(X, X)
    dXX_mean = dXX.mean()
    dYY = _pairwise_abs_matrix(Y, Y)
    dYY_mean = dYY.mean()

    return float(2 * dXY - dXX_mean - dYY_mean)


def energy_test(X, Y, n_perm: int = 1000, seed: int | None = None, standardize: str | None = None):
    """Two-sample permutation test on the energy distance statistic.

    The pooled pairwise absolute-distance matrix is built once and
    re-indexed for each permutation, rather than recomputing pairwise
    distances from scratch every time.

    Parameters
    ----------
    X, Y : array-like
        One-dimensional samples.
    n_perm : int, default 1000
        Number of random permutations for the null distribution. The
        smallest reportable p-value is ``1 / (n_perm + 1)``; with the
        default of 1000 permutations that is 0.000999, commonly quoted
        as a resolution limit of about 0.001.
    seed : int, optional
        Seed for the permutation random number generator, for
        reproducibility.
    standardize : {None, "pooled_std"}, optional
        See :func:`energy_distance`.

    Returns
    -------
    dict
        ``statistic``: observed energy distance.
        ``null_distribution``: array of length ``n_perm`` of energy
        distances computed on random relabellings of the pooled sample.
        ``p_value``: fraction of null draws >= the observed statistic
        (permutation p-value, with the +1 correction: (count + 1) /
        (n_perm + 1)).
    """
    X = np.asarray(X, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    if X.size == 0 or Y.size == 0:
        raise ValueError("X and Y must be non-empty")
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")

    X, Y = _standardize(X, Y, standardize)

    n, m = len(X), len(Y)
    pooled = np.concatenate([X, Y])

    # Pooled pairwise absolute-distance matrix, built once.
    D = _pairwise_abs_matrix(pooled, pooled)

    def _stat_from_labels(labels_x_mask: np.ndarray) -> float:
        idx_x = np.where(labels_x_mask)[0]
        idx_y = np.where(~labels_x_mask)[0]

        dXY = D[np.ix_(idx_x, idx_y)].mean()
        dXX_mean = D[np.ix_(idx_x, idx_x)].mean()
        dYY_mean = D[np.ix_(idx_y, idx_y)].mean()

        return 2 * dXY - dXX_mean - dYY_mean

    observed_mask = np.concatenate([np.ones(n, dtype=bool), np.zeros(m, dtype=bool)])
    observed = _stat_from_labels(observed_mask)

    rng = np.random.default_rng(seed)
    null_draws = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm_mask = rng.permutation(observed_mask)
        null_draws[i] = _stat_from_labels(perm_mask)

    p_value = (np.sum(null_draws >= observed) + 1) / (n_perm + 1)

    return {
        "statistic": float(observed),
        "null_distribution": null_draws,
        "p_value": float(p_value),
    }


def tost(x, y, bound: float, alpha: float = 0.05):
    """Two one-sided tests (TOST) for equivalence of two sample means.

    Tests the null hypothesis that the difference in means falls
    outside [-bound, +bound] against the alternative that it falls
    inside that range. Equivalence (both one-sided tests significant)
    supports the claim that the mean difference is smaller than
    ``bound`` in magnitude.

    A non-significant result on this test says nothing about
    equivalence in either direction -- it says the two one-sided
    hypotheses could not both be rejected at the chosen alpha. This is
    the complement of a conventional difference test: a non-significant
    difference test is not evidence of equivalence, which is the whole
    reason TOST exists.

    Parameters
    ----------
    x, y : array-like
        One-dimensional samples, already in the units the equivalence
        bound is expressed in. This function does not standardise its
        inputs; the caller supplies ``bound`` in the same units as
        ``x`` and ``y``.
    bound : float
        The equivalence bound, must be > 0. Interpreted symmetrically:
        equivalence is declared if the mean difference lies strictly
        within (-bound, +bound) at the given alpha.
    alpha : float, default 0.05
        Significance level for each one-sided test.

    Returns
    -------
    dict
        ``p_lower``: p-value for testing mean(x) - mean(y) > -bound.
        ``p_upper``: p-value for testing mean(x) - mean(y) < +bound.
        ``p_value``: max(p_lower, p_upper), the overall TOST p-value.
        ``equivalent``: True if ``p_value < alpha``.
        ``mean_difference``: mean(x) - mean(y).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 2 or y.size < 2:
        raise ValueError("x and y must each have at least 2 observations")
    if bound <= 0:
        raise ValueError("bound must be > 0")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")

    mean_diff = float(np.mean(x) - np.mean(y))

    # Two one-sided Welch t-statistics, one against each side of the bound.
    nx, ny = len(x), len(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    se = np.sqrt(var_x / nx + var_y / ny)
    if se == 0:
        raise ValueError("standard error of the mean difference is zero; cannot test")

    # Welch-Satterthwaite degrees of freedom.
    df = (var_x / nx + var_y / ny) ** 2 / (
        (var_x / nx) ** 2 / (nx - 1) + (var_y / ny) ** 2 / (ny - 1)
    )

    t_lower = (mean_diff - (-bound)) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)

    t_upper = (mean_diff - bound) / se
    p_upper = stats.t.cdf(t_upper, df)

    p_value = max(p_lower, p_upper)

    return {
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "p_value": float(p_value),
        "equivalent": bool(p_value < alpha),
        "mean_difference": mean_diff,
    }


def tost_variance_ratio(x, y, bound_low: float, bound_high: float, alpha: float = 0.05):
    """TOST for equivalence of variance, expressed as a ratio.

    Tests whether the ratio var(x) / var(y) lies within
    (bound_low, bound_high) using the F distribution, following the
    same two-one-sided-test logic as :func:`tost`.

    Parameters
    ----------
    x, y : array-like
        One-dimensional samples.
    bound_low, bound_high : float
        Lower and upper bounds on the variance ratio var(x)/var(y) that
        together define the equivalence region. Both must be > 0 and
        ``bound_low < 1 < bound_high`` is the typical, though not
        required, choice (e.g. 0.8 and 1.25).
    alpha : float, default 0.05
        Significance level for each one-sided test.

    Returns
    -------
    dict
        ``p_lower``: p-value for testing ratio > bound_low.
        ``p_upper``: p-value for testing ratio < bound_high.
        ``p_value``: max(p_lower, p_upper).
        ``equivalent``: True if ``p_value < alpha``.
        ``variance_ratio``: var(x) / var(y).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 2 or y.size < 2:
        raise ValueError("x and y must each have at least 2 observations")
    if bound_low <= 0 or bound_high <= 0:
        raise ValueError("bound_low and bound_high must be > 0")
    if bound_low >= bound_high:
        raise ValueError("bound_low must be < bound_high")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")

    nx, ny = len(x), len(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    if var_y == 0:
        raise ValueError("variance of y is zero; cannot form a ratio")

    ratio = var_x / var_y
    df1, df2 = nx - 1, ny - 1

    # F = ratio / bound tested against F(df1, df2).
    f_lower = ratio / bound_low
    p_lower = 1 - stats.f.cdf(f_lower, df1, df2)

    f_upper = ratio / bound_high
    p_upper = stats.f.cdf(f_upper, df1, df2)

    p_value = max(p_lower, p_upper)

    return {
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "p_value": float(p_value),
        "equivalent": bool(p_value < alpha),
        "variance_ratio": float(ratio),
    }
