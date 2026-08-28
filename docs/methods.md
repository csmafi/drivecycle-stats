# Methods

This note describes what each method in `drivecycle-stats` computes, where it comes from, what it assumes, and how it can fail. It contains no results from any private dataset and no reference to any driving-cycle generator. Every example number in this note comes from the public Vehicle Energy Dataset (VED) or from a synthetic trace built only to show the method.

## 1. Microtrip segmentation

**What it computes.** A microtrip is the part of a trip between the start of one stop and the start of the next. A stop is a run of consecutive seconds at or below a speed threshold (default 1.0 km/h), lasting at least a minimum duration (default 1 second).

**Source.** This is a standard idea used across the driving-cycle literature; the specific implementation here is new.

**Assumptions.**

- The input trace is at a fixed 1-second sampling rate. Raw loggers are often not evenly sampled (VED is one example, see Section 6); the caller must resample first.
- A single fixed speed threshold applies for the whole trace. A vehicle that idles at, say, 2 km/h due to sensor drift would not register as stopped with the default threshold.

**Failure modes.**

- Too low a threshold treats sensor noise as movement, and the trip never appears to stop.
- Too high a threshold merges genuinely separate low-speed manoeuvres (e.g. parking) into the same microtrip.
- Too short a minimum stop duration creates many very short microtrips from brief red lights or momentary GPS glitches.

## 2. Trip and microtrip descriptors

**What it computes.** A fixed set of summary numbers for one trip or microtrip: distance, duration, average speed (including idle seconds), running speed (excluding idle seconds), average positive acceleration, RMS acceleration, idle share, number of stops, mean stop duration, 95th-percentile speed, and mean positive vehicle-specific power.

**Source.** These are standard descriptors used across driving-cycle and vehicle-emissions literature. Different papers define some of them slightly differently (in particular, whether idle seconds are included in "average speed"), so each definition is stated explicitly in the code rather than assumed.

**Assumptions.**

- Acceleration is assumed already computed and supplied as a column; this module does not compute it from speed (see `io_ved.py`, Section 6, for the resampling and differencing step that produces it from VED).
- "Average positive acceleration" only counts seconds with acceleration above 0.1 m/s², to avoid diluting the average with near-zero noise.

**Failure modes.**

- Distance, if not supplied directly, is obtained by integrating speed over 1-second steps. This assumes exactly 1 Hz sampling; on unevenly sampled data it will be biased.
- The 95th-percentile speed can be sensitive to a small number of trips (a short trace has few points to estimate a percentile from).

## 3. Vehicle-specific power (VSP)

**What it computes.** An estimate of engine power demand per unit vehicle mass, in watts per kilogram, from speed and acceleration:

```
u = speed_kmh / 3.6
vsp = u * (1.1 * a + 0.132) + 0.000302 * u^3
```

**Source.** Jimenez-Palacios (1999), PhD thesis, Massachusetts Institute of Technology. The coefficients are the commonly cited light-duty-vehicle form.

**Assumptions.**

- Road grade is assumed zero. This is a limitation of the calculation, not a validated result: on graded roads, VSP will be biased low downhill and high uphill.
- The coefficients are a generic light-duty fit, not refit to any specific vehicle.

**Failure modes.** Applying this formula to a heavy-duty vehicle, or to a route with substantial elevation change, will give a systematically wrong VSP.

## 4. Energy distance and its permutation test

**What it computes.** The standard-form energy distance between two one-dimensional samples X and Y:

```
E(X, Y) = 2 * mean|x - y| - mean|x - x'| - mean|y - y'|
```

where the first term averages pairwise absolute distances between X and Y, and the second and third average pairwise distances within X and within Y. E is non-negative and is zero only when X and Y come from the same distribution.

The accompanying permutation test estimates a p-value by repeatedly relabelling the pooled sample at random and recomputing E, then comparing the observed E against that null distribution.

**Source.** Szekely and Rizzo (2013), "Energy statistics: A class of statistics based on distances," *Journal of Statistical Planning and Inference*, 143(8), 1249-1272. Written here directly from the published definition, not from any file in a private project.

**What it is sensitive to.** E responds to differences across the whole distribution, including shape and the tails, not only to a difference in means. Two samples with the same average can still separate clearly on E. This is the reason to prefer it over a t-test when the underlying question is "are these two samples the same distribution" rather than "do these two samples have the same average."

**Assumptions and a scaling note.** E is not scale-invariant: multiplying one column by a constant changes the answer. How the input columns are standardised therefore changes the result, and this package leaves that choice to the caller rather than picking a default silently. If comparing several descriptors together, decide on and record a standardisation before comparing across studies.

**Failure modes.**

- With `n_perm` permutations, the smallest reportable p-value is `1 / (n_perm + 1)`. With the default 1000 permutations that is about 0.001; a reported p-value of 0.001 should be read as "at least this small," not as an exact value.
- Small samples (a handful of points per group) give a very coarse null distribution and unreliable p-values regardless of how many permutations are run.

## 5. TOST equivalence testing

**What it computes.** Two one-sided tests (TOST) for whether the difference between two sample means falls within a pre-specified bound. Given samples x and y and a bound b > 0, TOST tests the null hypothesis that the true difference is outside (-b, +b) against the alternative that it is inside; both one-sided tests must be significant (at the chosen alpha) to declare equivalence. A variance-ratio version, `tost_variance_ratio`, applies the same logic to the ratio of variances using the F distribution.

**Source.** Schuirmann (1987), "A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability," *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657-680. Written here directly from the published procedure.

**Why this and not a plain difference test.** A conventional two-sample test (e.g. a t-test) asks "is there a significant difference?" A non-significant result does not mean the two samples are equivalent — it may simply mean the test lacked the power to detect a real difference. TOST instead asks the question directly: can we rule out a difference larger than the bound we care about?

**Assumptions.**

- The equivalence bound is supplied by the caller in the same units as the data. This package does not ship a default bound, a default choice of which descriptor to compare on, or a combined pass/fail rule across multiple descriptors — those are study-specific decisions the caller must make and record.
- The test as implemented assumes approximately normal sampling distributions of the means (via Welch's t-test with Welch-Satterthwaite degrees of freedom), which is reasonable for moderate-to-large samples by the central limit theorem but should not be assumed for very small or heavily skewed samples.

**Failure modes.** Choosing a bound after seeing the data (rather than specifying it in advance, based on what difference would matter practically) undermines the interpretation of the test as evidence of equivalence.

## 6. VED loader and cleaning

**What it does.** Reads raw VED per-second logger files, resamples each trip to a uniform 1 Hz grid, computes acceleration by differencing, clips physically implausible acceleration values, and drops trips shorter than a minimum duration. Every step is counted and returned in a report dictionary so the caller can see exactly what was dropped and why.

**Why resampling is necessary.** Raw VED timestamps are not evenly spaced. In one representative week file, the gap between consecutive samples within a single trip ranged from about 100 milliseconds to about 2800 milliseconds, with a median close to 1000 milliseconds but substantial spread around it. Differencing raw speed against these irregular gaps would produce noisy, physically meaningless acceleration values, so this package resamples to a uniform 1-second grid by linear interpolation before differencing.

**Assumptions.**

- Linear interpolation between samples is a reasonable approximation of the vehicle's speed in the gap. Over gaps close to 1 second this is a mild assumption; the few gaps that reach 2-3 seconds interpolate over more uncertainty.
- The acceleration clip threshold (default 5 m/s², symmetric) is a deliberately conservative, non-data-driven bound meant to catch clearly impossible values (e.g. from a GPS glitch), not to shape the distribution of real accelerations.

**A known inconsistency in the source data.** The two VED static vehicle-parameter files use different column names for the same field (`Vehicle Type` in one file, `EngineType` in the other). The loader renames both to `EngineType` when merging; this is a data-cleaning step, not a claim that the underlying values are equivalent between vehicle categories.

**Failure modes.**

- A trip with very large timestamp gaps will have its interior speed effectively guessed by interpolation; the report's row counts can be used to check how much of a given trip's duration this affects.
- Auxiliary VED columns (hybrid battery state, air conditioning power, and similar) are mostly missing for conventional gasoline vehicles by design; this loader does not use them and callers should expect high missingness if they read those columns directly.

## 7. Low-density outlier flag

**What it computes.** Fits a Gaussian kernel density estimate on a two-column array (e.g. a pair of descriptors for many trips), z-scores the resulting density, and flags the lowest percentile of points as candidate outliers.

**Source.** A standard idea (points in the sparsest region of an estimated density are candidate outliers); the implementation here is specific to this package.

**Assumptions and limits.** The percentile threshold is a tuning constant with no theoretical justification behind the exact number. Any conclusion that depends on which points get flagged should be checked at more than one percentile value to confirm it is not an artefact of that choice. This package deliberately does not ship a fixed pair of descriptors to run this on, nor a rule for combining flags across more than one descriptor pair — those choices belong to the analysis, not the tool.

**Failure modes.** With few points, the KDE bandwidth estimate is unstable and the flagged set can change substantially between runs on slightly different samples.

## References

- Jimenez-Palacios, J.L. (1999). *Understanding and Quantifying Motor Vehicle Emissions with Vehicle Specific Power and TILDAS Remote Sensing.* PhD thesis, Massachusetts Institute of Technology.
- Szekely, G.J. and Rizzo, M.L. (2013). "Energy statistics: A class of statistics based on distances." *Journal of Statistical Planning and Inference*, 143(8), 1249-1272.
- Schuirmann, D.J. (1987). "A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability." *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657-680.
- Oh, G., LeBlanc, D.J., Peng, H. (2020). "Vehicle Energy Dataset (VED), A Large-scale Dataset for Vehicle Energy Consumption Research." *IEEE Transactions on Intelligent Transportation Systems*. Repository: https://github.com/gsoh/VED (Apache-2.0 licence).
