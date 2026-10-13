# LM literature: problem inventory and additions

Reviewed **13 October 2026** against the current working tree. This is a
documentation backlog: no test cases, benchmark cases, or solver changes were
added during this inventory.

**Recommended first additions:** MGH10, generalized Rosenbrock with valley
exponents 2 and 3, Brooks's coupled curved valley, MGH09, Powell singular, and
Brown–Dennis. Then complete the NIST, Nielsen, and MINPACK-2 coverage below.
These first six model definitions are a starting point, not full literature parity.

## Scope and current coverage

The audit covers the references cited directly by the project's
[LM notes](levenberg-marquardt.md), [RNC-LM notes](rnc-lm.md),
[demo discussion](demo.md), and [SVI notes](svi_calib.md). It checks the papers'
evaluation sections and appendices, not just their abstracts or the project's
historical documentation.

| Current problem family | Evidence | Coverage assessment |
| --- | --- | --- |
| Standard Rosenbrock, two parameters | [Shared model](../Testing/problems/rosenbrock.h), [LM cases](../Testing/Cxx/TestLevenbergMarquardt.cpp), [backend cases](../Testing/Cxx/TestSolversBackend.cpp), [RNC cases](../Testing/Cxx/TestRncLmSolver.cpp) | Present; retain it, but one initial point does not establish robustness |
| Small linear, scalar quadratic, stationary, inactive-column and failure cases | [LM cases](../Testing/Cxx/TestLevenbergMarquardt.cpp), [RNC cases](../Testing/Cxx/TestRncLmSolver.cpp) | Useful implementation checks; do not substitute for canonical literature fixtures |
| Raw-SVI calibration, five parameters and 13 quotes | [SVI fixture](../Testing/Benchmark/svi_fixture.h) | Present as an application case; native LM/RNC run without bounds |
| Stiff generalized Rosenbrock, exponents 2 and 3 | [Manual comparison](rnc-lm.md#comparison-with-table-1a) | Documented and checked manually; no persistent first-party fixture or registered case |
| NIST nonlinear regression collection | No corresponding first-party model/data files | **0 of 27 datasets present** |
| Other canonical models listed below | No corresponding first-party model/data files | Missing; related toy regression checks are marked where relevant |

The 45 registered CTest cases checked in the previous change include many
option combinations and API checks. They are not 45 distinct fitting problems.
Historical references to `TestLeastSquaresModels.cpp`, Powell/exponential
fixtures, and `SolversBenchmark rnc-lm` describe code that is absent from the
current tree and must not be counted as current coverage.

## Literature checked

| ID | Primary reference | Evaluation scope relevant to this project |
| --- | --- | --- |
| M77 | [Moré, Argonne report underlying the 1978 LM chapter](https://www.osti.gov/servlets/purl/7256021), Section 8; [published chapter](https://doi.org/10.1007/BFb0067700) | Helical valley, Kowalik–Osborne, Bard, Brown–Dennis; three starting-vector scales |
| N99 | [Nielsen, *Damping Parameter in Marquardt's Method*](https://www2.imm.dtu.dk/documents/ftp/tr99/tr05_99.pdf), Section 3 and Appendix | 20 model definitions, evaluated in 30 dimension configurations |
| TS12 | [Transtrum–Sethna, *Improvements to the Levenberg–Marquardt algorithm*](https://arxiv.org/pdf/1201.5885), Appendix A/B | 17 reported comparison problems A–Q; Appendix A additionally describes R; perturbed starting-point ensembles |
| GA12 | [Transtrum–Sethna, *Geodesic acceleration and the small-curvature approximation*](https://arxiv.org/pdf/1207.4999) | Theory; numerical demonstrations are referred to other papers rather than supplied as a new independent suite |
| B24 | [Brooks, *Higher-Order Corrections to Optimisers based on Newton's Method*, v2](https://arxiv.org/pdf/2307.03820), Sections 6–7 | Coupled two-dimensional valley with an anisotropy sweep; electrostatic ion focusing |
| LZ26 | [Liu–Zhang, RNC-LM, v2](https://arxiv.org/html/2607.07623v2), Sections 2.2–2.4 | Generalized Rosenbrock, all 27 NIST nonlinear datasets, reaction–diffusion and three further PINN equations, potential-energy fitting |
| SVI20 | [Ferhati, *Robust Calibration For SVI Model Arbitrage Free*](https://www.researchgate.net/publication/340530301_Robust_Calibration_For_SVI_Model_Arbitrage_Free) | Application calibration; the project already has an SVI fixture |

NIST's [official catalog](https://itl.nist.gov/div898/strd/nls/nls_main.shtml)
and [ASCII reference files](https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/MGH10.dat)
provide model definitions, starts, observations and certified results.
The [Argonne MINPACK-2 collection](https://ftp.mcs.anl.gov/pub/MINPACK-2/tprobs.92/)
provides executable residual/Jacobian definitions for its fitting problems.

## Priority and counting rules

- **P0:** the six initial model definitions listed above. They expose plateau,
  curved-valley, singularity and nonzero-residual behavior with small models.
- **P1:** complete the classical coverage, including every NIST dataset, all
  Nielsen definitions/configurations, and the remaining MINPACK-2 cases.
- **P2:** larger scientific workloads and exact paper-specific reproductions.
- **Conditional:** a listed problem whose original data or complete setup must
  be obtained before an exact reproduction can be defined.

The core backlog contains **49 canonical definitions or explicit variants**:
27 NIST datasets, 16 additional Nielsen definitions, four additional MINPACK-2
definitions, and two geometric definitions. This count excludes standard
Rosenbrock because it exists already. It also excludes the later scientific
workloads. Multiple dimensions, starts, search budgets, and transformed variants
are separate required configurations, not new datasets.

## A. All 27 NIST nonlinear regression datasets

All rows are missing from the persistent first-party suite. Dimensions below
are `(number of parameters, number of observations)` from the
[NIST catalog](https://itl.nist.gov/div898/strd/nls/nls_main.shtml). LZ26 reports
evaluating this entire collection; the TS12 letters identify overlapping cases.

| ID | Dataset | Parameters | Observations | NIST difficulty | Priority / overlap |
| --- | --- | ---: | ---: | --- | --- |
| C01 | Misra1a | 2 | 14 | Lower | P1 |
| C02 | Chwirut2 | 3 | 54 | Lower | P1 |
| C03 | Chwirut1 | 3 | 214 | Lower | P1 |
| C04 | Lanczos3 | 6 | 24 | Lower | P1 |
| C05 | Gauss1 | 8 | 250 | Lower | P1 |
| C06 | Gauss2 | 8 | 250 | Lower | P1 |
| C07 | DanWood | 2 | 6 | Lower | P1 |
| C08 | Misra1b | 2 | 14 | Lower | P1 |
| C09 | Kirby2 | 5 | 151 | Average | P1 |
| C10 | Hahn1 | 7 | 236 | Average | P1 |
| C11 | Nelson | 3 | 128 | Average | P1 |
| C12 | MGH17 | 5 | 33 | Average | P1; Osborne1 / N99 #17 / TS12 D |
| C13 | Lanczos1 | 6 | 24 | Average | P1 |
| C14 | Lanczos2 | 6 | 24 | Average | P1 |
| C15 | Gauss3 | 8 | 250 | Average | P1 |
| C16 | Misra1c | 2 | 14 | Average | P1 |
| C17 | Misra1d | 2 | 14 | Average | P1 |
| C18 | Roszman1 | 4 | 25 | Average | P1 |
| C19 | ENSO | 9 | 168 | Average | P1 |
| C20 | MGH09 | 4 | 11 | Higher | **P0**; Kowalik–Osborne / N99 #9 / M77 #2 / TS12 G |
| C21 | Thurber | 7 | 37 | Higher | P1; TS12 I |
| C22 | BoxBOD | 2 | 6 | Higher | P1; TS12 J |
| C23 | Rat42 | 3 | 9 | Higher | P1; TS12 K |
| C24 | MGH10 | 3 | 16 | Higher | **P0**; Meyer / N99 #10 / TS12 F / LZ26 Table 1b |
| C25 | Eckerle4 | 3 | 35 | Higher | P1; TS12 L |
| C26 | Rat43 | 4 | 15 | Higher | P1; TS12 M |
| C27 | Bennett5 | 3 | 154 | Higher | P1; TS12 N |

Each dataset needs both supplied NIST starts and its certified residual sum of
squares. MGH10 additionally needs the plateau study from the first NIST start:
plain LM, LM-GA with and without acceleration-ratio control, and RNC orders 2–4
with the published curve-search budget. LZ26 also reports order 5, which the
current implementation does not support; that is an algorithm extension as
well as an evaluation gap. See the
[original MGH10 data and starts](https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/MGH10.dat).

## B. Nielsen definitions not already covered by NIST or Rosenbrock

These 16 entries complete N99's 20 definitions after retaining existing
Rosenbrock and mapping #9, #10 and #17 to C20, C24 and C12. The
[report's Appendix and numerical tables](https://www2.imm.dtu.dk/documents/ftp/tr99/tr05_99.pdf)
are the source for dimensions, data, starts and configuration variants.

| ID | Problem | Parameters / residuals | N99 number | Priority / required property |
| --- | --- | --- | ---: | --- |
| D01 | Linear, full rank | Variable `n`, `m >= n` | 1 | P1; canonical overdetermined/nonzero-residual case, beyond current toy checks |
| D02 | Linear, rank one | Variable `n`, `m >= n` | 2 | P1; nonunique solutions |
| D03 | Linear, rank one with zero rows and columns | Variable `n >= 3`, `m >= n` | 3 | P1; current inactive-column check covers only part of this behavior |
| D04 | Helical valley | 3 / 3 | 5 | P1; also M77 #1; preserve the angle branch convention |
| D05 | Powell singular | 4 / 4 | 6 | **P0**; singular Jacobian at the solution |
| D06 | Freudenstein–Roth | 2 / 2 | 7 | P1; competing local solutions |
| D07 | Bard | 3 / 15 | 8 | P1; also M77 #3; rational fitting |
| D08 | Watson | Variable `n`, 31 residuals | 11 | P1; reproduce the report's selected parameter counts |
| D09 | Box three-dimensional | 3 / variable `m >= 3` | 12 | P1; distinct from the two-parameter BoxBOD model |
| D10 | Jennrich–Sampson | 2 / variable `m >= 2` | 13 | P1; exponential fitting and the report's residual-count variants |
| D11 | Brown–Dennis | 4 / variable `m >= 4` | 14 | **P0**; also M77 #4 with `m=20`; large nonzero residual |
| D12 | Chebyquad / Chebyshev quadrature | Variable `n`, `m >= n` | 15 | P1; also TS12 H specifically at `n=8, m=11` |
| D13 | Brown almost-linear | Variable `n`, `m=n` | 16 | P1; nonlinear product term and alternative stationary points |
| D14 | Four-parameter exponential fit | 4 / 45 | 18 | P1; use the report's 45 observations, not a generic exponential surrogate |
| D15 | Two-parameter projected exponential fit | 2 / 45 | 19 | P1; same observations as D14, amplitudes obtained by an inner linear least-squares solve |
| D16 | Modified Meyer | 3 / 16 | 20 | P1; retain the report's reparameterization and residual scaling |

To match N99's study, implement its **30 dimension configurations**, not just
one arbitrary size per model. D12 needs both the N99 configurations and the
TS12 `8/11` configuration. D14 and D15 share data but are distinct optimization
formulations and must both be evaluated.

For M77's scaling study, D04, C20, D07 and D11 require `x0`, `10*x0`, and
`100*x0`, with the paper's scaling policies distinguished. Evaluating the same
problems using our custom options is useful but does not by itself reproduce
the published LM implementation.

## C. Further MINPACK-2 definitions from Transtrum–Sethna

The remaining small/medium models in
[TS12 Appendix A](https://arxiv.org/pdf/1201.5885) add four definitions beyond
sections A and B. Use the
[Argonne reference routines and data](https://ftp.mcs.anl.gov/pub/MINPACK-2/tprobs.92/).

| ID | Problem | Parameters / residuals | TS12 letter | Priority / required setup |
| --- | --- | --- | --- | --- |
| MP01 | Alpha-pinene isomerization, direct ODE formulation | 5 / 40 | A | P1; evaluate the published kinetic model and observation times |
| MP02 | Alpha-pinene isomerization, collocation formulation | 130 / 165 | B | P1; retain the residual penalty and the paper's penalty-strength convention |
| MP03 | Coating thickness standardization | 134 / 252 | C | P1; preserve the multiple-response model and data |
| MP04 | Gaussian/multi-peak data fitting | 11 / 65 | E | P1; separate from the NIST eight-parameter Gauss datasets |

TS12 D is C12; F is C24; G is C20; H is D12. I–N map to C21, C22, C23,
C25, C26 and C27, respectively. The complete A–N coverage therefore needs all the
mapped datasets plus MP01–MP04, without duplicating their aliases.

## D. Geometric problems for LM-GA and RNC-LM

| ID | Problem | Exact family / setup | Priority / current gap |
| --- | --- | --- | --- |
| G01 | Generalized Rosenbrock | `r = [x, A*(y - x^p/p)]`, `A=1e6`, `p=2,3`, start `(1,1/p)`, `0.5*||r||² < 1e-4` | **P0**; manual results exist, persistent fixture and complete method/search matrix are missing |
| G02 | Brooks coupled curved valley | `r = [x+y², K*(y-x²)]`, start `(pi,e)`, sweep `K=1,10,...,1e12` | **P0**; entirely missing; distinct from both Rosenbrock models |

G01 comes from [LZ26 Eq. (6) and Table 1a](https://arxiv.org/html/2607.07623v2).
Include full-step LM/LM-GA and RNC orders 2–4 with zero, one, two and three
allowed curve reductions. Keep the exponent separate from parameter dimension:
both cases have two parameters.

G02 comes from [B24 Section 6 and Table 1](https://arxiv.org/pdf/2307.03820).
Brooks's residual-linear higher-order method differs from RNC-LM's moving-tangent
recursion. Reusing its problem is an appropriate comparison; requiring RNC-LM
to match Brooks's iteration counts would conflate different algorithms.
The author supplies [C reference code](https://stephenbrooks.org/ap/src/optimtest.zip).

## E. Larger workloads needed for broader literature claims

These are not prerequisites for basic dense LM correctness. They are necessary
to support claims matching the corresponding papers' larger scientific results.
All are missing from the current first-party suite.

| ID | Workload | Source | Required setup / availability |
| --- | --- | --- | --- |
| W01 | Systems-biology kinetic ODE fit | TS12 O | 48 parameters; 68 observed data values plus the paper's weak prior residuals; obtain the exact model and prior weights |
| W02 | Avalanche-size scaling-function fit | TS12 P | 32 parameters and 398 data values; obtain the original scaling model and data |
| W03 | Concrete compressive-strength neural-network fit | TS12 Q | 81 weights, 1030 observed targets and weak priors; preserve architecture, normalization and starting ensembles |
| W04 | Quantum Monte Carlo Jastrow-parameter fit | TS12 R | 27 parameters, 2002 residuals; conditional on original wave-function/model/data availability |
| W05 | Electrostatic ion-bunch focusing | B24 Section 7 | 180 electrode parameters, 400 ions, 1200 centroid-subtracted coordinate outputs in the main text; use the author's tracker and initial conditions |
| W06 | Reaction–diffusion PINN | LZ26 Table 2 / Figure 2 | `rho=5`, `nu=5,6,10,20`; fixed full-domain setup; separate initialization and collocation robustness studies |
| W07 | Convection PINN | LZ26 Table 3 | `u_t + 40*u_x = 0`; recover the exact network, domain, loss weights and conditions |
| W08 | Reaction PINN | LZ26 Table 3 | `u_t - 5*u*(1-u) = 0`; recover the exact training setup |
| W09 | Wave PINN | LZ26 Table 3 | `u_tt - 4*u_xx = 0`, initial frequency parameter 5; recover initial/boundary conditions and training setup |
| W10 | Three-body H2O potential-energy fit | LZ26 Section 2.4 | 985160 configurations, 1000 descriptors, network `1000–55–55–55–1`; identical data split, initialization and training-error target |

Sources: [TS12 Appendix A](https://arxiv.org/pdf/1201.5885),
[B24 Section 7](https://arxiv.org/pdf/2307.03820), and
[LZ26 Sections 2.3–2.4](https://arxiv.org/html/2607.07623v2).

The TS12 plots report 17 cases A–Q, although the appendix additionally names R.
Track W04 separately instead of silently treating the plots as an 18-case study.
B24's ion-model main text specifies 1200 outputs, but one figure caption says
300; resolve this using the [author's tracker source](https://stephenbrooks.org/ap/src/tracker.zip)
before fixing an exact reference configuration. LZ26 references supplementary
training details not included in the arXiv source reviewed in the preceding
RNC audit; those details and datasets remain requirements for exact reproduction.

## Avoid duplicate models and transcription errors

| Names in different sources | Canonical backlog entry | What must remain explicit |
| --- | --- | --- |
| Meyer / thermistor resistance / MGH10 | C24 | Parameter sign/scale transformations and distinct starting points |
| Kowalik–Osborne / enzyme reaction / MGH09 | C20 | Independent-variable and parameter mapping between formulations |
| Osborne1 / exponential data fitting / MGH17 | C12 | Original sample times, parameter scaling and starts |
| Chebyquad / Chebyshev quadrature | D12 | `n,m` and residual scaling; TS12's `8/11` is a required configuration |
| Modified Meyer | D16 using C24's data | A separate transformed/scaled evaluation, not another observation dataset |
| Full and projected exponential fits | D14 and D15 | Shared data, different parameter counts and derivative contracts |
| Gaussian MINPACK-2 fit / NIST Gauss1–3 | MP04 versus C05/C06/C15 | Different parameter counts, observation counts and models; not aliases |

Some equations in TS12 Appendix A contain apparent transcription errors.
For example, its printed exponential-fitting expression has an additive third
amplitude where the Argonne implementation multiplies the second exponential.
Use the primary residual/Jacobian routines and NIST data definitions to resolve
such discrepancies. Preserve each source's independent-variable units; copying
a familiar equation with different sample times changes the problem.

## What constitutes being at parity

Adding names alone is insufficient. Every listed case needs a recorded model,
data source, parameterization, dimensions, starts, weights, derivative contract,
and reference solution or acceptance criterion.

1. **Reference fit:** compare a recomputed common residual sum of squares with
   the certified/best-known value, and parameters or predictions where useful.
   Most NIST fits have nonzero residuals; requiring zero cost is incorrect.
   Native LM returns `||r||²`, whereas RNC-LM returns half that quantity.
2. **Starting points:** both NIST starts; M77's three vector scales; N99's 30
   dimension configurations; TS12's deliberately perturbed Gaussian ensembles.
   If the original ensembles cannot be obtained, label replacements explicitly.
3. **Feature comparison:** plain LM with acceleration and bold acceptance off,
   LM-GA, bold-only LM, and their combination. Acceleration defaults to on in
   this project, so the baseline must be selected explicitly. Evaluate our
   available damping/linear-solver choices without presenting them as identical
   to every published algorithm.
4. **RNC comparison:** orders 1–4, the prescribed curve-search budgets, correct
   ordinary curve derivatives, and matching damping units. Fifth-order MGH10
   reproduction requires additional algorithm support.
5. **Work and outcomes:** final RSS, status, accepted updates, attempted curves,
   rejected trials, residual/Jacobian evaluations and failed starts. An iteration
   count cannot substitute for fit quality or reliability. Missing diagnostics
   must stay missing rather than being presented as measured zeros.
6. **Scientific prediction:** for the PINNs compare solution error and held-out
   PDE residuals, not training loss alone; for neural fits retain validation
   splits and a common error target. These are P2 workloads with separate setup
   requirements.

## Suggested implementation sequence

1. Add the six P0 model definitions and verify their reference fits/configurations.
2. Complete the 27 NIST fixtures and both starts, prioritizing the eight datasets
   classified as higher difficulty and MGH17.
3. Complete the 16 additional Nielsen definitions and the report's selected sizes;
   reuse the mapped NIST fixtures for the overlapping models.
4. Add MP01–MP04 and the TS12 starting-point ensembles. At this point the
   classical coverage is broad enough to assess the implemented LM extensions
   against their main small/medium-problem literature.
5. Add W01–W03, then the reproducible P2 PINN/physics workloads. W04 stays
   conditional until its original setup is available. Scientific-paper speedup
   claims require these workloads and comparable execution settings.

This backlog specifies future coverage. It does not claim that any missing
model, starting-point ensemble, or published result has already been reproduced.
