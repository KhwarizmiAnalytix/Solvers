# Current design review and implementation prompt

Reviewed baseline: commit `443186e`, 2026-09-27. This review supersedes the earlier eight-blocker audit where the code has changed. The implementation prompt below is intended to be given directly to a coding agent.

## Where the design stands

The initial redesign proposed a problem-oriented API, distinct least-squares and scalar-objective contracts, native solvers as the default, optional external optimizers, and shared evaluation/reporting contracts. The later Ceres plan proposed executable type erasure for AD, explicit derivative policies, safe evaluations, correct diagnostics, and tested packaging.

The current implementation has made substantial progress:

| Area | Current state |
|---|---|
| Problem-oriented API | Implemented: least-squares/objective types, options, traits, runtime dispatch, and structured result type. |
| Executable AD | Implemented: Ceres `DynamicAutoDiffCostFunction`, const factory creation, and correct conversion from row-major Jacobians. |
| Provider abstraction | New `JacobianProvider` and `GradientProvider` support analytic and numerical derivatives; AD Jacobians can also feed native solvers. This advances beyond the Ceres-only first delivery in the AD plan. |
| Model helper | `least_squares(model, n, m)` creates residuals; `problem.derivatives(auto_diff())` attaches AD. The old `std::any` and `make_ceres_autodiff_problem` APIs have been removed. |
| Derivative policy | Partial: provider source, explicit policy, optimizer selection, and actual execution still disagree. |
| Evaluation lifecycle | Partial: two evaluator interfaces have different failure contracts; AD recreates evaluators per call; residual callbacks and providers can represent different models. |
| Diagnostics | Improved Ceres termination mapping, but missing evaluation counts, budget enforcement, backend code propagation, and precise limit distinctions. |
| Build/package | Ceres integration target and numeric feature flags exist. Shared installation, consumers, examples, and benchmark validation remain incomplete. |
| Original broader solver redesign | Not certified by this review. General nonlinear constraints and matrix-free products are particularly incomplete; the API must reject capabilities the adapters do not execute. Native numerical kernels and root algorithms require separate evidence against the original plan. |

The architectural direction is sound. Preserve the provider API and make the existing abstractions obey one contract; do not add a third derivative abstraction or revert to passive `std::any` storage.

### Current evidence

The Ceres-enabled static Release build in `build_test` succeeded. Its full CTest run discovered 159 tests: 146 passed, 13 skipped because the relevant optional backends were unavailable. This establishes the existing tested paths, not completion of the design.

The shared Debug build in `build_ninja`, configured with Ceres, Ipopt, and PETSc enabled, also succeeded. Its selected dispatch test run had 22 passes and one skip. Despite its name, this directory was not a native-only configuration. A fresh backend-off build and installed consumers were not validated during this review; they remain explicit acceptance work below.

Standalone probes against the current source/library reproduced:

```text
Ceres automatic AD provider reports: supplied
false-return provider returned normally: r=123, J=456
NaN gradient check passed: 1
required AD without provider on native: converged
```

The false-return test initializes residual/Jacobian outputs to sentinels and uses a functor that returns false without writing them. `AutoDiffJacobianProvider::compute` returns normally with those sentinels unchanged.

### Remaining design defects

1. `src/api/dispatch.cpp::resolve_derivatives` labels every generic provider as `supplied`, ignoring `source()`. It only runs on the Ceres path. Native solving uses whichever provider is present regardless of explicit derivative policy. Its precedence also differs from the resolver's stated callback-first order.
2. The generic AD provider computes residuals and Jacobians together, but the native/Ceres callback bridges discard its residuals and separately evaluate `problem.residuals`. These can disagree after public fields are reassigned. Final diagnostics again use the separate callback.
3. `AutoDiffJacobianProvider::compute` and `residuals_only` handle fatal errors but ignore `invalid_trial`. Its evaluator is recreated for every call. The Ceres wrapper treats fatal and recoverable statuses alike and loses diagnostic context.
4. `run_petsc_tao_least_squares` only forwards `problem.jacobian`; it ignores `jacobian_provider` although traits can route provider-backed large problems to TAO.
5. The Ceres fallback remains a stored, self-capturing finite-difference closure. At an exact bound its step can become zero, followed by division by zero; the near-upper-bound branch still attempts a forward step. The generic finite-difference provider is scaled but not bound-aware.
6. Validation is fragmented. Provider/problem dimensions, integer conversion, matrix-size overflow, callback sizes/finiteness, empty engaged optional functions, and bound feasibility lack one reliable validation boundary.
7. `check_jacobian`/`check_gradient` can pass NaN outputs because comparisons against NaN never update the initially zero maximum error. Their reported maximum relative error is tied to the largest absolute error, not independently maximized.
8. Ceres summary handling uses a public `void*` bridge, leaves counters/backend status unset, and maps usable `NO_CONVERGENCE` to an iteration limit even when another limit caused it. Final callback evaluation remains outside the protected evaluation path. Requested evaluation budgets are not enforced or rejected.
9. The package template still reconstructs static imported libraries regardless of build type. The README/migration guide refer to removed APIs. The AD benchmark still default-constructs a model that requires data, and its target does not consume the AD integration target.
10. The AD suite now renames its stride test to four parameters instead of testing more than four. The exponential/false-return cases were removed rather than repaired. A passing suite therefore misses important original acceptance criteria.
11. General constraints are represented by counts only; Ipopt's implementation has zero general constraints. Hessian-vector products influence dispatch but are not forwarded to the TAO implementation. These capabilities must not silently disappear during solving.

## Agent prompt — complete the current derivative-provider design

You are working in the Solvers repository. Review and finish the derivative-provider integration, particularly Ceres automatic differentiation, against the original design and the current implementation. Implement the changes, add regression tests, run the supported validation matrix, and report evidence. Do not stop at an assessment or a list of proposed changes.

Read repository instructions and these design documents first:

- `docs/reviews/solver-redesign-2026-09-26.md`
- `docs/reviews/ceres-autodiff-plan-2026-09-27.md`
- `docs/reviews/current-design-agent-prompt-2026-09-27.md`

Inspect current HEAD before editing. Treat historical findings as hypotheses to revalidate; do not reintroduce removed APIs or redo fixes already present. Record a requirements-to-implementation checklist. Passing existing tests and commit messages are not proof that a requirement is complete.

### Target architecture and boundaries

Retain the problem-oriented API, native solvers, optional external optimizers, `JacobianProvider`, `GradientProvider`, and compile-time AD instantiation followed by executable type erasure. Ceres remains a forward-mode derivative engine and an optional optimizer; using Ceres-generated Jacobians must not force the Ceres optimizer when the selected native optimizer can consume them.

Keep these decisions distinct:

- Mathematical model and its dimensions.
- Derivative method: supplied/analytic, AD, finite differences, or none.
- Derivative engine: Ceres Jets, library numerical differentiation, user implementation, etc.
- Optimizer backend and algorithm.
- Per-solve evaluation state, counters, failures, and termination.

Recommended flow:

```text
Model + provider + solve options
  -> validate and resolve one execution plan
  -> create one evaluation session per solve
  -> selected optimizer consumes that session
  -> translate termination and return diagnostics from the same model/session
```

The public provider and internal factory may remain separate only with clear roles: an immutable description/factory constructs a solve-local executable evaluator. Use one error/status vocabulary. Do not create another overlapping interface.

Do not expand this task into implementing sparse block graphs, reverse-mode AD, general nonlinear constraints, Hessians, or matrix-free products. Reject unsupported requests truthfully and document deferred work. Do not rewrite unrelated numerical kernels. Fix touched native evaluation integration where necessary to honor derivative contracts.

### Milestone 1 — one derivative and capability resolver

Create one resolution path used by public selection helpers and execution. Resolve backend, algorithm, derivative method/engine, provider, availability, and required capabilities before callbacks execute.

Adopt and document these rules:

- In automatic derivative mode, an explicitly attached provider is authoritative; otherwise use a callable derivative callback; otherwise use the backend's documented fallback. This formalizes the newer provider API and deliberately updates the earlier callback-first proposal.
- Explicit `supplied` requires an analytic/supplied derivative source. A numerical or AD provider must not be relabeled supplied merely because the caller attached it.
- Explicit AD requires an executable AD source, but may run with a compatible native optimizer. Do not equate AD with `backend::ceres`.
- Explicit finite differences must actually run finite differences, even if AD or analytic sources are available. Count/instrument calls to prove this.
- Explicit derivative-free algorithms must not silently ignore a conflicting derivative requirement.
- No-provider explicit AD, empty engaged optional functions, unsupported backend/algorithm combinations, and incompatible provider capabilities must fail before callback execution.
- Automatic optimizer selection keeps ordinary unconstrained dense problems on native LM/GN where providers are usable. Preserve residual-only automatic POUNDERS behavior unless a documented compatibility change is deliberately made. Explicit finite differences imply an appropriate derivative-based route instead of blindly selecting POUNDERS.
- Size alone does not establish matrix-free or sparse capability. Route large problems only to paths that actually consume their derivative representation.
- Honor explicit backend selection or return a precise unavailable/unsupported result; never silently substitute another optimizer.

Use provider `source()` and capabilities rather than pointer presence or concrete provider casts. Derive the optional Ceres execution hook from the selected provider instead of maintaining independently mutable duplicate state. Replace heap-allocated error results with value-based resolution results.

Extend equivalent policy consistency to the existing scalar-gradient providers. If a derivative policy is intentionally least-squares-only, reject it explicitly for objective problems and document that contract instead of ignoring it.

Acceptance: a table-driven test matrix proves resolution and actual callback/provider invocation agree across automatic, explicit, unavailable, and incompatible configurations.

### Milestone 2 — canonical model, solve-local evaluation, and failure handling

Choose one authoritative residual evaluator for each solve. Residuals, Jacobians, objective values, and final diagnostics must describe the same model. Do not discard provider-computed residuals and silently pair the Jacobian with an unrelated callback. Define provider/callback coexistence and model replacement; reject ambiguous combinations or adapt them through a documented single-model contract.

Create evaluator/workspace once per solve and reuse it. Keep model lifetime owned and explicit. Do not cache mutable buffers or counters globally or in a shared provider that would make concurrent solves unsafe. Repeated/copy/move use must remain valid.

Propagate all statuses consistently:

- Success: finite outputs with correct dimensions.
- Recoverable invalid trial: reject the trial without consuming stale outputs or changing the last accepted iterate.
- Fatal evaluation failure: stop and preserve meaningful diagnostic context.

Never ignore a functor's false return. Catch application exceptions at callback boundaries, including backend worker threads. Where a native kernel cannot recover from a domain failure, return a structured failure; never present it as successful evaluation. Protect final diagnostic evaluation with the same rules.

Validate positive dimensions, provider/model agreement, input/output shapes, finite values, representable allocation sizes, and `int` conversions before entering Ceres. Validate callbacks that resize their outputs. Document that raw-pointer user code must still obey buffer bounds.

Acceptance: false returns, exceptions, NaN/Inf, mismatched dimensions, model replacement, repeated solves, and problem lifetime tests establish these contracts. Sentinel outputs must never be mistaken for computed derivatives.

### Milestone 3 — Ceres execution and numerical derivatives

For selected Ceres AD, use the typed `DynamicAutoDiffCostFunction` provider directly through a thin adapter. Preserve the corrected row-major conversion or use validated raw-buffer evaluation to remove redundant copies. Do not force the AD path through an additional double-only numerical wrapper.

For supplied derivatives, retain a validated callback/provider adapter. For ordinary Ceres numerical differentiation, use public `DynamicNumericDiffCostFunction` with a documented relative-step option as originally planned, unless a shared library provider demonstrably provides a required stronger contract; record any justified deviation.

For bounded/domain-limited numerical differentiation, use a deliberate feasible-stencil implementation or explicitly reject the unsupported combination. Bounds on optimizer iterates do not make arbitrary numerical perturbations safe. Fix zero steps at exact bounds, use backward differences at upper bounds and forward differences at lower bounds, and divide by the actual representable displacement. Define fixed-parameter behavior. Remove the retained self-capturing fallback lambda.

Apply lower/upper bounds independently. Validate exact lengths, ordering, NaNs, and initial feasibility before evaluation. Preserve accepted-iterate behavior.

Acceptance: analytic/AD/numerical agreement at ordinary points; exact lower/upper-bound tests; domain-restricted models; fixed parameters; residual-only requests; no unnecessary Jet evaluation for residual-only calls.

### Milestone 4 — adapter parity and honest capability handling

Make native LM/GN/L-BFGS and TAO residual paths consume the selected Jacobian provider. Make native, Ipopt, and TAO scalar paths honor the selected gradient source. Avoid backend-specific precedence differences.

Repair the TAO least-squares provider omission before allowing dispatch to send provider-only models there. If a path cannot consume a representation, reject the request before callbacks rather than omit its derivatives.

Reject general nonlinear constraints until real constraint callbacks and backend enforcement exist. Reject unimplemented Hessian-vector/matrix-free requests instead of using their presence only to choose a backend and then dropping them. These guards are in scope; implementing those numerical capabilities is not.

Acceptance: unavailable builds test precise statuses; available backends test provider invocation, constraints/capabilities, and explicit incompatibilities. Tests may skip truly unavailable optional libraries but must identify the skips.

### Milestone 5 — diagnostics, checks, and budgets

Replace the public `void*` summary bridge with a typed backend-neutral internal report. Preserve Ceres termination code, usability, message, meaningful iteration count, and available evaluation statistics. Distinguish convergence, iteration/time/evaluation limits, user termination, and numerical failure as supported by the API; extend statuses or report fields where needed. Do not infer every `NO_CONVERGENCE` is an iteration limit.

Report actual derivative method and engine on every backend, including native optimizers consuming Ceres AD. Do not call the source `automatic` after a concrete method ran. Preserve the actual optimizer/algorithm combination as well.

Define separate counters for residual-only requests, Jacobian/gradient requests, and underlying model invocations, including Jet passes and numerical perturbations. Include final evaluations. Represent unavailable diagnostics explicitly rather than leaving a plausible zero. Enforce nonzero evaluation budgets consistently or reject them as unsupported before solving; silent ignoring is unacceptable.

Fix `check_jacobian` and `check_gradient`: fail on nonfinite values and invalid dimensions/tolerances; independently compute absolute and relative maxima; provide a clear tolerance rule and meaningful error locations. A NaN derivative must never pass.

Acceptance: limit-hit but usable iterates, preserved backend codes, source assertions, instrumented call counts, unsupported-budget rejection or exact accounting, and checker NaN/Inf/scaling regressions.

### Milestone 6 — packaging, examples, and documentation

Keep Ceres optional. Core headers and analytic/numerical provider consumers must build without Ceres. Optional AD headers may require it only for AD instantiation. Consider moving the generic model helper out of the Ceres-only header so ordinary templated residual construction does not itself require an AD engine; do not introduce hidden Ceres dependencies.

Use `Solvers::CeresAutodiff` for consumers needing AD templates, including the benchmark. Carry dependencies through CMake targets. Repair installed static/shared targets and transitive dependencies; prefer real exports where feasible. Keep the native Bazel target functional.

Update README, migration guide, examples, and benchmark code to the current `least_squares`/provider API. Remove references to deleted helpers and stale backend claims. Compile the documented examples as external consumers. Repair benchmark model construction and ensure it checks successful comparable outcomes and actual derivative source before reporting timings. Do not publish speedups based on failed solves or unknown zero counters.

Acceptance: Ceres OFF/ON builds, static/shared installed consumers, and external application-defined functors work with documented targets. Benchmarks compile and run; unmeasured performance claims are removed.

### Milestone 7 — required regression and acceptance matrix

Add meaningful tests for:

1. The four reproduced defects in the review above.
2. Provider-versus-callback precedence and explicit supplied/AD/FD requests on every supported path.
3. Native optimizer with Ceres AD and Ceres optimizer with its direct AD provider path, both reporting the actual source.
4. A nonsymmetric rectangular Jacobian with at least five parameters and stride four, checked entry-by-entry against independent analytic derivatives at multiple points.
5. Rosenbrock, Powell, exponential fitting with valid data construction, and a noisy problem with nonzero optimal residual.
6. Instrumented double and Jet calls proving AD runs, multiple passes occur, and residual-only requests avoid unnecessary differentiation.
7. Functor false, exceptions, NaN/Inf, invalid shapes/dimensions, empty callbacks, provider/model mismatch, and malformed options.
8. Lower-only, upper-only, exact active bounds, fixed parameters, infeasible initial guesses, and bound/domain-aware numerical stencils.
9. Temporary functors, destroyed originals, copied/moved problems, repeated solves, and documented reentrant concurrent use where supported.
10. Truthful termination, counters, budgets, unsupported constraints/matrix-free requests, and installed package consumers.

Keep tests independent of implementation where possible: analytic expectations and mathematical residuals are stronger than convergence-only assertions. Do not delete a failing acceptance test because it is difficult; implement a valid namespace-scope templated functor and preserve the requirement. Do not rename a four-parameter test to imply multiple stride-four passes.

Run fresh configurations and inspect their feature flags; existing build-directory names are not evidence of enabled backends. At minimum validate native-only and Ceres-enabled builds, the full existing suite, the new regression suite, and static/shared installed consumers. Run appropriate sanitizers on evaluator/lifetime tests. Test optional backend execution where installed, and report untested configurations accurately.

### Delivery requirements

Work in focused increments: resolver and failing regressions; evaluation/status contract; adapter/numerical integration; reporting and budgets; package/docs/benchmarks. Preserve existing correct work and unrelated changes.

Deliver the implementation and a concise completion report containing:

- What changed and why.
- A checklist mapping each milestone to code and test evidence.
- Exact configurations, test counts, skipped tests, and consumer/benchmark results.
- Any deliberate departure from the original plan, with its reason and compatibility impact.
- Remaining limitations, distinguishing this derivative work from the broader initial numerical-kernel redesign.

Do not claim the whole original solver redesign is finished merely because this derivative work passes. Do not mark this task complete while any required derivative capability is silently ignored or misreported. Sparse blocks, reverse-mode AD, and advanced matrix-free support remain explicit follow-ups.
