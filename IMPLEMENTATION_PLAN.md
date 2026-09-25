# IMPLEMENTATION_PLAN.md — vol-forecast-model-comparison

Standalone quant-track project, self-paced. Suggested pacing assumes
roughly **5-6 hrs/week** — this project is smaller in scope than
`lob-alpha-engine` or `stat-arb-optimizer`, matching its "focused,
foundational" purpose (same spirit as `pairs-trading-honest-costs`).

## Week 1: Data & Target
- Phase 0, Phase 1
- Milestone: volatility target constructed, look-ahead-freedom verified
  by an actual test, not just design intention

## Week 2: Simple Baseline — record the naive number, then move on
- Phase 2
- Milestone: EWMA and GARCH(1,1) implemented, naive-split numbers
  computed and recorded (then set aside — they exist for Phase 5's
  comparison, not to guide the complex model's design)

## Week 3–4: Complex Model
- Phase 3
- Milestone: ML model built with a genuinely fair, reasonable feature set
  and hyperparameter search — not deliberately weakened, not artificially
  strengthened

## Week 5: Walk-Forward Validation
- Phase 4
- Milestone: all models re-evaluated under genuine walk-forward
  validation

## Week 6: Comparison & Verdict
- Phase 5
- This is the project's actual point — budget real time to state the
  verdict precisely and mechanistically, not just report a number
- Milestone: `docs/results_comparison.md` complete with a plain,
  specific verdict

## Week 7: CI + Documentation Polish
- Phase 6, Phase 7
- Milestone: CI green, README structured exactly per `PROJECT.md` §10,
  final review confirms every complexity claim points to specific numbers

## Dependency Notes

- Do not begin Phase 3 (complex model) with knowledge of how it needs to
  perform to "win" — Phase 2's naive baseline number should already be
  recorded and set aside per `TASKS.md`, but the complex model's design
  (features, architecture, hyperparameter search) should be chosen on its
  own reasonable merits, not reverse-engineered to beat a known target.
- Do not begin Phase 5 (comparison) before Phase 4 (walk-forward
  validation) has been applied to BOTH the simple and complex models —
  comparing a walk-forward-validated simple model against a naively-
  validated complex model (or vice versa) would itself be a methodology
  error of exactly the kind this project exists to avoid.
- If Phase 3's complex model requires trying several feature sets or
  hyperparameter configurations to get a fair, working version, keep a
  running log from the start — this becomes the required disclosure in
  `DECISIONS.md`, not something to reconstruct from memory at the end.
