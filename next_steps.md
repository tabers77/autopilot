# AutoPilotML – Time‑Series Research & Portfolio Contract

**Owner:** You
**Role Target:** AI Lead / Senior ML Architect
**Primary Goal:** Build a *credible, opinionated, non‑black‑box* experimentation framework for **time‑series forecasting**, with extensibility toward **agentic LLM components**, usable both as:

* a **portfolio flagship project**, and
* a **personal R&D library** for method research.

---

## 1. Non‑Goals (Explicit Scope Control)

To keep the project sharp and defensible, the following are **explicitly out of scope** for now:

* ❌ Competing with AutoML tools on *raw leaderboard performance*
* ❌ One‑click “best model” UX
* ❌ Broad support for every ML task (tabular/text only later)
* ❌ Production deployment / MLOps completeness
* ❌ LLMs as direct forecasters

This project optimizes **understanding, reproducibility, and controlled experimentation**.

---

## 2. Core Product Statement (Contractual)

> **AutoPilotML is a controlled experimentation framework for time‑series forecasting that enables fast, reproducible comparison of modeling assumptions (lags, features, CV, horizons, agents) while preventing temporal leakage and exposing why methods succeed or fail.**

If a feature does not strengthen this statement, it is deprioritized.

---

## 3. Design Principles (Must‑Hold Constraints)

1. **No hidden decisions**
   Every modeling choice must be inspectable, logged, and reproducible.

2. **Ablation‑first, not optimization‑first**
   The system is designed to answer *why*, not only *what wins*.

3. **Time‑series correctness over convenience**
   Leakage‑safe CV, horizon handling, and covariate alignment are non‑negotiable.

4. **Methods > Models**
   Pipelines, transformations, and assumptions are first‑class citizens.

5. **LLMs must justify their cost**
   Every agent has measurable value (accuracy delta, stability, insight quality).

---

## 4. Architecture Pillars (What You Are Really Building)

### Pillar A — Experiment Axes (Foundation)

Explicit, orthogonal experiment dimensions:

* Data split strategy (expanding / sliding / blocked)
* Lag window strategy
* Feature construction strategy
* Forecast horizon strategy
* Model family
* Agentic augmentation (on/off, versioned)

Each axis must be:

* enumerable
* comparable
* logged

---

### Pillar B — Leakage‑Safe Time‑Series Engine

This is your **technical moat**.

Required capabilities:

* Centralized time index handling
* Explicit cutoff timestamps per fold
* Guardrails for future covariates
* Automatic leakage diagnostics (basic but real)

---

### Pillar C — Experiment Artifacts & Insight

Every experiment produces:

* metrics (global + per‑horizon)
* stability measures
* cost metrics (time, later tokens)
* artifacts (plots, tables)
* a *human‑readable experiment card*

---

### Pillar D — Agentic Layer (Later, Additive)

Agents assist **analysis and interpretation**, not prediction.

---

## 5. Implementation Strategy & Priority Order

### PHASE 0 — Strategic Lock‑In (Required Before Coding)

**Goal:** Prevent scope creep and AutoML drift

Deliverables:

* [ ] README section: *“What this is NOT”*
* [ ] One‑paragraph philosophy statement
* [ ] Single canonical forecasting use case chosen

⏱️ Time: 1–2 days

---

### PHASE 1 — Time‑Series Core (Highest Priority)

**Goal:** Be technically correct and defensible

Implement **only** what is needed for forecasting:

1. TimeSeriesDataset abstraction

   * time index
   * target
   * optional covariates

2. SplitStrategy module

   * expanding window
   * sliding window
   * horizon‑aware

3. Lag & Feature Engine

   * configurable lag windows
   * rolling statistics
   * strict cutoff enforcement

4. Baseline Models

   * 1 classical (e.g. ETS/SARIMA)
   * 1 ML (e.g. tree‑based regression on lags)

**Acceptance criteria:**

* You can explain exactly why no leakage occurs.

⏱️ Time: ~2 weeks

---

### PHASE 2 — Experiment Runner & Axes (Critical)

**Goal:** Enable method research velocity

Implement:

1. ExperimentConfig (typed, serializable)
2. Factorial experiment execution
3. Budget controls (max runs / time cap)
4. Deterministic seeding

Outputs:

* leaderboard
* per‑axis comparisons
* CV variance

⏱️ Time: ~2 weeks

---

### PHASE 3 — Reporting & Insight Artifacts (Portfolio Multiplier)

**Goal:** Turn results into explanations

Implement:

* automatic plots:

  * horizon error decay
  * stability vs performance
* Pareto frontiers (error vs cost)
* experiment card (Markdown):

  * what changed
  * what stayed fixed
  * key insight

This is what interviewers will *read*.

⏱️ Time: ~1–2 weeks

---

### PHASE 4 — Agentic LLM Layer (Only After Core Is Solid)

**Goal:** Add intelligence, not noise

Agents to implement (in order):

1. Forecast failure explainer
2. Regime shift summarizer
3. Feature importance narrator

Rules:

* agent outputs are artifacts
* agent cost is logged
* agent is optional per axis

⏱️ Time: ~1–2 weeks

---

### PHASE 5 — Polish for Portfolio

**Goal:** Signal AI Lead maturity

* [ ] One end‑to‑end demo notebook/script
* [ ] Clean README with diagrams
* [ ] Clear comparison vs AutoML tools
* [ ] “What I learned” section

⏱️ Time: ~1 week

---

## 6. Recommended Implementation Order (Tactical Execution)

### Week 1: Foundation
- [ ] **Day 1-2**: Complete PHASE 0
  - [ ] Update README with "What This Is NOT" section
  - [ ] Add design philosophy statement to README
  - [ ] Confirm canonical use case (energy consumption forecasting)
  - [ ] Document scope boundaries
- [ ] **Day 3-4**: Time-Series Module Structure
  - [ ] Create `taberspilotml/time_series/` module
  - [ ] Implement `TimeSeriesDataset` class with temporal validation
  - [ ] Add basic temporal ordering tests
- [ ] **Day 5**: Initial Testing
  - [ ] Write unit tests for `TimeSeriesDataset`
  - [ ] Test with real energy consumption data
  - [ ] Validate temporal integrity

### Week 2: Core Time-Series Capabilities
- [ ] **Day 1-3**: Temporal Cross-Validation
  - [ ] Implement `TemporalCrossValidator` class
  - [ ] Add expanding window strategy
  - [ ] Test cutoff date enforcement
  - [ ] Log fold metadata (train size, test size, cutoff dates)
- [ ] **Day 4-5**: Feature Engineering
  - [ ] Implement `LagFeatureEngine`
  - [ ] Add rolling statistics features
  - [ ] Enforce cutoff-aware feature creation
  - [ ] Test leakage prevention

### Week 3: Baseline Models & Integration
- [ ] **Day 1-2**: Baseline Time-Series Models
  - [ ] Implement Naive forecaster
  - [ ] Implement Seasonal Naive forecaster
  - [ ] Add simple ML baseline (e.g., Ridge on lags)
  - [ ] Horizon-aware evaluation metrics
- [ ] **Day 3-4**: Integration with Auto_Mode
  - [ ] Add `mode='time_series'` parameter to autopilot_mode
  - [ ] Create time_series_steps dictionary
  - [ ] Integrate temporal pipeline with existing handlers
- [ ] **Day 5**: Documentation & Example
  - [ ] Create end-to-end example notebook
  - [ ] Document time-series config parameters
  - [ ] Add usage examples to README

### Week 4+: Iteration & Enhancement
- [ ] Add sliding window CV strategy
- [ ] Implement blocked CV
- [ ] Add leakage diagnostic tools
- [ ] Expand baseline model library
- [ ] Create experiment comparison utilities

**Critical Path Items** (must complete in order):
1. TimeSeriesDataset (enables everything else)
2. TemporalCrossValidator (enables proper evaluation)
3. LagFeatureEngine (enables ML models)
4. Baseline models (enables comparison)
5. Integration (makes it usable)

---

## 7. Success Criteria (Objective)

The project is **successful** if:

* You can answer *why* a forecasting method worked
* You can demonstrate leakage prevention convincingly
* You can add a new method/agent in < 30 minutes
* An interviewer can understand your design decisions without code

---

## 8. Kill Criteria (When to Stop Adding Features)

Stop expanding scope when:

* insights are repeating
* experiments answer your main questions
* added features don’t improve explanations

At that point, **polish, don’t grow**.

---

## 9. Final Contract Statement

This project exists to **demonstrate judgment, rigor, and systems thinking**—not to chase benchmarks.

If a decision improves clarity, reproducibility, or insight → **do it**.
If it only improves accuracy → **question it**.
