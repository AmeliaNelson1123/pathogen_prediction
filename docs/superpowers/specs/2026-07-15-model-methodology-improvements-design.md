# Model Methodology Improvements — Design

**Date:** 2026-07-15
**Context:** Response to the final report from the judges
**Scope chosen:** L2 — correctness + honesty + methodology depth.
**Deliverable:** This design, followed by an implementation plan (no code changes yet).
**Role:** Intermediary step to improve the results and address the review comments so the
project reports **accurate, defensible metrics**. A presentation-slide deck will be built
later as a separate effort — no Q&A/prose write-up is produced here.

---

## 1. Problem statement

The competition review scored the project 39/60 (65%). Reproducibility lost the most
points (5/10), and the Q&A exposed gaps in the modeling methodology. Reading the code, most
of the criticisms share a single root cause:

> **Models and hyperparameters are selected directly on the test set, with no
> cross-validation, using duplicated code that has drifted between the notebooks, the
> saved results, and the deployed models.**

### Evidence in the code
- `preparation/Run_and_Test_Models.ipynb` — `get_train_test` makes **one** 78/22 split
  (`random_state=42`, **no `stratify`**). Each `test_*` function fits on train and scores on
  that **same test set**.
- `preparation/Analyze_Models.ipynb` — the "best" model per family is chosen with
  `idxmax()` on **test accuracy**. Across ~480 NN configs / ~64 GBM configs, this keeps
  whichever config was luckiest on 137 test rows → optimistically biased, unstable metrics.
- No seed on GBM (`GradientBoostingClassifier(...)`), the NN, or the split-consuming models
  → non-deterministic; the README/PDF copied **stale** cached notebook outputs.
- `saving_selected_models_for_pipeline.py` hardcodes **weaker** hyperparameters than the
  reported "best" (GBM `lr=0.1,n=100` ≈ 85% vs reported `lr=0.2,n=800`; NN 1 layer/32/10
  epochs vs 4 layers/128/20; SVM `C=1` vs `C=4`). The website serves worse models than the
  report describes, and the NN is `pickle`-dumped / `joblib`-loaded (fragile across TF
  versions).
- `train_test_split` uses `Path(file_info.name)` in the notebook, stripping the directory
  (relative-path bug).
- `log of index` (a row-index artifact) is left in as a feature.
- `cluster_kmeans` / `scaled_cluster_kmeans` were computed on the full dataset **before** the
  split (leakage) and are undocumented — the "cluster variable" the judges asked about. The
  backend (`website/backend/main.py:595`) reconstructs these features at predict time from
  `kmeans_fitter.joblib`, but the deploy training script *drops* them → train/deploy mismatch.

### Verified facts
- Dataset: `data/ListeriaSoil_clean_log.csv`, **622 rows, balanced 50/50** (311 positive /
  311 negative). Accuracy is therefore not outright broken, but a **false negative (missed
  Listeria) is the costly error**, and the site turns predicted probability into a risk
  score — so **recall + calibrated probabilities** matter more than raw accuracy. Real-world
  soil prevalence is far below 50%, so a model trained on balanced data is **miscalibrated**
  for deployment unless corrected.
- True metrics (seeded, reproduced): GBM ≈ 87.6%, NN ≈ 90.5% — not the reported 94.2%.

### Mapping criticisms → fixes
| Review finding | Addressed by |
|---|---|
| Metrics not reproducible (94.2% vs ~87%) | §3 (CV + single holdout), §6 (seed, re-run, refresh Table 1) |
| DT slightly > RF ("unusual") | §3 (compare CV distributions, not single points) |
| Cluster variable unexplained | §4 (pipeline KMeans on train folds; brief inline note) |
| Which NN architecture / limited ML understanding | Methodology made defensible via §3–§5; the narrative is deferred to the later slide deck |
| Deployed model ≠ reported | §7 (deploy CV-selected configs from one source of truth) |
| sklearn API break, notebook interdependency, path bug | §1 refactor, §6 plumbing |
| Dependency / Python-version issues | §6 plumbing |

---

## 2. Decisions locked in

- **Validation strategy:** stratified k-fold **GridSearchCV** (RandomizedSearchCV for the
  large NN grid) on the training portion for selection, then **one held-out test set,
  touched exactly once**, for the reported number. (Chosen over nested CV for tractable
  runtime given the ~480-config NN grid.)
- **KMeans clusters:** **keep them, but fit KMeans inside the CV pipeline on training folds
  only** (no leakage), and add a brief inline note (code comment / one README line) of what
  the clusters represent — no formal write-up.
- **Selection metric:** **accuracy**, measured by cross-validation on the training data
  (never on the test set). Recall / PR-AUC / confusion matrix are reported alongside it.

---

## 3. Validation methodology (core fix)

- Single **stratified** train/test split up front (`stratify=y`, `random_state=42`); the test
  set is quarantined and used once at the very end.
- Model + hyperparameter selection via `GridSearchCV` / `RandomizedSearchCV` with
  `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the **training** portion.
- **Selection metric** (the score CV uses to pick the winner): **accuracy** — defensible on
  the balanced 50/50 data and the simplest to explain. The key change from the review is only
  that accuracy is now measured by **cross-validation on the training data**, never on the
  test set. Recall/PR-AUC/etc. are still reported alongside (§5).
- Seed everything: `random_state=42` on DecisionTree/GBM/RandomForest and all splits;
  `tf.keras.utils.set_random_seed(42)` (+ single-threaded determinism if needed) for the NN.
- NN grid shrunk and run via `RandomizedSearchCV` (or a small manual CV loop) so runtime
  stays in minutes.
- **DT-vs-RF question:** report each family's cross-validated score distribution (mean ± std)
  so the comparison is honest; the expectation is they overlap within noise.

## 4. Model registry, preprocessing & leakage hygiene

- Each model becomes an sklearn `Pipeline`:
  `SimpleImputer(strategy="median") → StandardScaler → [optional KMeans cluster feature] →
  estimator`. Fitting the whole pipeline inside each CV fold guarantees the imputer, scaler,
  and KMeans see **training data only**.
- The NN is wrapped as a pipeline-compatible estimator (KerasClassifier-style builder) so it
  goes through the same CV machinery.
- Drop `log of index` from the feature set (row-index artifact).
- Replace the ±99999 missing-value fill with median imputation inside the pipeline; document
  any fields that are structurally (not randomly) missing.
- KMeans clusters added via a custom transformer step so the label is computed from
  train-fold centroids only, then used as a feature. Persist the fitted KMeans with the model
  so the backend uses the identical transform.

## 5. Metric & calibration (use-case fit)

- **Accuracy stays a reported headline metric.** Report it *alongside* **recall
  (sensitivity), precision, F1, ROC-AUC, PR-AUC, and the confusion matrix** — accuracy is not
  removed or demoted; the other metrics are added so the picture is complete and recall (the
  cost-sensitive one) is visible.
- **Reporting vs. selecting** are separate: the change from the review is that we no longer
  *select* the winner on the **test set**. Selection happens via cross-validation on the
  training data (§3); see §3's selection-metric note for which score drives that choice.
- Add **probability calibration** (`CalibratedClassifierCV`, isotonic or sigmoid chosen by
  CV) around the final model, plus a **calibration curve + Brier score** in the analysis.
- Document the **prevalence caveat**: training is balanced 50/50; real-world prevalence is
  much lower, so deployed probabilities should be interpreted as relative risk (or
  base-rate-corrected). (Threshold re-tuning is out of scope for L2 → L3.)

## 6. Shared module & reproducibility plumbing

- Extract shared logic into `preparation/pipeline_utils.py`: `data_prep`, feature lists, the
  model registry, the CV/eval routine, and a single **`best_configs.json`** artifact. Both
  notebooks and `saving_selected_models_for_pipeline.py` import from it — the report and the
  deployment share one source of truth and cannot drift.
- Fix `LogisticRegression(C=c, l1_ratio=lr)` → correct `penalty`/`solver`
  (`solver="saga", penalty="elasticnet", l1_ratio=...`) or drop `l1_ratio` and use
  `penalty` in {`l1`,`l2`} with a compatible solver. Pin `scikit-learn`.
- Fix the `Path(file_info.name)` relative-path bug → use the real `data/` path.
- Re-execute both notebooks to refresh embedded outputs; regenerate
  `results_for_ListeriaSoil_clean_log.csv` and `top_3_scaled_models_summary.csv`; update
  README Table 1 (and note the corrected metrics for the PDF).
- `requirements.txt`: remove duplicate `numpy`, remove stdlib `pathlib`, add `plotly`, pin
  all versions, and add a Python-version constraint (`Requires Python 3.10 or 3.11`, since
  `tensorflow==2.15.1` does not support 3.12+).

## 7. Deployment alignment

- Rewrite `saving_selected_models_for_pipeline.py` to train the **CV-selected best configs**
  read from `best_configs.json` (seeded, identical to the report).
- Save the NN via Keras `model.save(...".keras")`; update `website/backend/main.py`
  `load_model` to load the Keras format for the neural net (keep joblib for sklearn models).
- Reconcile cluster handling: the same in-pipeline KMeans transform is persisted and used by
  the backend predict path, removing the current train/deploy mismatch.

---

## 8. Out of scope (candidates for L3)

- Decision-threshold optimization / cost-sensitive operating point.
- Class weighting, wider architecture search, stacked ensembles.
- New features or additional data sources.
- Front-end / GUI changes beyond loading the corrected models.
- Presentation slides and any Q&A narrative (separate, later effort).

## 9. Success criteria

1. `Run_and_Test_Models.ipynb` and `Analyze_Models.ipynb` run top-to-bottom independently on
   a clean env (pinned deps, Python 3.10/3.11) with no errors.
2. Reported metrics are reproducible from a seeded run and match the regenerated CSVs and
   README Table 1.
3. Model/hyperparameter selection uses stratified CV on train only; the test set is used
   exactly once.
4. No leakage: imputation, scaling, and KMeans are all fit inside CV folds; `log of index`
   removed.
5. The deployed models are the CV-selected best configs (report == deployment), NN saved and
   loaded via Keras format.
6. Analysis reports recall/PR-AUC + a calibration curve, and the regenerated metrics are
   accurate and defensible (ready to be turned into slides later).
