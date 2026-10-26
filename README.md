# Orange Paralelo

*[Leer en castellano](https://github.com/Krl05oP11/orange3-paralelo/blob/main/README.es.md)*

An add-on for [Orange](https://orangedatamining.com/) 3.40 with four widgets that spread the work across CPU cores, and a new model, **Linear SVM (liblinear)**. They sit next to the original widgets, in their own category, **Paralelo**:

| Widget | What runs in parallel |
|---|---|
| Test and Score (parallel) | The (fold × model) pairs of the evaluation, for every sampling method of the original. Adds **"Keep groups together:"**: cross-validation stratified by class and grouped by a variable (StratifiedGroupKFold), for data with several rows per group, such as several texts from the same source |
| Parameter Fitter (parallel) | The grid points |
| Permutation Plot (parallel) | The permutations |
| Random Forest (parallel) | The trees (scikit-learn's `n_jobs`) |
| Linear SVM (liblinear) | Nothing: it is a new model. A linear SVM built on liblinear, meant for text. It takes seconds per fold where Orange's SVM takes minutes or hours. Details below |

In the four parallel widgets, inputs, outputs and options are those of the original. Each one adds a **Parallel** box with the number of processes. The default is the number of physical cores minus one; with 1, the widget runs Orange's original code, serially.

**Linear SVM (liblinear):**
- **Models:** `LinearSVC` one-vs-one, like libsvm, for classification; `LinearSVR` for regression.
- **Deterministic.**
- **Calibrated probabilities:** a sigmoid fitted on 5 internal folds. The most probable class is always the predicted class.
- **Sparse data:** no imputation. Each column is scaled by its range, like Orange's SVMs do, but inside the model, without creating one Orange variable per column.

## Installation
In Orange: **Options → Add-ons… → Add more…**, type `orange3-paralelo`, click **Add**, then **OK**, and restart Orange.

Or with pip, in the Python environment where Orange is installed:
```bash
pip install orange3-paralelo
```
To uninstall it, untick it in the same dialog, or run `pip uninstall orange3-paralelo`. Orange itself is not touched.

## How much faster
Measured on a Ryzen 9 7900 (12 cores, 24 threads) with 11 processes. In every case the results were identical to Orange's:

| Case | Orange (serial) | Paralelo (11 processes) | Speed-up |
|---|---|---|---|
| Test and Score: text corpus, TF-IDF (2,699 × 70k) + logistic regression, 10-fold CV | 608 s | 140 s | **4.4×** |
| Test and Score: synthetic 50k × 50 + Random Forest (100 trees), 10-fold CV | 208 s | 25 s | **8.4×** |
| Parameter Fitter: synthetic 10k × 30 + Random Forest, 10 points (10–100 trees) | 101 s | 20 s | **5.2×** |
| Permutation Plot: heart_disease + logistic regression, 100 permutations | 8.2 s | 2.8 s | **3.0×** |
| Test and Score: titanic + logistic regression, 10-fold CV (runs serially by itself) | 0.0 s | 0.0 s | 1.0× |

- **Small jobs run serially by themselves.** The threshold is 1 million cells × models to fit (`motor.UMBRAL_TRABAJO`). Below it, the serial run takes less than about 1 s, which is what starting the processes costs.
- **The first parallel run is about 1 s slower,** both in each Orange session and after a cancel, because the processes start then. Later runs reuse them.
- **In cross-validation, the ceiling is the number of tasks.** 10 folds × 1 model are 10 tasks, and the slowest fold sets the pace. Parameter Fitter and Permutation Plot have more tasks, so they use more cores.

### With a real text corpus
The corpus has 2,699 texts in 18 classes, as 130,047 bag-of-words columns, built as an Orange3-Text Corpus. Test and Score used stratified 5-fold CV, and each model used the settings of its widget:

| Case | Before | With the add-on | Speed-up |
|---|---|---|---|
| Logistic regression (C = 1), 5 tasks | 7.8 min (Orange, serial) | 0.7 min | **11×**, identical classes and probabilities (1e-12) |
| Logistic regression + Orange's SVM, 10 tasks | ~113 min (an earlier version of the add-on); about 9 h estimated for serial Orange | **9.1 min** | **12×** |

With cross-validation grouped by text source (**"Keep groups together:"**):

| Model | Balanced accuracy | Time |
|---|---|---|
| Logistic regression (C = 1) | 0.293 | 0.7 min |
| Orange's SVM (linear kernel) | **0.332** | 9.4 min |
| Linear SVM (liblinear) | 0.322 | **0.25 min** |

Linear SVM (liblinear) comes within 0.010 of Orange's SVM and is about 38 times faster. Adding two features of Orange's SVM, range scaling and one-vs-one, raised its balanced accuracy from 0.287 to 0.322: +0.035, with a 95% CI of [+0.021, +0.049] from a bootstrap over groups. This was measured on a single text dataset.

## How it works
- **`motor.py` (no Qt):**
  - One pool of processes (loky) serves the whole Orange session, with 1 BLAS thread per process.
  - A lock lets one parallel job run at a time. If two widgets start together, the second one waits, and it can be cancelled while waiting.
  - Cancelling kills the processes, so nothing keeps running.
  - The table is written to disk once, and each process memory-maps it instead of receiving a copy with every task.
- **`evaluacion.py` (no Qt):** reproduces Orange 3.40's code and only replaces the serial loop with the engine. That covers:
  - Validation and TestOnTestData;
  - Parameter Fitter's `_search`;
  - Permutation Plot's `permutation`.
- **Each widget subclasses the original and changes a single point:**
  - **Test and Score:** the method that submits the evaluation (`__submit`);
  - **Parameter Fitter and Permutation Plot:** the task they start (`start`);
  - **Random Forest:** `create_learner`.
- **A Corpus is sent to the processes as a plain `Table`,** with the same domain and matrices.
  - Orange transforms data column by column (`SklImpute`, the SVM's `Scale`). With an Orange3-Text Corpus, it builds a new Corpus for every column, which is almost all of the time with a 130,000-column bag of words.
  - Orange's models do not use anything specific to a Corpus, so the results do not change, and the outputs still come from the original Corpus.
  - There is no conversion if Test and Score has a preprocessor or if a model is not one of Orange's.
- **Inside a process, a model with its own `n_jobs` uses 1,** such as the parallel Random Forest, because the cores are already shared out at the level above.

## Risks and limits
- **Tested with Orange 3.40.**
  - The add-on relies on Orange internals: `Validation`, `_mp_worker`, the function chain that Test and Score builds, and the names of the `run` functions.
  - On any other Orange version, the parallel widgets run the original code serially and show a warning (`evaluacion.VERSIONES_PROBADAS`). In Test and Score, "Keep groups together" is not available then, and a second warning says the cross-validation is the ordinary one. Linear SVM does not depend on Orange internals.
  - New releases of the add-on will follow new Orange versions. If an Orange release removes an internal the add-on imports, the widgets may fail to load until an update is out.
- **Subclassing widgets:** Orange 3.40 warns that this "will be disabled in the future".
  - The warning is silenced only for the four parallel classes. Linear SVM subclasses `OWBaseLearner`, which is open to subclassing.
  - If a future version forbids it, the widgets will be rewritten by composition.
- **Permutation Plot:** the shuffles are the original's (seed 0, chained).
  - The result is identical if the model does not draw from numpy's global random generator between shuffles, as with logistic regression or a Random Forest with a fixed seed.
  - With a model that does draw from it, the original is not reproducible either, and the parallel version fixes one seed per permutation.
- **SVM probabilities:** the `SVC(probability=True)` that Orange builds does not fix `random_state`.
  - Two runs of the same original Test and Score give the same predicted classes but different probabilities (up to 0.07 in a test with text).
  - For Orange's SVM, identity with the original therefore holds for the classes, not the probabilities.
- **Preprocessor in cross-validation:** each process applies it to its own training set, just as the original applies it to each one. A preprocessor that uses the global random generator without a seed can give a different result, as already happens between two runs of the original.

## Tests
```bash
pip install -e ".[test]"
python -m pytest tests/
```
There are 50 tests:
- **identity with the original** in every sampling method, with a preprocessor, with a failing model, with an Orange3-Text `Corpus`, in Parameter Fitter and in Permutation Plot;
- **Corpus as Table:** the processes receive a `Table` with identical results, the outputs are still the Corpus, and there is no conversion with a preprocessor or with a non-Orange model;
- **Linear SVM:**
  - probabilities sum to 1 and their top class is the predicted one;
  - it is deterministic;
  - regression works, and so does a class with a single example;
  - a bag of words gets no per-column variables;
  - scaling matches Orange's;
- **engine:** order, errors, cancelling without orphan processes, and two jobs at once;
- **widgets:** checked with Orange's test framework, including grouped cross-validation and the fallback on an untested Orange version.

Code comments and docstrings are in Spanish.

## License
GPL-3.0-or-later; see [LICENSE](https://github.com/Krl05oP11/orange3-paralelo/blob/main/LICENSE). Copyright © 2026 Carlos Ponce Schaller.

This add-on contains code adapted from Orange 3.40, © Bioinformatics Laboratory, Faculty of Computer and Information Science, University of Ljubljana, under GPL-3.0-or-later. It was modified in October 2026 to spread the work across processes:
- `evaluacion.py` takes parts of `Orange/evaluation/testing.py`, `owparameterfitter.py` and `owpermutationplot.py`;
- the widget icons are Orange's.

Developed by **Schaller & Ponce**.
