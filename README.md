# PATHS_cpm (FERN Lab @ UVM)
  
  This is a fork of [wwu-mmll/confound_corrected_cpm](https://github.com/wwu-mml
  l/confound_corrected_cpm) at v0.7.0. It adds options used 
  in our ABCD adversity CPM analyses, which follow Simon et al. (2026, *Commun
  Biol*, doi:10.1038/s42003-026-10822-8). Every option added is off by
  default, so with the defaults, the package behaves like the author's v0.7.0.

## What we added
  
  We made eight changes to v0.7.0.

  Two of them set how the author's analysis is run. A new FixedSplits class (splits.py), passed in
  as cv, supplies precomputed outer folds, so every model is trained and tested on the same
  subjects. The permutation_groups option (cpm_analysis.py) restricts the permutation test to
  shuffling the target within blocks, such as site.

  Four changes add outputs without altering the author's results. Each edge's mean and standard
  deviation of r across folds is now saved (results_manager.py). With effect_size='partial', that r
  is the partial rather than the semipartial correlation; the selected edges are unchanged
  (statistics.py, edge_selection.py). With pack_fold_edges=True, the per-fold edge masks are saved
  bit-packed in edges_packed.npz instead of edges.npy (results_manager.py). Finally, predictions
  are also scored pooled across all subjects in each repeat (Spearman and R²), with a permutation
  p-value computed on the median over repeats (results_manager.py, inference.py).

  The last two are opt-in and off in our main runs. edge_mask restricts edge selection to a
  predefined set of edges (edge_selection.py), and fixed_edges skips selection and fits each fold
  on given edge masks (cpm_analysis.py).

  ## Notes
  - v0.7.0 cannot take a text covariate column (such as site) alongside numeric ones. One-hot
  encode it first: `pd.get_dummies(covariates, drop_first=True, dtype=float)`.
  - `src/cccpm/run_cpm.py` is our cluster driver (one target per Slurm array task). It isn't part
  of the library.
  - This repo holds no ABCD data, subject IDs or results.

  ## Credit
  CCCPM is by Nils Winter, Juraj Howe and colleagues (MIT license). If you use this code, cite the
  original package; see `CITATION.cff`.

--------------------------------------------------------------------------------------------------------

[![GitHub Workflow Status](https://github.com/wwu-mmll/confound_corrected_cpm/actions/workflows/test.yml/badge.svg)](https://github.com/wwu-mmll/confound_corrected_cpm/actions/workflows/test.yml)
[![Coverage Status](https://coveralls.io/repos/github/wwu-mmll/confound_corrected_cpm/badge.svg)](https://coveralls.io/github/wwu-mmll/confound_corrected_cpm)
[![Github Contributors](https://img.shields.io/github/contributors-anon/wwu-mmll/confound_corrected_cpm?color=blue)](https://github.com/wwu-mmll/confound_corrected_cpm/graphs/contributors)
[![Github Commits](https://img.shields.io/github/commit-activity/y/wwu-mmll/confound_corrected_cpm)](https://github.com/wwu-mmll/confound_corrected_cpm/commits/main)

# Confound-Corrected Connectome-Based Predictive Modelling in Python
**Confound-Corrected Connectome-Based Predictive Modelling** is a Python package for performing connectome-based predictive modeling (CPM). This toolbox is designed for researchers in neuroscience and psychiatry, providing robust methods for building predictive models based on structural or functional connectome data. It emphasizes replicability, interpretability, and flexibility, making it a valuable tool for analyzing brain connectivity and its relationship to behavior or clinical outcomes.

---

## What is Connectome-Based Predictive Modeling?

Connectome-based predictive modeling (CPM) is a machine learning framework that leverages the brain's connectivity patterns to predict individual differences in behavior, cognition, or clinical status. By identifying key edges in the connectome, CPM creates models that link connectivity metrics with target variables (e.g., clinical scores). This approach is particularly suited for studying complex relationships in neuroimaging data and developing interpretable predictive models.

---

## Key Features

- **Univariate Edge Selection**: Supports methods like `pearson`, `spearman`, and their partial correlation counterparts, with options for p-threshold optimization and FDR correction.
- **Cross-Validation**: Implements nested cross-validation for robust model evaluation.
- **Edge Stability**: Selects stable edges across folds to improve model reliability.
- **Confound Adjustment**: Controls for covariates during edge selection and modeling.
- **Permutation Testing**: Assesses the statistical significance of models using robust permutation-based methods.

---

## Documentation

For detailed instructions on installation, usage, and advanced configurations, visit the [documentation website](https://wwu-mmll.github.io/confound_corrected_cpm/).

---

## Installation

Install the latest release from PyPI:

```bash
pip install cccpm
```

Or install the development version from GitHub:

```bash
git clone https://github.com/wwu-mmll/confound_corrected_cpm.git
cd confound_corrected_cpm
pip install .
```

CCCPM requires Python 3.10–3.14 and uses [PyTorch](https://pytorch.org/) for
computation (CPU by default; CUDA/MPS used automatically when available). See the
[installation guide](https://wwu-mmll.github.io/confound_corrected_cpm/installation/)
for platform-specific notes.

## Quick Example
Here's a quick overview of how to run a CPM analysis:

```python
from cccpm import CPMAnalysis, UnivariateEdgeSelection, PThreshold
from sklearn.model_selection import KFold

# Configure edge selection. `selection_input="residualized"` controls for the
# covariates while *selecting* edges.
univariate_edge_selection = UnivariateEdgeSelection(
    selection_statistic="pearson",
    selection_input="residualized",
    edge_selection=[PThreshold(threshold=[0.05], correction=["fdr_by"])]
)

# Create the CPM analysis object. `model_input` is the second, independent
# confound-control choice: deconfound the connectome the models consume.
cpm = CPMAnalysis(
    results_directory="results/",
    cv=KFold(n_splits=10, shuffle=True, random_state=42),
    edge_selection=univariate_edge_selection,
    model_input="residualized",
    n_permutations=100
)

# Run the analysis
X = ...           # Connectome data, shape (n_samples, n_features)
y = ...           # Target variable, shape (n_samples,)
covariates = ...  # Covariates to control for, shape (n_samples, n_covariates)
cpm.run(X=X, y=y, covariates=covariates)   # covariates are optional
```

## Contributing
Contributions are welcome! If you have ideas, feedback, or feature requests, feel free to open an issue or submit a pull request on the GitHub repository.

