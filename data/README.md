# Data

No data files are stored in this repository. Both datasets ship with scikit-learn and are loaded in memory by
`src/data_loader.py`:

| Experiment | Loader | Rows | Features | Target |
|---|---|---|---|---|
| Classification | `sklearn.datasets.load_breast_cancer` | 569 | 30 numeric | malignant (1) or benign (0) |
| Regression | `sklearn.datasets.load_diabetes(scaled=False)` | 442 | 10 numeric | disease progression after one year |

The breast cancer target is recoded from scikit-learn's encoding (`0 = malignant`) so that malignant is the
positive class. The diabetes features are used in their original units rather than scikit-learn's default
mean-centred and scaled version.

Kumo Tabular checkpoints are downloaded by `structured-data-models` into the Hugging Face cache
(`~/.cache/huggingface/hub` by default), not into this folder.
