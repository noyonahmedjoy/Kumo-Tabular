import importlib.metadata
import platform

import numpy as np
import pandas as pd
import torch

from src.data_loader import TARGET_COLUMN

QUANTILE_COLUMNS = [f"q{i:03d}" for i in range(1, 1000)]


class KumoUnavailableError(RuntimeError):
    pass


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(preference)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise KumoUnavailableError("CUDA was requested but no CUDA device is available")
    if device.type == "cuda" and (device.index or 0) >= torch.cuda.device_count():
        raise KumoUnavailableError(f"{device} was requested but only {torch.cuda.device_count()} CUDA device(s) exist")
    return device


def autocast_supported(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) >= (7, 0)


def environment_info(device: torch.device) -> dict:
    try:
        sdm_version = importlib.metadata.version("structured-data-models")
    except importlib.metadata.PackageNotFoundError:
        sdm_version = None
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "structured_data_models": sdm_version,
        "device": str(device),
        "gpu": None,
        "gpu_memory_gb": None,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info["gpu"] = properties.name
        info["gpu_memory_gb"] = round(properties.total_memory / 1024**3, 1)
    return info


def _import_sdm():
    try:
        import sdm
    except ImportError as error:
        raise KumoUnavailableError(
            "structured-data-models is not installed. Install it with "
            "'pip install structured-data-models'."
        ) from error
    return sdm


class KumoTabularPredictor:
    def __init__(
        self,
        task: str,
        size: str = "small",
        device: str = "auto",
        num_estimators: int = 8,
        estimator_batch_size: int | None = 1,
        autocast: bool | str = "auto",
    ) -> None:
        if task not in ("classification", "regression"):
            raise ValueError(f"Unsupported task '{task}'")
        self.sdm = _import_sdm()
        self.task = task
        self.size = size
        self.device = resolve_device(device)
        self.num_estimators = num_estimators
        self.estimator_batch_size = estimator_batch_size
        self.autocast = autocast_supported(self.device) if autocast == "auto" else bool(autocast)
        self.is_warm = False
        try:
            self.model = self.sdm.models.KumoTabular(task=task, size=size, device=self.device)
        except Exception as error:
            raise KumoUnavailableError(
                f"Could not load the Kumo Tabular '{size}' {task} checkpoint from "
                f"Hugging Face (nvidia/Kumo-Tabular): {error}"
            ) from error

    def _target_stype(self) -> str:
        return "categorical" if self.task == "classification" else "numerical"

    def _run(self, X_context: pd.DataFrame, y_context: pd.Series, X_query: pd.DataFrame, seed: int):
        if TARGET_COLUMN in X_query.columns:
            raise ValueError(f"Query rows must not contain the '{TARGET_COLUMN}' column")
        if list(X_context.columns) != list(X_query.columns):
            raise ValueError("Context and query feature columns differ")
        if len(X_context) != len(y_context):
            raise ValueError("Context features and targets have different lengths")

        context = X_context.assign(**{TARGET_COLUMN: y_context.to_numpy()})
        stypes = self.sdm.infer_stypes(context, overrides={TARGET_COLUMN: self._target_stype()})
        feature_stypes = {column: stype for column, stype in stypes.items() if column != TARGET_COLUMN}
        context_table = self.sdm.TableTensor.from_pandas(context, stypes, device=self.device)
        query_table = self.sdm.TableTensor.from_pandas(X_query, feature_stypes, device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        with torch.amp.autocast(self.device.type, dtype=torch.float16, enabled=self.autocast):
            output = self.model(
                x_context=context_table.drop_columns(TARGET_COLUMN),
                y_context=context_table[:, TARGET_COLUMN],
                x_query=query_table,
                num_estimators=self.num_estimators,
                estimator_batch_size=self.estimator_batch_size,
                generator=generator,
            )
        columns = list(output.columns[self.sdm.Stype.numerical])
        values = output.numerical.float().cpu().numpy()
        if values.shape != (len(X_query), len(columns)):
            raise ValueError(
                f"Kumo returned shape {values.shape}, expected ({len(X_query)}, {len(columns)})"
            )
        if not np.isfinite(values).all():
            raise ValueError("Kumo returned non-finite predictions")
        return pd.DataFrame(values, columns=columns, index=X_query.index)

    def warmup(self, X_context: pd.DataFrame, y_context: pd.Series) -> None:
        if self.is_warm:
            return
        self._run(X_context, y_context, X_context.iloc[:8], seed=0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.is_warm = True

    def predict_proba(
        self, X_context: pd.DataFrame, y_context: pd.Series, X_query: pd.DataFrame, seed: int = 0
    ) -> pd.DataFrame:
        if self.task != "classification":
            raise ValueError("predict_proba requires a classification predictor")
        output = self._run(X_context, y_context, X_query, seed)
        classes = np.sort(y_context.unique())
        expected_columns = [str(label) for label in classes]
        if sorted(output.columns) != sorted(expected_columns):
            raise ValueError(f"Kumo returned classes {list(output.columns)}, expected {expected_columns}")
        probabilities = output[expected_columns].set_axis(classes, axis=1)
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-3):
            raise ValueError("Kumo class probabilities do not sum to one")
        return probabilities

    def predict_quantiles(
        self, X_context: pd.DataFrame, y_context: pd.Series, X_query: pd.DataFrame, seed: int = 0
    ) -> pd.DataFrame:
        if self.task != "regression":
            raise ValueError("predict_quantiles requires a regression predictor")
        output = self._run(X_context, y_context, X_query, seed)
        if list(output.columns) != QUANTILE_COLUMNS:
            raise ValueError(f"Kumo returned {output.shape[1]} unexpected quantile columns")
        return output
