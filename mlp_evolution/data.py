from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from ucimlrepo import fetch_ucirepo

from mlp_evolution.config import SearchConfig


@dataclass(slots=True)
class DatasetBundle:
    input_dim: int
    num_classes: int
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_size: int
    val_size: int
    test_size: int
    feature_names: list[str]
    target_names: list[str]
    dataset_metadata: dict
    raw_variables: list[dict]


def load_dataset(config: SearchConfig) -> DatasetBundle:
    dataset = fetch_ucirepo(id=config.dataset_id)
    features = dataset.data.features
    targets = dataset.data.targets

    if features is None or targets is None:
        raise ValueError("UCI dataset must provide both features and targets.")

    x_df = pd.DataFrame(features).copy()
    y_series = _prepare_target_series(targets)

    valid_mask = ~y_series.isna()
    x_df = x_df.loc[valid_mask].reset_index(drop=True)
    y_series = y_series.loc[valid_mask].reset_index(drop=True)

    x_df = pd.get_dummies(x_df, dummy_na=True)
    x_df = x_df.fillna(0.0)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_series.astype(str)).astype(np.int64)
    x = x_df.to_numpy(dtype=np.float32)
    dataset_metadata = dict(dataset.metadata or {})
    raw_variables = _prepare_variable_records(dataset.variables)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=config.test_ratio,
        random_state=config.random_seed,
        stratify=y,
    )

    relative_val_ratio = config.val_ratio / (config.train_ratio + config.val_ratio)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_val_ratio,
        random_state=config.random_seed,
        stratify=y_train_val,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    train_loader = _build_loader(x_train, y_train, config.batch_size, shuffle=True)
    val_loader = _build_loader(x_val, y_val, config.batch_size, shuffle=False)
    test_loader = _build_loader(x_test, y_test, config.batch_size, shuffle=False)

    return DatasetBundle(
        input_dim=x.shape[1],
        num_classes=len(np.unique(y)),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_size=len(x_train),
        val_size=len(x_val),
        test_size=len(x_test),
        feature_names=x_df.columns.tolist(),
        target_names=[str(name) for name in label_encoder.classes_],
        dataset_metadata=dataset_metadata,
        raw_variables=raw_variables,
    )


def _prepare_target_series(targets) -> pd.Series:
    if isinstance(targets, pd.DataFrame):
        if targets.shape[1] != 1:
            raise ValueError("Only single-target classification datasets are supported.")
        return targets.iloc[:, 0]
    if isinstance(targets, pd.Series):
        return targets
    target_array = np.asarray(targets)
    if target_array.ndim == 2:
        if target_array.shape[1] != 1:
            raise ValueError("Only single-target classification datasets are supported.")
        target_array = target_array[:, 0]
    return pd.Series(target_array)


def _prepare_variable_records(variables) -> list[dict]:
    if variables is None:
        return []
    if isinstance(variables, pd.DataFrame):
        return variables.to_dict(orient="records")
    if isinstance(variables, list):
        return [item for item in variables if isinstance(item, dict)]
    return []


def _build_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(labels),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
