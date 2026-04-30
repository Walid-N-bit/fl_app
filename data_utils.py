import os
import time
import json
import pickle
import csv
from typing import Literal, Any
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime


def save_arbitrary_json(path: str, data: dict):
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def save_pkl(path: str, data):
    """
    save raw data into .pkl files.
    """
    # Convert Path object to string for compatibility with older os.path usage if needed
    path = str(path)
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pkl(path: str):
    """
    load data from a .pkl file.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


# ----------------------------------------------------------------------
# PARSING FUNCTIONS
# ----------------------------------------------------------------------


def parse_client_history_csv(train_replies: dict) -> pd.DataFrame:
    """
    Creates the History CSV.
    Cols: round, client-name, round-time, transmission-time, and epoch lists.
    Lists are preserved inside the cells.
    """
    rows = []

    # Keys that contain lists (History)
    history_keys = [
        "train-acc",
        "train-loss",
        "val-acc",
        "val-loss",
        "local-classes",
        "local-labels",
        "epoch",
        "classifier-lr",
        "features-lr",
    ]

    # Keys that are scalars but relevant for history context
    scalar_context_keys = ["client-name", "round-time", "transmission-time"]

    for round_id, client_list in train_replies.items():
        if not client_list:
            continue
        for client_data in client_list:
            row = {"round": round_id}

            # Add scalars
            for k in scalar_context_keys:
                row[k] = client_data.get(k)

            # Add Lists
            for k in history_keys:
                row[k] = client_data.get(k)

            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Reorder for readability
    priority = ["round", "client-name", "round-time", "transmission-time"]
    cols = priority + [c for c in df.columns if c not in priority]
    return df[cols]


def parse_client_evaluation_csv(train_replies: dict) -> pd.DataFrame:
    """
    Creates the Evaluation CSV.
    Cols: round, client-name, accuracy, precision, recall, f1, train-time,
          transmission-time, per-class-accuracy.
    """
    rows = []

    eval_keys = [
        "client-name",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "train-time",
        "transmission-time",
        "round-time",
        "per-class-accuracy",
        "num-examples",
    ]

    for round_id, client_list in train_replies.items():
        if not client_list:
            continue
        for client_data in client_list:
            row = {"round": round_id}

            for k in eval_keys:
                row[k] = client_data.get(k)

            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    priority = [
        "round",
        "client-name",
        "accuracy",
        "f1",
        "train-time",
        "transmission-time",
    ]
    cols = priority + [c for c in df.columns if c not in priority]
    return df[cols]


def parse_server_metrics_combined(
    train_metrics: dict[int, Any], eval_metrics: dict[int, Any]
) -> pd.DataFrame:
    """
    Merges server-side timing (train_metrics_clientapp)
    and global performance (evaluate_metrics_serverapp).
    """
    rows = []

    # Get all unique round numbers from both sources
    all_rounds = set(train_metrics.keys()) | set(eval_metrics.keys())

    for r in sorted(list(all_rounds)):
        row = {"round": r}

        # 1. Add training metrics (contains round-time, transmission-time if aggregated)
        if r in train_metrics:
            data = train_metrics[r]
            # Handle dict or MetricRecord-like objects
            if isinstance(data, dict):
                row.update(data)
            elif hasattr(data, "items"):
                row.update(dict(data.items()))
            elif hasattr(data, "__dict__"):
                row.update(
                    {k: getattr(data, k) for k in dir(data) if not k.startswith("_")}
                )

        # 2. Add evaluation metrics (contains global loss/acc)
        if r in eval_metrics:
            data = eval_metrics[r]
            if isinstance(data, dict):
                row.update(data)
            elif hasattr(data, "items"):
                row.update(dict(data.items()))
            elif hasattr(data, "__dict__"):
                row.update(
                    {k: getattr(data, k) for k in dir(data) if not k.startswith("_")}
                )

        rows.append(row)

    return pd.DataFrame(rows)


# def parse_raw_metrics(raw_metrics: dict[int, list[dict[str, Any]]]) -> pd.DataFrame:
#     # This function is now superseded by the specific parsers above,
#     # but kept here as it was in your provided code.
#     rows = []
#     METADATA_LIST_KEYS = {"local-classes", "local-labels", "per-class-accuracy"}

#     for round_id, client_list in raw_metrics.items():
#         for client_data in client_list:
#             if not client_data:
#                 continue
#             scalars = {}
#             metadata_lists = {}
#             series_data = {}

#             for k, v in client_data.items():
#                 if not isinstance(v, list):
#                     scalars[k] = v
#                 elif k in METADATA_LIST_KEYS:
#                     metadata_lists[k] = v
#                 else:
#                     series_data[k] = v

#             num_steps = 0
#             for v in series_data.values():
#                 if len(v) > 0:
#                     num_steps = len(v)
#                     break

#             if num_steps == 0:
#                 row = {"round": round_id, **scalars, **metadata_lists}
#                 rows.append(row)
#                 continue

#             for i in range(num_steps):
#                 row = {"round": round_id}
#                 row.update(scalars)
#                 row.update(metadata_lists)
#                 for key, values in series_data.items():
#                     row[key] = values[i] if i < len(values) else None
#                 rows.append(row)

#     df = pd.DataFrame(rows)
#     if df.empty:
#         return df
#     priority_cols = ["round", "client-name"]
#     existing_priority = [c for c in priority_cols if c in df.columns]
#     other_cols = [c for c in df.columns if c not in existing_priority]
#     return df[existing_priority + other_cols]


# def split_df_by_type(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
#     scalar_cols = []
#     list_cols = []
#     if not df.empty:
#         first_row = df.iloc[0]
#         for col in df.columns:
#             if isinstance(first_row[col], list):
#                 list_cols.append(col)
#             else:
#                 scalar_cols.append(col)
#     return df[scalar_cols], df[list_cols]


def json_serializer(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_experiment_data(
    save_dir: str,
    dataset_name: str,
    model_name: str,
    train_replies: dict,
    result: Any,
    state_dict: dict,
    config_dict: dict,
    global_metrics: dict,
):
    """
    Saves all experiment data into a single timestamped folder.
    """

    # 1. Create Unique Experiment Folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{dataset_name}_{model_name}_{timestamp}"
    exp_path = Path(save_dir) / folder_name
    exp_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Saving experiment data to: {exp_path}")
    print(f"{'='*50}")

    # 2. Save Model State Dict
    model_path = exp_path / "model.pt"
    torch.save(state_dict, model_path)
    print(f"Saved model state dict -> {model_path.name}")

    # 3. Save Raw Data (Pickle)
    raw_path = exp_path / "raw_data.pkl"
    save_pkl(raw_path, train_replies)
    print(f"Saved raw data -> {raw_path.name}")

    # 4. Save Client History (CSV) - Lists preserved
    try:
        df_history = parse_client_history_csv(train_replies)
        if not df_history.empty:
            df_history.to_csv(exp_path / "client_history.csv", index=False)
            print(f"Saved client history -> client_history.csv")
    except Exception as e:
        print(f"Failed to save client history: {e}")

    # 5. Save Client Evaluation (CSV) - Scalars + Per-Class
    try:
        df_eval = parse_client_evaluation_csv(train_replies)
        if not df_eval.empty:
            df_eval.to_csv(exp_path / "client_evaluation.csv", index=False)
            print(f"Saved client evaluation -> client_evaluation.csv")
    except Exception as e:
        print(f"Failed to save client evaluation: {e}")

    # 6. Save Server Metrics (CSV) - Combined Time + Global Metrics
    try:
        server_train = getattr(result, "train_metrics_clientapp", {})
        server_eval = getattr(result, "evaluate_metrics_serverapp", {})

        if server_train or server_eval:
            df_server = parse_server_metrics_combined(server_train, server_eval)
            df_server.to_csv(exp_path / "server_metrics.csv", index=False)
            print(f"Saved server metrics -> server_metrics.csv")

    except Exception as e:
        print(f"Failed to save server metrics: {e}")

    # 7. Save Metadata (JSON)
    try:
        metadata = config_dict.copy()
        metadata["final_global_metrics"] = global_metrics

        json_path = exp_path / "metadata.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4, default=json_serializer)
        print(f"Saved metadata -> {json_path.name}")
    except Exception as e:
        print(f"Failed to save metadata: {e}")

    return exp_path
