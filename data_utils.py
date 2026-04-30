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


#     return pd.DataFrame(rows)
def parse_client_history_csv(train_replies: dict) -> pd.DataFrame:
    """
    Explodes time-series lists into rows (one row per epoch).
    Excludes 'local-classes' and 'local-labels'.
    """
    rows = []

    # Keys that contain the time-series data to be exploded
    series_keys = [
        "epoch",
        "train-acc",
        "train-loss",
        "val-acc",
        "val-loss",
        "classifier-lr",
        "features-lr",
    ]

    # Scalar keys to repeat for every epoch row
    scalar_keys = ["client-name", "round-time", "transmission-time"]

    for round_id, client_list in train_replies.items():
        if not client_list:
            continue
        for client_data in client_list:
            # 1. Get the base scalars
            base_row = {"round": round_id}
            for k in scalar_keys:
                base_row[k] = client_data.get(k)

            # 2. Find the length of the lists (number of epochs)
            # Check 'epoch' or 'train-loss' to determine length
            num_epochs = 0
            for k in series_keys:
                val = client_data.get(k, [])
                if isinstance(val, list) and len(val) > num_epochs:
                    num_epochs = len(val)

            # 3. Create a row for each epoch
            if num_epochs == 0:
                # If no history data, just append the base row once
                rows.append(base_row)
            else:
                for i in range(num_epochs):
                    row = base_row.copy()
                    for k in series_keys:
                        val = client_data.get(k)
                        if isinstance(val, list) and i < len(val):
                            row[k] = val[i]
                        else:
                            row[k] = None  # Handle uneven lists
                    rows.append(row)

    df = pd.DataFrame(rows)
    # Reorder
    priority = ["round", "client-name", "epoch", "round-time", "transmission-time"]
    cols = priority + [c for c in df.columns if c not in priority]
    return df[cols]


def parse_client_evaluation_csv(train_replies: dict) -> pd.DataFrame:
    """
    Extracts only scalar evaluation metrics.
    Excludes 'per-class-accuracy' and other lists.
    """
    rows = []

    # Strict list of keys we want in this CSV (Scalars only)
    eval_scalar_keys = [
        "client-name",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "train-time",
        "transmission-time",
        "round-time",
        "num-examples",
    ]

    for round_id, client_list in train_replies.items():
        if not client_list:
            continue
        for client_data in client_list:
            row = {"round": round_id}

            for k in eval_scalar_keys:
                # Only add if it exists and is not a list (sanity check)
                val = client_data.get(k)
                if not isinstance(val, list):
                    row[k] = val
                else:
                    row[k] = None  # Explicitly ignore lists

            rows.append(row)

    df = pd.DataFrame(rows)
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
    train_metrics: dict, eval_metrics: dict
) -> pd.DataFrame:
    """
    Merges server-side timing and global performance.
    """
    rows = []
    all_rounds = sorted(list(set(train_metrics.keys()) | set(eval_metrics.keys())))

    for r in all_rounds:
        row = {"round": r}

        # From train_metrics_clientapp (Round Time)
        if r in train_metrics:
            data = train_metrics[r]
            if isinstance(data, dict):
                row.update(data)
            elif hasattr(data, "items"):
                row.update(dict(data.items()))

        # From evaluate_metrics_serverapp (Global Acc)
        if r in eval_metrics:
            data = eval_metrics[r]
            if isinstance(data, dict):
                row.update(data)
            elif hasattr(data, "items"):
                row.update(dict(data.items()))

        rows.append(row)

    return pd.DataFrame(rows)


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

    # 7. Save Metadata (JSON with Client Specifics)
    try:
        # Dictionary to hold data for each client
        # We iterate through all rounds. If a client appears in multiple rounds,
        # the data from the latest round will overwrite previous data (which is usually desired for 'final' state).
        client_configs = {}

        # Sort rounds to ensure we process in order (latest overwrites oldest)
        for round_id in sorted(train_replies.keys()):
            for client_data in train_replies[round_id]:
                name = client_data.get("client-name")
                if name:
                    client_configs[name] = {
                        "local-classes": client_data.get("local-classes"),
                        "local-labels": client_data.get("local-labels"),
                        "per-class-accuracy": client_data.get("per-class-accuracy"),
                    }

        metadata = config_dict.copy()
        metadata["final_global_metrics"] = global_metrics
        metadata["client_configs"] = client_configs

        json_path = exp_path / "metadata.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4, default=json_serializer)
        print(f"Saved metadata -> metadata.json")
    except Exception as e:
        print(f"Failed to save metadata: {e}")

    return exp_path
