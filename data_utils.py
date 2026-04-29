import os, time, json, pickle, csv
from typing import Literal, Any
import torch
import pandas as pd
from pathlib import Path, PosixPath
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

    :param path: file path
    :type path: str
    :param data: data to be saved
    """
    import pickle

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pkl(path: str):
    """
    load data from a .pkl file.

    :param path: file path
    :type path: str
    :return: raw data
    :rtype: Any
    """

    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def parse_raw_metrics(raw_metrics: dict[int, list[dict[str, Any]]]) -> pd.DataFrame:
    """
    Transforms nested raw metrics into a Pandas DataFrame dynamically.
    Handles both list-based metrics (epochs) and scalar values (metadata) automatically.

    Args:
        raw_metrics: Dictionary mapping round numbers to lists of client metric dicts.
                     Client dicts contain both scalar values (e.g., 'client-name')
                     and list values (e.g., 'train-loss' per epoch).

    Returns:
        A flat Pandas DataFrame where each row corresponds to one epoch of one client.
    """
    rows = []

    for round_id, client_list in raw_metrics.items():
        for client_data in client_list:
            if not client_data:
                continue

            # 1. Separate scalars (metadata) from lists (time-series metrics)
            # We assume metrics that vary per epoch are lists, and constants are scalars.
            scalars = {k: v for k, v in client_data.items() if not isinstance(v, list)}
            series_data = {k: v for k, v in client_data.items() if isinstance(v, list)}

            # 2. Determine the number of epochs (steps) from the first available list
            # If no lists exist (e.g., only scalars returned), we create 1 row.
            num_steps = 0
            for v in series_data.values():
                if len(v) > 0:
                    num_steps = len(v)
                    break

            if num_steps == 0 and scalars:
                # Handle case where client only returned scalars
                row = {"round": round_id, **scalars}
                rows.append(row)
                continue

            # 3. Flatten the lists into individual rows
            for i in range(num_steps):
                row = {"round": round_id}

                # Add metadata (repeated for each epoch row)
                row.update(scalars)

                # Add the specific value for this epoch index 'i'
                for key, values in series_data.items():
                    # Check bounds to prevent errors if lists are uneven
                    row[key] = values[i] if i < len(values) else None

                rows.append(row)

    # 4. Create DataFrame
    df = pd.DataFrame(rows)

    # Optional: Reorder columns to put important identifiers first
    # This is dynamic but puts 'round', 'client-name' at the front if they exist
    priority_cols = ["round", "client-name"]
    existing_priority = [c for c in priority_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_priority]

    return df[existing_priority + other_cols]


def parse_server_eval_metrics(metrics: dict[int, dict[str, Any]]) -> pd.DataFrame:
    """
    Transforms server evaluation metrics into a DataFrame.

    Args:
        metrics: Dictionary mapping round numbers to metric dictionaries.

    Returns:
        A Pandas DataFrame with one row per round.
    """
    # The simplest way to handle dynamic dict->DataFrame conversion
    # We add the round key to the dictionary for each row
    rows = [{"round": r, **data} for r, data in metrics.items()]

    return pd.DataFrame(rows)


def split_df_by_type(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into two based on column content types.

    Returns:
        df_scalars: Columns containing single numbers/strings (good for plots/stats).
        df_lists:   Columns containing lists/objects (good for inspection/exploding).
    """
    scalar_cols = []
    list_cols = []

    # Iterate over the first row to determine types
    # We assume the first row is representative of the whole column
    if not df.empty:
        first_row = df.iloc[0]
        for col in df.columns:
            # Check if the value is a list
            if isinstance(first_row[col], list):
                list_cols.append(col)
            else:
                scalar_cols.append(col)

    return df[scalar_cols], df[list_cols]


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

    Args:
        save_dir: Root directory where experiment folders will be created.
        dataset_name: Name of the dataset (used for folder naming).
        model_name: Name of the model (used for folder naming).
        train_replies: The raw dictionary of client training logs.
        result: The result object from strategy.start (contains server metrics).
        state_dict: The model state dictionary.
        config_dict: Dictionary of hyperparameters (epochs, lr, etc.).
        global_metrics: Dictionary containing final global metrics (tensors allowed).
    """

    # 1. Create Unique Experiment Folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{dataset_name}_{model_name}_{timestamp}"
    exp_path = Path(save_dir) / folder_name
    exp_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Saving experiment data to: {exp_path}")
    print(f"{'='*50}")

    # Helper to sanitize tensors for JSON
    def json_serializer(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.ndim == 0 else obj.tolist()
        if isinstance(obj, (Path, PosixPath)):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # 2. Save Model State Dict
    model_path = exp_path / "model.pt"
    torch.save(state_dict, model_path)
    print(f"Saved model state dict -> {model_path.name}")

    # 3. Save Raw Data (Pickle)
    raw_path = exp_path / "raw_data.pkl"
    save_pkl(raw_path, train_replies)
    print(f"Saved raw data -> {raw_path.name}")

    # 4. Save Client Metrics (CSV)
    try:
        client_df = parse_raw_metrics(train_replies)
        # Remove list columns for cleaner CSV analysis
        client_df_clean, _ = split_df_by_type(client_df)
        client_csv_path = exp_path / "client_metrics.csv"
        client_df_clean.to_csv(client_csv_path, index=False)
        print(f"Saved client metrics -> {client_csv_path.name}")
    except Exception as e:
        print(f"Failed to save client metrics: {e}")

    # 5. Save Server Evaluation Metrics (CSV)
    try:
        # Assuming result.evaluate_metrics_serverapp exists
        if (
            hasattr(result, "evaluate_metrics_serverapp")
            and result.evaluate_metrics_serverapp
        ):
            server_df = parse_server_eval_metrics(result.evaluate_metrics_serverapp)
            server_csv_path = exp_path / "server_eval_metrics.csv"
            server_df.to_csv(server_csv_path, index=False)
            print(f"Saved server metrics -> {server_csv_path.name}")
    except Exception as e:
        print(f"Failed to save server metrics: {e}")

    # 6. Save Metadata (JSON)
    try:
        metadata = config_dict.copy()
        # Add final global metrics to metadata
        metadata["final_global_metrics"] = global_metrics

        json_path = exp_path / "metadata.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4, default=json_serializer)
        print(f"Saved metadata -> {json_path.name}")
    except Exception as e:
        print(f"Failed to save metadata: {e}")

    # 7. Confusion Matrices
    # conf_matrix_path = exp_path / "confusion_matrices.json"
    # with open(conf_matrix_path, "w") as f:
    #     json.dump(confusion_data, f, default=json_serializer)

    return exp_path
