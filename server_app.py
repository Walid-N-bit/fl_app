import torch
from flwr.app import ArrayRecord, Context, MetricRecord, ConfigRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAvgM
from CustomClasses import CustomStrat, GlobalEvaluation
from utils import (
    cmd,
    save_pkl,
    parse_raw_metrics,
    parse_server_eval_metrics,
    generate_labels_map,
    readable_time,
    save_arbitrary_json,
    split_df_by_type,
)

from datetime import datetime
import os, time
from typing import Literal

from model_functions import (
    choose_model,
    get_true_and_pred_values,
    eval_per_class,
    # acc_per_class,
    # display_acc_logs,
)
from cifar10_data_prep import CIFAR10_CLASSES

server = ServerApp()

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEV)


def prep_phase(
    strategy: CustomStrat, grid: Grid, arrays: ArrayRecord, use_global_weights
) -> tuple:
    """
    send flag to clients to signify preparation phase of the training.
    receive and compile local classes from each client into a list of global classes.

    :param strategy: Description
    :type strategy: CustomStrat
    :param grid: Description
    :type grid: Grid
    :param arrays: Description
    :type arrays: ArrayRecord
    :return: Description
    :rtype: set
    """
    from wheat_data_utils import compute_class_weights

    def aggregate_data_summaries(data_info: list[dict] = None):
        if data_info:
            data_summary = data_info[0]
            for item in data_info[1:]:
                summary_keys = data_summary.keys()
                for key in item:
                    if key in summary_keys:
                        data_summary.update({key: data_summary[key] + item[key]})
                    else:
                        data_summary.update({key: item[key]})
            data_summary_list = dict(sorted(data_summary.items())).values()
            return list(data_summary_list)
        else:
            return None

    prep_conf = ConfigRecord(
        {"prep-phase": 1, "use-global-weights": use_global_weights}
    )
    prep_replies = strategy.prepare(grid, arrays, prep_config=prep_conf)
    global_classes = set()
    client_classes_list = []
    clients_configs = []
    global_data_info_lists = []
    for item in prep_replies:
        if not item.has_content():
            continue
        client_conf = item.content.get("config")
        client_classes = client_conf.get("local-classes")
        client_data_info = client_conf.get("local-data-info")
        if client_data_info:
            global_data_info_lists.append(client_data_info)
        else:
            global_data_info_lists = None
        clients_configs.append(client_conf)
        global_classes.update(set(client_classes))

        client_classes_list.append(client_classes)

    if global_data_info_lists:
        global_data_info_dict = [
            dict(zip(keys, values))
            for keys, values in zip(client_classes_list, global_data_info_lists)
        ]
        # print(f"\n{global_data_info_dict = }\n")
        data_summary = aggregate_data_summaries(global_data_info_dict)
    else:
        data_summary = None
    if data_summary:
        global_weights = compute_class_weights(data_summary).tolist()
    else:
        global_weights = None

    # print(f"\n{global_weights = }\n")

    return sorted(list(global_classes)), clients_configs, global_weights


def labels_map_per_client(global_classes: list, configs: list[dict]):

    labels_map = {i: c for i, c in enumerate(global_classes)}
    print("\nGlobal labels map: ", labels_map)
    contents = []
    for item in configs:
        node_name = item.get("node-name")
        node_id = item.get("node-id")
        node_classes = item.get("local-classes")
        node_labels_map = {k: v for k, v in labels_map.items() if (v in node_classes)}
        contents.append((int(node_id), {"labels": list(node_labels_map.keys())}))
    print("\nContents: ", contents)
    print(" ")
    return contents


def pick_test_dataloader(dataset_name: Literal["cifar10", "wheat"]):
    from wheat_data_prep import TESTING_DATA, data_loader as wheat_loader
    from cifar10_data_prep import CIFAR10_TEST, loader as cifar_loader

    if dataset_name == "wheat":
        test_dataloader = wheat_loader(TESTING_DATA, DEV, 128)
    elif dataset_name == "cifar10":
        test_dataloader = cifar_loader(CIFAR10_TEST, 128)

    return test_dataloader


@server.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    from CustomClasses import send_to_node, construct_messages_per_node

    model_name = context.run_config["model-name"]
    freeze = context.run_config["freeze"]
    batch_size = context.run_config["batch-size"]
    use_sampler = context.run_config["use-sampler"]
    num_workers = context.run_config["num-workers"]
    features_lr = context.run_config["features-lr"]
    classifier_lr = context.run_config["classifier-lr"]
    weight_decay = context.run_config["weight-decay"]
    sch_patience = context.run_config["sch-patience"]
    use_weights = context.run_config["use-weights"]
    epochs = context.run_config["local-epochs"]
    dataset_name = context.run_config["dataset-name"]
    mixer = context.run_config["mixer"]
    proximal_mu = context.run_config["proximal-mu"]
    use_custom_agg = context.run_config["use-custom-agg"]
    use_global_weights = context.run_config["use-global-weights"]
    use_loss_masking = context.run_config["use-loss-masking"]

    start_time = time.perf_counter()

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    fraction_train = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    momentum = context.run_config["momentum"]

    # Load global model
    # if dataset_name == "wheat":
    #     # from wheat_data_prep import CLASSES as wheat_classes

    #     # out_features = len(wheat_classes)
    #     out_features = 1
    # elif dataset_name == "cifar10":
    #     out_features = len(cifar_classes)

    temp_model = choose_model("cnn", freeze, 1).to(DEVICE)

    ## model_exists = file_exists(output_path)
    # if model_exists:
    #     global_model.load_state_dict(torch.load(output_path, weights_only=True))

    temp_arrays = ArrayRecord(temp_model.state_dict())

    ## Initialize FedAvg strategy
    # strategy = FedAvg(fraction_evaluate=fraction_evaluate)
    # strategy = CustomStrat(
    #     fraction_evaluate=fraction_evaluate, server_momentum=momentum
    # )

    # # for the custom strat based on FedAvg
    # strategy = CustomStrat(fraction_evaluate=fraction_evaluate)

    # for the custom strat based on FedProx
    strategy = CustomStrat(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        proximal_mu=proximal_mu,
    )

    # ====================================================================
    # prepare for training by receiving client arrays
    # ====================================================================

    global_classes, all_metrics, global_weights = prep_phase(
        strategy, grid, temp_arrays, use_global_weights
    )
    labels_maps = labels_map_per_client(global_classes, all_metrics)
    messages_to_clients = construct_messages_per_node(labels_maps)
    labels_msg_replies = send_to_node(grid, messages_to_clients)

    # print replies for sent labels
    for item in labels_msg_replies:
        print(
            f"\n--> {item.content.get("config").get("node-name")} have received assigned labels successfully."
        )
    print("")

    # ====================================================================
    # ====================================================================

    out_features = len(global_classes)
    global_model = choose_model(model_name, freeze, out_features).to(DEVICE)
    arrays = ArrayRecord(global_model.state_dict())

    print("\n### global classes: ", global_classes)
    print("\n### global classes: ", global_weights)

    # compile training configs
    train_configs = {
        "model-name": model_name,
        "freeze": freeze,
        "batch-size": batch_size,
        "use-sampler": use_sampler,
        "num-workers": num_workers,
        "features-lr": features_lr,
        "classifier-lr": classifier_lr,
        "weight-decay": weight_decay,
        "sch-patience": sch_patience,
        "use-weights": use_weights,
        "local-epochs": epochs,
        "dataset-name": dataset_name,
        "mixer": mixer,
        "out-features": out_features,
        "global-weights": global_weights if global_weights else [],
        "use-loss-masking": use_loss_masking,
    }
    # Start strategy, run FedAvg for `num_rounds`
    test_dataloader = pick_test_dataloader(dataset_name)
    train_replies, _, result = strategy.start(
        timeout=1e10,
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(train_configs),
        num_rounds=num_rounds,
        evaluate_fn=GlobalEvaluation(global_model, DEV, test_dataloader),
        use_custom_agg=use_custom_agg,
    )


    train_end = time.perf_counter()
    print(f"\n{'='*50}")
    print(f" Total training time: {readable_time(train_end - start_time)}")
    print(f"{'='*50}\n")

    # Save final model to disk
    state_dict = result.arrays.to_torch_state_dict()

    # evaluate accuracy per class
    global_model.load_state_dict(state_dict)
    global_labels_map = generate_labels_map(global_classes)
    # true_values, pred_values = get_true_and_pred_values(test_dataloader, global_model)
    # eval_results = acc_per_class(
    #     true_values, pred_values, out_features, global_labels_map
    # )
    # display_acc_logs(eval_results)
    global_metrics = eval_per_class(
        test_dataloader, global_model, out_features, global_labels_map
    )

    print(f"\n{global_metrics = }\n")

    # end time messages
    eval_end = time.perf_counter()
    print(f"\n{'='*50}")
    print(f" Total evaluation time: {readable_time(eval_end - train_end)}")
    print(f"{'='*50}")
    print(f" Total experiment time: {readable_time(eval_end - start_time)}")
    print(f"{'='*50}\n")

    the_time = datetime.strftime(datetime.now(), "%d.%m.26-%H:%M:%S")
    # model_path = f"/root/data/models/{dataset_name}_{model_name}_epochs:{epochs}_f-lr:{features_lr}_c-lr:{classifier_lr}_batch-size:{batch_size}_aug:{mixer}_{the_time}.pt"
    # os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # print("\nSaving final model to disk...")
    # torch.save(state_dict, model_path)

    # print("\n\nSaving Clients Metrics Data...\n")
    # data_name = f"{model_name}_epochs:{epochs}_f-lr:{features_lr}_c-lr:{classifier_lr}_batch-size:{batch_size}_aug:{mixer}_{the_time}"
    # raw_data_path = f"/root/data/metrics/{dataset_name}/{data_name}.pkl"
    # csv_data_path = f"/root/data/metrics/{dataset_name}/{data_name}.csv"
    # save_pkl(raw_data_path, train_replies)
    # data_df = parse_raw_metrics(train_replies)
    # data_df.to_csv(csv_data_path, index=False)

    print("\n\nSaving Server Evaluation Metrics Data...\n")
    # raw_eval_data_path = f"/root/data/metrics/{dataset_name}/{data_name}_eval.pkl"
    # csv_eval_data_path = f"/root/data/metrics/{dataset_name}/{data_name}_eval.csv"

    # == ignore these, they're just for testing=================================
    print(result.evaluate_metrics_serverapp)
    eval_data = parse_server_eval_metrics(result.evaluate_metrics_serverapp)
    print(eval_data)

    print(result.train_metrics_clientapp)
    print(f"\n{'-'*10} Training replies {'-'*20}\n")
    print(f"{train_replies = }")

    train_data = parse_raw_metrics(train_replies)
    train_data, _ = split_df_by_type(train_data)
    print(train_data)

    # ============================================================================

    # save_pkl(raw_eval_data_path, result.evaluate_metrics_serverapp)
    # eval_data_df = parse_server_eval_metrics(result.evaluate_metrics_serverapp)
    # eval_data_df.to_csv(csv_eval_data_path, index=False)

    # # save experiment data (configs + final results)
    # client_name = cmd("hostname").strip()
    # json_path = f"/root/data/experiments/{dataset_name}_{the_time}.json"
    # os.makedirs(os.path.dirname(json_path), exist_ok=True)
    # save_arbitrary_json(
    #     path=json_path,
    #     client_name=client_name,
    #     dataset_name=dataset_name,
    #     model_name=model_name,
    #     epochs=epochs,
    #     num_rounds=num_rounds,
    #     freeze=True if freeze else False,
    #     batch_size=batch_size,
    #     use_sampler=True if use_sampler else False,
    #     num_workers=num_workers,
    #     features_lr=features_lr,
    #     classifier_lr=classifier_lr,
    #     weight_decay=weight_decay,
    #     sch_patience=sch_patience,
    #     use_weights=True if use_weights else False,
    #     mixer=mixer,
    #     proximal_mu=proximal_mu,
    #     use_custom_agg=True if use_custom_agg else False,
    #     use_global_weights=True if use_global_weights else False,
    #     use_loss_masking=True if use_loss_masking else False,
    #     agg_metrics={i: dict(m) for i, m in result.evaluate_metrics_serverapp.items()},
    # )
