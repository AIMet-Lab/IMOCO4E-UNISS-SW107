
import argparse
import os
import logging

import onnx
import pynever.nodes as pyn_nodes
import pynever.networks as pyn_networks
import pynever.strategies.conversion as pyn_conv
import pynever.strategies.training as pyn_train
import torch.utils.data as pyt_data
import numpy as np

import utilities
import datasets


def make_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser("Training Script for SW-108.")

    data_path_help = "Path to the folder containing the data. The raw data should be in a subfolder " \
                     "named training/raw/."
    parser.add_argument("--data_path", type=str, help=data_path_help, default="data/")

    outputs_path_help = "Path to the folder which will contain the results of the training process."
    parser.add_argument("--outputs_path", type=str, help=outputs_path_help, default="outputs/")

    return parser


if __name__ == "__main__":

    # PARSE COMMAND LINE PARAMETERS.
    arg_parser = make_parser()
    args = arg_parser.parse_args()

    data_path = args.data_path
    outputs_path = args.outputs_path

    # CHECK FOLDERS.
    models_folder = outputs_path + "models/"
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)

    checkpoints_folder = outputs_path + "training_checkpoints/"
    if not os.path.exists(checkpoints_folder):
        os.mkdir(checkpoints_folder)

    logs_folder = outputs_path + "logs/"
    if not os.path.exists(logs_folder):
        os.mkdir(logs_folder)
    logs_file_path = logs_folder + "training_logs.csv"

    # LOGGERS INSTANTIATION
    logs_file_exist = os.path.exists(logs_file_path)
    stream_logger = logging.getLogger("pynever.strategies.training")
    file_logger = logging.getLogger("models_generation_file")

    file_handler = logging.FileHandler(logs_file_path)
    stream_handler = logging.StreamHandler()

    file_handler.setLevel(logging.INFO)
    stream_handler.setLevel(logging.INFO)

    file_logger.addHandler(file_handler)
    stream_logger.addHandler(stream_handler)

    file_logger.setLevel(logging.INFO)
    stream_logger.setLevel(logging.INFO)

    # DATA PREPROCESSING
    utilities.data_preprocessing(data_path)

    # PARAMETERS EXTRACTION
    (dataset_path, dataset_id, activation_functions, network_arch, test_percentage, optimizer_con, opt_params,
     loss_function, n_epochs, validation_percentage, train_batch_size, validation_batch_size, precision_metric, device,
     metric_params, test_batch_size) = utilities.training_config()

    # DATASET INSTANTIATION
    dataset = datasets.ComponentDegradationAD(dataset_path)
    test_len = int(np.floor(dataset.__len__() * test_percentage))
    train_len = dataset.__len__() - test_len
    training_dataset, test_dataset = pyt_data.random_split(dataset, (train_len, test_len))
    input_size = (dataset.__getitem__(0)[0].shape[0],)
    output_size = dataset.__getitem__(0)[1].shape[0]

    # IF LOG FILE DID NOT EXIST INSERT COLUMNS INFO
    if not logs_file_exist:
        file_logger.info("net_id,"
                         "optim,"
                         "lr,"
                         "loss_f,"
                         "n_epochs,"
                         "val_percentage,"
                         "train_b_size,"
                         "val_b_size,"
                         "precision_metric,"
                         "device,"
                         "test_b_size,"
                         "train_len,"
                         "test_len,"
                         "test_loss")

    for act_fun in activation_functions:

        for net_arch in network_arch:

            net_id = f"{dataset_id}_{act_fun.__name__}_{net_arch}".replace(", ", "-")
            network = pyn_networks.SequentialNetwork(identifier=net_id, input_id="X")

            node_index = 0
            in_dim = input_size
            for n_neurons in net_arch:
                fc_node = pyn_nodes.FullyConnectedNode(identifier=f"FC_{node_index}", in_dim=in_dim,
                                                       out_features=n_neurons)
                network.add_node(fc_node)
                node_index += 1

                act_node = act_fun(identifier=f"ACT_{node_index}", in_dim=fc_node.out_dim)
                network.add_node(act_node)
                in_dim = act_node.out_dim
                node_index += 1

            fc_out_node = pyn_nodes.FullyConnectedNode(identifier=f"FC_{node_index}", in_dim=in_dim,
                                                       out_features=output_size)
            network.add_node(fc_out_node)

            # == MODEL TRAINING == #

            stream_logger.info(f"NOW TRAINING MODEL: {network}")
            training_strategy = pyn_train.PytorchTraining(optimizer_con=optimizer_con, opt_params=opt_params,
                                                          loss_function=loss_function, n_epochs=n_epochs,
                                                          validation_percentage=validation_percentage,
                                                          train_batch_size=train_batch_size,
                                                          validation_batch_size=validation_batch_size,
                                                          precision_metric=precision_metric, device=device,
                                                          checkpoints_root=checkpoints_folder)

            trained_network = training_strategy.train(network, training_dataset)

            # == MODEL TESTING == #

            testing_strategy = pyn_train.PytorchTesting(metric=precision_metric, metric_params=metric_params,
                                                        test_batch_size=test_batch_size, device=device)

            test_loss = testing_strategy.test(trained_network, test_dataset)
            stream_logger.info(f"TEST LOSS: {test_loss}")

            # == MODEL SAVING == #
            onnx_net = pyn_conv.ONNXConverter().from_neural_network(trained_network).onnx_network
            onnx.save(onnx_net, models_folder + f"{trained_network.identifier}.onnx")

            file_logger.info(f"{net_id},{optimizer_con.__name__},{opt_params['lr']},{loss_function.__class__.__name__},"
                             f"{n_epochs},{validation_percentage},{train_batch_size},{validation_batch_size},"
                             f"{precision_metric.__class__.__name__},{device},{test_batch_size},{train_len},{test_len},"
                             f"{test_loss}")
