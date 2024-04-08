import os
import sys
import logging
import datetime

import torch
import onnx
import pandas
import numpy as np
import pynever.strategies.conversion as pyn_conv
import pynever.networks as pyn_networks
import sklearn.preprocessing as skl_prep
import pynever.nodes as pyn_nodes
import torch.optim as pyt_optim
import torch.nn as nn


def logger_instantiation(logger_origin: str, logs_filepath: str = None):

    stream_logger = logging.getLogger(logger_origin + "_stream")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_logger.addHandler(stream_handler)
    stream_logger.setLevel(logging.INFO)

    if logs_filepath is not None:
        file_logger = logging.getLogger(logger_origin + "_file")
        file_handler = logging.FileHandler(logs_filepath)
        file_handler.setLevel(logging.INFO)
        file_logger.addHandler(file_handler)
        file_logger.setLevel(logging.INFO)

    else:
        file_logger = None

    return stream_logger, file_logger


def get_model(model_path: str) -> (torch.nn.Module, str):

    onnx_net = pyn_conv.ONNXNetwork(model_path, onnx.load(model_path))
    pyn_net = pyn_conv.ONNXConverter().to_neural_network(onnx_net)
    pyt_net = pyn_conv.PyTorchConverter().from_neural_network(pyn_net).pytorch_network

    if '/' in model_path:
        model_id = model_path.split('/')[-1].replace('.onnx', '')
    else:
        model_id = model_path.replace('.onnx', '')

    return pyt_net, model_id


def data_cleaning(data_dict: dict, norm_info: pandas.DataFrame, norm_max: float, norm_min: float) -> torch.Tensor:

    clean_data = []
    for key, value in data_dict.items():
        if key != 'timestamp':
            lb = norm_info[key][0]
            ub = norm_info[key][1]
            value_std = (value - lb) / (ub - lb)
            value_norm = value_std * (norm_max - norm_min) + norm_min
            clean_data.append(value_norm)

    return torch.from_numpy(np.array(clean_data))


def compute_vre(model: torch.nn.Module, data: torch.Tensor) -> float:
    model.float()
    data = data.float()
    pred = model(data)
    losses = torch.nn.MSELoss(reduction='none')(pred, data)
    vre = losses.mean()
    return vre.item()


def parse_csv_name(name: str):
    year = 2018
    month = (9 + int(name[0:2]) - 1) % 12 + 1  # needed since the first month is actually october
    if 1 <= month < 10:
        year = 2019
    day = int(name[3:5])
    hour = int(name[6:8])
    minute = int(name[8:10])
    second = int(name[10:12])
    mode = int(name[21])
    starting_datetime = datetime.datetime(year, month, day, hour, minute, second)
    pandas_s_datetime = pandas.to_datetime(starting_datetime)
    return mode, pandas_s_datetime


def get_year_dataframe(csvs_path: str):
    csv_ids = sorted(os.listdir(csvs_path))
    csv_ids.remove("placeholder.txt")
    dataframe_list = []
    for csv_id in csv_ids:
        temp_dataframe = pandas.read_csv(csvs_path + csv_id)
        mode, start_datetime = parse_csv_name(csv_id)
        mode_list = np.array([mode for _ in range(temp_dataframe.__len__())])
        for i in range(temp_dataframe.__len__()):
            new_timestamp = start_datetime + pandas.to_timedelta(temp_dataframe['timestamp'][i], "seconds")
            temp_dataframe.loc[i, 'timestamp'] = new_timestamp

        temp_dataframe.insert(1, 'mode', mode_list)

        dataframe_list.append(temp_dataframe)

    year_dataframe = pandas.concat(dataframe_list, ignore_index=True)
    return year_dataframe


def normalize_data(df: pandas.DataFrame, columns: list):
    scalers = {}
    for c in columns:
        scalers[c] = skl_prep.MinMaxScaler((-1, 1)).fit(df[c].values.reshape(-1, 1))

    norm_df = df.copy()
    for c in columns:
        norm = scalers[c].transform(norm_df[c].values.reshape(-1, 1))
        norm_df[c] = norm

    return norm_df


def compute_loss(network: pyn_networks.SequentialNetwork, loss_f, sample, target):

    pyt_net = pyn_conv.PyTorchConverter().from_neural_network(network).pytorch_network
    pyt_net.to("mps")
    with torch.no_grad():
        pyt_net.float()
        sample = sample.float()
        target = target.float()
        sample, target = sample.to("mps"), target.to("mps")
        outputs = pyt_net(sample)
        loss = loss_f(outputs, target)
    return loss


def data_preprocessing(data_path: str):

    csvs_path = data_path + "raw/"
    year_csv_path = data_path + "year_data.csv"
    norm_year_path = data_path + "norm_year_data.csv"

    if not os.path.exists(year_csv_path):
        year_dataframe = get_year_dataframe(csvs_path)
        year_dataframe.to_csv(year_csv_path, sep=",", index=False)
    else:
        year_dataframe = pandas.read_csv(year_csv_path)

    norm_columns = year_dataframe.columns[2:].tolist()
    if not os.path.exists(norm_year_path):
        norm_year_df = normalize_data(year_dataframe, norm_columns)
        norm_year_df.to_csv(norm_year_path, sep=",", index=False)
    else:
        norm_year_df = pandas.read_csv(norm_year_path)


def training_config():

    dataset_path = "data/norm_year_data.csv"
    dataset_id = "CDAD"

    activation_functions = [pyn_nodes.ReLUNode]
    network_arch = [[32, 8, 32], [50, 10, 50], [64, 16, 64], [128, 32, 128]]
    test_percentage = 0.2

    optimizer_con = pyt_optim.Adam
    opt_params = {"lr": 0.001}
    loss_function = nn.MSELoss()
    n_epochs = 50
    validation_percentage = 0.3
    train_batch_size = 512
    validation_batch_size = 128
    precision_metric = nn.MSELoss()
    device = "mps"
    metric_params = {}
    test_batch_size = 128

    return (dataset_path, dataset_id, activation_functions, network_arch, test_percentage, optimizer_con, opt_params,
            loss_function, n_epochs, validation_percentage, train_batch_size, validation_batch_size, precision_metric,
            device, metric_params, test_batch_size)