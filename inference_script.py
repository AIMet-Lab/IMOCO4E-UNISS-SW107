
import argparse
import configparser
import os.path
import time
import datetime
import json

import pandas
import kafka

import utilities


def make_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser("Script for SW-108.")
    parser.add_argument("--bs", type=str, default="localhost:9092",
                        help="Bootstrap Servers for Kafka Consumer and Producer.")
    parser.add_argument("--ct", type=str, default="sensor-blade", help="Topic of the Kafka Consumer.")
    parser.add_argument("--pt", type=str, default="rnn-results", help="Topic of the Kafka Producer.")
    parser.add_argument("--config_path", type=str, default="configs/default_config.ini", help="Path to config file.")
    parser.add_argument("--log_perf", type=str, default=None, help="File for logging script performances.")

    return parser


if __name__ == "__main__":

    # EXTRACT COMMAND LINE ARGUMENTS
    arg_parser = make_parser()
    args = arg_parser.parse_args()

    bootstrap_servers = args.bs
    consumer_topic = args.ct
    producer_topic = args.pt
    config_path = args.config_path
    logging_file = args.log_perf

    # IF LOGGING PERFORMANCE INITIALIZE LOGGING FILE, ELSE GET STREAM LOGGER.
    if logging_file is not None:
        logging_file_exist = os.path.exists(logging_file)
    else:
        logging_file_exist = False

    stream_logger, file_logger = utilities.logger_instantiation("SW-107", logging_file)
    if not logging_file_exist and file_logger is not None:
        file_logger.info("msg_elab_time,msg_arr_freq")

    # EXTRACT CONFIG PARAMS
    config = configparser.ConfigParser()
    _ = config.read(config_path)

    ver_heuristic = config["DEFAULT"]["ver_heuristic"]
    verbose = config["DEFAULT"].getboolean("verbose")

    model_path = config["DEFAULT"]["model_path"]
    norm_info_path = config["DEFAULT"]["norm_info_path"]
    norm_max = config["DEFAULT"].getfloat("norm_max")
    norm_min = config["DEFAULT"].getfloat("norm_min")
    threshold = config["DEFAULT"].getfloat("threshold")

    norm_info_df = pandas.read_csv(norm_info_path)

    # LOAD MODEL
    model, model_id = utilities.get_model(model_path=model_path)

    # INSTANTIATE KAFKA PRODUCER AND CONSUMER
    consumer = kafka.KafkaConsumer(consumer_topic, bootstrap_servers=bootstrap_servers)
    producer = kafka.KafkaProducer(bootstrap_servers=bootstrap_servers)

    # If we are logging the performances we need to track the time of arrival of messages.
    if file_logger is not None:
        last_msg_arr = time.perf_counter()

    # START MESSAGE CYCLE STOPPABLE FROM KEYBOARD
    try:

        for msg in consumer:

            if file_logger is not None:
                msg_arr = time.perf_counter()
                msg_freq = msg_arr - last_msg_arr
                last_msg_arr = msg_arr
                start_msg_elab = time.perf_counter()

            # GET RAW MESSAGE DATA AND DECODE IN JSON. SAVE INFO TO BE PROVIDED IN RESPONSE.
            msg_id = "ADE_" + datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
            msg_type = "Anomaly Detection Event"
            device_id = "-"
            start_time = datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
            json_string = msg.value.decode('utf-8').replace("\r", "").replace("\n", "").replace("'", '"')[:-1]
            data_dict = json.loads(json_string)

            # CLEAN AND NORMALIZE DATA.
            input_data = utilities.data_cleaning(data_dict, norm_info_df, norm_max, norm_min)

            # COMPUTE VECTOR RECONSTRUCTION ERROR
            vre = utilities.compute_vre(model, input_data)

            # DETERMINE ANOMALY
            if vre > threshold:
                is_anomalous = True
            else:
                is_anomalous = False

            end_time = datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

            # PREPARE RESPONSE MESSAGE DATA
            out_dict = {
                "id": msg_id,
                "type": msg_type,
                "anomalyDetected": is_anomalous,
                "startTime": start_time,
                "endTime": end_time,
                "deviceId": device_id,
                "MLModelId": model_id,
                "threshold": threshold,
                "vre": vre
            }
            out_dict.update(data_dict)
            out_message = json.dumps(out_dict).encode('utf-8')

            # SEND RESPONSE WITH KAFKA PRODUCER
            producer.send(topic=producer_topic, value=out_message)

            # IF LOGGING PERFORMANCES COMPUTE MSG ELAB TIME AND SAVE IN LOG FILE
            if file_logger is not None:
                end_msg_elab = time.perf_counter()
                file_logger.info(f"{end_msg_elab - start_msg_elab},{msg_freq}")

    except KeyboardInterrupt:
        pass

    finally:
        consumer.close()
        producer.close()


