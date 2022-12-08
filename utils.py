import argparse
import logging
import os
from datetime import datetime
import torch
import wandb


def get_args():
    parser = argparse.ArgumentParser(description='wapirl augmentation optimizer')
    parser.add_argument('--project_name', type=str, default='first')
    parser.add_argument('--model_type', type=str, default='basic')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--input_size_xy', type=int, default=96)
    parser.add_argument('--num_cpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--decouple_input',  action='store_true')
    parser.add_argument('--best_ckpt', type=str, default='best.ckpt')
    return parser.parse_args()


def pre_requisite(args):
    num_gpu = args.num_cpu
    torch.cuda.device(num_gpu)

    # set timestamp
    t = datetime.now()
    args.now = f"{t.year}_{t.month}_{t.day}_{t.hour}_{t.minute}_{t.second}"

    # create directories
    args.path_logs = f"output/logs/{args.project_name}/{args.now}"
    args.path_ckpt = f"output/checkpoints/{args.project_name}/{args.now}"
    os.makedirs(args.path_ckpt, exist_ok=True)

    # set logger
    args.logger = get_logger(args, 'main')

    # init wandb
    run = wandb.init(project=args.project_name, config=args)
    run.name = args.now

    # confirm whether or not it's ready
    args.logger.info(f"Cuda Enabled : {torch.cuda.is_available()}")
    args.logger.info(f"GPU Info : num_gpu:{num_gpu}, name : {torch.cuda.get_device_name(num_gpu)}")
    assert torch.cuda.is_available()
    return run


def print_metric(epoch,  args, train_history, valid_history):
    print(
        "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training MultiAUPRC {:.2f} % AVG Test MultiAUPRC {:.2f} %".format(
            epoch + 1,
            args.epochs,
            train_history['loss'],
            valid_history['loss'],
            train_history['MultiAUPRC'],
            valid_history['MultiAUPRC']))


def get_logger(args, module_name):
    os.makedirs(args.path_logs, exist_ok=True)
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    fn_info_log = f"{module_name}_{args.now}.log"
    fn_err_Log = f"{module_name}_{args.now}.err"
    path_info_log = os.path.join(args.path_logs, fn_info_log)
    path_err_log = os.path.join(args.path_logs, fn_err_Log)
    formatter = logging.Formatter('%(asctime)s:%(module)s:%(levelname)s:%(message)s', '%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_info_handler = logging.FileHandler(path_info_log)
    file_info_handler.setLevel(logging.INFO)  # info
    file_info_handler.setFormatter(formatter)
    logger.addHandler(file_info_handler)

    file_error_handler = logging.FileHandler(path_err_log)
    file_info_handler.setLevel(logging.ERROR)  # error
    file_error_handler.setFormatter(formatter)
    logger.addHandler(file_error_handler)
    return logger


def make_description(history: dict, prefix: str = ''):
    new_history = {}
    for metric_name, metric_values in history.items():
        new_history[prefix + "_" + metric_name] = metric_values
    return new_history


if __name__ == '__main__':
    args = get_args()
    print(args)
    pre_requisite(args)




