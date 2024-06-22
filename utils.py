import torch
from transformers import AutoModelForCausalLM
from accelerate import dispatch_model
import logging
import os
import pandas as pd
def _device_map(num_gpus, num_layers):
    per_gpu_layers = (num_layers + 2) / num_gpus

    device_map = {
        'transformer.wte': 0,
        'transformer.ln_f': 0,
        'lm_head': num_gpus-1
    }

    used = 1
    gpu_target = 0
    for i in range(num_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0 if gpu_target < num_gpus-1 else 1
        assert gpu_target < num_gpus
        device_map[f'transformer.h.{i}'] = gpu_target
        used += 1

    return device_map


def load_model_on_gpus(model_name_or_path, num_gpus: int = 2):
    num_devices = torch.cuda.device_count()

    if num_gpus == 1:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto',
                                                     trust_remote_code=True).eval()
    elif 1 < num_gpus <= num_devices:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='cpu',
                                                     trust_remote_code=True).eval()
        num_layers = model.config.num_hidden_layers
        device_map = _device_map(num_gpus, num_layers)
        print(device_map)
        model = dispatch_model(model, device_map=device_map)
    else:
        raise KeyError

    return model

def get_logger(log_file_path):
    logger = logging.getLogger("my logger")
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    log_file_path = log_file_path + ".log"  # Specify the path to your log file.
    file_handler = logging.FileHandler(log_file_path)

    logger.addHandler(file_handler)

    logger.addHandler(handler)

    return logger

def get_target_data(dataset, mode=None):
    image_list = []
    if dataset == "advbench":
        csv_file = os.path.join("dataset", "advbench", "harmful_behaviors.csv")
        data = pd.read_csv(csv_file)["goal"].to_list()

    elif dataset == "safebench_tiny":
        csv_file = os.path.join("dataset", "safebench", "question", "SafeBench-Tiny.csv")
        images = os.path.join("dataset", "safebench", "question", "")
        data = pd.read_csv(csv_file)["question"].to_list()

    elif dataset == "mm-safebench":
        csv_file = os.path.join("dataset", "mm-safebench", "samples.csv")
        data = pd.read_csv(csv_file)["content"].to_list()

    elif dataset == "xstest":
        csv_file = os.path.join("dataset", "xstest", "xstest_v2_prompts.csv")
        data = pd.read_csv(csv_file)["prompt"].to_list()
        # data = []
        # for idx, row in pd.read_csv(csv_file).iterrows():
        #     if str(row["type"]) == "safe_contexts":
        #         data.append(str(row["prompt"]))

    elif "safebench" in dataset:
        csv_file = os.path.join("dataset", "safebench", "question", dataset+".csv")
        data = pd.read_csv(csv_file)["question"].to_list()



    if mode == "mm":
        return data, image_list
    else:
        return data
