import argparse
import json

import taichi as ti
import torch

from firmware import Firmware
from model import Model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_fn", type=str, default="config.json", help="config json")

    args = parser.parse_args()

    json_file = args.config_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch device: ", device)

    if torch.cuda.is_available():
        ti.init(arch=ti.cuda, device_memory_fraction=0.6)
    else:
        ti.init(arch=ti.cpu)

    with open(json_file, "r") as fn:
        json_data = json.load(fn)
        print("load json fn: ", json_file)

        firware_ins = Firmware(json_data['model']) 

        param = firware_ins.render()

        model_ins = Model(param, json_data['data'], device=device) 
        model_ins.process()
