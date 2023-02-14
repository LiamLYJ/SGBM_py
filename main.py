import argparse
import json

from firmware import Firmware
from model import Model
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_fn", type=str, default="config.json", help="config json")

    args = parser.parse_args()

    json_file = args.config_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    with open(json_file, "r") as fn:
        json_data = json.load(fn)
        print("load json fn: ", json_file)

        firware_ins = Firmware(json_data['model']) 

        param = firware_ins.render()

        model_ins = Model(param, json_data['data'], device=device) 
        model_ins.process()