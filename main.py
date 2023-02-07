import argparse
import json

from firmware import Firmware
from model import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_fn", type=str, default="config.json", help="config json")

    args = parser.parse_args()

    json_file = args.config_fn

    with open(json_file, "r") as fn:
        json_data = json.load(fn)

        firware_ins = Firmware(json_data['model']) 

        param = firware_ins.render()

        model_ins = Model(param, json_data['data']) 
        model_ins.process()