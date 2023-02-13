import argparse
import json
import os
from glob import glob

from firmware import Firmware
from model import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="res/driving_stereo", help="dir to save the results")
    parser.add_argument("--dataset_dir", type=str, default="dataset/DrivingStereo", help="dir to find datset")

    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    json_file = "config.json"
    with open(json_file, "r") as fn:
        json_data = json.load(fn)

        left_fns = glob(os.path.join(args.dataset_dir, "test-left-image", "*.jpg"))

        count = 0
        for left_fn in left_fns:
            right_fn = left_fn.replace("test-left-image", "test-right-image")

            json_data['data']['left_fn'] = left_fn
            json_data['data']['right_fn'] = right_fn

            _dir = left_fn.split("/")[-1].replace(".jpg", "")
            os.makedirs(os.path.join(args.work_dir, _dir), exist_ok=True)
            json_data['data']['out_fn'] = os.path.join(args.work_dir, _dir, "res.png")

            with open(os.path.join(args.work_dir, _dir, "res.json"), 'w') as fp:
                json.dump(json_data, fp, indent=4)

            firware_ins = Firmware(json_data['model']) 

            param = firware_ins.render()

            model_ins = Model(param, json_data['data']) 
            model_ins.process()

            if count > 3:
                break

            count += 1
