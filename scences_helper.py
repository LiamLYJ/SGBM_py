import argparse
import json
import os

from firmware import Firmware
from model import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="res/scences", help="dir to save the results")
    parser.add_argument("--dataset_dir", type=str, default="dataset/scences2001", help="dir to find datset")

    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    json_file = "config.json"
    with open(json_file, "r") as fn:
        json_data = json.load(fn)

        for root, dirs, files in os.walk(args.dataset_dir):
            for _dir in dirs:
                left_fn = os.path.join(root, _dir, "im2.png")
                right_fn = os.path.join(root, _dir, "im6.png")

                json_data['data']['left_fn'] = left_fn
                json_data['data']['right_fn'] = right_fn
                json_data['data']['out_fn'] = os.path.join(args.work_dir, _dir, "res.png")
                os.makedirs(os.path.join(args.work_dir, _dir), exist_ok=True)

                with open(os.path.join(args.work_dir, _dir, "res.json"), 'w') as fp:
                    json.dump(json_data, fp, indent=4)

                if 0:
                    firware_ins = Firmware(json_data['model']) 

                    param = firware_ins.render()

                    model_ins = Model(param, json_data['data']) 
                    model_ins.process()

# import cv2
# from glob import glob

# fns = glob("dataset/scences2001/*/*.ppm")
# h_scence = 380
# w_scence = 430
# for fn in fns:
#     if "im2" in fn or "im6" in fn:
#         img = cv2.imread(fn)[:h_scence, :w_scence, :]
#         cv2.imwrite(fn.replace(".ppm", ".png"), img)
