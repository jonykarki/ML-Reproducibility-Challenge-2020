# @jkarki, run all the tests and save the results in a csv file
import subprocess
import os
import torch

trained_models = os.listdir("trained_model")

dict_ = {
    "CD": "color",
    "GD": "gray",
}

untested_models = []
for i in trained_models:
    # check if trained all 300 epochs
    epochs = torch.load(os.path.join('trained_model', f"{i}", "ckpt"), map_location=torch.device('cpu'))['epoch']
    if epochs >= 300 and not os.path.exists(os.path.join("trained_model", f"{i}", "test_log.txt")):
        untested_models.append(i)

untested_models = sorted(untested_models)
for i in untested_models:
   print(f"Testing {i}")
   file = dict_[i.split("_")[1]]
   noise_level = i.split("_")[3]
   subprocess.run(f"python3 test_{file}.py --mode {i.split('_')[2]} --model_name {os.path.join('trained_model', i, 'ckpt')} --noise_level {noise_level} > {os.path.join('trained_model', i, 'test_log.txt')}", shell=True)
