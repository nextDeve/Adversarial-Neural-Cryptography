import torch
from Fine_tune.train import train
import os
import json
import numpy as np

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open('../config.json') as f:
        config = json.load(f)
    config['device'] = device
    os.makedirs('./model_dict' , exist_ok=True)
    os.makedirs('./loss_data', exist_ok=True)
    k = np.array([-1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1])
    k = k.astype(np.float32)
    k = torch.from_numpy(k)
    train(config, k)
