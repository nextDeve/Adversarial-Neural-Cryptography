import torch
from model_code.train import train
import os
import json

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open('../config.json') as f:
        config = json.load(f)
    config['device'] = device
    os.makedirs('.'+config['loss_path'], exist_ok=True)
    os.makedirs('.'+config['model_path'], exist_ok=True)
    train(config)
