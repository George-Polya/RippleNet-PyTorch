import argparse
from os import device_encoding
import numpy as np
from data_loader import load_data
from train import train
import torch

from parser import parse_args

if __name__ == "__main__":
    global args, device
    np.random.seed(2022)
    args = parse_args()
    device = torch.device("cuda" if args.use_cuda else torch.device("cpu"))
    show_loss = False
    data_info = load_data(args)
    train(args, data_info, show_loss)
