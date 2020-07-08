import torch
import torch.nn as nn

def resave(input_path, output_path):
    net = torch.load(input_path)
    torch.save(net['model'], output_path)
