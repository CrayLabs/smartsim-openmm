#import torch

def intermediate(tensor):
    A = torch.norm(tensor)
    return A

def test_function(tensor):
    return intermediate(tensor*2.0)*tensor
