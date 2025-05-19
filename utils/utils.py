import re
import torch
import random
import numpy as np


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def calc_score(text):
    return round(sum(map(float, re.findall(r'\b\d+\.\d+\b', text))) / len(re.findall(r'\b\d+\.\d+\b', text)), 2)

def print_latex(text):
    print(" & ".join(map(str, re.findall(r'\b\d+\.\d+\b', text))))



if __name__ == "__main__":
    pass

