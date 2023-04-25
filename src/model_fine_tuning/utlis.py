import torch
import gc


def report_gpu():
    torch.cuda.empty_cache()
    gc.collect()
