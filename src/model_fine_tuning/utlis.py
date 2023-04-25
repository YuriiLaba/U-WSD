import torch
import gc


def report_gpu():
    torch.cuda.empty_cache()
    gc.collect()


class AverageMeter:
    def __init__(self, name):
        self.val = None
        self.avg = None
        self.sum_ = None
        self.count = None
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum_ = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum_ += val * n
        self.count += n
        self.avg = self.sum_ / self.count

    def __call__(self):
        return self.val, self.avg
