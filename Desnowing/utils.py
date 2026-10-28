import time


class Adder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.num = 0.0

    def __call__(self, num):
        self.count += 1
        self.num += float(num)

    def average(self):
        return self.num / self.count if self.count else 0.0


class Timer:
    def __init__(self, option='s'):
        self.option = option
        self.divider = {'s': 1, 'm': 60}.get(option, 3600)
        self.tm = 0.0

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.divider


def check_lr(optimizer):
    return optimizer.param_groups[-1]['lr']
