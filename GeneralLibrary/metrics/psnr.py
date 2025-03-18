from torch import mean, log10

class PSNR(object):
    def __init__(self, max_val=1.0):
        self.numerator = max_val * max_val
        pass
        
    def __call__(self, o, t):
        mse = mean((o - t) ** 2) + 1.0e-6

        return 10 * log10(self.numerator / mse)
    
    def __repr__(self):
        return "PSNR"