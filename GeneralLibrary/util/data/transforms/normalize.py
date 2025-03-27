from .basetransform import BaseTransform
from torchvision.transforms.v2.functional import normalize

class Normalize(BaseTransform):
    def __init__(self, max_val=255.0):
        self.max_val = max_val

    def __call__(self, source, target):
        source /= self.max_val
        target /= self.max_val

        return source, target