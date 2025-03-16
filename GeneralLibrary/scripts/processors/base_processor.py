from abc import ABC, abstractmethod
from torch.nn import Module
from torch.utils.data import Dataset
from util.logtool.mainlogger import MainLogger


class BaseProcessor(ABC):
    model: Module = None
    logger: MainLogger = None
    metrics: dict = None
    datasets: list[Dataset] = None # [train valid test]
    device: str = None

    def __init__(self):
        pass
        
    @abstractmethod
    def process(self, device, **kwargs):
        NotImplementedError()