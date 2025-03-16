from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    def __init__(self):
        pass

    def __call__(self, *args, **kwds):
        super().__call__(*args, **kwds)
        self.process(kwds)
        
    @abstractmethod
    def process(self, kwds):
        NotImplementedError()