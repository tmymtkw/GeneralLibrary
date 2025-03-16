from .base_processor import BaseProcessor

class Tester(BaseProcessor):
    def __init__(self):
        super().__init__()

    def process(self, device, **kwargs):
        self.logger.debug("-----test------")
        