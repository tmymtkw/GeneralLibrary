from scripts.trainer import Trainer

class Validator(Trainer):
    def __init__(self):
        super().__init__()

    def Validate(self):
        self.Debug("function called: Validation")