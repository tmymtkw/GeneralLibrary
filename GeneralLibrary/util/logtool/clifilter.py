FILL = "\033[7m"
CLOSE = "\033[0m"

class CLIFilter():

    def __init__(self, length=40):
        print("CLIFilter init")
        self.length = length

    def filter(self, record):
        if hasattr(record, "status"):
            record.msg = self.GetCLIMsg(record.status)
        return True
    
    def GetCLIMsg(self, status):
        epoch_bar = self.length * status["cur_epoch"] // status["max_epoch"]
        itr_bar = self.length * status["cur_itr"] // status["max_itr"]

        msg = ("\n[epoch]\t|" + FILL + ("_" * epoch_bar) + CLOSE + "_" * (self.length - epoch_bar) + "|\t\n"
               + "[ itr ]\t|" + FILL + ("_" * itr_bar) + CLOSE + "_" * (self.length - itr_bar) + "|\n\n"
               + f"[curr] lr : {status['lr']:<12f}\tloss : {status['loss']:<12f}\n"
               + f"[best]")

        return msg