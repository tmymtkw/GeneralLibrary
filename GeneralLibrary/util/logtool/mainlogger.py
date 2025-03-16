from logging import Logger
import os
from datetime import date
from logging import StreamHandler, FileHandler, Formatter, DEBUG, INFO
from .logfilter import LogFilter
from .clifilter import CLIFilter

class MainLogger(Logger):
    # 日時フォーマット
    datefmt = "%m-%d %H:%M:%S"
    # ファイル出力のフォーマット
    filefmt = "[%(levelname)-6s | %(asctime)s] %(message)s"
    # コンソール出力のフォーマット
    clifmt = "\033[%(n)dA[%(asctime)s] %(message)s\n"
    # 出力ファイル名
    file_name = "output.log"

    def __init__(self, dir, name="main_logger", level = 0):
        super().__init__(name, level)

        # dirが存在しなければ作成
        if not os.path.exists(dir):
            os.mkdir(dir)

        # ログファイルを作成
        abs_file = os.path.join(dir, self.file_name)
        with open(os.path.join(abs_file), mode="w") as f:
            f.write(f"[Running Date: {date.today().strftime('%Y-%m-%d')}]\n\n")
        
        # ファイル出力の設定
        file_formatter = self.get_formatter(self.filefmt)
        file_handler = FileHandler(os.path.join(dir, self.file_name), mode="a+", encoding="utf_8")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(DEBUG)
        file_filter = LogFilter()
        file_handler.addFilter(file_filter)

        # コンソール出力の設定
        cli_formatter = self.get_formatter(self.clifmt)
        cli_handler = StreamHandler()
        cli_handler.terminator = ""
        cli_handler.setFormatter(cli_formatter)
        cli_handler.setLevel(INFO)
        cli_filter = CLIFilter()
        cli_handler.addFilter(cli_filter)

        self.addHandler(cli_handler)
        self.addHandler(file_handler)
        self.setLevel(DEBUG)

    def displayStatus(self, cur_epoch, cur_itr, max_epoch, max_itr, lr=0.0, loss=0.0):
        """学習状況の標準出力

        同じフォーマットで描画を更新する
        """

        self.info(msg="", 
                  extra={"status": {"cur_epoch": cur_epoch,
                                    "cur_itr": cur_itr,
                                    "max_epoch": max_epoch,
                                    "max_itr": max_itr,
                                    "lr": lr,
                                    "loss": loss},
                         "n": 6})
        
    def info(self, msg, extra = {"n": 0}):
        super().info(msg, extra=extra)

    def setLogDigits(self, epoch, iteration):
        if not self.hasHandlers():
            self.debug("[NOTICE] SetLogDigits failed. main_logger has no handlers.")
            return
        assert isinstance(self.handlers[1].filters[0], LogFilter), f"[ERROR] bad filter type: {type(self.handlers[1].filters[0])}"
        self.handlers[1].filters[0].SetDigits(epoch, iteration)

    def get_formatter(self, kind):
        return Formatter(kind, self.datefmt)