from .base_runner import BaseRunner
import os, json
from datetime import datetime
from util.logtool.mainlogger import MainLogger
from util.parser.mainparser import MainParser

from .processors import BaseProcessor, Trainer, Tester
from model import Net
from metrics import PSNR, SSIM
from util.data.dataset import ImageToImageDataset
from torch.optim import Adam
from scheduler.linear_cos import LinearCosineScheduler

class Runner(BaseRunner):
    def __init__(self):
        super().__init__()

    def run(self):
        self.setup()

        self.build()

        mode = self.get_cfg_val("option", "mode")

        assert isinstance(mode, str), f"\n[ERROR] bad mode type: {type(mode)}"
        # 小文字に統一
        mode = mode.lower()
        assert (mode == "train" or mode == "test" or mode == "analyze"), f"\n[ERROR] bad mode: {mode}"

        # if mode == "train":
        #     self.trainer()
        #     self.tester()
        # elif mode == "test":
        #     self.tester(self.model, self.metrics)
        # elif mode == "analyze":
        #     self.analyzer(self.model)

    def setup(self):
        # 1. コマンドライン引数受け取り
        parser = MainParser()
        args = parser.parse_args()

        # 2. コンフィグ設定
        self.load_cfg(args.cfg_path)

        # 3. コンフィグ書き換え
        self.cfg["option"]["mode"] = args.mode

        # 4. ログの設定
        logger = MainLogger(save_dir=self.get_path("output"))
        # コンフィグ状況を出力
        logger.debug(self.show_cfg())
        BaseProcessor.logger = logger
        logger.info("runner setup completed")
        self.down_cursor(1)

    def build(self):
        BaseProcessor.logger.info("building...")
        self.down_cursor(1)
        # BaseProcessorのクラス変数を定義(全てのインスタンスで共有)
        model_class = globals()[self.get_cfg_val("model", "name")]
        BaseProcessor.device = self.get_cfg_val("option", "device")
        BaseProcessor.model = model_class().to(BaseProcessor.device)
        BaseProcessor.datasets = self.build_dataset()
        # 評価指標を定義
        BaseProcessor.metrics = {}
        for m in self.get_hparam("metrics"):
            metrics_class = globals()[m]
            BaseProcessor.metrics[m] = metrics_class()
            BaseProcessor.logger.debug(f"metrics added: {m} {type(BaseProcessor.metrics[m])}")


        # trainer
        self.trainer = Trainer()
        # インスタンス変数(trainer)
        self.trainer.build_dataloader(batch_size=self.get_hparam("batch_size"),
                                      shuffle=True,
                                      num_workers=self.get_cfg_val("option", "num_workers"),
                                      pin_memory=True,
                                      drop_last=True)
        # TODO
        self.trainer.optimizer = Adam(params=self.trainer.model.parameters(),
                                      lr=self.get_hparam("lr"))
        self.trainer.scheduler = LinearCosineScheduler(optimizer=self.trainer.optimizer)
        # TODO: loss handler
        self.trainer.build_loss_handler(losses=self.get_hparam("loss"))
        self.trainer.epochs = self.get_hparam("epoch")
        self.trainer.iteration = len(self.trainer.datasets[0]) // self.get_hparam("batch_size")
        self.trainer.val_interval = self.get_cfg_val("option", "val_interval")
        self.trainer.log_interval = self.get_cfg_val("option", "log_interval")
        self.trainer.save_interval = self.get_cfg_val("option", "save_interval")
        self.trainer.itr_digit = len(str(self.trainer.iteration))

        BaseProcessor.logger.setLogDigits(epoch=self.get_hparam("epoch"), iteration=self.trainer.iteration)

        # tester
        self.tester = Tester(save_dir=self.get_path("output"))
        self.trainer.logger.debug(f"\n{self.trainer.model}")
        self.trainer.logger.debug("*** check global objects ***")
        self.trainer.logger.debug(f"model: {id(self.trainer.model)} {id(self.tester.model)}")
        self.trainer.logger.debug(f"logger: {id(self.trainer.logger)} {id(self.tester.logger)}")
        self.trainer.logger.info("build completed")
        self.down_cursor(1)

    def build_dataset(self):
        train_dataset = ImageToImageDataset(img_dir=self.get_path("dataset") + self.get_path("train"),
                                            input_dir=self.get_path("input"),
                                            target_dir=self.get_path("target"))
        valid_dataset = ImageToImageDataset(img_dir=self.get_path("dataset") + self.get_path("valid"),
                                            input_dir=self.get_path("input"),
                                            target_dir=self.get_path("target"))
        test_dataset = ImageToImageDataset(img_dir=self.get_path("dataset") + self.get_path("test"),
                                           input_dir=self.get_path("input"),
                                           target_dir=self.get_path("target"))
        return [train_dataset, valid_dataset, test_dataset]

    def load_cfg(self, path):
        assert os.path.exists(path), f"\n[Error] ファイルが見つかりません \
                                       \nInput: {path} \
                                       \nAbs path: {os.path.abspath(path)}\n"

        # jsonファイルをロード
        with open(path, "r") as f:
            self.cfg = json.load(f)

        # ファイル出力用ディレクトリがあるか確認
        assert os.path.isdir(self.get_path("output")), f"\n[ERROR] bad output dir: {self.get_path('output')}"

        # 出力ディレクトリ名を日時で更新
        d = datetime.now().strftime("%Y-%m%d-%H%M%S") + "/"
        self.cfg["path"]["output"] += d

    def show_cfg(self):
        msg = ""
        for key, category in self.cfg.items():
            msg += f"\n[{key}]"

            for name, val in category.items():
                msg += f"\n    {name}: {val}"
        return msg

    def get_cfg_val(self, category: str, name: str):
        return self.cfg[category][name]
    
    def get_path(self, name: str):
        return self.get_cfg_val("path", name)
    
    def get_hparam(self, name: str):
        return self.get_cfg_val("hyperparam", name)
    
    def down_cursor(self, n=0):
        print(f"\033[{n}B")