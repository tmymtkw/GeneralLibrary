from .base_runner import BaseRunner
import os, json
from datetime import datetime
from util.logtool.mainlogger import MainLogger
from util.parser.mainparser import MainParser

class Runner(BaseRunner):
    def __init__(self):
        super().__init__()

    def setup(self):
        # 1. コマンドライン引数受け取り
        parser = MainParser()
        args = parser.parse_args()

        # 2. コンフィグ設定
        self.load_cfg(args.cfg_path)

        # 3. コンフィグ書き換え
        self.cfg["option"]["mode"] = args.mode

        # 4. ログの設定
        self.logger = MainLogger(dir=self.get_path("output"))
        # コンフィグ状況を出力
        self.logger.debug(self.show_cfg())
        self.logger.info("runner setup completed")

    def run(self):
        self.setup()

        mode = self.get_cfg_val("option", "mode")

        assert isinstance(mode, str), f"[ERROR] bad mode type: {type(mode)}"
        # 小文字に統一
        mode = mode.lower()
        assert (mode == "train" or mode == "test" or mode == "analyze"), f"[ERROR] bad mode: {mode}"
        return
        if mode == "train":
            self.trainer(self.model, self.metrics)
        elif mode == "test":
            self.tester(self.model, self.metrics)
        elif mode == "analyze":
            self.analyzer(self.model)

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

# from scripts.analyzer import Analyzer
# from loss.mse import MSELoss
# from model.net import Net
# from torch.optim import Adam

# TRAIN = 0
# TEST = 1

# class Runner(Analyzer):
#     def __init__(self):
#         super().__init__()
#         self.is_train = (self.args.mode == TRAIN)
#         print(self.is_train)
        
#     def Run(self):
#         self.Debug(msg="program beginning...")
        
#         # 環境設定
#         # データセット作成
#         self.SetDataset(self.cfg.GetPath("dataset"),
#                         self.cfg.GetPath("input"),
#                         self.cfg.GetPath("target"))
#         # データローダー作成
#         self.SetDataLoader(batch_size=self.cfg.GetHyperParam("batch_size"),
#                            shuffle=self.is_train,
#                            num_workers=self.cfg.GetHyperParam("num_workers"),
#                            pin_memory=True,
#                            drop_last=self.is_train)
#         # ログ書式設定
#         self.SetLogDigits(self.epochs, len(self.train_dataset) // self.cfg.GetHyperParam("batch_size") + 1)
        
#         # モデル定義
#         self.model = Net()
#         self.model.to(self.device)
#         # オプティマイザ定義
#         self.optimizer = Adam(self.model.parameters(), lr=self.cfg.GetHyperParam("lr"))
#         # 損失関数設定
#         self.criteria = MSELoss()
#         # 使用プロセッサ設定
#         self.SetDevice(device=self.cfg.GetInfo("option", "device"))

#         # メイン処理実行
#         self.Operate()

#         # 終了処理
#         self.Debug(msg="program finished.")

#     def Operate(self):
#         self.Info(f"running mode: {self.args.mode}")
#         print("\n"*9)

#         # Train
#         if self.is_train:
#             self.Train()
#         # Test
#         elif self.args.mode == TEST:
#             pass
#         else:
#             self.Analyze()