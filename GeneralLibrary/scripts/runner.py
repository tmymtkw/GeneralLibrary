from .base_runner import BaseRunner
from util.logtool.mainlogger import MainLogger
from util.parser.mainparser import MainParser
from util.parser.cfgloader import ConfigLoader

class Runner(BaseRunner):
    def __init__(self):
        super().__init__()

    def setup(self):
        # 1. コンフィグ設定

        # 2. コマンドライン引数受け取り
        parser = MainParser()
        args = parser.parse_args()

        # 3. コンフィグ書き換え
        

        # 4. ログの設定
        

    def run(self, mode="TRAIN"):
        mode = mode.lower()

        # 小文字に統一
        if mode == "train":
            self.trainer(self.model, self.metrics)
        elif mode == "test":
            self.tester(self.model, self.metrics)
        elif mode == "analyze":
            self.analyzer(self.model)

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