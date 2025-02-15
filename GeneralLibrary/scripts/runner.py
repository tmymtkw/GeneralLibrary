from scripts.analyzer import Analyzer

TRAIN = 0
VALIDATION = 1
TEST = 2

class Runner(Analyzer):
    def __init__(self):
        super().__init__()
        self.is_train = (self.args.mode == TRAIN)
        print(self.is_train)
        
    def Run(self):
        self.Debug(msg="program beginning...")
        
        # 環境設定
        # データセット作成
        self.SetDataset(self.cfg.GetPath("dataset"),
                        self.cfg.GetPath("input"),
                        self.cfg.GetPath("target"))
        # データローダー作成
        self.SetDataLoader(batch_size=self.cfg.GetHyperParam("batch_size"),
                           shuffle=self.is_train,
                           num_workers=self.cfg.GetHyperParam("num_workers"),
                           pin_memory=True,
                           drop_last=self.is_train)
        
        # モデル定義
        self.SetModel()
        # オプティマイザ定義
        # 使用プロセッサ設定
        self.SetDevice(device=self.cfg.GetInfo("option", "device"))

        # メイン処理実行
        self.Operate()

        # 終了処理
        self.Debug(msg="program finished.")

    def Operate(self):
        self.Info(f"running mode: {self.args.mode}")
        print("\n"*9)

        # Train
        if self.is_train:
            self.Train()
        # Validation
        elif self.args.mode == VALIDATION:
            self.Validate()
        # Test
        elif self.args.mode == TEST:
            pass
        else:
            self.Analyze()