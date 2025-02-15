from torch.nn import Module
from torch.utils.data import DataLoader
from util.data.dataset import ImageToImageDataset
from scripts.recorder import Recorder
from model.net import Net

class Trainer(Recorder):
    def __init__(self):
        super().__init__()
        
        self.model: Module = None
        self.dataset: ImageToImageDataset = None
        self.dataloader: DataLoader = None
        self.optimizer = None
        self.criteria = None
        self.SetDevice(self.cfg.GetInfo("option", "device"))
        self.epochs = self.cfg.GetHyperParam("epoch")

    # TODO
    def Train(self):
        """学習を行う関数

        epochs分だけProcess()を実行する

        :epochs int エポック数
        """
        self.Debug("function called: Train")

        assert self.model is not None, "\n[ERROR] model is not defined"

        for epoch in range(self.epochs):
            self.Process(epoch=epoch, is_train=True)

            # TODO: 比較と保存処理

    def Process(self, epoch, is_train=True):
        """1エポック分の学習を実施する関数

        is_trainがFalseの時は逆伝播を行わない

        :is_train bool 学習モードの切り替え
        """

        if is_train:
            self.model.train()
        else:
            self.model.eval()

        for i, (img_input, img_target) in enumerate(self.dataloader):
            # GPU(CPU)にデータを移動
            img_input = img_input.to(self.device)
            img_target = img_target.to(self.device)

            # 順伝播
            img_output = self.model(img_input)
            # 損失の計算
            # loss = self.criteria(img_output, img_target)
            # 逆伝播
            # loss.backward()
            # オプティマイザの更新
            
            # TODO
            if i % self.cfg.GetInfo("option", "interval") != 0:
                continue
            self.DisplayStatus(epoch, i, self.epochs, len(self.dataset) // self.cfg.GetHyperParam("batch_size"), self.cfg.GetHyperParam("lr"))

    def SetModel(self, model_class=Net):        
        self.model = model_class()

        assert isinstance(self.model, Module), f"\n[ERROR] incorrect model class: {type(model_class)}"
        self.Debug("Model created.")

    def SetDataset(self, img_dir, input_dir, target_dir):
        self.dataset = ImageToImageDataset(img_dir, input_dir, target_dir)
        self.Debug("Dataset created.")

    def SetDataLoader(self,
                      batch_size=32,
                      shuffle=True,
                      num_workers=4,
                      pin_memory=True,
                      drop_last=True):
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=num_workers,
                                     pin_memory=pin_memory,
                                     drop_last=drop_last)
        self.Debug("Dataloader created.")
        
    def SetDevice(self, device):
        assert (device == "cuda" or device == "cpu"), \
            f"\n[ERROR] incorrevt device type : {device}"
        
        self.Debug(f"setting device: {device}")

        self.device = device
    
    def DisplayStatus(self, cur_epoch, cur_itr, max_epoch, max_itr, lr=0.0, loss=0.0):
        """学習状況の標準出力

        同じフォーマットで描画を更新する
        """

        self.Info(msg="", extra={"status": {"cur_epoch": cur_epoch+1,
                                            "cur_itr": cur_itr+1,
                                            "max_epoch": max_epoch,
                                            "max_itr": max_itr,
                                            "lr": lr,
                                            "loss": loss},
                                "n": 6})