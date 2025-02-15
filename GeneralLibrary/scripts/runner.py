from util.parser.mainparser import MainParser
from util.parser.cfgloader import ConfigLoader
from scripts.trainer import Trainer

import os
import torch
from torchvision.utils import save_image

class Runner(Trainer):
    def __init__(self):
        super().__init__()
        self.parser = MainParser()
        # コマンド受け取り
        self.args = self.parser.parse_args()
        self.loader = ConfigLoader(self.args.cfg_path)
        self.is_train = (self.args.mode == 0)
        
    def Run(self):
        self.Show()
        
        # 環境設定
        # データセット・データローダー作成
        self.SetDataset(self.loader.GetPath("dataset"),
                        self.loader.GetPath("input"),
                        self.loader.GetPath("target"))
        self.SetDataLoader(batch_size=self.loader.GetHyperParam("batch_size"),
                           shuffle=self.is_train,
                           num_workers=self.loader.GetHyperParam("num_workers"),
                           pin_memory=True,
                           drop_last=self.is_train)
        print("dataset and dataloader created")
        # モデル定義
        # オプティマイザ定義
        self.SetDevice(device=self.loader.GetInfo("mode", "device"))

        # 学習
        self.Operate()

        # 終了処理

    def Operate(self):
        # Train
        if self.is_train:
            epoch = self.loader.GetHyperParam("epoch")

            for i in range(epoch):
                self.Train()
        # Test
        else:
            print("test mode")
            for i in range(10):
                img_input, img_target = self.dataset[i]
                img_input = img_input.to(self.device, torch.float32)
                img_target = img_target.to(self.device, torch.float32)
                print(type(img_input), img_input.dtype, img_input.device, img_input.shape)
                print(type(img_target), img_target.dtype, img_target.device, img_target.shape)

                img_input /= 255
                img_target /= 255
                img_input = img_input.unsqueeze(0).to("cpu")
                img_target = img_target.unsqueeze(0).to("cpu")
                save_image(img_input, os.path.join("./", self.loader.GetInfo("path", "output"), f"debug_input_{i}.jpg"))
                save_image(img_target, os.path.join("./", self.loader.GetInfo("path", "output"), f"debug_target{i}.jpg"))
                print(img_input.shape)

    def Show(self):
        print(self.args)