from .base_processor import BaseProcessor
from torch import no_grad, mean, save
from torch.utils.data import DataLoader
from importlib import import_module

# from torch.optim import Optimizer
# from torch.optim.lr_scheduler import LRScheduler

class Trainer(BaseProcessor):
    def __init__(self):
        super().__init__()

        self.dataloader:DataLoader = None
        self.optimizer = None
        self.scheduler = None
        self.loss_handler: dict = None

        self.epochs: int = 10
        self.iteration: int = None
        self.val_interval: int = 1
        self.log_interval: int = 3
        self.save_interval: int = 1

        self.itr_digit = 5

    def process(self):
        self.logger.debug("function called: TRAIN")
        print(self.model)
        print("\n"* 10)

        assert self.model is not None, "\n[ERROR] model is not defined"

        for epoch in range(1, self.epochs+1):
            # train 1 epoch
            self.train(epoch=epoch)
            # validation
            if (epoch % self.val_interval == 0):
                self.validate()
            # 重み保存
            if (epoch % self.save_interval == 0):
                self.save_model(epoch)

    def train(self, epoch):
        '''1エポック分の学習を行う関数
        
        引数
            - epoch -- エポック数
        '''
        self.logger.debug("-----train-----")

        # 学習モードに変更
        self.model.train()

        # 損失のリセット
        # loss = None

        for i, (source, target) in enumerate(self.dataloader, 1):
            # deviceにデータを移動
            source = source.to(self.device)
            target = target.to(self.device)
            # 勾配情報をリセット
            self.optimizer.zero_grad()
            # 順伝播
            output = self.model(source)
            # 損失の計算
            # loss = self.loss_handler(output, target)
            loss_dict = {}
            loss = 0
            for key, func in self.loss_handler.items():
                loss_dict[key] = func(output, target)
                loss += loss_dict[key]
            # 逆伝播
            loss.backward()
            # オプティマイザの更新
            self.optimizer.step()
            # スケジューラの更新(warmup)
            if self.scheduler is not None:
                try:
                    self.scheduler.warm()
                except AttributeError:
                    pass
            # 標準出力
            if i == 1:
                self.logger.debug(f"require_grad: {output.requires_grad}")
            if i == 1 or i % self.log_interval == 0:
                self.logger.displayStatus(epoch, i,
                                          self.epochs, self.iteration,
                                          self.optimizer.param_groups[0]["lr"],
                                          loss=loss.item())

        # スケジューラの更新(step)
        self.scheduler.step()

    def validate(self):
        self.logger.debug("-----valid-----")

        # 推論モードに変更
        self.model.eval()

        # 損失・精度のリセット
        # loss = None
        accr = dict()
        for key in self.metrics:
            accr[key] = 0
        
        # 評価
        with no_grad():
            for i, (source, target) in enumerate(self.datasets[1], 1):
                # データの移動
                source = source.to(self.device)
                target = target.to(self.device)
                source = source.unsqueeze(0)
                target = target.unsqueeze(0)
                # 順伝播
                output = self.model(source)
                # 損失の計算
                # loss = self.loss_handler(output, target)
                # 精度の計算
                for key, metric in self.metrics.items():
                    accr[key] += mean(metric(output, target)).item()
                
                # 出力
                if i == 1:
                    self.logger.debug(f"require_grad: {output.requires_grad}")
                if i == 1 or i % (self.log_interval * self.dataloader.batch_size) == 0:
                    msg = f"[valid] [{i:>{self.itr_digit}}/{len(self.datasets[1]):>{self.itr_digit}}] loss: {.0:.9f} "
                    for key, val in accr.items():
                        msg += f"| {key}: {val/i:.9f} "
                    self.logger.info(msg)

    def save_model(self, epoch, loss=0.):
        self.logger.debug("save model")
        if self.device == "cuda":
            save(obj={"epoch": epoch,
                    "model_state_dict": self.model.to("cpu").state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss},
                f= self.out_dir + f"weight_{epoch}.pth")
            # .to()によるメモリ移動を戻す
            self.model.to("cuda")
        else:
            save(obj={"epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss},
                f= self.out_dir + f"weight_{epoch}.pth")
        
        self.logger.debug(f"model at epoch {epoch} weight is saved.")

    def build_dataloader(self, batch_size=4, shuffle=True, num_workers=2, pin_memory=True, drop_last=True):
        self.dataloader = DataLoader(self.datasets[0],
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=num_workers,
                                     pin_memory=pin_memory,
                                     drop_last=drop_last)
        
    def build_loss_handler(self, losses: list[str]):
        self.loss_handler = {}
        module = import_module(name="loss")
        for loss in losses:
            try:
                loss_class = getattr(module, loss)
                self.loss_handler[f"{loss}"] = loss_class()
                self.logger.debug(f"loss added: {loss} {type(self.loss_handler[loss])}")
            except AttributeError:
                self.logger.debug(f"fatal: {loss}")