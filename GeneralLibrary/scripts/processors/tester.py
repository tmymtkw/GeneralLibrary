from .base_processor import BaseProcessor
import os
from torch import no_grad, mean, cat
from torchvision.utils import save_image

class Tester(BaseProcessor):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir + "imgs/"

    def process(self):
        print("\n")
        self.logger.debug("-----test------")
        
        self.model.eval()

        # 画像保存用ディレクトリを作成
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        cur_accr = dict()
        all_accr = dict()
        for key in self.metrics:
            cur_accr[key] = 0
            all_accr[key] = 0

        with no_grad():
            for i, (source, target) in enumerate(self.datasets[2], 1):
                source = source.to(self.device)
                target = target.to(self.device)
                source = source.unsqueeze(0)
                target = target.unsqueeze(0)
                output = self.model(source)
                for key, metric in self.metrics.items():
                    cur_accr[key] = mean(metric(output, target)).item()
                    all_accr[key] += cur_accr[key] / len(self.datasets[2])

                self.logger.info(self.make_cli_msg(accr=cur_accr, img_num=i, img_shape=source.shape), extra={"n":1})

                output = cat((source, output, target), dim=0)
                save_image(output, self.save_dir+f"{i}.png")
        
        print("\033[3B")
        self.logger.info("[result]" + self.metric_to_str(all_accr))

    def make_cli_msg(self, accr: dict, img_num: int, img_shape) -> str:
        msg = f"image {img_num}: {img_shape} |" + self.metric_to_str(accr)

        return msg

    def metric_to_str(self, accr: dict):
        msg = ""

        for key, val in accr.items():
            msg += f" {key}: {val:.9f} |"

        return msg