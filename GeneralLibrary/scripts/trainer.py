from torch.utils.data import DataLoader
from util.data.dataset import ImageToImageDataset

class Trainer(object):
    def __init__(self):
        self.model = None
        self.dataset = None
        self.dataloader = None
        self.optimizer = None
        self.device = None

    # TODO
    def Train(self):
        for i, (img_input, img_target) in enumerate(self.dataloader):
            if i % 10 != 0:
                continue
            print(i, img_input.size())
            continue

    def SetDataset(self, img_dir, input_dir, target_dir):
        self.dataset = ImageToImageDataset(img_dir, input_dir, target_dir)

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
        
    def SetDevice(self, device):
        assert (device == "cuda" or device == "cpu"), \
            f"\n[ERROR] incorrevt device type : {device}"
        
        print("setting device")
        self.device = device
    