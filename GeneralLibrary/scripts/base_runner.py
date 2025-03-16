from abc import ABC, abstractmethod

class BaseRunner(ABC):
    def __init__(self):
        # processor
        self.trainer = None
        # self.validator
        self.tester = None
        self.analyzer = None

        # config
        self.cfg = None
        # logger
        self.logger = None

        # data
        self.dataset = None
        self.dataloader = None

        # necessary objects
        self.model = None
        self.metrics = None
        self.losses = None
        self.dictionary = None

    @abstractmethod
    def setup(self):
        '''run()を実行する前の処理

        1. コマンドライン引数の受け取り
        2. コンフィグの設定
        3. コンフィグ設定の書き換え
        4. ログの設定
        5. インスタンス変数設定
        '''
        NotImplementedError()

    @abstractmethod
    def run(self, mode):
        '''main処理
        
        引数:
            - mode -- 実行モード'''
        NotImplementedError()

    @abstractmethod
    def load_cfg(self, path: str):
        NotImplementedError()