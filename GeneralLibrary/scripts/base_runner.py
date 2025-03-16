from abc import ABC, abstractmethod

class BaseRunner(ABC):
    def __init__(self):
        # config
        self.cfg = None

        # processor
        self.processor = None
        self.trainer = None
        self.tester = None
        self.analyzer = None

        # necessary objects
        self.dictionary = None

    @abstractmethod
    def run(self, mode):
        '''main処理
        
        引数:
            - mode -- 実行モード'''
        raise NotImplementedError()

    @abstractmethod
    def setup(self):
        '''run()を実行する前の処理

        1. コマンドライン引数の受け取り
        2. コンフィグの設定
        3. コンフィグ設定の書き換え
        4. ログの設定
        5. インスタンス変数設定
        '''
        raise NotImplementedError()

    @abstractmethod
    def build(self):
        raise NotImplementedError()

    @abstractmethod
    def load_cfg(self, path: str):
        raise NotImplementedError()