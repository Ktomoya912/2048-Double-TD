import logging

import torch
import torch.optim as optim

from common.config import BAT_SIZE, DEVICE
from common.utils import write_make_input

from .common import Trainer

logger = logging.getLogger(__name__)


class D_TDA_X_Trainer(Trainer):
    # 学習用の関数
    def train(self, records: list[dict], pack: dict, count: int = 1):
        other_values = []
        boards = []
        for record in records:
            board = record["board"]
            other_value = record["other_value"]
            boards.append(board)
            other_values.append(other_value)

        # メインネットワークはターゲットから得られた評価値を使用して学習する
        logger.info(
            f"train {pack['name']} {count=}, {len(boards)=}, {len(other_values)=}"
        )
        if len(boards) == 0:
            logger.warning("No records to train.")
            return
        if len(boards) != len(other_values):
            logger.error(f"Length mismatch: {len(boards)=}, {len(other_values)=}")
            return
        model = pack["model"]
        optimizer: optim.Adam = pack["optimizer"]

        # with open(f"tmp_train_{pack['name']}.txt", "a", encoding="utf-8") as f:
        #     f.write(f"{pack['name']}\n{boards=}\n{other_values=}\n\n")
        model.train()  # モデルを学習モードに設定
        optimizer.zero_grad()  # 勾配をゼロに初期化
        tmp = torch.zeros(len(boards), 99, device="cpu")
        for i in range(len(boards)):
            write_make_input(boards[i], tmp[i, :])
        inputs = tmp.to(DEVICE)
        # ネットワークにデータを入力し、順伝播を行う
        outputs = model.forward(inputs)
        targets = torch.as_tensor(other_values, dtype=torch.float32)
        targets = targets.reshape(-1, 1)  # ターゲットの形状を調整
        targets = targets.to(DEVICE)
        loss = self.criterion(outputs, targets)  # 損失を計算
        loss.backward()  # 逆伝播を行い、各パラメータの勾配を計算
        optimizer.step()
        logger.debug(f"loss : {loss.item()}")

    def batch_trainer(self, pack: dict):
        logger.info(f"Starting batch training for {pack['name']}...")
        train_count = 0
        records = []
        while not self.stop_event.is_set():
            train_count += 1
            while len(records) != BAT_SIZE and not self.stop_event.is_set():
                records.append(pack["queue"].get())
            self.train(records, pack, train_count)

            records.clear()
        logger.info(
            f"Batch training completed for {pack['name']} after {train_count} iterations."
        )
        return train_count
