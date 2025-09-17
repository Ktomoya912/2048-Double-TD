import logging

import torch

from common.args import args
from common.config import BAT_SIZE, DEVICE, MAIN_NETWORK, TARGET_NETWORK
from common.utils import write_make_input

from .common import Trainer

logger = logging.getLogger(__name__)


class D_TDA_C_Trainer(Trainer):
    def train(self, records: list[dict], pack: dict, count: int = 1):
        # inputsには盤面の情報、targetsには評価値が入る
        target_values = []
        boards = []
        for record in records:
            board = record["board"]
            target_value = record["other_value"]
            boards.append(board)
            target_values.append(target_value)

        # メインネットワークはターゲットから得られた評価値を使用して学習する
        logger.info(f"train {count=}, {len(boards)=}, {len(target_values)=}")
        if len(boards) == 0:
            logger.warning("No records to train.")
            return
        if len(boards) != len(target_values):
            logger.error(f"Length mismatch: {len(boards)=}, {len(target_values)=}")
            return

        model = pack["model"]
        optimizer = pack["optimizer"]

        model.train()  # モデルを学習モードに設定
        optimizer.zero_grad()  # 勾配をゼロに初期化
        tmp = torch.zeros(len(boards), 99, device="cpu")
        for i in range(len(boards)):
            write_make_input(boards[i], tmp[i, :])
        inputs = tmp.to(DEVICE)
        # ネットワークにデータを入力し、順伝播を行う
        outputs = model.forward(inputs)
        targets = torch.as_tensor(target_values, dtype=torch.float32)
        targets = targets.reshape(-1, 1)  # ターゲットの形状を調整
        targets = targets.to(DEVICE)
        loss = self.criterion(outputs, targets)  # 損失を計算
        loss.backward()  # 逆伝播を行い、各パラメータの勾配を計算
        optimizer.step()
        logger.debug(f"loss : {loss.item()}")

    def batch_trainer(self, pack: dict):
        train_count = 0
        records = []
        # 重み更新の頻度
        update_target_every = args.target_update_freq
        while not self.stop_event.is_set():
            train_count += 1
            while len(records) != BAT_SIZE and not self.stop_event.is_set():
                records.append(pack["queue"].get())
            self.train(records, pack, train_count)

            if train_count % update_target_every == 0:
                TARGET_NETWORK.load_state_dict(MAIN_NETWORK.state_dict())
                logger.info(f"Update target network at {train_count=}, {pack['name']}")

            records.clear()
        return train_count
