import logging

import numpy as np
import torch

from common.config import BAT_SIZE
from common.utils import get_one_values
from game_2048_3_3 import State

from .common import Trainer

# ゲームごとに貯めて学習
logger = logging.getLogger(__name__)


class TDA_Trainer(Trainer):
    def _play(self, packs, canmov, bd: State, last_board):
        main_values = get_one_values(canmov, bd.clone(), packs[0]["model"])
        # main_valuesから最大の評価値を持つインデックスを取得
        main_max_index = np.argmax(main_values)
        bd.play(main_max_index)
        if last_board is not None:
            self.put_queue(
                last_board.copy(),
                self_value=main_values[main_max_index],
                other_value=torch.tensor(0),
                packs=packs,
            )

    def _train(self, records: list[dict], pack: dict, count: int = 1):
        # inputsには盤面の情報、targetsには評価値が入る
        values = []
        boards = []
        for record in records:
            board = record["board"]
            value = record["self_value"]
            boards.append(board)
            values.append(value)

        logger.info(f"train {pack['name']} {count=}, {len(boards)=}, {len(values)=}")

        # メインネットワークはターゲットから得られた評価値を使用して学習する
        logger.info(f"train {count=}, {len(boards)=}, {len(values)=}")
        if len(boards) == 0:
            logger.warning("No records to train.")
            return None, None
        if len(boards) != len(values):
            logger.error(f"Length mismatch: {len(boards)=}, {len(values)=}")
            return None, None

        return boards, values

    def batch_trainer(self, pack: dict):
        train_count = 0
        records = []
        # 重み更新の頻度
        while not self.stop_event.is_set():
            train_count += 1
            while len(records) != BAT_SIZE and not self.stop_event.is_set():
                records.append(pack["queue"].get())
            self.train(records, pack, train_count)

            records.clear()
        return train_count
