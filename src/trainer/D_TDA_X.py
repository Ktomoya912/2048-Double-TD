import logging

from common.config import BAT_SIZE

from .common import Trainer

logger = logging.getLogger(__name__)


class D_TDA_X_Trainer(Trainer):
    # 学習用の関数
    def _train(self, records: list[dict], pack: dict, count: int = 1):
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
            return None, None
        if len(boards) != len(other_values):
            logger.error(f"Length mismatch: {len(boards)=}, {len(other_values)=}")
            return None, None
        return boards, other_values

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
