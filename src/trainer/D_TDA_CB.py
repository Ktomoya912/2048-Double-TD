import logging

from common.args import args
from common.config import BAT_SIZE, MAIN_NETWORK, TARGET_NETWORK

from .common import Trainer

logger = logging.getLogger(__name__)


class D_TDA_CB_Trainer(Trainer):
    def _train(self, records: list[dict], pack: dict, count: int = 1):
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
            return None, None
        if len(boards) != len(target_values):
            logger.error(f"Length mismatch: {len(boards)=}, {len(target_values)=}")
            return None, None

        return boards, target_values

    def batch_trainer(self, pack: dict):
        train_count = 0
        records = []
        # 重み更新の頻度
        update_target_every = args.target_update_freq
        weights_before_2 = None
        while not self.stop_event.is_set():
            train_count += 1
            while len(records) != BAT_SIZE and not self.stop_event.is_set():
                records.append(pack["queue"].get())
            self.train(records, pack, train_count)

            if train_count % update_target_every == 0:
                weights = MAIN_NETWORK.state_dict()
                if weights_before_2 is not None:
                    TARGET_NETWORK.load_state_dict(weights_before_2)
                weights_before_2 = weights
                logger.info(f"Update target network at {train_count=}, {pack['name']}")

            records.clear()
        return train_count
