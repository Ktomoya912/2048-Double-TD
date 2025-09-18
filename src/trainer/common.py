import logging
import threading

import numpy as np
import torch
from torch import nn, optim

from common.args import args
from common.config import DEVICE, MAIN_NETWORK, TARGET_NETWORK
from common.utils import (
    board_print,
    get_eval,
    get_values,
    mirror_board,
    rotate_board,
    write_make_input,
)
from game_2048_3_3 import State

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, packs):
        self.stop_event = threading.Event()
        self.criterion = nn.MSELoss()
        self.packs: list[dict] = packs

    def put_queue(
        self, board: np.ndarray, self_value: float, other_value: float, packs
    ):
        board_cp = board.copy()
        queue = packs[0]["queue"]

        if args.symmetry:
            bd_list: list[np.ndarray] = [board]
            for _ in range(3):
                board = rotate_board(board)
                bd_list.append(board)
            board = mirror_board(board)
            bd_list.append(board)
            for _ in range(3):
                board = rotate_board(board)
                bd_list.append(board)

            logger.debug(f"{bd_list=}\n\t{board_cp=}")
            for bd in bd_list:
                queue.put(
                    {
                        "board": bd,
                        "self_value": self_value,
                        "other_value": other_value,
                    }
                )
        else:
            queue.put(
                {
                    "board": board,
                    "self_value": self_value,
                    "other_value": other_value,
                }
            )

    def _train(self) -> tuple[list[np.ndarray], float]:
        raise NotImplementedError

    def train(self, records: list[dict], pack: dict, count: int = 1):
        boards, values = self._train(records, pack, count)
        if boards is None or values is None:
            return
        model = pack["model"]
        optimizer: optim.Adam = pack["optimizer"]

        model.train()  # モデルを学習モードに設定
        optimizer.zero_grad()  # 勾配をゼロに初期化
        tmp = torch.zeros(len(boards), 99, device="cpu")
        for i in range(len(boards)):
            write_make_input(boards[i], tmp[i, :])
        inputs = tmp.to(DEVICE)
        # ネットワークにデータを入力し、順伝播を行う
        outputs = model.forward(inputs)
        targets = torch.as_tensor(values, dtype=torch.float32)
        targets = targets.reshape(-1, 1)  # ターゲットの形状を調整
        targets = targets.to(DEVICE)
        loss = self.criterion(outputs, targets)  # 損失を計算
        loss.backward()  # 逆伝播を行い、各パラメータの勾配を計算
        optimizer.step()
        logger.debug(f"loss : {loss.item()}")

    def _play(self, packs, canmov, bd: State, last_board):
        self_values, other_values = get_values(canmov, bd.clone(), packs)
        # 自分自身の評価値を取得
        self_max_index = np.argmax(self_values)
        if args.consensus:
            # self_valuesとother_valuesのそれぞれを足し合わせる
            values = np.array(self_values) + np.array(other_values)
            sample_idx = np.argmax(values)
            bd.play(sample_idx)
        else:
            bd.play(self_max_index)
        if last_board is not None:
            self.put_queue(
                last_board.copy(),
                self_value=self_values[self_max_index],
                other_value=other_values[self_max_index],
                packs=packs,
            )

    def play_game(self, thread_id: int):
        packs = self.packs.copy()
        try:
            games = 0
            while not self.stop_event.is_set():
                games += 1
                states = []
                bd = State()
                bd.initGame()
                turn = 0
                if args.trainer == "D_TDA_X":
                    packs.reverse()
                for count in range(10_000):
                    last_board = None
                    init_eval_1 = 0
                    init_eval_2 = 0
                    while not self.stop_event.is_set():
                        turn += 1
                        canmov = [bd.canMoveTo(i) for i in range(4)]
                        self._play(packs, canmov, bd, last_board)
                        last_board = bd.clone().board
                        if turn == 1:
                            init_eval_1 = get_eval(bd.board, MAIN_NETWORK)
                            init_eval_2 = get_eval(bd.board, TARGET_NETWORK)
                        bd.putNewTile()
                        states.append(bd.clone())
                        if bd.isGameOver():
                            board_print(bd)
                            self.put_queue(
                                last_board.copy(),
                                torch.tensor(0),
                                torch.tensor(0),
                                packs=packs,
                            )
                            logger.info(
                                f"GAMEOVER: {thread_id=:02d} {count=:03d} {bd.score=:04d} {turn=:04d} {packs[0]['name'].ljust(7)}queue_size={packs[0]['queue'].qsize():04d} {init_eval_1=:.2f} {init_eval_2=:.2f}"
                            )
                            break
                    if args.restart and len(states) > 10:
                        bd = states[len(states) // 2]
                        turn -= len(states) // 2
                        states = []
                    else:
                        break
        except Exception as e:
            logger.exception(e)
            self.stop_event.set()

    def batch_trainer(self, pack: dict):
        raise NotImplementedError
