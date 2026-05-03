# src/trainer/policy_mixin.py
import logging

import numpy as np
import torch
from queue import Queue

from common.utils import ACTION_TRANSFORM, rotate_board, mirror_board, write_make_input

logger = logging.getLogger(__name__)


class PolicyMixin:
    """Adds policy data collection and training to any Trainer subclass.

    Requires the subclass to set self.policy_pack (dict with 'model', 'optimizer', 'queue')
    before calling put_policy_queue. When self.policy_pack is None, all methods are no-ops.
    """

    def put_policy_queue(self, state: np.ndarray, action: int):
        """Enqueue (state, action) pairs for policy training.
        When --symmetry is active, enqueues all 8 augmented pairs with transformed actions.
        Skips silently if self.policy_pack is None.
        """
        if not hasattr(self, "policy_pack") or self.policy_pack is None:
            return

        from common.args import args

        queue: Queue = self.policy_pack["queue"]

        if args.symmetry:
            board = state.copy()
            boards = [board]
            for _ in range(3):
                board = rotate_board(board)
                boards.append(board)
            board = mirror_board(board)
            boards.append(board)
            for _ in range(3):
                board = rotate_board(board)
                boards.append(board)
            for sym_idx, bd in enumerate(boards):
                queue.put({"board": bd.copy(), "action": ACTION_TRANSFORM[sym_idx][action]})
        else:
            queue.put({"board": state.copy(), "action": action})

    def policy_batch_trainer(self, policy_pack: dict):
        """Batch training loop for the separate policy model (CNN_DEEP_POLICY).
        Runs in its own thread. Uses CrossEntropyLoss.
        """
        from common.config import BAT_SIZE, DEVICE
        from torch import nn

        criterion = nn.CrossEntropyLoss()
        train_count = 0
        records: list[dict] = []

        while not self.stop_event.is_set():
            train_count += 1
            while len(records) != BAT_SIZE and not self.stop_event.is_set():
                records.append(policy_pack["queue"].get())

            model = policy_pack["model"]
            optimizer = policy_pack["optimizer"]

            boards = [r["board"] for r in records]
            actions = [r["action"] for r in records]

            tmp = torch.zeros(len(boards), 99, device="cpu")
            for i, board in enumerate(boards):
                write_make_input(board, tmp[i])
            inputs = tmp.to(DEVICE)
            targets = torch.tensor(actions, dtype=torch.long, device=DEVICE)

            model.train()
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            logger.debug(f"policy loss: {loss.item():.4f}")
            records.clear()

        return train_count
