# src/trainer/multi_head.py
import logging

import numpy as np
import torch
from torch import nn, optim

from common.args import args
from common.config import BAT_SIZE, DEVICE
from common.utils import write_make_input
from game_2048_3_3 import State

from .policy_mixin import PolicyMixin, ACTION_TRANSFORM
from .common import Trainer

logger = logging.getLogger(__name__)


class MultiHeadTrainer(PolicyMixin, Trainer):
    """Trainer for multi-head models (CNN_DEEP_MULTI, ALPHA_ZERO_STATE).

    The model takes state (before action) as input and outputs (value, pi_logits).
    Value is trained with TD(0): V(s_t) <- V(s_{t+1}).
    Policy is trained with CrossEntropy on best_action from TD-Afterstate evaluation.
    During play, afterstates are fed through the model's value head for action selection.
    """

    def __init__(self, packs):
        super().__init__(packs, policy_pack=None)
        self.value_criterion = nn.MSELoss()
        self.policy_criterion = nn.CrossEntropyLoss()

    def _play(self, packs, canmov, bd: State, last_board):
        state_before = bd.board.copy()
        model = packs[0]["model"]
        model.eval()

        # Evaluate each afterstate through model's value head for action selection
        inputs = torch.zeros(4, 99, device="cpu")
        sub_list = []
        for i in range(4):
            copy_bd = bd.clone()
            copy_bd.play(i)
            sub_list.append(copy_bd.score - bd.score)
            write_make_input(copy_bd.board, inputs[i])

        with torch.no_grad():
            value_out, _ = model.forward(inputs.to(DEVICE))

        values = [-1e10] * 4
        for i in range(4):
            if canmov[i]:
                values[i] = float(value_out[i]) + sub_list[i]

        best_action = int(np.argmax(values))
        bd.play(best_action)

        # Enqueue previous step's data (state, action, next_state)
        # next_state is known on the NEXT call (after putNewTile), so we use
        # last_board as previous-step state and the current afterstate as next placeholder.
        # Actual next_state (after new tile) is set in play_game override.
        if last_board is not None and hasattr(self, "_last_state_before"):
            if self._last_state_before is not None:
                packs[0]["queue"].put({
                    "state": self._last_state_before,
                    "action": self._last_best_action,
                    "next_state": state_before,  # state before current move = after last new tile
                    "is_terminal": False,
                })

        self._last_state_before = state_before
        self._last_best_action = best_action
        return best_action

    def play_game(self, thread_id: int):
        packs = self.packs.copy()
        self._last_state_before = None
        self._last_best_action = None
        try:
            while not self.stop_event.is_set():
                bd = State()
                bd.initGame()
                turn = 0
                while not self.stop_event.is_set():
                    turn += 1
                    canmov = [bd.canMoveTo(i) for i in range(4)]
                    last_board = bd.board.copy() if turn > 1 else None
                    self._play(packs, canmov, bd, last_board)
                    bd.putNewTile()
                    if bd.isGameOver():
                        # Terminal state: value target = 0
                        if self._last_state_before is not None:
                            packs[0]["queue"].put({
                                "state": self._last_state_before,
                                "action": self._last_best_action,
                                "next_state": bd.board.copy(),
                                "is_terminal": True,
                            })
                        self._last_state_before = None
                        self._last_best_action = None
                        logger.info(f"GAMEOVER: {thread_id=:02d} {turn=:04d} score={bd.score}")
                        break
        except Exception as e:
            logger.exception(e)
            self.stop_event.set()

    def batch_trainer(self, pack: dict):
        """Combined value+policy training loop for multi-head models."""
        train_count = 0
        records: list[dict] = []

        while not self.stop_event.is_set():
            train_count += 1
            while len(records) != BAT_SIZE and not self.stop_event.is_set():
                records.append(pack["queue"].get())

            model = pack["model"]
            optimizer = pack["optimizer"]
            batch_size = len(records)

            states_t = torch.zeros(batch_size, 99, device="cpu")
            next_states = torch.zeros(batch_size, 99, device="cpu")
            for i, r in enumerate(records):
                write_make_input(r["state"], states_t[i])
                write_make_input(r["next_state"], next_states[i])
            states_t = states_t.to(DEVICE)
            next_states = next_states.to(DEVICE)

            actions = torch.tensor([r["action"] for r in records], dtype=torch.long, device=DEVICE)
            is_terminal = torch.tensor([r["is_terminal"] for r in records], dtype=torch.float32, device=DEVICE)

            # Compute TD targets: V(s_{t+1}), zeroed for terminal states
            model.eval()
            with torch.no_grad():
                next_values, _ = model.forward(next_states)
            td_targets = next_values.detach() * (1.0 - is_terminal.unsqueeze(1))

            # Train both heads
            model.train()
            optimizer.zero_grad()
            values, pi_logits = model.forward(states_t)

            value_loss = self.value_criterion(values, td_targets)
            policy_loss = self.policy_criterion(pi_logits, actions)
            loss = value_loss + policy_loss
            loss.backward()
            optimizer.step()
            logger.debug(f"multi-head loss={loss.item():.4f} value={value_loss.item():.4f} policy={policy_loss.item():.4f}")
            records.clear()

        return train_count

    def _train(self, records, pack, count=1):
        raise NotImplementedError("MultiHeadTrainer uses batch_trainer directly")
