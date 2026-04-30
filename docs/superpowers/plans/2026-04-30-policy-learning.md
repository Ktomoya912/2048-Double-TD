# Policy Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** TD-Afterstate 価値学習と並行して、state → policy のPolicy学習（CNN_DEEP_POLICY・CNN_DEEP_MULTI・ALPHA_ZERO_STATE）を `--with_policy` フラグで有効化する。

**Architecture:** 分離型（CNN_DEEP + CNN_DEEP_POLICY）では既存の価値学習を変更せず、別スレッドでCrossEntropyベースのPolicy学習を追加する。マルチヘッド型（CNN_DEEP_MULTI・ALPHA_ZERO_STATE）では、State入力の共有バックボーンからValueヘッド（TD(0)）とPolicyヘッド（CrossEntropy）を同時学習する。

**Tech Stack:** PyTorch, Python 3.11+, threading, numpy

---

## ファイルマップ

| 操作 | ファイル | 役割 |
|---|---|---|
| 新規 | `tests/conftest.py` | sys.path設定 |
| 新規 | `tests/test_models_policy.py` | モデル出力形状テスト |
| 新規 | `tests/test_action_transform.py` | 行動変換テーブルテスト |
| 新規 | `src/models/CNN_DEEP_POLICY.py` | 分離型Policyモデル |
| 新規 | `src/models/CNN_DEEP_MULTI.py` | マルチヘッド（CNN_DEEP系） |
| 新規 | `src/models/ALPHA_ZERO_STATE.py` | マルチヘッド（ResNet系） |
| 新規 | `src/trainer/policy_mixin.py` | put_policy_queue・policy_batch_trainer |
| 新規 | `src/trainer/multi_head.py` | MultiHeadTrainer |
| 変更 | `src/common/args.py` | `--with_policy`, `--load_policy` 追加 |
| 変更 | `src/common/config.py` | `POLICY_NETWORK`, 検証ロジック追加 |
| 変更 | `src/common/utils.py` | `ACTION_TRANSFORM` テーブル追加 |
| 変更 | `src/trainer/common.py` | `_play()` に state_before / best_action 捕捉追加 |
| 変更 | `src/trainer/TDA.py` | PolicyMixin継承追加 |
| 変更 | `src/trainer/D_TDA_C.py` | PolicyMixin継承追加 |
| 変更 | `src/trainer/D_TDA_CB.py` | PolicyMixin継承追加 |
| 変更 | `src/trainer/D_TDA_X.py` | PolicyMixin継承追加 |
| 変更 | `src/trainer/__init__.py` | policy_pack・MultiHeadTrainer・保存ロジック追加 |

---

## Task 1: テストインフラ + CNN_DEEP_POLICY モデル

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_models_policy.py`
- Create: `src/models/CNN_DEEP_POLICY.py`

- [ ] **Step 1.1: conftest.py を作成してsys.pathを設定**

```python
# tests/conftest.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

- [ ] **Step 1.2: CNN_DEEP_POLICY のテストを書く**

```python
# tests/test_models_policy.py
import torch
import pytest


def test_cnn_deep_policy_output_shape():
    from models.CNN_DEEP_POLICY import Model
    model = Model()
    x = torch.zeros(4, 99)
    out = model(x)
    assert out.shape == (4, 4), f"Expected (4,4), got {out.shape}"


def test_cnn_deep_policy_output_is_logits():
    """Output should be raw logits (no softmax), allowing CrossEntropyLoss."""
    from models.CNN_DEEP_POLICY import Model
    model = Model()
    x = torch.randn(2, 99)
    out = model(x)
    # Raw logits can be any float, not bounded to [0,1]
    assert out.dtype == torch.float32


def test_cnn_deep_multi_output_shape():
    from models.CNN_DEEP_MULTI import Model
    model = Model()
    x = torch.zeros(4, 99)
    value, pi = model(x)
    assert value.shape == (4, 1), f"Expected (4,1), got {value.shape}"
    assert pi.shape == (4, 4), f"Expected (4,4), got {pi.shape}"


def test_alpha_zero_state_output_shape():
    from models.ALPHA_ZERO_STATE import Model
    model = Model()
    x = torch.zeros(2, 99)
    value, pi = model(x)
    assert value.shape == (2, 1), f"Expected (2,1), got {value.shape}"
    assert pi.shape == (2, 4), f"Expected (2,4), got {pi.shape}"
```

- [ ] **Step 1.3: テストを実行して FAIL を確認**

```bash
cd C:\Users\iceto\Documents\research\2048-Double-TD
.venv\Scripts\python -m pytest tests/test_models_policy.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'models.CNN_DEEP_POLICY'`

- [ ] **Step 1.4: CNN_DEEP_POLICY を実装**

```python
# src/models/CNN_DEEP_POLICY.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        chw0, chw1, chw2, chw3, chw4, chw5 = 64, 128, 232, 256, 256, 256
        self.conv0 = nn.Conv2d(11, chw0, kernel_size=1)
        self.conv1 = nn.Conv2d(chw0, chw1, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(chw1, chw2, kernel_size=2)
        self.conv3 = nn.Conv2d(chw2, chw3, kernel_size=2)
        self.conv4 = nn.Conv2d(chw3, chw4, kernel_size=2)
        self.fc1 = nn.Linear(chw4, chw5)
        self.fc2 = nn.Linear(chw5, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 11, 3, 3)
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

- [ ] **Step 1.5: テストを実行して CNN_DEEP_POLICY のテストが PASS することを確認**

```bash
.venv\Scripts\python -m pytest tests/test_models_policy.py::test_cnn_deep_policy_output_shape tests/test_models_policy.py::test_cnn_deep_policy_output_is_logits -v
```

Expected: 2 passed

- [ ] **Step 1.6: コミット**

```bash
git add tests/conftest.py tests/test_models_policy.py src/models/CNN_DEEP_POLICY.py
git commit -m "feat: CNN_DEEP_POLICYモデルを追加"
```

---

## Task 2: CNN_DEEP_MULTI モデル

**Files:**
- Create: `src/models/CNN_DEEP_MULTI.py`

- [ ] **Step 2.1: テストを実行して FAIL を確認（Task 1 で書き済み）**

```bash
.venv\Scripts\python -m pytest tests/test_models_policy.py::test_cnn_deep_multi_output_shape -v
```

Expected: `ModuleNotFoundError: No module named 'models.CNN_DEEP_MULTI'`

- [ ] **Step 2.2: CNN_DEEP_MULTI を実装**

```python
# src/models/CNN_DEEP_MULTI.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """CNN_DEEP backbone with shared features, value head and policy head.
    Input: 99-dim state (before action). Output: (value [N,1], pi_logits [N,4]).
    """

    def __init__(self):
        super().__init__()
        chw0, chw1, chw2, chw3, chw4, chw5 = 64, 128, 232, 256, 256, 256
        self.conv0 = nn.Conv2d(11, chw0, kernel_size=1)
        self.conv1 = nn.Conv2d(chw0, chw1, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(chw1, chw2, kernel_size=2)
        self.conv3 = nn.Conv2d(chw2, chw3, kernel_size=2)
        self.conv4 = nn.Conv2d(chw3, chw4, kernel_size=2)
        self.fc1 = nn.Linear(chw4, chw5)
        self.value_head = nn.Linear(chw5, 1)
        self.policy_head = nn.Linear(chw5, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1, 11, 3, 3)
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 256)
        features = F.relu(self.fc1(x))
        return self.value_head(features), self.policy_head(features)
```

- [ ] **Step 2.3: テストを実行して PASS を確認**

```bash
.venv\Scripts\python -m pytest tests/test_models_policy.py::test_cnn_deep_multi_output_shape -v
```

Expected: 1 passed

- [ ] **Step 2.4: コミット**

```bash
git add src/models/CNN_DEEP_MULTI.py
git commit -m "feat: CNN_DEEP_MULTIマルチヘッドモデルを追加"
```

---

## Task 3: ALPHA_ZERO_STATE モデル

**Files:**
- Create: `src/models/ALPHA_ZERO_STATE.py`

- [ ] **Step 3.1: テストを実行して FAIL を確認（Task 1 で書き済み）**

```bash
.venv\Scripts\python -m pytest tests/test_models_policy.py::test_alpha_zero_state_output_shape -v
```

Expected: `ModuleNotFoundError: No module named 'models.ALPHA_ZERO_STATE'`

- [ ] **Step 3.2: ALPHA_ZERO_STATE を実装**

```python
# src/models/ALPHA_ZERO_STATE.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, num_filters: int) -> None:
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        return F.relu(out)


class Model(nn.Module):
    """ResNet backbone with value head and policy head.
    Input: 99-dim state (before action). Output: (value [N,1], pi_logits [N,4]).
    Fixes the value_head bug in the original ALPHA_ZERO where value_head was overwritten.
    """

    def __init__(self, num_res_block: int = 19, num_filters: int = 128) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(11, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*[ResNetBlock(num_filters) for _ in range(num_res_block)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 3 * 3, 4),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 3 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1, 11, 3, 3)
        features = self.res_blocks(self.conv_block(x))
        return self.value_head(features), self.policy_head(features)
```

- [ ] **Step 3.3: テストをすべて実行して PASS を確認**

```bash
.venv\Scripts\python -m pytest tests/test_models_policy.py -v
```

Expected: 4 passed

- [ ] **Step 3.4: コミット**

```bash
git add src/models/ALPHA_ZERO_STATE.py
git commit -m "feat: ALPHA_ZERO_STATEマルチヘッドモデルを追加"
```

---

## Task 4: 行動変換テーブルを utils.py に追加

**Files:**
- Create: `tests/test_action_transform.py`
- Modify: `src/common/utils.py`

- [ ] **Step 4.1: テストを書く**

```python
# tests/test_action_transform.py
"""
Verify action transformation table for 8-way symmetry augmentation.
Actions: 0=Up, 1=Right, 2=Down, 3=Left
Symmetry sequence (matches put_queue in Trainer):
  0=original, 1=CW1, 2=CW2, 3=CW3,
  4=M(CW3), 5=CW∘M(CW3), 6=CW2∘M(CW3), 7=CW3∘M(CW3)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from common.utils import ACTION_TRANSFORM, rotate_board, mirror_board
from game_2048_3_3 import State



def test_action_transform_table_has_8_rows():
    from common.utils import ACTION_TRANSFORM
    assert len(ACTION_TRANSFORM) == 8
    for row in ACTION_TRANSFORM:
        assert len(row) == 4


def test_action_transform_original_is_identity():
    from common.utils import ACTION_TRANSFORM
    assert ACTION_TRANSFORM[0] == [0, 1, 2, 3]


def test_action_transform_cw1_shifts_by_1():
    from common.utils import ACTION_TRANSFORM
    assert ACTION_TRANSFORM[1] == [1, 2, 3, 0]


def test_action_transform_cw2_shifts_by_2():
    from common.utils import ACTION_TRANSFORM
    assert ACTION_TRANSFORM[2] == [2, 3, 0, 1]


def test_action_transform_cw3_shifts_by_3():
    from common.utils import ACTION_TRANSFORM
    assert ACTION_TRANSFORM[3] == [3, 0, 1, 2]


def test_action_transform_mirror_swaps_left_right():
    from common.utils import ACTION_TRANSFORM
    row = ACTION_TRANSFORM[4]
    # Mirror(CW3): Up↔Right swap, Down↔Left swap
    assert row[0] == 1  # Up→Right
    assert row[1] == 0  # Right→Up
    assert row[2] == 3  # Down→Left
    assert row[3] == 2  # Left→Down


def test_action_transform_all_rows_are_permutations():
    from common.utils import ACTION_TRANSFORM
    for row in ACTION_TRANSFORM:
        assert sorted(row) == [0, 1, 2, 3], f"Row {row} is not a permutation of 0-3"


def test_action_transform_verified_with_game():
    """Verify CW rotation mapping using actual game state."""
    from common.utils import ACTION_TRANSFORM, rotate_board, write_make_input
    import numpy as np

    # Board where only Left (3) is a valid non-trivial move:
    # 0 0 1
    # 0 0 2
    # 0 0 3
    board = np.array([0, 0, 1, 0, 0, 2, 0, 0, 3], dtype="int64")
    original_action = 3  # Left

    # Apply Left to original board
    bd_orig = State(board.copy(), 0)
    assert bd_orig.canMoveTo(3)
    bd_orig.play(3)
    result_orig = bd_orig.board.copy()
    # Expected: [1,0,0, 2,0,0, 3,0,0]
    assert result_orig[0] == 1 and result_orig[3] == 2 and result_orig[6] == 3

    # Apply CW rotation, then ACTION_TRANSFORM[1][3]=0 (Up)
    rotated = rotate_board(board)
    transformed_action = ACTION_TRANSFORM[1][original_action]  # should be 0 (Up)
    assert transformed_action == 0

    bd_rot = State(rotated.copy(), 0)
    assert bd_rot.canMoveTo(transformed_action)
    bd_rot.play(transformed_action)
    result_rotated = bd_rot.board.copy()

    # Rotate the original result and compare
    expected = rotate_board(result_orig)
    np.testing.assert_array_equal(result_rotated, expected)
```

- [ ] **Step 4.2: テストを実行して FAIL を確認**

```bash
.venv\Scripts\python -m pytest tests/test_action_transform.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'ACTION_TRANSFORM' from 'common.utils'`

- [ ] **Step 4.3: ACTION_TRANSFORM を utils.py に追加**

`src/common/utils.py` の末尾に以下を追記（既存コードはそのまま）:

```python
# ACTION_TRANSFORM[symmetry_idx][original_action] = transformed_action
# Sequence matches put_queue() in Trainer:
#   0=original, 1=CW1, 2=CW2, 3=CW3,
#   4=M(CW3), 5=CW∘M(CW3), 6=CW2∘M(CW3), 7=CW3∘M(CW3)
# Derived from game_2048_3_3.py: 0=Up,1=Right,2=Down,3=Left
# CW rotation: action → (action+1)%4; LR mirror: Up↔Up, Right↔Left, Down↔Down
ACTION_TRANSFORM: list[list[int]] = [
    [0, 1, 2, 3],  # original
    [1, 2, 3, 0],  # CW×1
    [2, 3, 0, 1],  # CW×2
    [3, 0, 1, 2],  # CW×3
    [1, 0, 3, 2],  # Mirror(CW×3)
    [2, 1, 0, 3],  # CW×1 ∘ Mirror(CW×3)
    [3, 2, 1, 0],  # CW×2 ∘ Mirror(CW×3)
    [0, 3, 2, 1],  # CW×3 ∘ Mirror(CW×3)
]
```

- [ ] **Step 4.4: テストを実行して PASS を確認**

```bash
.venv\Scripts\python -m pytest tests/test_action_transform.py -v
```

Expected: 8 passed

- [ ] **Step 4.5: コミット**

```bash
git add src/common/utils.py tests/test_action_transform.py
git commit -m "feat: 対称性拡張用の行動変換テーブルをutils.pyに追加"
```

---

## Task 5: args.py / config.py に --with_policy を追加

**Files:**
- Modify: `src/common/args.py`
- Modify: `src/common/config.py`

- [ ] **Step 5.1: args.py に2つの引数を追加**

`src/common/args.py` の `parser.add_argument("--load_target", ...)` の後に追加:

```python
parser.add_argument(
    "--with_policy",
    action="store_true",
    help="Policyモデルの同時学習を有効にする",
)
parser.add_argument(
    "--load_policy",
    type=Path,
    help="Policyモデルのパス（--with_policy時のみ有効）",
)
```

- [ ] **Step 5.2: config.py に POLICY_NETWORK とバリデーションを追加**

`src/common/config.py` の末尾（`if args.load_target:` ブロックの後）に追加:

```python
MULTI_HEAD_MODELS = {"CNN_DEEP_MULTI", "ALPHA_ZERO_STATE"}
POLICY_SUPPORTED_MODELS = {"CNN_DEEP", "CNN_DEEP_MULTI", "ALPHA_ZERO_STATE"}

if args.model in MULTI_HEAD_MODELS and not args.with_policy:
    raise ValueError(
        f"--model {args.model} はマルチヘッドモデルです。--with_policy フラグが必要です。"
    )

POLICY_NETWORK: torch.nn.Module | None = None
if args.with_policy:
    if args.model not in POLICY_SUPPORTED_MODELS:
        raise ValueError(
            f"--with_policy は {POLICY_SUPPORTED_MODELS} のみ対応しています。"
            f" 指定されたモデル: {args.model}"
        )
    if args.model not in MULTI_HEAD_MODELS:
        exec("from models import CNN_DEEP_POLICY as policy_modeler")
        POLICY_NETWORK = policy_modeler.Model().to(DEVICE)  # noqa: F821
        logger.info("POLICY_NETWORK (CNN_DEEP_POLICY) initialized.")
        if args.load_policy:
            if data := get_trained_model(args.load_policy, DEVICE):
                POLICY_NETWORK.load_state_dict(data)
```

- [ ] **Step 5.3: 動作確認（--with_policy なしで既存モデルが通ることを確認）**

```bash
cd C:\Users\iceto\Documents\research\2048-Double-TD\src
..\venv\Scripts\python -c "import sys; sys.argv=['x','--model','CNN_DEEP','--trainer','TDA']; from common import config; print('OK')"
```

Expected: `OK`（エラーなし）

- [ ] **Step 5.4: コミット**

```bash
git add src/common/args.py src/common/config.py
git commit -m "feat: --with_policy / --load_policy 引数とPOLICY_NETWORKを追加"
```

---

## Task 6: PolicyMixin を実装

**Files:**
- Create: `src/trainer/policy_mixin.py`

- [ ] **Step 6.1: policy_mixin.py を実装**

```python
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
```

- [ ] **Step 6.2: PolicyMixin の単体テストを書いて実行**

```bash
.venv\Scripts\python -c "
import sys; sys.path.insert(0, 'src')
import sys; sys.argv=['x','--model','CNN_DEEP','--trainer','TDA']
import numpy as np
from queue import Queue
from trainer.policy_mixin import PolicyMixin

class MockTrainer(PolicyMixin):
    def __init__(self):
        self.policy_pack = {'queue': Queue(100), 'model': None, 'optimizer': None}
        import threading
        self.stop_event = threading.Event()

t = MockTrainer()

# No symmetry test
import argparse
from common import args as args_mod
args_mod.args.symmetry = False
board = np.zeros(9, dtype='int64')
t.put_policy_queue(board, 2)
assert t.policy_pack['queue'].qsize() == 1
rec = t.policy_pack['queue'].get()
assert rec['action'] == 2
print('PolicyMixin no-symmetry: OK')

# Symmetry test
args_mod.args.symmetry = True
t.put_policy_queue(board, 0)
assert t.policy_pack['queue'].qsize() == 8, f'Expected 8, got {t.policy_pack[\"queue\"].qsize()}'
actions = [t.policy_pack['queue'].get()['action'] for _ in range(8)]
assert actions[0] == 0  # original
assert actions[1] == 1  # CW1: Up→Right
print('PolicyMixin symmetry: OK')
"
```

Expected: `PolicyMixin no-symmetry: OK` and `PolicyMixin symmetry: OK`

- [ ] **Step 6.3: コミット**

```bash
git add src/trainer/policy_mixin.py
git commit -m "feat: PolicyMixinを追加（put_policy_queue・policy_batch_trainer）"
```

---

## Task 7: 既存Trainerの _play() に state_before / best_action を捕捉

**Files:**
- Modify: `src/trainer/common.py`
- Modify: `src/trainer/TDA.py`
- Modify: `src/trainer/D_TDA_C.py`
- Modify: `src/trainer/D_TDA_CB.py`
- Modify: `src/trainer/D_TDA_X.py`

- [ ] **Step 7.1: Trainer.__init__ に policy_pack パラメータを追加し、_play() を変更**

`src/trainer/common.py` の `Trainer` クラスを以下のように変更:

```python
class Trainer:
    def __init__(self, packs, policy_pack=None):
        self.stop_event = threading.Event()
        self.criterion = nn.MSELoss()
        self.packs: list[dict] = packs
        self.policy_pack = policy_pack  # None when --with_policy is not set
```

同ファイルの `Trainer._play()` メソッド全体を置き換え:

```python
def _play(self, packs, canmov, bd: State, last_board):
    state_before = bd.board.copy()
    self_values, other_values = get_values(canmov, bd.clone(), packs)
    self_max_index = np.argmax(self_values)
    if args.consensus:
        values = np.array(self_values) + np.array(other_values)
        sample_idx = np.argmax(values)
        bd.play(sample_idx)
        best_action = int(sample_idx)
    else:
        bd.play(self_max_index)
        best_action = int(self_max_index)
    if last_board is not None:
        self.put_queue(
            last_board.copy(),
            self_value=self_values[self_max_index],
            other_value=other_values[self_max_index],
            packs=packs,
        )
    if self.policy_pack is not None and any(canmov):
        self.put_policy_queue(state_before, best_action)
```

- [ ] **Step 7.2: TDA_Trainer._play() を変更**

`src/trainer/TDA.py` の `_play()` メソッド全体を置き換え:

```python
def _play(self, packs, canmov, bd: State, last_board):
    state_before = bd.board.copy()
    main_values = get_one_values(canmov, bd.clone(), packs[0]["model"])
    main_max_index = np.argmax(main_values)
    bd.play(main_max_index)
    if last_board is not None:
        self.put_queue(
            last_board.copy(),
            self_value=main_values[main_max_index],
            other_value=torch.tensor(0),
            packs=packs,
        )
    if self.policy_pack is not None and any(canmov):
        self.put_policy_queue(state_before, int(main_max_index))
```

`src/trainer/TDA.py` の import の末尾に追加:

```python
from .policy_mixin import PolicyMixin
```

`TDA_Trainer` クラス定義を変更:

```python
class TDA_Trainer(PolicyMixin, Trainer):
```

- [ ] **Step 7.3: D_TDA_C, D_TDA_CB, D_TDA_X に PolicyMixin を継承追加**

`src/trainer/D_TDA_C.py` の先頭 import に追加し、クラス定義を変更:

```python
from .policy_mixin import PolicyMixin

# クラス定義を変更
class D_TDA_C_Trainer(PolicyMixin, Trainer):
```

`src/trainer/D_TDA_CB.py` も同様:

```python
from .policy_mixin import PolicyMixin

class D_TDA_CB_Trainer(PolicyMixin, Trainer):
```

`src/trainer/D_TDA_X.py` も同様:

```python
from .policy_mixin import PolicyMixin

class D_TDA_X_Trainer(PolicyMixin, Trainer):
```

- [ ] **Step 7.4: 動作確認（既存コードが壊れていないことを確認）**

```bash
cd C:\Users\iceto\Documents\research\2048-Double-TD\src
..\venv\Scripts\python -c "
import sys; sys.argv=['x','--model','CNN_DEEP','--trainer','TDA']
from trainer.TDA import TDA_Trainer
from trainer.D_TDA_C import D_TDA_C_Trainer
from trainer.D_TDA_CB import D_TDA_CB_Trainer
from trainer.D_TDA_X import D_TDA_X_Trainer
from trainer.policy_mixin import PolicyMixin
assert issubclass(TDA_Trainer, PolicyMixin)
assert issubclass(D_TDA_C_Trainer, PolicyMixin)
print('Mixin inheritance: OK')
"
```

Expected: `Mixin inheritance: OK`

- [ ] **Step 7.5: コミット**

```bash
git add src/trainer/common.py src/trainer/TDA.py src/trainer/D_TDA_C.py src/trainer/D_TDA_CB.py src/trainer/D_TDA_X.py
git commit -m "feat: 既存TrainerにPolicyMixin継承とstate_before/best_action捕捉を追加"
```

---

## Task 8: MultiHeadTrainer を実装

**Files:**
- Create: `src/trainer/multi_head.py`

- [ ] **Step 8.1: multi_head.py を実装**

```python
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
```

- [ ] **Step 8.2: MultiHeadTrainer のインポートが通ることを確認**

```bash
cd C:\Users\iceto\Documents\research\2048-Double-TD\src
..\venv\Scripts\python -c "
import sys; sys.argv=['x','--model','CNN_DEEP_MULTI','--trainer','TDA','--with_policy']
from trainer.multi_head import MultiHeadTrainer
print('MultiHeadTrainer import: OK')
"
```

Expected: `MultiHeadTrainer import: OK`

- [ ] **Step 8.3: コミット**

```bash
git add src/trainer/multi_head.py
git commit -m "feat: MultiHeadTrainerを実装（Value+Policy同時学習）"
```

---

## Task 9: trainer/__init__.py を更新

**Files:**
- Modify: `src/trainer/__init__.py`

- [ ] **Step 9.1: trainer/__init__.py 全体を書き換え**

既存の `src/trainer/__init__.py` を以下に置き換え（既存の内容をベースに拡張）:

```python
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue

import numpy as np
import torch
from torch import optim

from common.args import args
from common.config import (
    LOG_PATH,
    MAIN_NETWORK,
    MODEL_DIR,
    MULTI_HEAD_MODELS,
    POLICY_NETWORK,
    TARGET_NETWORK,
    TIME_LIMIT,
)

from .common import Trainer
from .D_TDA_C import D_TDA_C_Trainer
from .D_TDA_CB import D_TDA_CB_Trainer
from .D_TDA_X import D_TDA_X_Trainer
from .TDA import TDA_Trainer
from .multi_head import MultiHeadTrainer

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

logger = logging.getLogger(__name__)
tasks = min(os.cpu_count() - 2, 6)

optimizer_main = optim.Adam(MAIN_NETWORK.parameters(), lr=0.001)
optimizer_target = optim.Adam(TARGET_NETWORK.parameters(), lr=0.001)
pack_main = {
    "model": MAIN_NETWORK,
    "optimizer": optimizer_main,
    "name": "main",
    "queue": Queue(tasks * 2),
}
pack_target = {
    "model": TARGET_NETWORK,
    "optimizer": optimizer_target,
    "name": "target",
    "queue": Queue(tasks * 2),
}

policy_pack = None
if args.with_policy and POLICY_NETWORK is not None:
    optimizer_policy = optim.Adam(POLICY_NETWORK.parameters(), lr=0.001)
    policy_pack = {
        "model": POLICY_NETWORK,
        "optimizer": optimizer_policy,
        "name": "policy",
        "queue": Queue(tasks * 2),
    }


def clear_queues():
    for pack in [pack_main, pack_target, policy_pack]:
        if pack is None:
            continue
        while pack["queue"].qsize() > 0:
            try:
                pack["queue"].get_nowait()
            except Exception:
                break
    logger.info("Queues cleared.")


def save_models(save_count: int = -1):
    main_path = MODEL_DIR / f"{LOG_PATH.stem}_{save_count:02d}_main.pth"
    target_path = MODEL_DIR / f"{LOG_PATH.stem}_{save_count:02d}_target.pth"
    torch.save(MAIN_NETWORK.state_dict(), main_path)
    logger.info(f"saved {main_path.name}")
    torch.save(TARGET_NETWORK.state_dict(), target_path)
    logger.info(f"saved {target_path.name}")
    if policy_pack is not None:
        policy_path = MODEL_DIR / f"{LOG_PATH.stem}_{save_count:02d}_policy.pth"
        torch.save(POLICY_NETWORK.state_dict(), policy_path)
        logger.info(f"saved {policy_path.name}")


def submit_batches(trainer: Trainer, executor: ThreadPoolExecutor):
    for i in range(tasks):
        executor.submit(trainer.play_game, i)
    executor.submit(trainer.batch_trainer, pack_main)
    if args.trainer == "D_TDA_X":
        executor.submit(trainer.batch_trainer, pack_target)
    if policy_pack is not None:
        executor.submit(trainer.policy_batch_trainer, policy_pack)
    return executor


def main():
    try:
        packs = [pack_main, pack_target]
        is_multi_head = args.model in MULTI_HEAD_MODELS

        if is_multi_head:
            trainer = MultiHeadTrainer(packs)
        elif args.trainer == "D_TDA_C":
            trainer = D_TDA_C_Trainer(packs, policy_pack=policy_pack)
        elif args.trainer == "D_TDA_CB":
            trainer = D_TDA_CB_Trainer(packs, policy_pack=policy_pack)
        elif args.trainer == "TDA":
            packs = [pack_main]
            trainer = TDA_Trainer(packs, policy_pack=policy_pack)
        elif args.trainer == "D_TDA_X":
            trainer = D_TDA_X_Trainer(packs, policy_pack=policy_pack)
        else:
            raise ValueError(f"Unknown trainer type: {args.trainer}")

        worker_count = tasks + (2 if args.trainer == "D_TDA_X" else 1)
        if policy_pack is not None:
            worker_count += 1
        executor = ThreadPoolExecutor(max_workers=worker_count)
        executor = submit_batches(trainer, executor)

        start_time = datetime.now()
        save_count = 0
        last_save_time = start_time
        save_interval = timedelta(hours=24)
        while (
            datetime.now() - start_time < TIME_LIMIT and not trainer.stop_event.is_set()
        ):
            if TIME_LIMIT.total_seconds() >= save_interval.total_seconds():
                if datetime.now() - last_save_time >= save_interval:
                    save_count += 1
                    save_models(save_count)
                    last_save_time = datetime.now()
            time.sleep(1)

        trainer.stop_event.set()
        save_models()
        clear_queues()
        logger.info("All threads have been successfully terminated.")
    except Exception as e:
        logger.exception(e)
        trainer.stop_event.set()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, stopping threads...")
        trainer.stop_event.set()
    finally:
        trainer.stop_event.set()
        clear_queues()
        executor.shutdown(wait=True)
        logger.info("All threads have been successfully terminated.")
        if args.play_after_train:
            from play import main as play_main
            play_main()
```

- [ ] **Step 9.2: MULTI_HEAD_MODELS を config.py から正しくインポートできることを確認**

`src/common/config.py` で `MULTI_HEAD_MODELS` が定義されていることを確認。定義されていない場合は Task 5 で追加済みのはず。確認コマンド:

```bash
cd C:\Users\iceto\Documents\research\2048-Double-TD\src
..\venv\Scripts\python -c "
import sys; sys.argv=['x','--model','CNN_DEEP','--trainer','TDA']
from common.config import MULTI_HEAD_MODELS, POLICY_NETWORK
print('MULTI_HEAD_MODELS:', MULTI_HEAD_MODELS)
print('POLICY_NETWORK:', POLICY_NETWORK)
print('import OK')
"
```

Expected: `MULTI_HEAD_MODELS: {'CNN_DEEP_MULTI', 'ALPHA_ZERO_STATE'}`, `POLICY_NETWORK: None`

- [ ] **Step 9.3: コミット**

```bash
git add src/trainer/__init__.py
git commit -m "feat: trainer/__init__.pyをpolicy_pack・MultiHeadTrainer対応に更新"
```

---

## Task 10: 統合スモークテスト

- [ ] **Step 10.1: 既存モデル（--with_policy なし）が壊れていないことを確認**

```bash
cd C:\Users\iceto\Documents\research\2048-Double-TD\src
..\venv\Scripts\python train.py --model CNN_DEEP --trainer TDA --hours 0 --log INFO 2>&1 | head -20
```

Expected: ゲームが開始されログが流れる。エラーなし。5秒後に Ctrl+C で停止。

- [ ] **Step 10.2: 分離型 Policy 学習（CNN_DEEP + CNN_DEEP_POLICY）のスモークテスト**

```bash
cd C:\Users\iceto\Documents\research\2048-Double-TD\src
..\venv\Scripts\python train.py --model CNN_DEEP --trainer TDA --with_policy --hours 0 --log DEBUG 2>&1 | head -40
```

Expected: 
- `POLICY_NETWORK (CNN_DEEP_POLICY) initialized.` がログに表示される
- `policy loss:` がログに表示される
- エラーなし

- [ ] **Step 10.3: CNN_DEEP_MULTI マルチヘッドのスモークテスト**

```bash
cd C:\Users\iceto\Documents\research\2048-Double-TD\src
..\venv\Scripts\python train.py --model CNN_DEEP_MULTI --trainer TDA --with_policy --hours 0 --log DEBUG 2>&1 | head -40
```

Expected:
- `multi-head loss=` がログに表示される
- エラーなし

- [ ] **Step 10.4: 対称性あり + Policy のスモークテスト**

```bash
cd C:\Users\iceto\Documents\research\2048-Double-TD\src
..\venv\Scripts\python train.py --model CNN_DEEP --trainer TDA --with_policy --symmetry --hours 0 --log DEBUG 2>&1 | head -40
```

Expected: `policy loss:` が表示される。エラーなし。

- [ ] **Step 10.5: バリデーションエラーのテスト**

```bash
cd C:\Users\iceto\Documents\research\2048-Double-TD\src
..\venv\Scripts\python train.py --model CNN_DEEP_MULTI --trainer TDA --hours 0 2>&1 | head -5
```

Expected: `ValueError: --model CNN_DEEP_MULTI はマルチヘッドモデルです。--with_policy フラグが必要です。`

- [ ] **Step 10.6: 全テストを実行**

```bash
cd C:\Users\iceto\Documents\research\2048-Double-TD
.venv\Scripts\python -m pytest tests/ -v
```

Expected: 12 passed (または同等)

- [ ] **Step 10.7: 最終コミット**

```bash
git add -A
git commit -m "feat: Policy学習機能を実装（CNN_DEEP_POLICY・CNN_DEEP_MULTI・ALPHA_ZERO_STATE）"
```
