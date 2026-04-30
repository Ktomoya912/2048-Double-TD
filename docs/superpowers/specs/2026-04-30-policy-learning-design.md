# Policy Learning Design

**Date:** 2026-04-30  
**Project:** 2048 Deep TD Learning (3×3 board)

---

## Overview

Existing system learns a Value function via TD-Afterstate learning. This spec adds Policy learning alongside Value learning. The Policy network is trained but **not used during play** — play continues to rely solely on the Value (Afterstate) function.

A new `--with_policy` flag activates Policy training. Without it, behavior is unchanged.

---

## Models

Three new model files in `src/models/`:

### 1. `CNN_DEEP_POLICY.py`
- Input: State before action (99-dim one-hot)
- Output: 4 logits (policy over 4 actions)
- Structure: identical to CNN_DEEP, only `fc2` changed from `Linear(chw5, 1)` to `Linear(chw5, 4)`
- Used with: `--model CNN_DEEP --with_policy` (separate from value network)

### 2. `CNN_DEEP_MULTI.py`
- Input: State before action (99-dim one-hot)
- Output: tuple `(value: scalar, pi_logits: 4-dim)`
- Structure: CNN_DEEP backbone shared, splits into value head `Linear(..., 1)` and policy head `Linear(..., 4)`
- Used with: `--model CNN_DEEP_MULTI --with_policy`

### 3. `ALPHA_ZERO_STATE.py`
- Input: State before action (99-dim one-hot)
- Output: tuple `(value: scalar, pi_logits: 4-dim)`
- Structure: ResNet backbone (similar to existing ALPHA_ZERO but with working value head), value head and policy head
- Fixes the existing ALPHA_ZERO bug where `value_head = nn.Sequential()` overwrites the real value head
- Used with: `--model ALPHA_ZERO_STATE --with_policy`

### Usage Matrix

| `--model` | `--with_policy` | Value training | Policy training |
|---|---|---|---|
| CNN_DEEP | no | CNN_DEEP (afterstate) | — |
| CNN_DEEP | yes | CNN_DEEP (afterstate, unchanged) | CNN_DEEP_POLICY (separate) |
| CNN_DEEP_MULTI | yes (required) | CNN_DEEP_MULTI value head (state) | CNN_DEEP_MULTI policy head |
| ALPHA_ZERO_STATE | yes (required) | ALPHA_ZERO_STATE value head (state) | ALPHA_ZERO_STATE policy head |

`CNN_DEEP_MULTI` and `ALPHA_ZERO_STATE` without `--with_policy` raise `ValueError`.

---

## Data Pipeline

### Policy Label
- **Type:** hard label (one-hot), CrossEntropyLoss
- **Source:** argmax over TD-Afterstate value evaluations for all 4 moves

### Queue Contents

**Separate model case (CNN_DEEP + CNN_DEEP_POLICY):**
- Existing value queue: `{board: afterstate, self_value, other_value}` — unchanged
- New policy queue: `{board: state_before_move, action: int}`

**Multi-head model case (CNN_DEEP_MULTI / ALPHA_ZERO_STATE):**
- Single queue: `{state: state_before_move, action: int, next_state: state_after_new_tile}`
- Value target: `V(next_state)` evaluated with `model.eval()` + `no_grad()` (TD(0))
- Game over: value target = 0

### Data Collection Change in `play_game()`

```python
# before _play():
state_before = bd.board.copy()

# inside _play() — return best_action in addition to existing logic
best_action = argmax(...)

# after _play():
if args.with_policy and best_action is not None:
    self.put_policy_queue(state_before, best_action, ...)
```

Game-over states: `put_policy_queue()` is skipped (no valid action).

### Symmetry Augmentation for Policy

When `--symmetry` is enabled, `put_policy_queue()` generates 8 augmented (board, action) pairs. The action label is transformed according to the board transformation:

| Augmentation | Action transformation |
|---|---|
| Original | A |
| CW × 1 | (A + 1) % 4 |
| CW × 2 | (A + 2) % 4 |
| CW × 3 | (A + 3) % 4 |
| Mirror(CW×3) | [0→1, 1→0, 2→3, 3→2] |
| CW×1 ∘ Mirror(CW×3) | [0→2, 1→1, 2→0, 3→3] |
| CW×2 ∘ Mirror(CW×3) | [0→3, 1→2, 2→1, 3→0] |
| CW×3 ∘ Mirror(CW×3) | [0→0, 1→3, 2→2, 3→1] |

Transformation table derived from `game_2048_3_3.py` action implementations (0=Up, 1=Right, 2=Down, 3=Left) and `rotate_board` (CW, k=-1) / `mirror_board` (np.fliplr).

---

## Training Infrastructure

### New Files

| File | Purpose |
|---|---|
| `src/models/CNN_DEEP_POLICY.py` | Separate policy model |
| `src/models/CNN_DEEP_MULTI.py` | Multi-head CNN_DEEP |
| `src/models/ALPHA_ZERO_STATE.py` | Multi-head ResNet |
| `src/trainer/policy_mixin.py` | `put_policy_queue()`, `policy_batch_trainer()` shared logic |
| `src/trainer/multi_head.py` | `MultiHeadTrainer` for CNN_DEEP_MULTI / ALPHA_ZERO_STATE |

### Modified Files

| File | Change |
|---|---|
| `src/trainer/common.py` | `_play()` captures state_before / best_action; calls put_policy_queue when with_policy |
| `src/trainer/__init__.py` | Creates policy_pack / policy_queue when with_policy; adds policy_batch_trainer thread |
| `src/common/args.py` | Adds `--with_policy` flag, `--load_policy` path argument |
| `src/common/config.py` | Creates `POLICY_NETWORK` when with_policy (for separate model case) |

### Trainer Class Hierarchy

```
Trainer (common.py)
  └─ PolicyMixin (policy_mixin.py)    ← new
       ├─ TDA_Trainer
       ├─ D_TDA_C_Trainer
       ├─ D_TDA_CB_Trainer
       └─ D_TDA_X_Trainer

MultiHeadTrainer (trainer/multi_head.py)   ← new, for CNN_DEEP_MULTI / ALPHA_ZERO_STATE
```

### Thread Layout (--with_policy, separate model)

```
ThreadPoolExecutor
  ├── play_game × N
  ├── value_batch_trainer       (existing)
  └── policy_batch_trainer      (new)
```

Multi-head model uses only `value_batch_trainer` (handles both heads, combined loss).

### Loss Functions

- Separate model: `MSELoss` (value), `CrossEntropyLoss` (policy)
- Multi-head model: `total_loss = MSELoss(value) + CrossEntropyLoss(policy)`

---

## Model Saving / Loading

- `save_models()` extended to save `POLICY_NETWORK` as `..._policy.pth` when `--with_policy`
- New `--load_policy <path>` argument for loading a pre-trained policy model
- Multi-head models save as a single file (both heads included)

---

## Error Handling

| Condition | Behavior |
|---|---|
| `--model CNN_DEEP_MULTI` without `--with_policy` | `ValueError` on startup |
| `--model ALPHA_ZERO_STATE` without `--with_policy` | `ValueError` on startup |
| Game-over state reached | `put_policy_queue()` skipped |
| All moves invalid (`canmov` all False) | `put_policy_queue()` skipped |
| `--load_policy` path not found | Warning logged, random init used |
