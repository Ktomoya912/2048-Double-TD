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
    from game_2048_3_3 import State
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
