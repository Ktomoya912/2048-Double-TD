import logging

import numpy as np
import torch

from game_2048_3_3 import State

from .config import DEVICE

logger = logging.getLogger(__name__)


# デバッグ用の関数
def board_print(bd: State, level=logging.DEBUG):
    for i in range(3):
        logger.log(
            level, f"{bd.board[i * 3 + 0]} {bd.board[i * 3 + 1]} {bd.board[i * 3 + 2]}"
        )


def rotate_board(board: np.ndarray):
    """
    盤面を90度回転させる（NumPy操作で最適化）
    """
    board_2d = board.reshape(3, 3)
    rotated_2d = np.rot90(board_2d, k=-1)  # 時計回りに90度回転
    return rotated_2d.flatten()


def mirror_board(board: np.ndarray):
    """
    盤面を左右反転させる（NumPy操作で最適化）
    """
    board_2d = board.reshape(3, 3)
    mirrored_2d = np.fliplr(board_2d)  # 左右反転
    return mirrored_2d.flatten()


def get_eval(board: np.ndarray, model: torch.nn.Module):
    x = torch.zeros(1, 99, device=DEVICE)
    write_make_input(board, x[0, :])
    model.eval()  # モデルを評価モードに設定
    eval = model.forward(x)
    return eval.item()


def save_models(save_count: int = -1):
    # torch.save(TARGET_NETWORK.state_dict(), target_model_path)
    # logger.info(f"save {target_model_path.name} {save_count=}")
    pass


def calc_progress(board: np.ndarray):
    """
    ボード状態から進捗度（progress）を計算
    各タイルの値の2の累乗和を2で割った値を返す。
    """
    return sum(2**i for i in board if i > 0) // 2


def write_make_input(board: np.ndarray, x: torch.Tensor):
    for j in range(9):
        x[board[j] * 9 + j] = 1


def get_values(canmov: list[bool], bd: State, packs: list):
    """移動可能な方向の評価値を計算する。
    移動可能な方向に対して、評価値を計算し、最大の評価値とその方向を返す。

    Args:
        canmov (list[bool]): 移動可能な方向のリスト
        bd (State): ゲームの状態を表すStateオブジェクト

    Returns:
        tuple: メインネットワークの評価値、ターゲットネットワークの評価値
    """
    packs[0]["model"].eval()
    packs[1]["model"].eval()
    inputs = torch.zeros(4, 99, device="cpu")
    sub_list = []
    for i in range(4):
        copy_bd = bd.clone()
        copy_bd.play(i)
        sub_list.append(copy_bd.score - bd.score)
        write_make_input(copy_bd.board, inputs[i, :])
    inputs = inputs.to(DEVICE)
    main_result: torch.Tensor = packs[0]["model"].forward(inputs)
    target_result: torch.Tensor = packs[1]["model"].forward(inputs)
    main_value = [-1e10] * 4
    target_value = [-1e10] * 4
    for i in range(4):
        if canmov[i]:
            main_value[i] = float(main_result.data[i]) + sub_list[i]
            target_value[i] = float(target_result.data[i]) + sub_list[i]

    return main_value, target_value


def get_one_values(canmov: list[bool], bd: State, model: torch.nn.Module):
    """移動可能な方向の評価値を計算する。
    移動可能な方向に対して、評価値を計算し、最大の評価値とその方向を返す。

    Args:
        canmov (list[bool]): 移動可能な方向のリスト
        bd (State): ゲームの状態を表すStateオブジェクト
        model (torch.nn.Module): 評価を行うニューラルネットワークモデル

    Returns:
        list: 各方向の評価値のリスト
    """
    model.eval()
    inputs = torch.zeros(4, 99, device="cpu")
    sub_list = []
    for i in range(4):
        copy_bd = bd.clone()
        copy_bd.play(i)
        sub_list.append(copy_bd.score - bd.score)
        write_make_input(copy_bd.board, inputs[i, :])
    inputs = inputs.to(DEVICE)
    result: torch.Tensor = model.forward(inputs)
    values = [-1e10] * 4
    for i in range(4):
        if canmov[i]:
            values[i] = float(result.data[i]) + sub_list[i]

    return values
