import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EvalAndHandProgress:
    evals: list[float]  # 長さ4のリスト
    prg: int  # progress

    @property
    def idx(self) -> list[int]:
        """
        evalsの最大値のindexのリストを返す。
        """
        max_eval = max(self.evals)
        return [i for i, eval in enumerate(self.evals) if eval == max_eval]


@dataclass
class PlotData:
    x_label: str
    y_label: str
    data: dict[str, "GraphData"]


@dataclass
class GraphData:
    x: list[float]
    y: list[float]


def get_eval_and_hand_progress(eval_file: Path):
    """
    ファイルから
    評価値、選択した手、progressを取得する。
    """
    eval_txt = eval_file.read_text("utf-8")
    subed_eval_txt = re.sub(r"game.*\n?", "", eval_txt)
    eval_lines = subed_eval_txt.splitlines()
    eval_and_hand_progress = [
        EvalAndHandProgress(
            evals=list(map(float, line.split()[:4])),
            prg=int(
                float(line.split()[4])
            ),  # progress を double(float)で受け取ってから int に変換
        )
        for line in eval_lines
    ]
    return eval_and_hand_progress


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")
