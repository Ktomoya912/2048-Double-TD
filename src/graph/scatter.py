import random
import re
from pathlib import Path

import matplotlib.pyplot as plt

from common import get_eval_and_hand_progress


def get_evals(eval_file: Path):
    eval_txt = eval_file.read_text("utf-8")
    subed_eval_txt = re.sub(r"game.*\n?", "", eval_txt)
    eval_lines = subed_eval_txt.splitlines()
    return [float(line) for line in eval_lines]


def plot_scatter(
    horizontal_files: list[Path],
    vertical_files: list[Path],
    output: Path,
    is_show: bool = True,
    config: dict = {},
):
    """
    縦軸は基準となるプレイ: vertical_files
    横軸はそのプレイを評価したプレイ: horizontal_files
    """
    for i, (vertical_file, horizontal_file) in enumerate(
        zip(sorted(vertical_files), sorted(horizontal_files))
    ):
        ver_eval_and_hand_progress = get_eval_and_hand_progress(vertical_file)
        hor_eval_and_hand_progress = get_eval_and_hand_progress(horizontal_file)

        assert len(ver_eval_and_hand_progress) == len(
            hor_eval_and_hand_progress
        ), f"データ数が異なります。{len(ver_eval_and_hand_progress)=}, {len(hor_eval_and_hand_progress)=}"

        scatter_data = [
            (ver_eval.evals[ver_eval.idx[0]], hor_eval.evals[ver_eval.idx[0]])
            for ver_eval, hor_eval in zip(
                ver_eval_and_hand_progress, hor_eval_and_hand_progress
            )
        ]
        # 1000個のデータをランダムで取得
        scatter_data = random.sample(scatter_data, 1000)

        # 散布図のdotの大きさを指定
        plt.scatter(
            [d[0] for d in scatter_data],
            [d[1] for d in scatter_data],
            s=5,
        )
        # 直線を引く
        plt.plot(
            [0, 6000],
            [0, 6000],
            color="gray",
            linestyle="dashed",
        )
        plt.xlabel(
            config.get(vertical_file.parent.name, {}).get(
                "label", vertical_file.parent.name
            )
        )
        plt.ylabel(
            config.get(horizontal_file.parent.name, {}).get(
                "label", horizontal_file.parent.name
            )
        )
        plt.tight_layout()
        plt.savefig(output.with_stem(f"{output.stem}_{horizontal_file.parent.name}"))
        if is_show:
            plt.show()
        plt.close()
    return None


if __name__ == "__main__":
    plot_scatter(
        vertical_files=[
            Path(
                r"board_data\[model-CNN_DEEP][seed-1][symmetry-True]\eval-state-PP.txt"
            )
        ],
        horizontal_files=[Path("board_data/PP/eval.txt")],
        output=Path("dist/scatter_plot.pdf"),
    )
