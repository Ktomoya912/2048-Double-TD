import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from queue import Queue

import numpy as np
import torch
from torch import optim

from common.args import args
from common.config import LOG_PATH, MAIN_NETWORK, MODEL_DIR, TARGET_NETWORK, TIME_LIMIT

from .common import Trainer
from .D_TDA_C import D_TDA_C_Trainer
from .D_TDA_CB import D_TDA_CB_Trainer
from .D_TDA_X import D_TDA_X_Trainer
from .TDA import TDA_Trainer

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


def clear_queues():
    while pack_main["queue"].qsize() > 0 and pack_target["queue"].qsize() > 0:
        pack_main["queue"].get()
        pack_target["queue"].get()
    logger.info("Queues cleared, stopping threads...")


def save_models(save_count: int = -1):
    main_model_path = MODEL_DIR / f"{LOG_PATH.stem}_{save_count:02d}_main.pth"
    target_model_path = MODEL_DIR / f"{LOG_PATH.stem}_{save_count:02d}_target.pth"

    torch.save(MAIN_NETWORK.state_dict(), main_model_path)
    logger.info(f"save {main_model_path.name} {save_count=}")
    torch.save(TARGET_NETWORK.state_dict(), target_model_path)
    logger.info(f"save {target_model_path.name} {save_count=}")


def submit_batches(trainer: Trainer, executor: ThreadPoolExecutor):
    for i in range(tasks):
        executor.submit(trainer.play_game, i)
    for pack in trainer.packs:
        executor.submit(trainer.batch_trainer, pack)
    return executor


def main():
    try:
        packs = [pack_main, pack_target]
        if args.trainer == "D-TDA-C":
            trainer = D_TDA_C_Trainer(packs)
        elif args.trainer == "D-TDA-CB":
            trainer = D_TDA_CB_Trainer(packs)
        elif args.trainer == "TDA":
            packs = [pack_main]
            trainer = TDA_Trainer(packs)
        elif args.trainer == "D-TDA-X":
            trainer = D_TDA_X_Trainer(packs)
        else:
            raise ValueError(f"Unknown trainer type: {args.trainer}")

        executor = ThreadPoolExecutor(max_workers=tasks + len(packs))
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
            time.sleep(1)  # 少し待ってから終了処理を行う

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
