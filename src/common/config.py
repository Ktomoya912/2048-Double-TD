import logging
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

import torch

from .args import args, game_conf

logger = logging.getLogger(__name__)

try:
    exec(f"from models import {args.model} as modeler")
    exec(f"from models import {args.model} as modeler2")
except ImportError as e:
    logger.error(f"Model {args.model} not found in models module: {e}")
    raise ImportError(
        f"Model {args.model} not found. Please check the models module."
    ) from e


def get_trained_model(model_path: Path, device) -> OrderedDict:
    pth_file = MODEL_DIR / model_path
    if not pth_file.exists():
        pth_file = model_path
    if not pth_file.exists():
        logger.warning(f"Model file {pth_file} does not exist.")
        return None
    state_dict = torch.load(pth_file, map_location=device, weights_only=True)
    logger.info(f"Model loaded: {pth_file.name}")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def get_model_name():
    # game_confからモデル名を取得
    models_names = []
    for k, v in game_conf.items():
        models_names.append(f"[{k}-{v}]")
    return "".join(models_names)


start_time = datetime.now()
TIME_LIMIT = timedelta(hours=args.hours) if args.hours > 0 else timedelta(days=1)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
LOG_PATH = Path(f"log/{start_time.strftime('%Y%m%dT%H%M%S')}_{get_model_name()}.log")
LOG_PATH.parent.mkdir(exist_ok=True)
FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
logging.basicConfig(
    level=args.log,
    format=FORMAT,
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        # RichHandler(show_time=False, show_level=False, show_path=False),
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger("config_2048")
for key, value in args._get_kwargs():
    logger.info(f"{key} : {value}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")
MAIN_NETWORK: torch.nn.Module = modeler.Model().to(DEVICE)  # noqa: F821
TARGET_NETWORK: torch.nn.Module = modeler2.Model().to(DEVICE)  # noqa: F821
BAT_SIZE = 1024
if args.load_main:
    if data := get_trained_model(args.load_main, DEVICE):
        MAIN_NETWORK.load_state_dict(data[0])
if args.load_target:
    if data := get_trained_model(args.load_target, DEVICE):
        TARGET_NETWORK.load_state_dict(data[0])

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
