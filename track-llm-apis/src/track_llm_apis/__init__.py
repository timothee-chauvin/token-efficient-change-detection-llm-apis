import os
from pathlib import Path

from beartype.claw import beartype_this_package
from dotenv import load_dotenv

os.environ.setdefault(
    "VLLM_LOGGING_CONFIG_PATH",
    str(Path(__file__).with_name("assets") / "vllm_logging.json"),
)

beartype_this_package()

# TODO remove the load_dotenv calls in other files once there's a unified way to invoke code
load_dotenv()


def get_assets_dir() -> Path:
    return Path(__file__).with_name("assets")
