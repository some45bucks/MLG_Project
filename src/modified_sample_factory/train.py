from typing import Tuple

from modified_sample_factory.algo.runners.runner import Runner
from modified_sample_factory.algo.runners.runner_parallel import ParallelRunner
from modified_sample_factory.algo.runners.runner_serial import SerialRunner
from modified_sample_factory.algo.utils.misc import ExperimentStatus
from modified_sample_factory.cfg.arguments import maybe_load_from_checkpoint
from modified_sample_factory.utils.typing import Config


def make_runner(cfg: Config) -> Tuple[Config, Runner]:
    if cfg.restart_behavior == "resume":
        # if we're resuming from checkpoint, we load all of the config parameters from the checkpoint
        # unless they're explicitly specified in the command line
        cfg = maybe_load_from_checkpoint(cfg)

    if cfg.serial_mode:
        runner_cls = SerialRunner
    else:
        runner_cls = ParallelRunner

    runner = runner_cls(cfg)

    return cfg, runner


def run_rl(cfg: Config):
    cfg, runner = make_runner(cfg)

    # here we can register additional message or summary handlers
    # see sf_examples/dmlab/train_dmlab.py for example

    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    return status
