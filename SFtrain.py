"""
Example of how to use VizDoom env API to use your own custom VizDoom environment with Sample Factory.

To train:
python -m viz_doom_sf.train_custom_vizdoom_env --algo=APPO --env=doom_my_custom_env --experiment=doom_my_custom_env_example --save_every_sec=5 --experiment_summaries_interval=10

After training for a desired period of time, evaluate the policy by running:
python -m viz_doom_sf.enjoy_custom_vizdoom_env --algo=APPO --env=doom_my_custom_env --experiment=doom_my_custom_env_example

"""
import argparse
import functools
import os
import sys
from os.path import join

from modified_sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from modified_sample_factory.envs.env_utils import register_env
from modified_sample_factory.train import run_rl
from viz_doom_sf.doom.action_space import doom_action_space_extended
from viz_doom_sf.doom.doom_params import add_doom_env_args, doom_override_defaults
from viz_doom_sf.doom.doom_utils import DoomSpec, make_doom_env_from_spec
from viz_doom_sf.train_vizdoom import register_vizdoom_envs
from viz_doom_sf.vizdoom.doom.DoomModel import register_model_components


def add_custom_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--my_custom_doom_arg", type=int, default=300, help="Any custom arguments users might define")


def register_custom_doom_env(custom_timeout):
    # absolute path needs to be specified, otherwise Doom will look in the SampleFactory scenarios folder
    scenario_absolute_path = join(os.path.dirname(__file__), "custom_env", "custom_doom_env.cfg")

    spec = DoomSpec(
        "doom_my_custom_env",
        scenario_absolute_path,  # use your custom cfg here
        doom_action_space_extended(),
        reward_scaling=0.01,
        default_timeout=custom_timeout,
    )

    # register the env with Sample Factory
    make_env_func = functools.partial(make_doom_env_from_spec, spec)
    register_env(spec.name, make_env_func)


def main():

    register_vizdoom_envs()
    register_model_components()

    parser, cfg = parse_sf_args()
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    add_custom_args(parser)
    # second parsing pass yields the final configuration
    cfg = parse_full_cfg(parser)

    register_custom_doom_env(custom_timeout=cfg.my_custom_doom_arg)

    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())