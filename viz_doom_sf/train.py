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
from viz_doom_sf.doom.doom_utils import register_vizdoom_envs
from viz_doom_sf.doom.DoomModel import register_model_components


def main():

    register_vizdoom_envs()
    register_model_components()

    parser, cfg = parse_sf_args()
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    # second parsing pass yields the final configuration
    cfg = parse_full_cfg(parser)

    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())