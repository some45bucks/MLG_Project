import sys

from modified_sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from modified_sample_factory.enjoy import enjoy
from viz_doom_sf.doom.doom_params import add_doom_env_args, add_doom_env_eval_args, doom_override_defaults
from viz_doom_sf.doom.doom_utils import register_vizdoom_envs
from viz_doom_sf.doom.DoomModel import register_model_components


def main():
    """Script entry point."""
    register_vizdoom_envs()
    register_model_components()

    parser, cfg = parse_sf_args(evaluation=True)
    add_doom_env_args(parser)
    add_doom_env_eval_args(parser)
    doom_override_defaults(parser)
    cfg = parse_full_cfg(parser)

    status = enjoy(cfg)
    return status

    



if __name__ == "__main__":
    sys.exit(main())