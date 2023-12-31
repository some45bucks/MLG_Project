import datetime
import os
from os.path import join
from typing import Optional
import functools
from gymnasium.spaces import Discrete
from modified_sample_factory.envs.env_utils import register_env
from modified_sample_factory.envs.env_wrappers import (
    PixelFormatChwWrapper,
    RecordingWrapper,
    ResizeWrapper,
    RewardScalingWrapper,
    TimeLimitWrapper,
)
from modified_sample_factory.utils.utils import debug_log_every_n
from viz_doom_sf.action_space import (
    doom_action_space,
    doom_action_space_basic,
    doom_action_space_discretized_no_weap,
    doom_action_space_extended,
    doom_action_space_full_discretized,
    doom_turn_and_attack_only,
)
from viz_doom_sf.doom_gym import VizdoomEnv, VizdoomEnvGeo
from viz_doom_sf.wrappers.additional_input import DoomAdditionalInput
from viz_doom_sf.wrappers.multiplayer_stats import MultiplayerStatsWrapper
from viz_doom_sf.wrappers.observation_space import SetResolutionWrapper, resolutions
from viz_doom_sf.wrappers.reward_shaping import (
    REWARD_SHAPING_BATTLE,
    REWARD_SHAPING_DEATHMATCH_V0,
    REWARD_SHAPING_DEATHMATCH_V1,
    DoomRewardShapingWrapper,
    true_objective_frags,
    true_objective_winning_the_game,
)
from viz_doom_sf.wrappers.scenario_wrappers.gathering_reward_shaping import DoomGatheringRewardShaping

def register_vizdoom_envs():
    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)

class DoomSpec:
    def __init__(
        self,
        name,
        env_spec_file,
        action_space,
        reward_scaling=1.0,
        default_timeout=-1,
        num_agents=1,
        num_bots=0,
        respawn_delay=0,
        timelimit=4.0,
        extra_wrappers=None,
    ):
        self.name = name
        self.env_spec_file = env_spec_file
        self.action_space = action_space
        self.reward_scaling = reward_scaling
        self.default_timeout = default_timeout

        # 1 for singleplayer, >1 otherwise
        self.num_agents = num_agents

        self.num_bots = num_bots

        self.respawn_delay = respawn_delay
        self.timelimit = timelimit

        # expect list of tuples (wrapper_cls, wrapper_kwargs)
        self.extra_wrappers = extra_wrappers


ADDITIONAL_INPUT = (DoomAdditionalInput, {})  # health, ammo, etc. as input vector
BATTLE_REWARD_SHAPING = (
    DoomRewardShapingWrapper,
    dict(reward_shaping_scheme=REWARD_SHAPING_BATTLE, true_objective_func=None),
)  # "true" reward None means just the env reward (monster kills)
BOTS_REWARD_SHAPING = (
    DoomRewardShapingWrapper,
    dict(reward_shaping_scheme=REWARD_SHAPING_DEATHMATCH_V0, true_objective_func=true_objective_frags),
)
DEATHMATCH_REWARD_SHAPING = (
    DoomRewardShapingWrapper,
    dict(reward_shaping_scheme=REWARD_SHAPING_DEATHMATCH_V1, true_objective_func=true_objective_winning_the_game),
)


DOOM_ENVS = [
    DoomSpec(
        "doom_basic",
        "basic.cfg",
        Discrete(1 + 3),  # idle, left, right, attack
        reward_scaling=0.01,
        default_timeout=300,
    ),
    DoomSpec(
        "doom_two_colors_easy",
        "two_colors_easy.cfg",
        doom_action_space_basic(),
        extra_wrappers=[(DoomGatheringRewardShaping, {})],  # same as https://arxiv.org/pdf/1904.01806.pdf
    ),
    DoomSpec(
        "doom_two_colors_hard",
        "two_colors_hard.cfg",
        doom_action_space_basic(),
        extra_wrappers=[(DoomGatheringRewardShaping, {})],
    ),
    DoomSpec(
        "doom_dm",
        "cig.cfg",
        doom_action_space(),
        1.0,
        int(1e9),
        num_agents=8,
        extra_wrappers=[ADDITIONAL_INPUT, DEATHMATCH_REWARD_SHAPING],
    ),
    DoomSpec(
        "doom_dwango5",
        "dwango5_dm.cfg",
        doom_action_space(),
        1.0,
        int(1e9),
        num_agents=8,
        extra_wrappers=[ADDITIONAL_INPUT, DEATHMATCH_REWARD_SHAPING],
    ),
    # <==== Environments used in the paper ====>
    # this is for comparison with other frameworks (wall-time test)
    DoomSpec("doom_my_way_home_flat_actions", "my_way_home.cfg", Discrete(1 + 4), 1.0),
    DoomSpec("doom_defend_the_center_flat_actions", "defend_the_center.cfg", Discrete(1 + 3), 1.0),
    # "basic" single-player envs
    DoomSpec("doom_my_way_home", "my_way_home.cfg", doom_action_space_basic(), 1.0),
    DoomSpec("doom_deadly_corridor", "deadly_corridor.cfg", doom_action_space_extended(), 0.01),
    DoomSpec("doom_defend_the_center", "defend_the_center.cfg", doom_turn_and_attack_only(), 1.0),
    DoomSpec("doom_defend_the_line", "defend_the_line.cfg", doom_turn_and_attack_only(), 1.0),
    DoomSpec(
        "doom_health_gathering",
        "health_gathering.cfg",
        Discrete(1 + 4),
        1.0,
        extra_wrappers=[(DoomGatheringRewardShaping, {})],  # same as https://arxiv.org/pdf/1904.01806.pdf
    ),
    DoomSpec(
        "doom_health_gathering_supreme",
        "health_gathering_supreme.cfg",
        Discrete(1 + 4),
        1.0,
        extra_wrappers=[(DoomGatheringRewardShaping, {})],  # same as https://arxiv.org/pdf/1904.01806.pdf
    ),
    # "challenging" single-player envs
    DoomSpec(
        "doom_battle",
        "battle_continuous_turning.cfg",
        doom_action_space_discretized_no_weap(),
        1.0,
        2100,
        extra_wrappers=[ADDITIONAL_INPUT, BATTLE_REWARD_SHAPING],
    ),
    DoomSpec(
        "doom_battle2",
        "battle2_continuous_turning.cfg",
        doom_action_space_discretized_no_weap(),
        1.0,
        2100,
        extra_wrappers=[ADDITIONAL_INPUT, BATTLE_REWARD_SHAPING],
    ),
    # multi-player envs with bots as opponents (still only one agent)
    DoomSpec(
        "doom_duel_bots",
        "ssl2.cfg",
        doom_action_space_full_discretized(with_use=True),
        1.0,
        int(1e9),
        num_agents=1,
        num_bots=1,
        respawn_delay=2,
        extra_wrappers=[ADDITIONAL_INPUT, BOTS_REWARD_SHAPING],
    ),
    DoomSpec(
        "doom_deathmatch_bots",
        "dwango5_dm_continuous_weap.cfg",
        doom_action_space_full_discretized(),
        1.0,
        int(1e9),
        num_agents=1,
        num_bots=7,
        extra_wrappers=[ADDITIONAL_INPUT, BOTS_REWARD_SHAPING],
    ),
    # full multiplayer environments for self-play and PBT experiments
    DoomSpec(
        "doom_duel",
        "ssl2.cfg",
        doom_action_space_full_discretized(with_use=True),
        1.0,
        int(1e9),
        num_agents=2,
        num_bots=0,
        respawn_delay=2,
        extra_wrappers=[ADDITIONAL_INPUT, DEATHMATCH_REWARD_SHAPING],
    ),
    DoomSpec(
        "doom_deathmatch_full",
        "freedm.cfg",
        doom_action_space_full_discretized(with_use=True),
        1.0,
        int(1e9),
        num_agents=4,
        num_bots=4,
        respawn_delay=2,
        extra_wrappers=[ADDITIONAL_INPUT, DEATHMATCH_REWARD_SHAPING],
    ),
    # benchmark environment, this is the same doom_battle that we're using in the paper, but without extra input spaces
    # for measurements, and with a more simple action space, just so it is easier to use with other codebases
    # we measure throughput with 128x72 input resolution, 4-frameskip and original game resolution of 160x120
    # (no widescreen)
    DoomSpec("doom_benchmark", "battle.cfg", Discrete(1 + 8), 1.0, 2100),
]


def doom_env_by_name(name):
    for cfg in DOOM_ENVS:
        if cfg.name == name:
            return cfg
    raise RuntimeError("Unknown Doom env")


# noinspection PyUnusedLocal
def make_doom_env_impl(
    doom_spec,
    cfg=None,
    env_config=None,
    skip_frames=None,
    episode_horizon=None,
    player_id=None,
    num_agents=None,
    max_num_players=None,
    num_bots=0,  # for multi-agent
    custom_resolution=None,
    render_mode: Optional[str] = None,
    **kwargs,
):
    skip_frames = skip_frames if skip_frames is not None else cfg.env_frameskip

    fps = cfg.fps if "fps" in cfg else None
    async_mode = fps == 0

    if player_id is None:
        if cfg.use_geo:
            env = VizdoomEnvGeo(
                doom_spec.action_space,
                doom_spec.env_spec_file,
                skip_frames=skip_frames,
                async_mode=async_mode,
                render_mode=render_mode,
            )
        else:
            env = VizdoomEnv(
                doom_spec.action_space,
                doom_spec.env_spec_file,
                skip_frames=skip_frames,
                async_mode=async_mode,
                render_mode=render_mode,
            )

    record_to = cfg.record_to if "record_to" in cfg else None
    should_record = False
    if env_config is None:
        should_record = True
    elif env_config.worker_index == 0 and env_config.vector_index == 0 and (player_id is None or player_id == 0):
        should_record = True

    if record_to is not None and should_record:
        env = RecordingWrapper(env, record_to, player_id)

    env = MultiplayerStatsWrapper(env)

    # # BotDifficultyWrapper no longer in use
    # if num_bots > 0:
    #     bot_difficulty = cfg.start_bot_difficulty if "start_bot_difficulty" in cfg else None
    #     env = BotDifficultyWrapper(env, bot_difficulty)

    resolution = custom_resolution
    if resolution is None:
        resolution = "256x144" if cfg.wide_aspect_ratio else "160x120"

    assert resolution in resolutions
    env = SetResolutionWrapper(env, resolution)  # default (wide aspect ratio)

    if cfg.use_geo:
        h, w, channels = env.observation_space.spaces['obs'].shape
    else:
        h, w, channels = env.observation_space.shape
    if w != cfg.res_w or h != cfg.res_h:
        env = ResizeWrapper(env, cfg.res_w, cfg.res_h, grayscale=False)

    debug_log_every_n(50, "Doom resolution: %s, resize resolution: %r", resolution, (cfg.res_w, cfg.res_h))

    # randomly vary episode duration to somewhat decorrelate the experience
    timeout = doom_spec.default_timeout
    if episode_horizon is not None and episode_horizon > 0:
        timeout = episode_horizon
    if timeout > 0:
        env = TimeLimitWrapper(env, limit=timeout, random_variation_steps=0)

    pixel_format = cfg.pixel_format if "pixel_format" in cfg else "HWC"
    if pixel_format == "CHW":
        env = PixelFormatChwWrapper(env)

    if doom_spec.extra_wrappers is not None:
        for wrapper_cls, wrapper_kwargs in doom_spec.extra_wrappers:
            env = wrapper_cls(env, **wrapper_kwargs)

    if doom_spec.reward_scaling != 1.0:
        env = RewardScalingWrapper(env, doom_spec.reward_scaling)

    return env

def make_doom_env(env_name, cfg, env_config, render_mode: Optional[str] = None, **kwargs):
    spec = doom_env_by_name(env_name)
    return make_doom_env_from_spec(spec, env_name, cfg, env_config, render_mode, **kwargs)


def make_doom_env_from_spec(spec, _env_name, cfg, env_config, render_mode: Optional[str] = None, **kwargs):
    """
    Makes a Doom environment from a DoomSpec instance.
    _env_name is unused but we keep it, so functools.partial(make_doom_env_from_spec, env_spec) can registered
    in Sample Factory (first argument in make_env_func is expected to be the env_name).
    """

    if "record_to" in cfg and cfg.record_to:
        tstamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        cfg.record_to = join(cfg.record_to, f"{cfg.experiment}", tstamp)
        if not os.path.isdir(cfg.record_to):
            os.makedirs(cfg.record_to)
    else:
        cfg.record_to = None


    return make_doom_env_impl(spec, cfg=cfg, env_config=env_config, render_mode=render_mode, **kwargs)
