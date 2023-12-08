import copy
from dataclasses import asdict, dataclass
import os
import random
import re
import time
from os.path import join
from threading import Thread
from typing import Dict, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
from filelock import FileLock, Timeout
from gymnasium.utils import seeding
from vizdoom.vizdoom import AutomapMode, DoomGame, Mode, ScreenResolution
from torch_geometric.data import HeteroData

from modified_sample_factory.algo.utils.spaces.discretized import Discretized
from modified_sample_factory.utils.utils import log, project_tmp_dir


def doom_lock_file(max_parallel):
    """
    Doom instances tend to have problems starting when a lot of them are initialized in parallel.
    This is not a problem during normal execution once the envs are initialized.

    The "sweet spot" for the number of envs that can be initialized in parallel is about 5-10.
    Here we use file locking mechanism to ensure that only a limited amount of envs are being initialized at the same
    time.
    This tends to be more of a problem for multiplayer envs.

    This also has an advantage of working across completely independent process groups, e.g. different experiments.
    """
    lock_filename = f"doom_{random.randrange(0, max_parallel):03d}.lockfile"

    tmp_dir = project_tmp_dir()
    lock_path = join(tmp_dir, lock_filename)
    return lock_path


def key_to_action_default(key):
    """
    MOVE_FORWARD
    MOVE_BACKWARD
    MOVE_RIGHT
    MOVE_LEFT
    SELECT_WEAPON1
    SELECT_WEAPON2
    SELECT_WEAPON3
    SELECT_WEAPON4
    SELECT_WEAPON5
    SELECT_WEAPON6
    SELECT_WEAPON7
    ATTACK
    SPEED
    TURN_LEFT_RIGHT_DELTA
    """
    from pynput.keyboard import Key

    # health gathering
    action_table = {
        Key.left: 0,
        Key.right: 1,
        Key.up: 2,
        Key.down: 3,
    }

    # action_table = {
    #     Key.up: 0,
    #     Key.down: 1,
    #     Key.alt: 6,
    #     Key.ctrl: 11,
    #     Key.shift: 12,
    #     Key.space: 13,
    #     Key.right: 'turn_right',
    #     Key.left: 'turn_left',
    # }

    return action_table.get(key, None)


class VizdoomEnv(gym.Env):
    def __init__(
        self,
        action_space,
        config_file,
        coord_limits=None,
        max_histogram_length=200,
        show_automap=False,
        skip_frames=1,
        async_mode=False,
        record_to=None,
        render_mode: Optional[str] = None,
    ):
        self.initialized = False

        # essential game data
        self.game = None
        self.state = None
        self.curr_seed = 0
        self.rng = None
        self.skip_frames = skip_frames
        self.async_mode = async_mode

        # optional - for topdown view rendering and visitation heatmaps
        self.show_automap = show_automap
        self.coord_limits = coord_limits

        # can be adjusted after the environment is created (but before any reset() call) via observation space wrapper
        self.screen_w, self.screen_h, self.channels = 640, 480, 3
        self.screen_resolution = ScreenResolution.RES_640X480
        self.calc_observation_space()

        self.black_screen = None

        # provided as a part of environment definition, since these depend on the scenario and
        # can be quite complex multi-discrete spaces
        self.action_space = action_space
        self.composite_action_space = hasattr(self.action_space, "spaces")

        self.delta_actions_scaling_factor = 7.5

        if os.path.isabs(config_file):
            self.config_path = config_file
        else:
            scenarios_dir = join(os.path.dirname(__file__), "scenarios")
            self.config_path = join(scenarios_dir, config_file)
            if not os.path.isfile(self.config_path):
                log.warning(
                    "File %s not found in scenarios dir %s. Consider providing absolute path?",
                    config_file,
                    scenarios_dir,
                )

        self.variable_indices = self._parse_variable_indices(self.config_path)

        # only created if we call render() method
        self.screen = None

        # record full episodes using VizDoom recording functionality
        self.record_to = record_to
        self.curr_demo_dir = None

        self.is_multiplayer = False  # overridden in derived classes

        # (optional) histogram to track positional coverage
        # do not pass coord_limits if you don't need this, to avoid extra calculation
        self.max_histogram_length = max_histogram_length
        self.current_histogram, self.previous_histogram = None, None
        if self.coord_limits:
            x = self.coord_limits[2] - self.coord_limits[0]
            y = self.coord_limits[3] - self.coord_limits[1]
            if x > y:
                len_x = self.max_histogram_length
                len_y = int((y / x) * self.max_histogram_length)
            else:
                len_x = int((x / y) * self.max_histogram_length)
                len_y = self.max_histogram_length
            self.current_histogram = np.zeros((len_x, len_y), dtype=np.int32)
            self.previous_histogram = np.zeros_like(self.current_histogram)

        # helpers for human play with pynput keyboard input
        self._terminate = False
        self._current_actions = []
        self._actions_flattened = None

        self._prev_info = None
        self._last_episode_info = None

        self._num_episodes = 0

        self.mode = "algo"

        self.render_mode = render_mode

        self.seed()

    def seed(self, seed: Optional[int] = None):
        """
        Used to seed the actual Doom env.
        If None is passed, the seed is generated randomly.
        """
        self.rng, self.curr_seed = seeding.np_random(seed=seed)
        self.curr_seed = self.curr_seed % (2**32)  # Doom only supports 32-bit seeds
        return [self.curr_seed, self.rng]

    def calc_observation_space(self):
        self.observation_space = gym.spaces.Box(0, 255, (self.screen_h, self.screen_w, self.channels), dtype=np.uint8)

    def _set_game_mode(self, mode):
        if mode == "replay":
            self.game.set_mode(Mode.PLAYER)
        else:
            if self.async_mode:
                log.info("Starting in async mode! Use this only for testing, otherwise PLAYER mode is much faster")
                self.game.set_mode(Mode.ASYNC_PLAYER)
            else:
                self.game.set_mode(Mode.PLAYER)

    def _create_doom_game(self, mode):
        self.game = DoomGame()

        self.game.load_config(self.config_path)
        self.game.set_screen_resolution(self.screen_resolution)
        self.game.set_seed(self.curr_seed)

        if mode == "algo":
            self.game.set_window_visible(False)
        elif mode == "human" or mode == "replay":
            self.game.add_game_args("+freelook 1")
            self.game.set_window_visible(True)
        else:
            raise Exception("Unsupported mode")

        self._set_game_mode(mode)

    def _game_init(self, with_locking=True, max_parallel=10):
        lock_file = lock = None
        if with_locking:
            lock_file = doom_lock_file(max_parallel)
            lock = FileLock(lock_file)

        init_attempt = 0
        while True:
            init_attempt += 1
            try:
                if with_locking:
                    with lock.acquire(timeout=20):
                        self.game.init()
                else:
                    self.game.init()

                break
            except Timeout:
                if with_locking:
                    log.debug(
                        "Another process currently holds the lock %s, attempt: %d",
                        lock_file,
                        init_attempt,
                    )
            except Exception as exc:
                log.warning("VizDoom game.init() threw an exception %r. Terminate process...", exc)
                from modified_sample_factory.envs.env_utils import EnvCriticalError

                raise EnvCriticalError()

    def initialize(self):
        self._create_doom_game(self.mode)

        # (optional) top-down view provided by the game engine
        if self.show_automap:
            self.game.set_automap_buffer_enabled(True)
            self.game.set_automap_mode(AutomapMode.OBJECTS)
            self.game.set_automap_rotate(False)
            self.game.set_automap_render_textures(False)

            # self.game.add_game_args("+am_restorecolors")
            # self.game.add_game_args("+am_followplayer 1")
            background_color = "ffffff"
            self.game.add_game_args("+viz_am_center 1")
            self.game.add_game_args("+am_backcolor " + background_color)
            self.game.add_game_args("+am_tswallcolor dddddd")
            # self.game.add_game_args("+am_showthingsprites 0")
            self.game.add_game_args("+am_yourcolor " + background_color)
            self.game.add_game_args("+am_cheat 0")
            self.game.add_game_args("+am_thingcolor 0000ff")  # player color
            self.game.add_game_args("+am_thingcolor_item 00ff00")
            # self.game.add_game_args("+am_thingcolor_citem 00ff00")

        self._game_init()
        self.initialized = True

    def _ensure_initialized(self):
        if not self.initialized:
            self.initialize()

    @staticmethod
    def _parse_variable_indices(config):
        with open(config, "r") as config_file:
            lines = config_file.readlines()
        lines = [ln.strip() for ln in lines]

        variable_indices = {}

        for line in lines:
            if line.startswith("#"):
                continue  # comment

            variables_syntax = r"available_game_variables[\s]*=[\s]*\{(.*)\}"
            match = re.match(variables_syntax, line)
            if match is not None:
                variables_str = match.groups()[0]
                variables_str = variables_str.strip()
                variables = variables_str.split(" ")
                for i, variable in enumerate(variables):
                    variable_indices[variable] = i
                break

        return variable_indices

    def _black_screen(self):
        if self.black_screen is None:
            self.black_screen = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return self.black_screen

    def _game_variables_dict(self, state):
        game_variables = state.game_variables
        variables = {}
        for variable, idx in self.variable_indices.items():
            variables[variable] = game_variables[idx]
        return variables

    @staticmethod
    def demo_path(episode_idx, record_to):
        demo_name = f"e{episode_idx:03d}.lmp"
        demo_path_ = join(record_to, demo_name)
        demo_path_ = os.path.normpath(demo_path_)
        return demo_path_

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        if "seed" in kwargs:
            self.seed(kwargs["seed"])

        self._ensure_initialized()

        episode_started = False
        if self.record_to and not self.is_multiplayer:
            # does not work in multiplayer (uses different mechanism)
            if not os.path.exists(self.record_to):
                os.makedirs(self.record_to)

            demo_path = self.demo_path(self._num_episodes, self.record_to)
            self.curr_demo_dir = os.path.dirname(demo_path)
            log.warning(f"Recording episode demo to {demo_path=}")

            if len(demo_path) > 101:
                log.error(f"Demo path {len(demo_path)=}>101, will not record demo")
                log.error(
                    "This seems to be a bug in VizDoom, please just use a shorter demo path, i.e. set --record_to to /tmp/doom_recs"
                )
            else:
                self.game.new_episode(demo_path)
                episode_started = True

        if self._num_episodes > 0 and not episode_started:
            # no demo recording (default)
            self.game.new_episode()

        self.state = self.game.get_state()
        img = None
        try:
            img = self.state.screen_buffer
        except AttributeError:
            # sometimes Doom does not return screen buffer at all??? Rare bug
            pass

        if img is None:
            log.error("Game returned None screen buffer! This is not supposed to happen!")
            img = self._black_screen()

        # Swap current and previous histogram
        if self.current_histogram is not None and self.previous_histogram is not None:
            swap = self.current_histogram
            self.current_histogram = self.previous_histogram
            self.previous_histogram = swap
            self.current_histogram.fill(0)

        self._actions_flattened = None
        self._last_episode_info = copy.deepcopy(self._prev_info)
        self._prev_info = None

        self._num_episodes += 1

        return np.transpose(img, (1, 2, 0)), {}  # since Gym 0.26.0, we return dict as second return value

    def _convert_actions(self, actions):
        """Convert actions from gym action space to the action space expected by Doom game."""

        if self.composite_action_space:
            # composite action space with multiple subspaces
            spaces = self.action_space.spaces
        else:
            # simple action space, e.g. Discrete. We still treat it like composite of length 1
            spaces = (self.action_space,)
            actions = (actions,)

        actions_flattened = []
        for i, action in enumerate(actions):
            if isinstance(spaces[i], Discretized):
                # discretized continuous action
                # check discretized first because it's a subclass of gym.spaces.Discrete
                # the order of if clauses here matters! DON'T CHANGE THE ORDER OF IFS!

                continuous_action = spaces[i].to_continuous(action)
                actions_flattened.append(continuous_action)
            elif isinstance(spaces[i], gym.spaces.Discrete):
                # standard discrete action
                num_non_idle_actions = spaces[i].n - 1
                action_one_hot = np.zeros(num_non_idle_actions, dtype=np.uint8)
                if action > 0:
                    action_one_hot[action - 1] = 1  # 0th action in each subspace is a no-op

                actions_flattened.extend(action_one_hot)
            elif isinstance(spaces[i], gym.spaces.Box):
                # continuous action
                actions_flattened.extend(list(action * self.delta_actions_scaling_factor))
            else:
                raise NotImplementedError(f"Action subspace type {type(spaces[i])} is not supported!")

        return actions_flattened

    def _vizdoom_variables_bug_workaround(self, info, done):
        """Some variables don't get reset to zero on game.new_episode(). This fixes it (also check overflow?)."""
        if done and "DAMAGECOUNT" in info:
            log.info("DAMAGECOUNT value on done: %r", info.get("DAMAGECOUNT"))

        if self._last_episode_info is not None:
            bugged_vars = ["DEATHCOUNT", "HITCOUNT", "DAMAGECOUNT"]
            for v in bugged_vars:
                if v in info:
                    info[v] -= self._last_episode_info.get(v, 0)

    def _process_game_step(self, state, done, info):
        if not done:
            observation = np.transpose(state.screen_buffer, (1, 2, 0))
            game_variables = self._game_variables_dict(state)
            info.update(self.get_info(game_variables))
            self._update_histogram(info)
            self._prev_info = copy.copy(info)
        else:
            observation = self._black_screen()

            # when done=True Doom does not allow us to call get_info, so we provide info from the last frame
            info.update(self._prev_info)

        self._vizdoom_variables_bug_workaround(info, done)

        return observation, done, info

    def step(self, actions) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Action is either a single value (discrete, one-hot), or a tuple with an action for each of the
        discrete action subspaces.
        """
        if self._actions_flattened is not None:
            # provided externally, e.g. via human play
            actions_flattened = self._actions_flattened
            self._actions_flattened = None
        else:
            actions_flattened = self._convert_actions(actions)

        default_info = {"num_frames": self.skip_frames}
        reward = self.game.make_action(actions_flattened, self.skip_frames)
        state = self.game.get_state()
        done = self.game.is_episode_finished()

        observation, done, info = self._process_game_step(state, done, default_info)

        # Gym 0.26.0 changes
        terminated = done
        truncated = False
        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        mode = self.render_mode
        if mode is None:
            return

        try:
            img = self.game.get_state().screen_buffer
            img = np.transpose(img, [1, 2, 0])
            if mode == "rgb_array":
                return img

            h, w = img.shape[:2]
            render_h, render_w = h, w
            max_w = 1280
            if w < max_w:
                render_w = max_w
                render_h = int(max_w * h / w)
                img = cv2.resize(img, (render_w, render_h))

            import pygame

            if self.screen is None:
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode((render_w, render_h))

            pygame.surfarray.blit_array(self.screen, img.swapaxes(0, 1))
            pygame.display.update()

            return img
        except AttributeError:
            return None

    def close(self):
        try:
            if self.game is not None:
                self.game.close()
        except RuntimeError as exc:
            log.warning("Runtime error in VizDoom game close(): %r", exc)

        # if self.viewer is not None:
        #     self.viewer.close()
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

    def get_info(self, variables=None):
        if variables is None:
            variables = self._game_variables_dict(self.game.get_state())

        info_dict = {"pos": self.get_positions(variables)}
        info_dict.update(variables)
        return info_dict

    def get_info_all(self, variables=None):
        if variables is None:
            variables = self._game_variables_dict(self.game.get_state())
        info = self.get_info(variables)
        if self.previous_histogram is not None:
            info["previous_histogram"] = self.previous_histogram
        return info

    def get_positions(self, variables):
        return self._get_positions(variables)

    @staticmethod
    def _get_positions(variables):
        have_coord_data = True
        required_vars = ["POSITION_X", "POSITION_Y", "ANGLE"]
        for required_var in required_vars:
            if required_var not in variables:
                have_coord_data = False
                break

        x = y = a = np.nan
        if have_coord_data:
            x = variables["POSITION_X"]
            y = variables["POSITION_Y"]
            a = variables["ANGLE"]

        return {"agent_x": x, "agent_y": y, "agent_a": a}

    def get_automap_buffer(self):
        if self.game.is_episode_finished():
            return None
        state = self.game.get_state()
        map_ = state.automap_buffer
        map_ = np.swapaxes(map_, 0, 2)
        map_ = np.swapaxes(map_, 0, 1)
        return map_

    def _update_histogram(self, info, eps=1e-8):
        if self.current_histogram is None:
            return
        agent_x, agent_y = info["pos"]["agent_x"], info["pos"]["agent_y"]

        # Get agent coordinates normalized to [0, 1]
        dx = (agent_x - self.coord_limits[0]) / (self.coord_limits[2] - self.coord_limits[0])
        dy = (agent_y - self.coord_limits[1]) / (self.coord_limits[3] - self.coord_limits[1])

        # Rescale coordinates to histogram dimensions
        # Subtract eps to exclude upper bound of dx, dy
        dx = int((dx - eps) * self.current_histogram.shape[0])
        dy = int((dy - eps) * self.current_histogram.shape[1])

        self.current_histogram[dx, dy] += 1

    def _key_to_action(self, key):
        if hasattr(self.action_space, "key_to_action"):
            return self.action_space.key_to_action(key)
        else:
            return key_to_action_default(key)

    def _keyboard_on_press(self, key):
        from pynput.keyboard import Key

        if key == Key.esc:
            self._terminate = True
            return False

        action = self._key_to_action(key)
        if action is not None:
            if action not in self._current_actions:
                self._current_actions.append(action)

    def _keyboard_on_release(self, key):
        action = self._key_to_action(key)
        if action is not None:
            if action in self._current_actions:
                self._current_actions.remove(action)

    # noinspection PyProtectedMember
    @staticmethod
    def play_human_mode(env, skip_frames=1, num_episodes=3, num_actions=None):
        from pynput.keyboard import Listener

        doom = env.unwrapped
        doom.skip_frames = 1  # handled by this script separately

        # noinspection PyProtectedMember
        def start_listener():
            with Listener(on_press=doom._keyboard_on_press, on_release=doom._keyboard_on_release) as listener:
                listener.join()

        listener_thread = Thread(target=start_listener)
        listener_thread.start()

        for episode in range(num_episodes):
            doom.mode = "human"
            env.reset()
            last_render_time = time.time()
            time_between_frames = 1.0 / 35.0

            total_rew = 0.0

            while not doom.game.is_episode_finished() and not doom._terminate:
                num_actions = 14 if num_actions is None else num_actions
                turn_delta_action_idx = num_actions - 1

                actions = [0] * num_actions
                for action in doom._current_actions:
                    if isinstance(action, int):
                        actions[action] = 1  # 1 for buttons currently pressed, 0 otherwise
                    else:
                        if action == "turn_left":
                            actions[turn_delta_action_idx] = -doom.delta_actions_scaling_factor
                        elif action == "turn_right":
                            actions[turn_delta_action_idx] = doom.delta_actions_scaling_factor

                for frame in range(skip_frames):
                    doom._actions_flattened = actions
                    _, rew, _, _, _ = env.step(actions)

                    new_total_rew = total_rew + rew
                    if new_total_rew != total_rew:
                        log.info("Reward: %.3f, total: %.3f", rew, new_total_rew)
                    total_rew = new_total_rew
                    state = doom.game.get_state()

                    verbose = True
                    if state is not None and verbose:
                        info = doom.get_info()
                        print(
                            "Health:",
                            info["HEALTH"],
                            # 'Weapon:', info['SELECTED_WEAPON'],
                            # 'ready:', info['ATTACK_READY'],
                            # 'ammo:', info['SELECTED_WEAPON_AMMO'],
                            # 'pc:', info['PLAYER_COUNT'],
                            # 'dmg:', info['DAMAGECOUNT'],
                        )

                    time_since_last_render = time.time() - last_render_time
                    time_wait = time_between_frames - time_since_last_render

                    if doom.show_automap and state.automap_buffer is not None:
                        map_ = state.automap_buffer
                        map_ = np.swapaxes(map_, 0, 2)
                        map_ = np.swapaxes(map_, 0, 1)
                        cv2.imshow("ViZDoom Automap Buffer", map_)
                        if time_wait > 0:
                            cv2.waitKey(int(time_wait) * 1000)
                    else:
                        if time_wait > 0:
                            time.sleep(time_wait)

                    last_render_time = time.time()

            if doom.show_automap:
                cv2.destroyAllWindows()

        log.debug("Press ESC to exit...")
        listener_thread.join()

    # noinspection PyProtectedMember
    @staticmethod
    def replay(env, rec_path):
        doom = env.unwrapped
        doom.mode = "replay"
        doom._ensure_initialized()
        doom.game.replay_episode(rec_path)

        episode_reward = 0
        start = time.time()

        while not doom.game.is_episode_finished():
            doom.game.advance_action()
            r = doom.game.get_last_reward()
            episode_reward += r
            log.info("Episode reward: %.3f, time so far: %.1f s", episode_reward, time.time() - start)

        log.info("Finishing replay")
        doom.close()

class VizdoomEnvGeo(VizdoomEnv):
    

    def __init__(
        self,
        action_space,
        config_file,
        coord_limits=None,
        max_histogram_length=200,
        show_automap=False,
        skip_frames=1,
        async_mode=False,
        record_to=None,
        render_mode: Optional[str] = None,
    ):
        super().__init__(action_space,config_file,coord_limits,max_histogram_length,show_automap,skip_frames,async_mode,record_to,render_mode)
        self.reset_graph_data()


    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        if "seed" in kwargs:
            self.seed(kwargs["seed"])

        self._ensure_initialized()
        
        episode_started = False
        if self.record_to and not self.is_multiplayer:
            # does not work in multiplayer (uses different mechanism)
            if not os.path.exists(self.record_to):
                os.makedirs(self.record_to)

            demo_path = self.demo_path(self._num_episodes, self.record_to)
            self.curr_demo_dir = os.path.dirname(demo_path)
            log.warning(f"Recording episode demo to {demo_path=}")

            if len(demo_path) > 101:
                log.error(f"Demo path {len(demo_path)=}>101, will not record demo")
                log.error(
                    "This seems to be a bug in VizDoom, please just use a shorter demo path, i.e. set --record_to to /tmp/doom_recs"
                )
            else:
                self.game.new_episode(demo_path)
                episode_started = True

        if self._num_episodes > 0 and not episode_started:
            # no demo recording (default)
            self.game.new_episode()

        self.state = self.game.get_state()
        img = None
        try:
            img = self.state.screen_buffer
        except AttributeError:
            # sometimes Doom does not return screen buffer at all??? Rare bug
            pass

        if img is None:
            log.error("Game returned None screen buffer! This is not supposed to happen!")
            img = self._black_screen()

        # Swap current and previous histogram
        if self.current_histogram is not None and self.previous_histogram is not None:
            swap = self.current_histogram
            self.current_histogram = self.previous_histogram
            self.previous_histogram = swap
            self.current_histogram.fill(0)

        self._actions_flattened = None
        self._last_episode_info = copy.deepcopy(self._prev_info)
        self._prev_info = None

        self._num_episodes += 1

        self.sectors = self.split_sectors(self.state.sectors)
        self.sectors_bb = []
        self.sectors_connections = self.find_sector_connections(self.sectors)

        for sec in self.sectors:
            self.sectors_bb.append(self.calculate_bounding_box(sec))

        self.totBBsize = self.bounding_box_of_bounding_boxes(self.sectors_bb)

        self.reset_graph_data()

        self.create_scene_graph(self.state)

        observation = {
            "obs":np.transpose(img, (1, 2, 0)),
        }

        for j in self.graph_data.keys():
                for i in self.graph_data[j].keys():
                    observation[f"geo_{j}_{i}"] = self.graph_data[j][i]["data"]
        return observation, {}  # since Gym 0.26.0, we return dict as second return value
    
    def reset_graph_data(self):
        self.graph_data = {
            "N": {
                "player":{
                    "data":-np.ones((2,3)),
                    "graph":{},
                    "prev":0
                },
                "items":{
                    "data":-np.ones((self.MAX_NODES+1,4)),
                    "graph":{},
                    "prev":0
                },
                "enemies":{
                    "data":-np.ones((self.MAX_NODES+1,4)),
                    "graph":{},
                    "prev":0
                },
                "dead":{
                    "data":-np.ones((self.MAX_NODES+1,4)),
                    "graph":{},
                    "prev":0
                },   
                "sector":{
                    "data":-np.ones((self.MAX_NODES+1,6)),
                    "graph":{},
                    "prev":0
                }, 
                "special" :{
                    "data":-np.ones((self.MAX_NODES+1,4)),
                    "graph":{},
                    "prev":0
                }
            },
            "E":{
                "player-sees-items":{
                    "data":-np.ones((self.MAX_EDGES+1,2)),
                    "graph":{},
                    "prev":0
                },
                "player-sees-enemies":{
                    "data":-np.ones((self.MAX_EDGES+1,2)),
                    "graph":{},
                    "prev":0
                },
                "enemies-sees-player":{
                    "data":-np.ones((self.MAX_EDGES+1,2)),
                    "graph":{},
                    "prev":0
                },  
                "player-sees-dead":{
                    "data":-np.ones((self.MAX_EDGES+1,2)),
                    "graph":{},
                    "prev":0
                },
                "player-sees-special":{
                    "data":-np.ones((self.MAX_EDGES+1,2)),
                    "graph":{},
                    "prev":0
                },  
                "player-inside-sector": {
                    "data":-np.ones((2,2)),
                    "graph":{},
                    "prev":0
                },
                "items-inside-sector": {
                    "data":-np.ones((self.MAX_EDGES+1,2)),
                    "graph":{},
                    "prev":0
                }, 
                "enemies-inside-sector": {
                    "data":-np.ones((self.MAX_EDGES+1,2)),
                    "graph":{},
                    "prev":0
                },
                "dead-inside-sector": {
                    "data":-np.ones((self.MAX_EDGES+1,2)),
                    "graph":{},
                    "prev":0
                },
                "special-inside-sector": {
                    "data":-np.ones((self.MAX_EDGES+1,2)),
                    "graph":{},
                    "prev":0
                },   
                "sector-connected-sector": {
                    "data":-np.ones((self.MAX_EDGES+1,2)),
                    "graph":{},
                    "prev":0
                }
            }

            
        }
        
        for key in self.graph_data["N"]:
            self.add_node_to_graph(-1,key,[0,0])

        for key in self.graph_data["E"]:
            keys = key.split("-")
            self.connect_nodes(self.graph_data["N"][keys[0]]["graph"][-1],self.graph_data["N"][keys[2]]["graph"][-1],keys[1])

    def _game_init(self, with_locking=True, max_parallel=10):
        lock_file = lock = None
        if with_locking:
            lock_file = doom_lock_file(max_parallel)
            lock = FileLock(lock_file)

        init_attempt = 0
        while True:
            init_attempt += 1
            self.game.set_labels_buffer_enabled(True)
            self.game.set_objects_info_enabled(True)
            self.game.set_sectors_info_enabled(True)
            try:
                if with_locking:
                    with lock.acquire(timeout=20):
                        self.game.init()
                else:
                    self.game.init()

                break
            except Timeout:
                if with_locking:
                    log.debug(
                        "Another process currently holds the lock %s, attempt: %d",
                        lock_file,
                        init_attempt,
                    )
            except Exception as exc:
                log.warning("VizDoom game.init() threw an exception %r. Terminate process...", exc)
                from modified_sample_factory.envs.env_utils import EnvCriticalError

                raise EnvCriticalError()

    def _black_screen(self):
        if self.black_screen is None:
            self.black_screen = np.zeros(self.observation_space["obs"].shape, dtype=np.uint8)
        return self.black_screen

    def calc_observation_space(self):
        self.reset_graph_data()

        space = {
            "obs":gym.spaces.Box(0, 255, (self.screen_h, self.screen_w, self.channels), dtype=np.uint8)
            }
        
        for j in self.graph_data.keys():
            for i in self.graph_data[j].keys():
                space[f"geo_{j}_{i}"] = gym.spaces.Box(low=-1, high=1, shape=self.graph_data[j][i]["data"].shape)

        self.observation_space = gym.spaces.Dict(space)
        self.metadata = [list(self.graph_data["N"].keys()),[tuple(i.split("-")) for i in self.graph_data["E"].keys()]]

    def step(self, actions) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Action is either a single value (discrete, one-hot), or a tuple with an action for each of the
        discrete action subspaces.
        """
        if self._actions_flattened is not None:
            # provided externally, e.g. via human play
            actions_flattened = self._actions_flattened
            self._actions_flattened = None
        else:
            actions_flattened = self._convert_actions(actions)

        default_info = {"num_frames": self.skip_frames}
        reward = self.game.make_action(actions_flattened, self.skip_frames)
        state = self.game.get_state()
        done = self.game.is_episode_finished()

        observation_img, done, info = self._process_game_step(state, done, default_info)

        self.create_scene_graph(state)

        observation = {
            "obs":observation_img,
        }

        for j in self.graph_data.keys():
                for i in self.graph_data[j].keys():
                    observation[f"geo_{j}_{i}"] = self.graph_data[j][i]["data"]

        # Gym 0.26.0 changes
        terminated = done
        truncated = False
        return observation, reward, terminated, truncated, info
    
    MAX_NODES = 50
    MAX_EDGES = 50
    

    graph_ignore = [
        "BulletPuff",
        "Blood",
        "DoomPlayer"
    ]

    graph_remap = {
        "Zombieman":("enemies",0),
        "ShotgunGuy":("enemies",1),
        "ChaingunGuy":("enemies",2),
        "Shotgun":("items",0),
        "Chaingun":("items",1),
        "Clip":("items",2),
        "Medikit":("items",3),
        "GreenArmor":("special",0),
        "TeleportFog":("special",1)
    }

    def node_in_graph(self,id,type):
        return id in self.graph_data["N"][type]["graph"]
        
    def add_node_to_graph(self,id,type,data):
        self.graph_data["N"][type]["graph"][id] = self.node_obj(id,type,data)

    def update_node(self,data,id,type):
        self.get_node_from_graph(id,type).data = data

    def get_node_from_graph(self,id,type):
        if type != None:
            return self.graph_data["N"][type]["graph"][id]
        else:
            for t in self.graph_data["N"]:
                if self.node_in_graph(id,t):
                    return self.graph_data["N"][t]["graph"][id]

    def get_all_nodes_from_graph(self,exept):
        nodes = []

        for section in [i for i in self.graph_data["N"].keys() if not i in exept]:
            nodes += [item for item in self.graph_data["N"][section]["graph"].values()]

        return nodes

    def delete_node(self,id,type):
        if id != -1:
            del self.graph_data["N"][type]["graph"][id]

                
    def connect_nodes(self,n1,n2,type):
        self.graph_data["E"][n1.type+"-"+type+"-"+n2.type]["graph"][str(n1.id)+"-"+type+"-"+str(n2.id)] = self.edge_obj(type,n1,n2)

    def disconnect_nodes(self,n1,n2,type):
        if self.are_nodes_connected(n1,n2,type) and n1.id != -1 and n2.id != -1:
            del self.graph_data["E"][n1.type+"-"+type+"-"+n2.type]["graph"][str(n1.id)+"-"+type+"-"+str(n2.id)]
                
    def are_nodes_connected(self,n1,n2,type):
        return str(n1.id)+"-"+type+"-"+str(n2.id) in self.graph_data["E"][n1.type+"-"+type+"-"+n2.type]["graph"]
    
    def get_node_to_connections(self,id,type=None,edge_type=None,other_type=None):
        edges = []

        for f_edge_type in self.graph_data["E"].keys():
            if (( type==None or type+"-" in f_edge_type) and
                (edge_type==None or "-"+edge_type+"-" in f_edge_type) and
                (other_type==None or "-"+other_type in f_edge_type)):

                edges += [item for item in self.graph_data["E"][f_edge_type]["graph"].values() if item.n1.id == id]

        return edges
    
    def get_node_from_connections(self,id,type=None,edge_type=None,other_type=None):
        edges = []

        for f_edge_type in self.graph_data["E"].keys():
            if ((type==None or type+"-" in f_edge_type) and
                (edge_type==None or "-"+edge_type+"-" in f_edge_type) and
                (other_type==None or "-"+other_type in f_edge_type)):

                edges += [item for item in self.graph_data["E"][f_edge_type]["graph"].values() if item.n2.id == id]

        return edges
    
    def get_all_node_connections(self,id,type=None,edge_type=None,other_type=None):
        return self.get_node_to_connections(id,type,edge_type,other_type) + self.get_node_from_connections(id,other_type,edge_type,type)
        
    def is_edge_valid(self,n1,n2,type):
        if self.are_nodes_connected(n1,n2,type):
            return not n1.deleted and not n2.deleted
        return False

    def create_scene_graph(self,state):
        if state != None:
            
            labels = state.labels
            label_ids = [i.object_id for i in labels]
            objects = state.objects
            object_ids = [i.id for i in objects]

            player_pos = [0,0]
            player_node = None

            obj = [o for o in objects if o.name == "DoomPlayer"][0]

            if not self.node_in_graph(obj.id,"player"):
                self.add_node_to_graph(obj.id,"player",[-1,-1,-1])

            player_node = self.get_node_from_graph(obj.id,"player")

            data = [self.normalize_point(obj.position_x,self.totBBsize),
                    self.normalize_point(obj.position_y,self.totBBsize),
                    self.normalize_angle(obj.angle)]
            
            self.update_node(data,obj.id,"player")

            player_pos[0] = obj.position_x
            player_pos[1] = obj.position_y
            player_id = obj.id

            for i in range(555,((len(self.sectors)-1)*10000)+555,9999):
                bb = self.sectors_bb[(i-555)//9999]
                is_in = self.is_position_within_bounding_box(player_pos,bb)

                if is_in:
                
                    if not self.node_in_graph(i,"sector"):
                        data = [self.normalize_point((bb[0][0] + bb[1][0])/2,self.totBBsize),
                                self.normalize_point((bb[0][1] + bb[1][1])/2,self.totBBsize),
                                self.normalize_point(bb[0][0],self.totBBsize),
                                self.normalize_point(bb[0][1],self.totBBsize),
                                self.normalize_point(bb[1][0],self.totBBsize),
                                self.normalize_point(bb[1][1],self.totBBsize)]
                        self.add_node_to_graph(i,"sector",data)

                    sector_node = self.get_node_from_graph(i,"sector")
                    
                    for connection in self.get_all_node_connections(player_id,"sector","inside","player"):
                        self.disconnect_nodes(connection.n1,connection.n2,"inside")

                    self.connect_nodes(player_node,sector_node,"inside")

                    for other_i in self.sectors_connections[(i-555)/9999]:
                        other_bb = self.sectors_bb[other_i]
                        other_i = other_i*9999 + 555
                        if not self.node_in_graph(other_i,"sector"):
                            data = [self.normalize_point((other_bb[0][0] + other_bb[1][0])/2,self.totBBsize),
                            self.normalize_point((other_bb[0][1] + other_bb[1][1])/2,self.totBBsize),
                            self.normalize_point(other_bb[0][0],self.totBBsize),
                            self.normalize_point(other_bb[0][1],self.totBBsize),
                            self.normalize_point(other_bb[1][0],self.totBBsize),
                            self.normalize_point(other_bb[1][1],self.totBBsize)]
                            self.add_node_to_graph(other_i,"sector",data)

                            other_node = self.get_node_from_graph(other_i,"sector")

                            self.connect_nodes(other_node,sector_node,"connected")
                            self.connect_nodes(sector_node,other_node,"connected")

            for connection in self.get_all_node_connections(player_node.id,"player","sees"):
                self.disconnect_nodes(connection.n1,connection.n2,"sees")

            for obj in objects:

                if obj.name in self.graph_ignore:
                        continue
                if "Dead" in obj.name:
                    node_type = "dead"
                    _, index = self.graph_remap[obj.name.split("Dead")[1]]
                else:
                    node_type, index = self.graph_remap[obj.name]

                if obj.id in label_ids:

                    if not self.node_in_graph(obj.id,node_type):
                        self.add_node_to_graph(obj.id,node_type,[-1,-1,-1,-1])
                        if node_type == "dead":
                            for connection in self.get_all_node_connections(obj.id,"enemies"):
                                self.disconnect_nodes(connection.n1,connection.n2,connection.type)
                            if self.node_in_graph(obj.id,"enemies"):
                                self.delete_node(obj.id,"enemies")
                    
                    node = self.get_node_from_graph(obj.id,node_type)
                    
                    data = [
                        self.normalize_point(obj.position_x,self.totBBsize),
                        self.normalize_point(obj.position_y,self.totBBsize),
                        self.normalize_angle(obj.angle),
                        index
                    ]

                    self.update_node(data,node.id,node_type)

                    self.connect_nodes(player_node,node,"sees")

                    if node_type == "enemies":
                        self.connect_nodes(node,player_node,"sees")

                    for i in range(555,(len(self.sectors)-1)*10000 + 555,9999):
                        bb = self.sectors_bb[(i-555)//9999]
                        is_in = self.is_position_within_bounding_box((obj.position_x,obj.position_y),bb)

                        if is_in:
                            if not self.node_in_graph(i,"sector"):
                                data = [self.normalize_point((bb[0][0] + bb[1][0])/2,self.totBBsize),
                                    self.normalize_point((bb[0][1] + bb[1][1])/2,self.totBBsize),
                                    self.normalize_point(bb[0][0],self.totBBsize),
                                    self.normalize_point(bb[0][1],self.totBBsize),
                                    self.normalize_point(bb[1][0],self.totBBsize),
                                    self.normalize_point(bb[1][1],self.totBBsize)]
                                
                                self.add_node_to_graph(i,"sector",data)

                            sector_node = self.get_node_from_graph(i,"sector")

                            for connection in self.get_all_node_connections(node.id,"sector","inside",node_type):
                                self.disconnect_nodes(connection.n1,connection.n2,"inside")

                            self.connect_nodes(node,sector_node,"inside")

            for node in self.get_all_nodes_from_graph(["sector","player"]):

                if not node.id in object_ids:
                    if self.node_in_graph(node.id,node_type):
                        for connection in self.get_all_node_connections(node.id):
                            self.disconnect_nodes(connection.n1,connection.n2,connection.type)

                        self.delete_node(node.id,node_type)

            for type in self.graph_data["N"]:
                count = 1
                for node in self.graph_data["N"][type]["graph"].values():
                    if node.id == -1:
                        node.pos = 0
                        for i in range(len(self.graph_data["N"][type]["data"][count])):
                            self.graph_data["N"][type]["data"][0][i] = 0
                    else:
                        node.pos = count
                        for i in range(len(self.graph_data["N"][type]["data"][count])):
                            self.graph_data["N"][type]["data"][count][i] = node.data[i]
                        count += 1

                for z in range(max(0,self.graph_data["N"][type]["prev"]-count)):
                    for i in range(len(self.graph_data["N"][type]["data"][count+z])):
                        self.graph_data["N"][type]["data"][count+z][i] = -1
                
                self.graph_data["N"][type]["prev"] = count

            for type in self.graph_data["E"]:
                count = 1
                for edge in self.graph_data["E"][type]["graph"].values():
                    if edge.n1.id == -1 and edge.n2.id == -1:
                        self.graph_data["E"][type]["data"][0][0] = 0
                        self.graph_data["E"][type]["data"][0][1] = 0
                    else:
                        self.graph_data["E"][type]["data"][count][0] = edge.n1.pos
                        self.graph_data["E"][type]["data"][count][1] = edge.n2.pos
                        count += 1

                for z in range(max(0,self.graph_data["E"][type]["prev"]-count)):
                    self.graph_data["E"][type]["data"][count+z][0] = -1
                    self.graph_data["E"][type]["data"][count+z][1] = -1

                self.graph_data["E"][type]["prev"] = count

    @dataclass
    class graph_node_obj:
        type: str
        pos: -1
        data: []
        id: int
        def __str__(self):
            return f"{self.type}:{self.pos}"
        
    def node_obj(self,id,type,data):

        return self.graph_node_obj(type,-1,data,id)
    
    def edge_obj(self,type,n1,n2):
        @dataclass
        class graph_edge_obj:
            type: str
            n1: self.graph_node_obj
            n2: self.graph_node_obj
            def __str__(self):
                return f"Node1:{self.n1}->{self.type}->Node2:{self.n2}"

        return graph_edge_obj(type,n1,n2)

    def is_position_within_bounding_box(self,position, bounding_box):
        # Unpack the position and bounding box
        pos_x, pos_y = position
        (min_x, min_y), (max_x, max_y) = bounding_box
        
        # Check if the position is within the bounding box boundaries
        return min_x <= pos_x <= max_x and min_y <= pos_y <= max_y

    def bounding_box_of_bounding_boxes(self,bounding_boxes):
        # Initialize min and max values to the first bounding box values
        min_x, min_y = bounding_boxes[0][0]
        max_x, max_y = bounding_boxes[0][1]

        # Iterate through each bounding box and update min and max values
        for (box_min_x, box_min_y), (box_max_x, box_max_y) in bounding_boxes:
            min_x = min(min_x, box_min_x)
            min_y = min(min_y, box_min_y)
            max_x = max(max_x, box_max_x)
            max_y = max(max_y, box_max_y)

        # Return the encompassing bounding box
        return (min_x, min_y), (max_x, max_y)

    def normalize_point(self,x, bounding_box):
        min_x, min_y = bounding_box[0]
        max_x, max_y = bounding_box[1]

        # Determine the maximum side length of the bounding box
        max_side_length = max(max_x - min_x, max_y - min_y)

        # Normalize both coordinates using the max side length
        normalized_x = x / max_side_length

        return normalized_x
    
    def normalize_angle(self,value):

        return value/360

    def split_sectors(self,sectors):
        new_sectors = []

        for sector in sectors:
            # Assuming each sector has a 'lines' attribute that is a list of Line objects
            connected_components = self.find_connected_components(sector.lines.copy())

            # Create new sectors from connected components
            for component in connected_components:
                new_sectors.append(component)

        return new_sectors

    # Helper function to find connected components within a sector
    def find_connected_components(self,lines):
        def find_connected(line, lines_set):
            for other_line in lines_set:
                if line is not other_line and \
                ((line.x1 == other_line.x1 and line.y1 == other_line.y1) or \
                    (line.x1 == other_line.x2 and line.y1 == other_line.y2) or \
                    (line.x2 == other_line.x1 and line.y2 == other_line.y1) or \
                    (line.x2 == other_line.x2 and line.y2 == other_line.y2)):
                    return other_line
            return None

        def dfs(start_line, lines_set):
            stack = [start_line]
            component = []
            visited = set()  # Keep track of visited lines

            while stack:
                line = stack.pop()
                if line not in visited:
                    visited.add(line)
                    component.append(line)
                    lines_set.discard(line)  # Use discard to avoid KeyError

                    connected_line = find_connected(line, lines_set)
                    while connected_line:
                        stack.append(connected_line)
                        lines_set.discard(connected_line)  # Use discard to avoid KeyError
                        connected_line = find_connected(connected_line, lines_set)

            return component

        components = []
        lines_set = set(lines)  # Use a set for efficient removal
        while lines_set:
            start_line = next(iter(lines_set))  # Get an arbitrary line to start
            component = dfs(start_line, lines_set)
            components.append(component)
        return components

    def calculate_bounding_box(self,sector):
        min_x = min(min(line.x1, line.x2) for line in sector)
        max_x = max(max(line.x1, line.x2) for line in sector)
        min_y = min(min(line.y1, line.y2) for line in sector)
        max_y = max(max(line.y1, line.y2) for line in sector)
        
        return (min_x, min_y), (max_x, max_y)

    def plot_sectors(self,sectors, file_name='sectors_plot.png'):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        fig, ax = plt.subplots()

        # Plot lines and bounding boxes for each sector
        for sector in sectors:
            # Draw lines
            for line in sector:
                ax.plot([line.x1, line.x2], [line.y1, line.y2], 'k-')

            # Calculate and draw bounding box
            bbox = self.calculate_bounding_box(sector)
            lower_left = bbox[0]
            upper_right = bbox[1]
            width = upper_right[0] - lower_left[0]
            height = upper_right[1] - lower_left[1]
            rect = patches.Rectangle(lower_left, width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        # Set equal scaling by choosing the larger range in the x or y domain
        x_min = min(min(line.x1, line.x2) for sector in sectors for line in sector)
        x_max = max(max(line.x1, line.x2) for sector in sectors for line in sector)
        y_min = min(min(line.y1, line.y2) for sector in sectors for line in sector)
        y_max = max(max(line.y1, line.y2) for sector in sectors for line in sector)

        ax.set_xlim(x_min - 10, x_max + 10)
        ax.set_ylim(y_min - 10, y_max + 10)
        ax.set_aspect('equal', adjustable='box')

        # Save the figure
        plt.savefig(file_name)

        # Display the plot
        plt.show()
        
    def find_sector_connections(self,sectors):
        connections = {}

        # Define a function to check if lines are equal
        def lines_equal(line1, line2):
            return ((line1.x1, line1.y1) == (line2.x1, line2.y1) and (line1.x2, line1.y2) == (line2.x2, line2.y2)) or \
                ((line1.x1, line1.y1) == (line2.x2, line2.y2) and (line1.x2, line1.y2) == (line2.x1, line2.y1))

        for i, sector_a in enumerate(sectors):
            for j, sector_b in enumerate(sectors[i+1:], i+1):  # Avoid comparing a sector with itself and duplicate checks
                for line_a in sector_a:
                    if any(lines_equal(line_a, line_b) for line_b in sector_b):
                        connections.setdefault(i, set()).add(j)
                        connections.setdefault(j, set()).add(i)
                        # Once a connection is found, no need to compare the rest of the lines
                        break

        # Convert sets back to lists for output consistency (optional)
        for key in connections:
            connections[key] = list(connections[key])

        return connections

