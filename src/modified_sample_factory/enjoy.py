import time
from collections import deque
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

import torch
from torch_geometric.data import HeteroData

from modified_sample_factory.algo.learning.learner import Learner
from modified_sample_factory.algo.sampling.batched_sampling import preprocess_actions
from modified_sample_factory.algo.utils.action_distributions import argmax_actions
from modified_sample_factory.algo.utils.env_info import extract_env_info
from modified_sample_factory.algo.utils.make_env import make_env_func_batched
from modified_sample_factory.algo.utils.misc import ExperimentStatus
from modified_sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from modified_sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from modified_sample_factory.cfg.arguments import load_from_checkpoint
from modified_sample_factory.model.actor_critic import create_actor_critic, create_actor_critic_geo
from modified_sample_factory.model.model_utils import get_rnn_size
from modified_sample_factory.utils.attr_dict import AttrDict
from modified_sample_factory.utils.typing import Config, StatusCode
from modified_sample_factory.utils.utils import debug_log_every_n, experiment_dir, log


def visualize_policy_inputs(normalized_obs: Dict[str, Tensor]) -> None:
    """
    Display actual policy inputs after all wrappers and normalizations using OpenCV imshow.
    """
    import cv2

    if "obs" not in normalized_obs.keys():
        return

    obs = normalized_obs["obs"]
    # visualize obs only for the 1st agent
    obs = obs[0]
    if obs.dim() != 3:
        # this function is only for RGB images
        return

    # convert to HWC
    obs = obs.permute(1, 2, 0)
    # convert to numpy
    obs = obs.cpu().numpy()
    # convert to uint8
    obs = cv2.normalize(
        obs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1
    )  # this will be different frame-by-frame but probably good enough to give us an idea?
    scale = 5
    obs = cv2.resize(obs, (obs.shape[1] * scale, obs.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("policy inputs", obs)
    cv2.waitKey(delay=1)

def visualize_heterograph(hetero_data):
    import cv2
    # Create a blank image (you might need to adjust the size)
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)

    # Define colors for different types of nodes and edges
    node_colors = {
        'player': (255, 215, 0),   # Gold
        'items': (0, 255, 127),    # Spring Green
        'enemies': (255, 69, 0),   # Red-Orange
        'dead': (128, 128, 128),   # Gray (for dead entities)
        'sector': (0, 191, 255),   # Deep Sky Blue
        'special': (148, 0, 211),  # Dark Violet
    }

    edge_colors = {
        'sees': (255, 255, 255),   # White (for visibility, clear against black)
        'inside': (255, 105, 180), # Hot Pink (contrasting with other edge types)
        'connected': (32, 178, 170), # Light Sea Green (distinct from player and item colors)
    }

    flip = np.array([1,-1])

    # Draw edges
    for edge_type in hetero_data.edge_types:
        edge_indices = hetero_data[edge_type].edge_index 
        color = edge_colors.get(edge_type[1], (255, 255, 255))  
        for i in range(edge_indices.shape[1]):
            source, target = edge_indices[:, i]
            if source.item() == -1 and target.item() == -1:
                continue
            source_pos = hetero_data[edge_type[0]].x[source.item()][:2].numpy()
            target_pos = hetero_data[edge_type[2]].x[target.item()][:2].numpy()
            source_pos = tuple(map(float, source_pos))
            target_pos = tuple(map(float, target_pos))
            source_pos = (int(850*source_pos[0]) + 100,int(-850*source_pos[1] + 500))
            target_pos = (int(850*target_pos[0]) + 100,int(-850*target_pos[1] + 500))
            cv2.line(img, source_pos, target_pos, color, 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Draw nodes
    for node_type in reversed(hetero_data.node_types):
        node_positions = hetero_data[node_type].x[:, :2].numpy()  # Remove the batch dimension
        color = node_colors.get(node_type, (0, 0, 255))  # Default to red if type not found
        for pos in node_positions:
            pos_m = tuple(map(float, pos))
            pos_m = (int(850*pos[0]) + 100,int(-850*pos[1] + 500))
            pos = tuple(map(int, pos*100))
            cv2.putText(img, f'({pos[0]},{pos[1]})', (pos_m[0] + 10, pos_m[1]+10), font, font_scale, color, font_thickness)
            cv2.circle(img, pos_m, 10, color, -1)

    # Add a legend (key)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    y_offset = 15
    for idx, (key, color) in enumerate(node_colors.items()):
        cv2.putText(img, f'{key}', (10, 800 + idx * y_offset), font, font_scale, color, font_thickness)
    cv2.putText(img, "------------------", (10, 800 + (idx+1) * y_offset), font, font_scale, color, font_thickness)    
    for idx, (key, color) in enumerate(edge_colors.items(), start=len(node_colors)):
        cv2.putText(img, f'{key}', (10, 810 + idx * y_offset), font, font_scale, color, font_thickness)

    # Show the image
    cv2.imshow('Heterogeneous Graph', img)
    cv2.waitKey(delay=1)


def render_frame(cfg, env, video_frames, num_episodes, last_render_start) -> float:
    render_start = time.time()

    if cfg.save_video:
        need_video_frame = len(video_frames) < cfg.video_frames or cfg.video_frames < 0 and num_episodes == 0
        if need_video_frame:
            frame = env.render()
            if frame is not None:
                video_frames.append(frame.copy())
    else:
        if not cfg.no_render:
            target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
            current_delay = render_start - last_render_start
            time_wait = target_delay - current_delay

            if time_wait > 0:
                # log.info("Wait time %.3f", time_wait)
                time.sleep(time_wait)

            try:
                env.render()
            except (gym.error.Error, TypeError) as ex:
                debug_log_every_n(1000, f"Exception when calling env.render() {str(ex)}")

    return render_start


def enjoy(cfg: Config) -> Tuple[StatusCode, float]:
    verbose = False

    cfg = load_from_checkpoint(cfg)

    eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert (
        cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    log.debug(f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for evaluation")

    cfg.num_envs = 1

    render_mode = "human"
    if cfg.save_video:
        render_mode = "rgb_array"
    elif cfg.no_render:
        render_mode = None

    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode
    )
    env_info = extract_env_info(env, cfg)

    if hasattr(env.unwrapped, "reset_on_init"):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    if cfg.use_geo:
        actor_critic = create_actor_critic_geo(cfg, env.observation_space, env.action_space, env.metadata)
    else:
        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict["model"])

    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    true_objectives = [deque([], maxlen=100) for _ in range(env.num_agents)]
    num_frames = 0

    last_render_start = time.time()

    def max_frames_reached(frames):
        return cfg.max_num_frames is not None and frames > cfg.max_num_frames

    reward_list = []

    obs, infos = env.reset()
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = None
    finished_episode = [False for _ in range(env.num_agents)]

    video_frames = []
    num_episodes = 0

    with torch.no_grad():
        while not max_frames_reached(num_frames):
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)

            if not cfg.no_render:
                visualize_policy_inputs(normalized_obs)

                hetero_data = HeteroData()

                for name in normalized_obs.keys():
                    if "geo" in name and "_N_" in name:
                        real_name = name.split("_")[-1]
                        arr = normalized_obs[name].squeeze(0).cpu()
                        last_num = 0
                        for a in range(len(arr)):
                            last_num = a+1 if arr[a][0] != -1 and arr[a][1] != -1 else last_num

                        hetero_data[real_name].x = arr[:last_num]
                    elif "_E_" in name:
                        real_parts = name.split("_")[-1].split("-")
                        arr = normalized_obs[name].int().squeeze(0).cpu()
                        last_num = 0
                        for a in range(len(arr)):
                            last_num = a+1 if arr[a][0] != -1 and arr[a][1] != -1 else last_num
                        
                        hetero_data[real_parts[0], real_parts[1], real_parts[2]].edge_index = arr[:last_num].t() 
                visualize_heterograph(hetero_data)

            policy_outputs = actor_critic(normalized_obs, rnn_states)

            # sample actions from the distribution by default
            actions = policy_outputs["actions"]

            if cfg.eval_deterministic:
                action_distribution = actor_critic.action_distribution()
                actions = argmax_actions(action_distribution)

            # actions shape should be [num_agents, num_actions] even if it's [1, 1]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)

            rnn_states = policy_outputs["new_rnn_states"]

            for _ in range(render_action_repeat):
                last_render_start = render_frame(cfg, env, video_frames, num_episodes, last_render_start)

                obs, rew, terminated, truncated, infos = env.step(actions)
                dones = make_dones(terminated, truncated)
                infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

                if episode_reward is None:
                    episode_reward = rew.float().clone()
                else:
                    episode_reward += rew.float()

                num_frames += 1
                if num_frames % 100 == 0:
                    log.debug(f"Num frames {num_frames}...")

                dones = dones.cpu().numpy()
                for agent_i, done_flag in enumerate(dones):
                    if done_flag:
                        finished_episode[agent_i] = True
                        rew = episode_reward[agent_i].item()
                        episode_rewards[agent_i].append(rew)

                        true_objective = rew
                        if isinstance(infos, (list, tuple)):
                            true_objective = infos[agent_i].get("true_objective", rew)
                        true_objectives[agent_i].append(true_objective)

                        if verbose:
                            log.info(
                                "Episode finished for agent %d at %d frames. Reward: %.3f, true_objective: %.3f",
                                agent_i,
                                num_frames,
                                episode_reward[agent_i],
                                true_objectives[agent_i][-1],
                            )
                        rnn_states[agent_i] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)
                        episode_reward[agent_i] = 0

                        if cfg.use_record_episode_statistics:
                            # we want the scores from the full episode not a single agent death (due to EpisodicLifeEnv wrapper)
                            if "episode" in infos[agent_i].keys():
                                num_episodes += 1
                                reward_list.append(infos[agent_i]["episode"]["r"])
                        else:
                            num_episodes += 1
                            reward_list.append(true_objective)

                # if episode terminated synchronously for all agents, pause a bit before starting a new one
                if all(dones):
                    render_frame(cfg, env, video_frames, num_episodes, last_render_start)
                    time.sleep(0.05)

                if all(finished_episode):
                    finished_episode = [False] * env.num_agents
                    avg_episode_rewards_str, avg_true_objective_str = "", ""
                    for agent_i in range(env.num_agents):
                        avg_rew = np.mean(episode_rewards[agent_i])
                        avg_true_obj = np.mean(true_objectives[agent_i])

                        if not np.isnan(avg_rew):
                            if avg_episode_rewards_str:
                                avg_episode_rewards_str += ", "
                            avg_episode_rewards_str += f"#{agent_i}: {avg_rew:.3f}"
                        if not np.isnan(avg_true_obj):
                            if avg_true_objective_str:
                                avg_true_objective_str += ", "
                            avg_true_objective_str += f"#{agent_i}: {avg_true_obj:.3f}"

                    log.info(
                        "Avg episode rewards: %s, true rewards: %s", avg_episode_rewards_str, avg_true_objective_str
                    )
                    log.info(
                        "Avg episode reward: %.3f, avg true_objective: %.3f",
                        np.mean([np.mean(episode_rewards[i]) for i in range(env.num_agents)]),
                        np.mean([np.mean(true_objectives[i]) for i in range(env.num_agents)]),
                    )

                # VizDoom multiplayer stuff
                # for player in [1, 2, 3, 4, 5, 6, 7, 8]:
                #     key = f'PLAYER{player}_FRAGCOUNT'
                #     if key in infos[0]:
                #         log.debug('Score for player %d: %r', player, infos[0][key])

            if num_episodes >= cfg.max_num_episodes:
                break

    env.close()
    
    return ExperimentStatus.SUCCESS, sum([sum(episode_rewards[i]) for i in range(env.num_agents)]) / sum(
        [len(episode_rewards[i]) for i in range(env.num_agents)]
    )
