import time
from collections import deque
from typing import Any, Dict, List, Optional, Type, Union
import os
import cv2

import gym
import numpy as np
import torch as th
from gym.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   Schedule)
from stable_baselines3.common.utils import (configure_logger, obs_as_tensor,
                                            safe_mean)
from stable_baselines3.common.vec_env import DummyVecEnv


def make_video_from_image_dir(vid_path, img_folder, video_name="trajectory", fps=5):
    """
    Create a video from a directory of images
    """
    images = [img for img in os.listdir(img_folder) if img.endswith(".png")]
    images.sort()

    rgb_imgs = []
    for i, image in enumerate(images):
        img = cv2.imread(os.path.join(img_folder, image))
        rgb_imgs.append(img)

    make_video_from_rgb_imgs(rgb_imgs, vid_path, video_name=video_name, fps=fps)


def make_video_from_rgb_imgs(
    rgb_arrs, vid_path, video_name="trajectory", fps=5, format="mp4v", resize=None
):
    """
    Create a video from a list of rgb arrays
    """
    print("Rendering video...")
    if vid_path[-1] != "/":
        vid_path += "/"
    video_path = vid_path + video_name + ".mp4"

    if resize is not None:
        width, height = resize
    else:
        frame = rgb_arrs[0]
        height, width, _ = frame.shape
        resize = width, height
    width, height = 320, 240
    resize = width, height

    fourcc = cv2.VideoWriter_fourcc(*format)
    video = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))

    for i, image in enumerate(rgb_arrs):
        percent_done = int((i / len(rgb_arrs)) * 100)
        if percent_done % 50 == 0:
            print("\t...", percent_done, "% of frames rendered")
        # import matplotlib.pyplot as plt
        # plt.cla()
        # plt.imshow(image, interpolation="nearest")
        # plt.savefig("./videos/render.png")
        # Always resize, without this line the video does not render properly.
        image = cv2.resize(image, resize, interpolation=cv2.INTER_NEAREST)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        video.write(image)

    video.release()


class DummyGymEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space


class IndependentPPO(OnPolicyAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        num_agents: int,
        env: GymEnv,
        learning_rate: Union[float, Schedule] = 1e-4,
        n_steps: int = 1000,
        batch_size: int = 6000,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 40,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        exp_name: str = "",
    ):
        self.env = env
        self.num_agents = num_agents
        self.num_envs = env.num_envs // num_agents
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_steps = n_steps
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self._logger = None
        env_fn = lambda: DummyGymEnv(self.observation_space, self.action_space)
        dummy_env = DummyVecEnv([env_fn] * self.num_envs)
        self.policies = [
            PPO(
                policy=policy,
                env=dummy_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                clip_range_vf=clip_range_vf,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                target_kl=target_kl,
                use_sde=use_sde,
                sde_sample_freq=sde_sample_freq,
                policy_kwargs=policy_kwargs,
                verbose=verbose,
                device=device,
            )
            for _ in range(self.num_agents)
        ]
        self.exp_name = exp_name

    def learn(
        self,
        total_timesteps: int,
        callbacks: Optional[List[MaybeCallback]] = None,
        log_interval: int = 1,
        tb_log_name: str = "", #IndependentPPO
        reset_num_timesteps: bool = True,
    ):

        num_timesteps = 0
        all_total_timesteps = []
        if not callbacks:
            callbacks = [None] * self.num_agents
        self._logger = configure_logger(
            self.verbose,
            self.tensorboard_log,
            tb_log_name + self.exp_name,
            reset_num_timesteps,
        )
        logdir = self.logger.dir

        # Setup for each policy
        for polid, policy in enumerate(self.policies):
            policy.start_time = time.time()
            if policy.ep_info_buffer is None or reset_num_timesteps:
                policy.ep_info_buffer = deque(maxlen=100)
                policy.ep_success_buffer = deque(maxlen=100)

            if policy.action_noise is not None:
                policy.action_noise.reset()

            if reset_num_timesteps:
                policy.num_timesteps = 0
                policy._episode_num = 0
                all_total_timesteps.append(total_timesteps)
                policy._total_timesteps = total_timesteps
            else:
                # make sure training timestamps are ahead of internal counter
                all_total_timesteps.append(total_timesteps + policy.num_timesteps)
                policy._total_timesteps = total_timesteps + policy.num_timesteps

            policy._logger = configure_logger(
                policy.verbose,
                logdir,
                "policy",
                reset_num_timesteps,
            )

            callbacks[polid] = policy._init_callback(callbacks[polid])

        for callback in callbacks:
            callback.on_training_start(locals(), globals())

        last_obs = self.env.reset()
        for policy in self.policies:
            policy._last_episode_starts = np.ones((self.num_envs,), dtype=bool)

        last_checkpoint = 0
        while num_timesteps < total_timesteps:
            last_obs = self.collect_rollouts(last_obs, callbacks)
            num_timesteps += self.num_envs * self.n_steps
            for polid, policy in enumerate(self.policies):
                policy._update_current_progress_remaining(
                    policy.num_timesteps, total_timesteps
                )
                if log_interval is not None and num_timesteps % log_interval == 0:
                    fps = int(policy.num_timesteps / (time.time() - policy.start_time))
                    policy.logger.record("policy_id", polid, exclude="tensorboard")
                    policy.logger.record(
                        "time/iterations", num_timesteps, exclude="tensorboard"
                    )
                    if (
                        len(policy.ep_info_buffer) > 0
                        and len(policy.ep_info_buffer[0]) > 0
                    ):
                        policy.logger.record(
                            "rollout/ep_rew_mean",
                            safe_mean(
                                [ep_info["r"] for ep_info in policy.ep_info_buffer]
                            ),
                        )
                        policy.logger.record(
                            "rollout/ep_len_mean",
                            safe_mean(
                                [ep_info["l"] for ep_info in policy.ep_info_buffer]
                            ),
                        )
                    policy.logger.record("time/fps", fps)
                    policy.logger.record(
                        "time/time_elapsed",
                        int(time.time() - policy.start_time),
                        exclude="tensorboard",
                    )
                    policy.logger.record(
                        "time/total_timesteps",
                        policy.num_timesteps,
                        exclude="tensorboard",
                    )
                    policy.logger.dump(step=policy.num_timesteps)

                policy.train()
            if num_timesteps >= last_checkpoint + total_timesteps // 10:
                self.save(logdir)
                last_checkpoint = num_timesteps

        for callback in callbacks:
            callback.on_training_end()

    def collect_rollouts(self, last_obs, callbacks):

        all_last_episode_starts = [None] * self.num_agents
        all_obs = [None] * self.num_agents
        all_last_obs = [None] * self.num_agents
        all_rewards = [None] * self.num_agents
        all_dones = [None] * self.num_agents
        all_infos = [None] * self.num_agents
        steps = 0

        for polid, policy in enumerate(self.policies):
            for envid in range(self.num_envs):
                assert (
                    last_obs[envid * self.num_agents + polid] is not None
                ), f"No previous observation was provided for env_{envid}_policy_{polid}"
            all_last_obs[polid] = np.array(
                [
                    last_obs[envid * self.num_agents + polid]
                    for envid in range(self.num_envs)
                ]
            )
            policy.policy.set_training_mode(False)
            policy.rollout_buffer.reset()
            callbacks[polid].on_rollout_start()
            all_last_episode_starts[polid] = policy._last_episode_starts

        while steps < self.n_steps:
            all_actions = [None] * self.num_agents
            all_values = [None] * self.num_agents
            all_log_probs = [None] * self.num_agents
            all_clipped_actions = [None] * self.num_agents
            with th.no_grad():
                for polid, policy in enumerate(self.policies):
                    obs_tensor = obs_as_tensor(all_last_obs[polid], policy.device)
                    (
                        all_actions[polid],
                        all_values[polid],
                        all_log_probs[polid],
                    ) = policy.policy.forward(obs_tensor)
                    clipped_actions = all_actions[polid].cpu().numpy()
                    if isinstance(self.action_space, Box):
                        clipped_actions = np.clip(
                            clipped_actions,
                            self.action_space.low,
                            self.action_space.high,
                        )
                    elif isinstance(self.action_space, Discrete):
                        # get integer from numpy array
                        clipped_actions = np.array(
                            [action.item() for action in clipped_actions]
                        )
                    all_clipped_actions[polid] = clipped_actions

            all_clipped_actions = (
                np.vstack(all_clipped_actions).transpose().reshape(-1)
            )  # reshape as (env, action)
            obs, rewards, dones, infos = self.env.step(all_clipped_actions)

            for polid in range(self.num_agents):
                all_obs[polid] = np.array(
                    [
                        obs[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )
                all_rewards[polid] = np.array(
                    [
                        rewards[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )
                all_dones[polid] = np.array(
                    [
                        dones[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )
                all_infos[polid] = np.array(
                    [
                        infos[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )
            #print(all_infos)

            for policy in self.policies:
                policy.num_timesteps += self.num_envs

            for callback in callbacks:
                callback.update_locals(locals())
            if not [callback.on_step() for callback in callbacks]:
                break

            for polid, policy in enumerate(self.policies):
                policy._update_info_buffer(all_infos[polid])

            steps += 1

            # add data to the rollout buffers
            for polid, policy in enumerate(self.policies):
                if isinstance(self.action_space, Discrete):
                    all_actions[polid] = all_actions[polid].reshape(-1, 1)
                all_actions[polid] = all_actions[polid].cpu().numpy()
                policy.rollout_buffer.add(
                    all_last_obs[polid],
                    all_actions[polid],
                    all_rewards[polid],
                    all_last_episode_starts[polid],
                    all_values[polid],
                    all_log_probs[polid],
                )
            all_last_obs = all_obs
            all_last_episode_starts = all_dones

        with th.no_grad():
            for polid, policy in enumerate(self.policies):
                obs_tensor = obs_as_tensor(all_last_obs[polid], policy.device)
                _, value, _ = policy.policy.forward(obs_tensor)
                policy.rollout_buffer.compute_returns_and_advantage(
                    last_values=value, dones=all_dones[polid]
                )

        for callback in callbacks:
            callback.on_rollout_end()

        for polid, policy in enumerate(self.policies):
            policy._last_episode_starts = all_last_episode_starts[polid]

        return obs

    @classmethod
    def load(
        cls,
        path: str,
        policy: Union[str, Type[ActorCriticPolicy]],
        num_agents: int,
        env: GymEnv,
        n_steps: int,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        **kwargs,
    ) -> "IndependentPPO":
        model = cls(
            policy=policy,
            num_agents=num_agents,
            env=env,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            **kwargs,
        )
        env_fn = lambda: DummyGymEnv(env.observation_space, env.action_space)
        dummy_env = DummyVecEnv([env_fn] * (env.num_envs // num_agents))
        for polid in range(num_agents):
            model.policies[polid] = PPO.load(
                path=path + f"/policy_{polid + 1}/model", env=dummy_env, **kwargs
            )
        return model

    def save(self, path: str) -> None:
        for polid in range(self.num_agents):
            self.policies[polid].save(path=path + f"/policy_{polid + 1}/model")

    def evaluate(
        self,
        num_episodes: int = 5,
        max_timesteps: int = 1000,
        deterministic: bool = True,
        render: bool = False,
        video_path: str = None,
        video_name: str = "evaluation",
        fps: int = 10,
        render_mode: str = "rgb_array",
        verbose: bool = True,
    ):
        """
        Evaluate a trained model.
        
        Args:
            num_episodes: Number of episodes to evaluate over
            max_timesteps: Maximum timesteps per episode
            deterministic: Whether to use deterministic actions
            render: Whether to render the environment
            video_path: Path to save video (if None, no video is saved)
            video_name: Name of the video file
            fps: Frames per second for video
            render_mode: Render mode to use (rgb_array for video recording)
            verbose: Whether to print evaluation progress
            
        Returns:
            Dictionary with evaluation metrics
        """
        if render and video_path is not None:
            frames = []
            # Ensure video_path directory exists
            os.makedirs(video_path, exist_ok=True)
            
        episode_rewards = [[] for _ in range(self.num_agents)]
        episode_lengths = [[] for _ in range(self.num_agents)]
        
        for episode in range(num_episodes):
            if verbose:
                print(f"Evaluating episode {episode+1}/{num_episodes}")
                
            # Reset env at the start of each episode
            obs = self.env.reset()
            done = [False] * (self.num_envs * self.num_agents)
            states = None
            episode_reward = [0.0] * (self.num_envs * self.num_agents)
            episode_length = 0
            
            for step in range(max_timesteps):
                all_actions = [None] * self.num_agents
                
                # Get actions from all policies
                for polid, policy in enumerate(self.policies):
                    # Extract observations for this policy
                    policy_obs = np.array([
                        obs[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ])
                    
                    # Get action from policy
                    action, states = policy.predict(
                        policy_obs,
                        state=states,
                        deterministic=deterministic
                    )
                    
                    all_actions[polid] = action
                
                # Reshape actions to match environment expectations
                all_actions = np.vstack(all_actions).transpose().reshape(-1)
                
                # Step environment
                obs, rewards, dones, infos = self.env.step(all_actions)
                
                # Update episode rewards and check if done
                for i in range(len(episode_reward)):
                    episode_reward[i] += rewards[i]
                    
                # Render if requested
                if render and video_path is not None:
                    frame = self.env.render(mode=render_mode)
                    frames.append(frame)
                    
                episode_length += 1
                
                # Check if all environments are done
                if all(dones):
                    break
                    
            # Record episode metrics
            for polid in range(self.num_agents):
                agent_rewards = [episode_reward[envid * self.num_agents + polid] for envid in range(self.num_envs)]
                episode_rewards[polid].extend(agent_rewards)
                episode_lengths[polid].extend([episode_length] * self.num_envs)
                
        # Save video if requested
        if render and video_path is not None and frames:
            make_video_from_rgb_imgs(frames, video_path, video_name=video_name, fps=fps)
            if verbose:
                print(f"Video saved to {video_path}/{video_name}.mp4")
        
        # Calculate metrics
        mean_reward = [np.mean(rewards) for rewards in episode_rewards]
        std_reward = [np.std(rewards) for rewards in episode_rewards]
        mean_length = [np.mean(lengths) for lengths in episode_lengths]
        
        # Print metrics
        if verbose:
            for polid in range(self.num_agents):
                print(f"Policy {polid+1}: Mean reward: {mean_reward[polid]:.2f} +/- {std_reward[polid]:.2f}")
                print(f"Policy {polid+1}: Mean episode length: {mean_length[polid]:.2f}")
        
        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_length": mean_length
        }
