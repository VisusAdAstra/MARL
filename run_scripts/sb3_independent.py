import argparse
import datetime
import gym
import supersuit as ss
import torch
import torch.nn.functional as F
# pip install git+https://github.com/Rohan138/marl-baselines3
from marl_baselines3 import IndependentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from torch import nn
import os
import yaml
import shutil
import numpy as np

from social_dilemmas.envs.pettingzoo_env import parallel_env

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def copy_args_to_yaml(args, filename):
    try:
        with open(filename, 'w') as yaml_file:
            yaml.dump(args, yaml_file, default_flow_style=False)
        print(f"Arguments successfully written to {filename}")
    except Exception as e:
        print(f"Error: {e}")

def parse_args():
    parser = argparse.ArgumentParser("MARL-Baselines3 PPO with Independent Learning")
    parser.add_argument(
        "--env-name",
        type=str,
        default="harvest",
        choices=["harvest", "cleanup"],
        help="The SSD environment to use",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=5,
        help="The number of agents",
    )
    parser.add_argument(
        "--num-inequity",
        type=int,
        default=5,
        help="The number of inequity",
    )
    parser.add_argument(
        "--rollout-len",
        type=int,
        default=1000,
        help="length of training rollouts AND length at which env is reset",
    )
    parser.add_argument(
        "--total-timesteps",
        type=float,
        default=5e8,
        help="Number of environment timesteps",
    )
    parser.add_argument(
        "--use-collective-reward",
        type=bool,
        default=False,
        help="Give each agent the collective reward across all agents",
    )
    parser.add_argument(
        "--inequity-averse-reward",
        type=bool,
        default=False,
        help="Use inequity averse rewards from 'Inequity aversion \
            improves cooperation in intertemporal social dilemmas'",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=5,
        help="Advantageous inequity aversion factor",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="Disadvantageous inequity aversion factor",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="",
        help="Experiment name",
    )
    parser.add_argument(
        "--train",
        type=bool,
        default=False,
        help="Train or evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Gpu device",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="The number of parallel envs",
    )
    args = parser.parse_args()
    return args


# Use this with lambda wrapper returning observations only
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim=128,
        view_len=7,
        num_frames=6,
        fcnet_hiddens=[1024, 128],
    ):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        flat_out = num_frames * 6 * (view_len * 2 - 1) ** 2
        self.conv = nn.Conv2d(
            in_channels=num_frames * 3,  # Input: (3 * 4) x 15 x 15
            out_channels=num_frames * 6,  # Output: 24 x 13 x 13
            kernel_size=3,
            stride=1,
            padding="valid",
        )
        self.fc1 = nn.Linear(in_features=flat_out, out_features=fcnet_hiddens[0])
        self.fc2 = nn.Linear(in_features=fcnet_hiddens[0], out_features=fcnet_hiddens[1])

    def forward(self, observations) -> torch.Tensor:
        # Convert to tensor, rescale to [0, 1], and convert from B x H x W x C to B x C x H x W
        observations = observations.permute(0, 3, 1, 2)
        features = torch.flatten(F.relu(self.conv(observations)), start_dim=1)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        return features


def main(args):
    # Config
    env_name = args.env_name
    num_agents = args.num_agents
    num_inequity = args.num_inequity
    rollout_len = args.rollout_len
    total_timesteps = args.total_timesteps
    use_collective_reward = args.use_collective_reward
    inequity_averse_reward = args.inequity_averse_reward
    alpha = args.alpha
    beta = args.beta
    exp_name = args.exp_name
    device = args.device

    # Training
    num_cpus = 4  # number of cpus
    num_envs = args.num_envs  # number of parallel multi-agent environments; # 12
    num_frames = 6  # number of frames to stack together; use >4 to avoid automatic VecTransposeImage
    features_dim = (
        128  # output layer of cnn extractor AND shared layer for policy and value functions
    )
    fcnet_hiddens = [1024, 128]  # Two hidden layers for cnn extractor
    ent_coef = 0.001  # entropy coefficient in loss
    batch_size = rollout_len * num_envs // 2  # This is from the rllib baseline implementation
    lr = 0.0001
    n_epochs = 30
    gae_lambda = 1.0
    gamma = 0.99
    target_kl = 0.01
    grad_clip = 40
    verbose = 3

    env = parallel_env(
        max_cycles=rollout_len,
        env=env_name,
        num_agents=num_agents,
        num_inequity=num_inequity,
        use_collective_reward=use_collective_reward,
        inequity_averse_reward=inequity_averse_reward,
        alpha=alpha,
        beta=beta,
    )
    env = ss.observation_lambda_v0(env, lambda x, _: x["curr_obs"], lambda s: s["curr_obs"])
    env = ss.frame_stack_v1(env, num_frames)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=num_envs, num_cpus=num_cpus, base_class="stable_baselines3"
    )
    env = VecMonitor(env)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(
            features_dim=features_dim, num_frames=num_frames, fcnet_hiddens=fcnet_hiddens
        ),
        net_arch=[features_dim],
    )

    tensorboard_log = "./results/sb3/ppo_independent"

    model = IndependentPPO(
        "CnnPolicy",
        num_agents=num_agents,
        env=env,
        learning_rate=lr,
        n_steps=rollout_len,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        max_grad_norm=grad_clip,
        target_kl=target_kl,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        exp_name=exp_name,
        device=device,
    )
    if(args.train):
        print(f"start training {datetime.datetime.now()}")
        os.makedirs(tensorboard_log + "/" + exp_name, exist_ok=True)
        copy_args_to_yaml(args, tensorboard_log + "/" + exp_name + "/config.yaml")
        model.learn(total_timesteps=total_timesteps)
        logdir = model.logger.dir

        model.save(logdir)
        shutil.copytree(logdir, logdir.rsplit("_", 1)[0], dirs_exist_ok=True)
        copy_args_to_yaml(args, f"{logdir}/config.yaml")
        del model
        model = IndependentPPO.load(  # noqa: F841
            logdir, "CnnPolicy", num_agents, env, rollout_len, policy_kwargs, tensorboard_log, verbose
        )
    else:
        logdir = tensorboard_log + "/" + exp_name + "_1"
        print(f"load model {logdir}")
        model = IndependentPPO.load(  # noqa: F841
            logdir, "CnnPolicy", num_agents, env, rollout_len, policy_kwargs, tensorboard_log, verbose
        )

        # Evaluate the model and generate a video
        eval_results = model.evaluate(
            num_episodes=2,
            max_timesteps=1000,
            deterministic=False,
            render=True,
            video_path="./videos",
            video_name=exp_name,
            fps=30,
            verbose=True
        )
        
        # Print overall results
        print("\nEvaluation complete!")
        print(f"Overall mean reward: {np.mean(eval_results['mean_reward']):.2f}")


if __name__ == "__main__":
    args = parse_args()
    print("args.train", args.train)
    main(args)
