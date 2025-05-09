import argparse
import yaml


def copy_args_to_yaml(args, filename):
    try:
        with open(filename, 'w') as yaml_file:
            yaml.dump(args, yaml_file, default_flow_style=False)
        print(f"Arguments successfully written to {filename}")
    except Exception as e:
        print(f"Error: {e}")


def get_args():
    parser = argparse.ArgumentParser("Hyperparameter Setting for Experiment")
    # Experiment arguments
    parser.add_argument("--exp_name", type=str, default="run_demo", help="Name of this experiment")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")

    # Tools arguments
    parser.add_argument("--use_wandb", default=False, action="store_true", help="Whether to use wandb")
    parser.add_argument("--wandb_api_key", type=str, help="Wandb API Key")

    # Training arguments
    parser.add_argument("--train_episode", type=int, default=4, help=" Maximum number of training episode")
    parser.add_argument("--train_inner_steps", type=int, default=512, help=" Maximum number of steps per training episode")
    parser.add_argument("--save_freq", type=int, default=128, help="The number of episodes per saving")
    parser.add_argument("--batch_size", type=int, default=2, help="The number of episodes for sampling for one training")
    parser.add_argument("--mini_batch_size", type=int, default=1, help="The number of episodes used for one training epoch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of NNs")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=1.0, help="GAE lambda")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=30, help="The number of update for one training")

    # Evaluation arguments
    parser.add_argument("--model_name", type=str, default=None, help="Evaluate Model name")
    parser.add_argument("--evaluate_inner_steps", type=int, default=512, help="Maximum number of steps per evaluation episode")

    # File arguments
    parser.add_argument("--img_dir", type=str, default="img", help="Folder to save images")
    parser.add_argument("--save_csv", default=True, help="Folder to save images")
    parser.add_argument("--csv_dir", type=str, default="csv", help="Folder to save csv files")

    # Environment arguments
    parser.add_argument("--use_rgb_norm", type=bool, default=True, help="map 0-255 to 0-1")
    parser.add_argument("--env_parallel_num", default=4, type=int, help="The number of parallel envs (i.e. the size of 'batch' in  the input shape of NN (batch, channel, height, width))")
    parser.add_argument("--use_collective_reward", default=False, action="store_true", help="Whether to use collective reward")
    parser.add_argument("--use_inequity_averse_reward", default=False, action="store_true", help="Whether to use inequity averaging reward")
    parser.add_argument("--num_inequity", type=int, default=0, help="Number of inequity agents")
    parser.add_argument("--alpha", type=float, default=5.0, help="inequity_averse_reward alpha")
    parser.add_argument("--beta", type=float, default=0.05, help="inequity_averse_reward beta")

    # Agent arguments
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--use_rnn", default=True, action="store_true", help="Whether to use RNN")
    parser.add_argument("--network_layer", type=int, default=2, help="The layer number of policy network and value network")

    # PPO tricks
    parser.add_argument("--use_adv_norm", type=bool, default=False, help="Trick: advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick: state normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.001, help="Trick: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="Trick:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=False, help="Trick: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=False, help="Trick: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=False, help="Trick: tanh activation function")
    
    args = parser.parse_args()
    return args
