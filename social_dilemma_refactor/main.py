import matplotlib.pyplot as plt
import numpy as np
from args import get_args, copy_args_to_yaml
from runners.clean_up_ppo_runner import CleanupPPORunner
from datetime import datetime


args = get_args()
runner = CleanupPPORunner(args)
if args.model_name:
    runner.evaluate()
# else:
#     copy_args_to_yaml(args, f'./results/{args.exp_name}/config.yaml') #/{datetime.now().strftime("%m%d")}
#     runner.train()
#     runner.data_collector.save_to_csv(f'./results/{args.exp_name}')
