import matplotlib.pyplot as plt
import numpy as np
from args import get_args, copy_args_to_yaml
from runners.clean_up_ppo_runner import CleanupPPORunner
from datetime import datetime
import os


args = get_args()
runner = CleanupPPORunner(args)
if args.model_name:
    runner.evaluate()
else:
    path = f'./results/{args.exp_name}' #/{datetime.now().strftime("%m%d")}
    os.makedirs(path, exist_ok=True)
    copy_args_to_yaml(args, f'{path}/config.yaml')
    runner.train()
    runner.data_collector.save_to_csv(path)
