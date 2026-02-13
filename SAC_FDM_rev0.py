import numpy as np
from Point_reaching_environment import Custom_point_reaching
from stable_baselines3 import SAC
# from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Callable, Dict, List, Optional, Tuple, Union, Type
from stable_baselines3.common.utils import get_schedule_fn
import gym
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU only for everything

env_temp = Custom_point_reaching()
# env = gym.make('FDM_module3_2D_point_reaching')
env = DummyVecEnv([lambda: env_temp])

# run multiple processes
# num_process = 8
# env = make_vec_env(lambda: env_temp, n_envs=num_process)

global_path = 'trained_policy/'
# Callback function definition
checkpoint_callback = CheckpointCallback(
    save_freq=2000,                # save every this timesteps
    save_path=global_path,
    name_prefix='SAC_FDM_env_rev1_'
)
LEARNING_TIMESTEPS = 7_000_000    # to run the single model.learn on this many steps

length_desired_trajectory = env_temp.episode_length
model = SAC('MlpPolicy', env, verbose=1,
            learning_rate=0.0001, buffer_size=1000000, learning_starts=2 * length_desired_trajectory,
            batch_size=10, ent_coef='auto', device='cpu',
            tensorboard_log=global_path + 'logging_dir'
            )
model.learn(total_timesteps=LEARNING_TIMESTEPS, callback=checkpoint_callback, tb_log_name='SAC')

# -------------------------------------------------------------------- #



