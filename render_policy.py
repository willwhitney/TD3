import numpy as np
import torch
import gym
import argparse
import os
from baselines import bench
import sys

import dm_control2gym


import utils
import TD3
import EmbeddedTD3
import OurDDPG
import DDPG
from DummyDecoder import DummyDecoder

import sys
# so it can find the action decoder class and LinearPointMass
sys.path.insert(0, '../action-embedding')
from pointmass import point_mass

# so it can find SparseReacher
sys.path.insert(0, '../pytorch-a2c-ppo-acktr')
import envs


# from pyvirtualdisplay import Display
# display_ = Display(visible=0, size=(550, 500))
# display_.start()


# Runs policy for X episodes and returns average reward
# def evaluate_policy(policy, eval_episodes=10):
#     avg_reward = 0.
#     for episode in range(eval_episodes):
#         obs = env.reset()
#         policy.reset()
#         done = False
#         while not done:
#             if isinstance(policy, EmbeddedTD3.EmbeddedTD3):
#                 action = policy.select_action(np.array(obs))
#             else:
#                 action = policy.select_action(np.array(obs))
#             # import ipdb; ipdb.set_trace()
#             obs, reward, done, _ = env.step(action)
#             avg_reward += reward


#     avg_reward /= eval_episodes

#     print("---------------------------------------")
#     print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
#     print("---------------------------------------")
#     return avg_reward

def render_policy(policy, filename, eval_episodes=5):
    frames = []
    avg_reward = 0.
    for episode in range(eval_episodes):
        obs = env.reset()
        policy.reset()
        frames.append(env.render(mode='rgb_array'))
        done = False
        while not done:
            if isinstance(policy, EmbeddedTD3.EmbeddedTD3):
                action, _, _ = policy.select_action(np.array(obs))
            else:
                action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            frame = env.render(mode='rgb_array')

            frame[:, :, 1] = (frame[:, :, 1].astype(float) + reward * 100).clip(0, 255)
            frames.append(frame)

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")

    utils.save_gif('{}/{}.mp4'.format(filename),
                   [torch.tensor(frame.copy()).float()/255 for frame in frames],
                   color_last=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None)                         # Job name
    parser.add_argument("--policy_name", default="TD3")                 # Policy name
    parser.add_argument("--env_name", default="HalfCheetah-v1")         # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=float)   # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)         # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e7, type=float)     # Max time steps to run environment for
    parser.add_argument("--no_save_models", action="store_true")        # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)        # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)          # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)         # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)             # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)      # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)        # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)           # Frequency of delayed policy updates

    parser.add_argument("--decoder", default=None, type=str)            # Name of saved decoder
    parser.add_argument("--dummy_decoder", action="store_true")         # use a dummy decoder that repeats actions
    parser.add_argument('--dummy_traj_len', type=int, default=1)        # traj_len of dummy decoder
    parser.add_argument("--replay_size", default=1e6, type=int)         # Size of replay buffer
    parser.add_argument("--render_freq", default=5e3, type=float)       # How often (time steps) we render
    args = parser.parse_args()

    if args.env_name.startswith('dm'):
        _, domain, task = args.env_name.split('.')
        env = dm_control2gym.make(domain_name=domain, task_name=task)
        env_max_steps = 1000
    else:
        env = gym.make(args.env_name)
        env_max_steps = env._max_episode_steps

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    # import ipdb; ipdb.set_trace()

    # Initialize policy
    if args.decoder is not None:
        decoder = torch.load(
                "../action-embedding/results/{}/{}/decoder.pt".format(
                args.env_name.strip("Super").strip("Sparse"),
                args.decoder))
    elif args.dummy_decoder:
        decoder = DummyDecoder(action_dim, args.dummy_traj_len, env.action_space)
    if args.policy_name == "EmbeddedTD3": policy = EmbeddedTD3.EmbeddedTD3(state_dim, action_dim, max_action, decoder)
    elif args.policy_name == "TD3": policy = TD3.TD3(state_dim, action_dim, max_action)
    elif args.policy_name == "OurDDPG": policy = OurDDPG.DDPG(state_dim, action_dim, max_action)
    elif args.policy_name == "DDPG": policy = DDPG.DDPG(state_dim, action_dim, max_action)


    policy.load('policy', 'results/{}'.format(args.name))

    render_policy(policy, args.name)
