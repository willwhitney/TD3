import numpy as np
import torch
import gym
import argparse
import os
from baselines import bench
import sys
import time

import utils
import TD3
import EmbeddedTD3
import RandomPolicy
import OurDDPG
import DDPG
from DummyDecoder import DummyDecoder
from RandomPolicy import RandomPolicy, ConstantPolicy
from RandomEmbeddedPolicy import RandomEmbeddedPolicy

import sys
# so it can find the action decoder class and LinearPointMass
# sys.path.insert(0, '../action-embedding')
from pointmass import point_mass

import reacher_family

def render_policy(policy, filename, render_mode='rgb_array', eval_episodes=5):
    frames = []
    avg_reward = 0.
    for episode in range(eval_episodes):
        obs = env.reset()
        policy.reset()
        frames.append(env.render(mode=render_mode))
        done = False
        while not done:
            if any([isinstance(policy, EmbeddedTD3.EmbeddedTD3),
                    isinstance(policy, RandomEmbeddedPolicy)]):
                action, _, _ = policy.select_action(np.array(obs))
            else:
                action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            frame = env.render(mode=render_mode)
            # frame[:, :, 1] = (frame[:, :, 1].astype(float) + reward * 100).clip(0, 255)

            frames.append(frame)
            if render_mode == 'human':
                time.sleep(0.05)

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")

    utils.save_gif('{}.mp4'.format(filename),
                   [torch.tensor(frame.copy()).float()/255 for frame in frames],
                   color_last=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None)                         # Job name
    parser.add_argument("--policy_name", default="TD3")                 # Policy name
    parser.add_argument("--env_name", default="HalfCheetah-v1")         # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds

    parser.add_argument("--decoder", default=None, type=str)            # Name of saved decoder
    parser.add_argument("--dummy_decoder", action="store_true")         # use a dummy decoder that repeats actions
    parser.add_argument('--dummy_traj_len', type=int, default=1)        # traj_len of dummy decoder
    parser.add_argument('--human', action="store_true")                 # render interactively
    args = parser.parse_args()

    if args.env_name.startswith('dm'):
        import dm_control2gym
        _, domain, task = args.env_name.split('.')
        env = dm_control2gym.make(domain_name=domain, task_name=task)
        env_max_steps = 1000
    else:
        env = gym.make(args.env_name)
        env_max_steps = env._max_episode_steps

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if args.policy_name == 'TD3':
        policy = TD3.load('policy', 'results/{}'.format(args.name))
    elif args.policy_name == 'EmbeddedTD3':
        policy = EmbeddedTD3.load('policy', 'results/{}'.format(args.name))
    elif args.policy_name == 'random':
        if args.decoder:
            decoder = load_decoder(args.env_name, args.decoder)
            policy = RandomEmbeddedPolicy(1, decoder, 4)
        elif args.dummy_decoder:
            decoder = DummyDecoder(action_dim, args.dummy_traj_len, env.action_space)
            policy = RandomEmbeddedPolicy(1, decoder, 1)
        else:
            policy = RandomPolicy(env.action_space)
    elif args.policy_name == 'constant':
        policy = ConstantPolicy(env.action_space)
    else:
        assert False


    render_mode = 'human' if args.human else 'rgb_array'
    render_policy(policy, "{}_{}".format(args.env_name, args.name), render_mode)
