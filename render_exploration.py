import numpy as np
import torch
import gym
import argparse
import os
from baselines import bench
import sys
import time

import pandas as pd
import altair as alt

import utils
import TD3
import EmbeddedTD3
from RandomPolicy import RandomPolicy, ConstantPolicy
from RandomEmbeddedPolicy import RandomEmbeddedPolicy
import OurDDPG
import DDPG
from DummyDecoder import DummyDecoder

import sys
# so it can find the action decoder class and LinearPointMass
sys.path.insert(0, '../action-embedding')
from pointmass import point_mass

import reacher_family

def render_exploration(env, policy, filename, eval_episodes=10):
    avg_reward = 0.
    uenv = env.unwrapped
    start_obs = env.reset()

    # this is only sufficient since I don't care about the goal/target
    start_state = [uenv.sim.data.qpos.copy(), uenv.sim.data.qvel.copy()]

    visited = []
    for episode in range(eval_episodes):
        env.reset()
        uenv.set_state(*start_state)
        obs = start_obs
        visited.append([episode, *(uenv.sim.data.qpos[:2] - start_state[0][:2])])
        policy.reset()
        done = False
        while not done:
            if any([isinstance(policy, EmbeddedTD3.EmbeddedTD3),
                    isinstance(policy, RandomEmbeddedPolicy)]):
                action, _, _ = policy.select_action(np.array(obs))
            else:
                action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            visited.append([episode, *(uenv.sim.data.qpos[:2] - start_state[0][:2])])
            avg_reward += reward

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")

    data = pd.DataFrame(visited, columns=['episode', 'x', 'y'])
    chart = alt.Chart(data).mark_circle().encode(
        x='x', y='y', color='episode:N',
    ).interactive().properties(width=400, height=300)
    chart.save("{}.html".format(filename))


def load_decoder(env_name, name):
    if 'SparsishPointMass' in env_name: base_env_name = 'SparsishPointMass-v0'
    elif 'PointMass' in env_name: base_env_name = 'LinearPointMass-v0'
    elif 'ReacherVertical' in env_name: base_env_name = 'ReacherVertical-v2'
    elif 'ReacherPush' in env_name: base_env_name = 'ReacherVertical-v2'
    elif 'ReacherSpin' in env_name: base_env_name = 'ReacherVertical-v2'
    elif 'ReacherTest' in env_name: base_env_name = 'ReacherTest-v2'
    elif 'Reacher' in env_name: base_env_name = 'Reacher-v2'
    elif 'Striker' in env_name: base_env_name = 'Pusher-v2'
    elif 'Thrower' in env_name: base_env_name = 'Pusher-v2'
    elif 'dm.manipulator' in env_name: base_env_name = 'dm.manipulator.bring_ball'
    else: base_env_name = env_name.strip("Super").strip("Sparse")
    decoder_path = "../action-embedding/results/{}/{}/decoder.pt".format(base_env_name, name)
    print("Loading decoder from {}".format(decoder_path))
    decoder = torch.load(decoder_path, map_location='cpu')
    return decoder


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None)                         # Job name
    parser.add_argument("--policy_name", default="TD3")                 # Policy name
    parser.add_argument("--env_name", default="HalfCheetah-v1")         # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds

    parser.add_argument("--decoder", default=None, type=str)            # Name of saved decoder
    parser.add_argument("--dummy_decoder", action="store_true")         # use a dummy decoder that repeats actions
    parser.add_argument('--dummy_traj_len', type=int, default=1)        # traj_len of dummy decoder
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
            policy = RandomEmbeddedPolicy(1, decoder)
        else:
            policy = RandomPolicy(env.action_space)
    elif args.policy_name == 'constant':
        policy = ConstantPolicy(env.action_space)
    else:
        assert False

    render_exploration(env, policy, "{}_{}".format(args.env_name, args.name))
