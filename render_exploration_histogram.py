import numpy as np
import torch
import gym
import argparse
import os
# from baselines import bench
import sys
import time

import pandas as pd
import scipy.stats
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
from pointmass import point_mass
sys.path.insert(0, '../action-embedding')

import reacher_family

def render_exploration(env, policy, filename, eval_episodes=10, max_steps=1000, query_dim=0, save=True):
    avg_reward = 0.
    uenv = env.unwrapped
    start_obs = env.reset()

    # this is only sufficient since I don't care about the goal/target
    start_state = [uenv.sim.data.qpos.copy(), uenv.sim.data.qvel.copy()]
    print("Start state: ", start_state)

    visited = []
    for episode in range(eval_episodes):
        env.reset()
        uenv.set_state(*start_state)
        obs = start_obs
        # visited.append([episode, *(uenv.sim.data.qpos[:2] - start_state[0][:2])])
        policy.reset()
        done = False
        for _ in range(max_steps):
            if any([isinstance(policy, EmbeddedTD3.EmbeddedTD3),
                    isinstance(policy, RandomEmbeddedPolicy)]):
                action, _, _ = policy.select_action(np.array(obs))
            else:
                action = policy.select_action(np.array(obs))
            # if episode == 0: print(action)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            if done: break
        visited.append([episode, (uenv.sim.data.qpos[query_dim] - start_state[0][query_dim])])

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")

    # data = pd.DataFrame(visited, columns=['episode', 'x', 'y'])
    # chart = alt.Chart(data).mark_circle(clip=True).encode(
    #     x=alt.X('x', scale=alt.Scale(domain=[-0.16, 0.16], nice=False)),
    #     y=alt.Y('y', scale=alt.Scale(domain=[-0.16, 0.16], nice=False)),
    #     color='episode:N',
    # ).interactive().properties(width=400, height=400)
    # chart.save("{}.html".format(filename))

    # import ipdb; ipdb.set_trace()
    data = pd.DataFrame(visited, columns=['episode', 'x'])
    hist = alt.Chart(data).mark_bar().encode(
        x=alt.X('x', bin=alt.Bin(maxbins=100)),
        y=alt.Y('count()')
        # x=alt.X('x', bin=alt.Bin(step=2), scale=alt.Scale(domain=[-60, 60])),
        # y=alt.Y('count()', scale=alt.Scale(domain=[0, 110]))
    ).interactive().properties(width=400, height=400)
    if save: hist.save("{}_hist.html".format(filename))

    data_np = np.array([d[1] for d in visited])
    data_hist, _ = np.histogram(data_np, bins=np.linspace(-0.25, 0.25, 100), density=True)
    print("Entropy of state distribution: ", scipy.stats.entropy(data_hist))
    return hist


def load_decoder(env_name, name):
    if 'SparsishPointMass' in env_name: base_env_name = 'SparsishPointMass-v0'
    elif 'Linear1DPointMass' in env_name: base_env_name = 'Linear1DPointMass-v0'
    elif '1DPointMass' in env_name: base_env_name = '1DPointMass-v0'
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
    parser.add_argument("--episodes", default=10, type=int)             # how many trials to run
    parser.add_argument("--max_steps", default=1000, type=int)          # how many steps to run each episode for
    parser.add_argument("--query_dim", default=0, type=int)             # which dimension to histogram

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
            policy = RandomEmbeddedPolicy(1, decoder, None)
        elif args.dummy_decoder:
            decoder = DummyDecoder(action_dim, args.dummy_traj_len, env.action_space)
            policy = RandomEmbeddedPolicy(1, decoder, 1)
        else:
            policy = RandomPolicy(env.action_space)
    elif args.policy_name == 'constant':
        policy = ConstantPolicy(env.action_space)
    else:
        assert False

    render_exploration(env, policy, "{}_{}".format(args.env_name, args.name),
                       eval_episodes=args.episodes, max_steps=args.max_steps,
                       query_dim=args.query_dim)
