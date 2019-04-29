import numpy as np
import torch
from torch import nn
import gym
import argparse
import os
from baselines import bench
import sys
import skimage.transform
import random

import utils
import TD3_pixels
import OurDDPG
import DDPG
from pixel_wrapper import PixelObservationWrapper

from torch.utils.data import DataLoader
import torch.nn.functional as F

# so it can find the action decoder class and LinearPointMass
sys.path.insert(0, '../action-embedding')
from pointmass import point_mass

import reacher_family

class StateRegressor(nn.Module):
    def __init__(self, arch, state_dim, stack=3, img_width=32):
        super().__init__()
        self.conv_layers, self.conv_output_dim = TD3_pixels.build_conv(arch, img_width, stack)
        self.lin_layers = nn.ModuleList([
            nn.Linear(self.conv_output_dim, 200),
            nn.Linear(200, 200),
            nn.Linear(200, state_dim),
        ])

    def forward(self, x):
        for layer in self.conv_layers:
            x = F.relu(layer(x))

        x = x.view(-1, self.conv_output_dim)
        for i, layer in enumerate(self.lin_layers):
            x = layer(x)
            if i < (len(self.lin_layers) - 1): x = F.relu(x)

        return x

def get_state(env):
    return env.unwrapped._get_obs()
    # return np.concatenate([
    #     env.unwrapped.sim.data.qpos.flat,
    #     env.unwrapped.model.body_pos[-1, (-3, -1)],    # target's (x, z) coords
    # ])

def run_epoch(model, loader, train=False):
    if train: model.train()
    else: model.eval()

    mean_loss = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            x, y, _, _, _ = batch
            x = x.cuda()
            y = y.cuda()
            optim.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            if train:
                loss.backward()
                optim.step()
            mean_loss += loss.item()
    # print(mean_loss / len(loader), len(loader))
    # import ipdb; ipdb.set_trace()
    mean_loss = mean_loss / len(loader)

    if train:
        print("Train loss:", epoch, mean_loss)
    else:
        scheduler.step(mean_loss)
        print("Eval loss:", epoch, mean_loss)
        print(utils.flat_str(pred[0]))
        print(utils.flat_str(y[0]))
        # print(torch.stack([pred[0], y[0]], dim=1))

def generate_dataset(size):
    dataset = utils.ReplayDataset(max_size=size)
    total_timesteps = 0
    episode_num = 0
    done = True

    while total_timesteps < size:
        if total_timesteps % 10000 == 0: print("{}/{}".format(total_timesteps, size))
        if done:
            # Reset environment
            obs = env.reset()
            true_state = get_state(env)
            done = False
            episode_timesteps = 0
            episode_num += 1

        # import ipdb; ipdb.set_trace()
        action = env.action_space.sample()

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        new_true_state = get_state(env)
        done_bool = 0 if episode_timesteps + 1 == env_max_steps else float(done)
        # import ipdb; ipdb.set_trace()
        # Store data in replay buffer
        dataset.add((obs, true_state, action, reward, done_bool))

        obs = new_obs
        true_state = new_true_state

        episode_timesteps += 1
        total_timesteps += 1
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None)                         # Job name
    parser.add_argument("--policy_name", default="TD3")                 # Policy name
    parser.add_argument("--env_name", default="Reacher-v2")             # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=float)   # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)         # How often (time steps) we evaluate
    parser.add_argument("--train_timesteps", default=1e4, type=int)     # Max time steps to run environment for
    parser.add_argument("--eval_timesteps", default=1e3, type=int)      # Max time steps to run environment for
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
    
    parser.add_argument("--init", action="store_true")                  # use the initialization from DDPG for networks
    parser.add_argument("--arch", default="mine")                       # which network architecture to use (mine or one from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L176)
    parser.add_argument("--stack", default=3, type=int)                 # frames to stack together as input
    parser.add_argument("--img_width", default=32, type=int)            # size of frames
    args = parser.parse_args()
    args.save_models = not args.no_save_models

    if args.name is None:
        args.name = "Regress{}_{}_seed{}".format(args.env_name, args.policy_name, args.seed)

    # file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (args.name))
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    if args.env_name.startswith('dm'):
        import os
        os.environ["MUJOCO_GL"] = 'osmesa'

        import dm_control2gym
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
    random.seed(args.seed)

    # add a Monitor and log the command-line options
    log_dir = "results/{}/".format(args.name)
    os.makedirs(log_dir, exist_ok=True)
    env = PixelObservationWrapper(env, stack=args.stack, img_width=args.img_width)
    env = bench.Monitor(env, log_dir, allow_early_resets=True)
    utils.write_options(args, log_dir)
    # import ipdb; ipdb.set_trace()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # model = StateRegressor(args.arch, len(env.unwrapped.sim.data.qpos) + 2).cuda()
    model = StateRegressor(args.arch, state_dim, stack=args.stack, img_width=args.img_width).cuda()
    print(model)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', verbose=True)

    data_path = "data/{}_{}_{}".format(args.env_name, args.img_width, args.train_timesteps)
    try:
        replay_buffer = utils.ReplayDataset(max_size=args.replay_size)
        replay_buffer.load(data_path)
    except FileNotFoundError:
        replay_buffer = generate_dataset(args.train_timesteps)
        replay_buffer.save(data_path)

    sample = replay_buffer[0]
    # import ipdb; ipdb.set_trace()



    # replay_buffer = utils.ReplayBuffer(max_size=args.replay_size)
    # data_path = "data/{}_{}_{}.pt".format(args.env_name, args.img_width, args.train_timesteps)
    # try:
    #     replay_buffer = torch.load(data_path)
    # except FileNotFoundError:
    #     replay_buffer = generate_dataset(args.train_timesteps)
    #     torch.save(replay_buffer, data_path)

    eval_replay_buffer = generate_dataset(args.eval_timesteps)
    loader = DataLoader(replay_buffer, num_workers=1, shuffle=True, batch_size=args.batch_size, drop_last=True)
    eval_loader = DataLoader(eval_replay_buffer, num_workers=1, shuffle=True, batch_size=args.batch_size, drop_last=True)

    for epoch in range(200):
        run_epoch(model, loader, train=True)
        run_epoch(model, eval_loader, train=False)
        print()
        if epoch % 10 == 0: torch.save(model, log_dir + "regress.pt")

    torch.save(model, log_dir + "regress.pt")

