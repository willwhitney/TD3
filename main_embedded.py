import numpy as np
import torch
import gym
import argparse
import os
from baselines import bench

import utils
import TD3
from EmbeddedTD3 import EmbeddedTD3
from RandomEmbeddedPolicy import RandomEmbeddedPolicy
import OurDDPG
import DDPG
from DummyDecoder import DummyDecoder

import sys
# so it can find the action decoder class and LinearPointMass
sys.path.insert(0, '../action-embedding')
from pointmass import point_mass

import reacher_family

# so it can find SparseReacher
# sys.path.insert(0, '../pytorch-a2c-ppo-acktr')
# import envs
# print("imports done")
# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        policy.reset()
        done = False
        while not done:
            action, _, _ = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward


def render_policy(policy, log_dir, total_timesteps, eval_episodes=5):
	frames = []
	for episode in range(eval_episodes):
		obs = env.reset()
		policy.reset()
		frames.append(env.render(mode='rgb_array'))
		done = False
		while not done:
			action, _, _ = policy.select_action(np.array(obs))
			obs, reward, done, _ = env.step(action)
			frame = env.render(mode='rgb_array')

			frame[:, :, 1] = (frame[:, :, 1].astype(float) + reward * 100).clip(0, 255)
			frames.append(frame)

	utils.save_gif('{}/{}.mp4'.format(log_dir, total_timesteps),
				   [torch.tensor(frame.copy()).float()/255 for frame in frames],
				   color_last=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None)                         # Job name
    parser.add_argument("--policy_name", default="EmbeddedTD3")         # Policy name
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
    parser.add_argument("--max_e_action", default=None, type=int)       # Clip the scale of the action embeddings
    parser.add_argument("--render_freq", default=5e3, type=float)       # How often (time steps) we render
    args = parser.parse_args()
    args.save_models = not args.no_save_models

    if args.name is None:
        args.name = "{}_{}_seed{}".format(args.env_name, args.policy_name, args.seed)

    # file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (args.name))
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    if args.env_name.startswith('dm'):
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

    # add a Monitor and log the command-line options
    log_dir = "results/{}/".format(args.name)
    os.makedirs(log_dir, exist_ok=True)
    env = bench.Monitor(env, log_dir, allow_early_resets=True)
    utils.write_options(args, log_dir)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if args.decoder is not None:
        if 'SparsishPointMass' in args.env_name: base_env_name = 'SparsishPointMass-v0'
        elif 'PointMass' in args.env_name: base_env_name = 'LinearPointMass-v0'
        elif 'ReacherVertical' in args.env_name: base_env_name = 'ReacherVertical-v2'
        elif 'ReacherPush' in args.env_name: base_env_name = 'ReacherVertical-v2'
        elif 'ReacherSpin' in args.env_name: base_env_name = 'ReacherVertical-v2'
        elif 'ReacherTurn' in args.env_name: base_env_name = 'ReacherVertical-v2'
        elif 'ReacherTest' in args.env_name: base_env_name = 'ReacherTest-v2'
        elif 'Reacher' in args.env_name: base_env_name = 'Reacher-v2'
        elif 'Striker' in args.env_name: base_env_name = 'Pusher-v2'
        elif 'Thrower' in args.env_name: base_env_name = 'Pusher-v2'
        elif 'dm.manipulator' in args.env_name: base_env_name = 'dm.manipulator.bring_ball'
        else: base_env_name = args.env_name.strip("Super").strip("Sparse")
        decoder_path = "../action-embedding/results/{}/{}/decoder.pt".format(base_env_name, args.decoder)
        print("Loading decoder from {}".format(decoder_path))
        decoder = torch.load(decoder_path)
    elif args.dummy_decoder:
        decoder = DummyDecoder(action_dim, args.dummy_traj_len, env.action_space)
    decoder.max_embedding = float(decoder.max_embedding)

    if args.max_e_action is not None: decoder.max_embedding = min(decoder.max_embedding, args.max_e_action)
    args.policy_noise = args.policy_noise * decoder.max_embedding / decoder.traj_len
    args.expl_noise = args.expl_noise * decoder.max_embedding

    policy = EmbeddedTD3(state_dim, action_dim, max_action, decoder)
    random_policy = RandomEmbeddedPolicy(max_action, decoder)

    replay_buffer = utils.EmbeddedReplayBuffer(max_size=args.replay_size)

    # Evaluate untrained policy
    evaluations = [(0, 0, evaluate_policy(policy))]

    total_timesteps = 0
    timesteps_since_eval = 0
    timesteps_since_render = 0
    episode_num = 0
    done = True

    while total_timesteps < args.max_timesteps:

        if done:

            if total_timesteps != 0:
                print("Total T: %d Episode Num: %d Episode T: %d Reward: %f" % (total_timesteps, episode_num, episode_timesteps, episode_reward))
                policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau,
                             args.policy_noise, args.noise_clip, args.policy_freq)

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluations.append((episode_num, total_timesteps, evaluate_policy(policy)))

                if args.save_models: policy.save("policy", directory=log_dir)
                np.save("{}/eval.npy".format(log_dir), np.stack(evaluations))

            if timesteps_since_render >= args.render_freq:
                timesteps_since_render %= args.render_freq
                render_policy(policy, log_dir, total_timesteps)


            # Reset environment
            obs = env.reset()
            policy.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            if done: random_policy.reset()
            action, e_action, plan_step = random_policy.select_action(np.array(obs))
        else:
            action, e_action, plan_step = policy.select_action(np.array(obs), args.expl_noise)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == env_max_steps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, e_action, plan_step, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        timesteps_since_render += 1

    # Final evaluation
    evaluations.append((episode_num, total_timesteps, evaluate_policy(policy)))
    np.save("{}/eval.npy".format(log_dir), np.stack(evaluations))
    render_policy(policy, log_dir, total_timesteps)
    if args.save_models: policy.save("policy", directory=log_dir)
