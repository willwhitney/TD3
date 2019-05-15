import numpy as np
import torch
import gym
import argparse
import os
import time
from baselines import bench
import sys
import skimage.transform
import torch.nn.functional as F

import utils
import TD3
import OurDDPG
import DDPG
from pixel_wrapper import PixelObservationWrapper

# so it can find the action decoder class and LinearPointMass
sys.path.insert(0, '../action-embedding')
from pointmass import point_mass

import reacher_family

def encode(obs):
    with torch.no_grad():
        obs = torch.tensor(obs).float().unsqueeze(0).cuda()
        regressed_obs = model.encode_state(obs)[0][0].cpu().numpy()
        # print(regressed_obs.max(), regressed_obs.min())
    return regressed_obs


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for episode in range(eval_episodes):
        obs = env.reset()
        obs = encode(obs)
        policy.reset()
        done = False
        while not done:
            action = policy.select_action(obs)
            # import ipdb; ipdb.set_trace()
            obs, reward, done, _ = env.step(action)
            obs = encode(obs)
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
        obs = encode(obs)
        policy.reset()

        frame = env.render_obs(color_last=True) * 255
        frames.append(frame)
        done = False
        while not done:
            action = policy.select_action(obs)
            obs, reward, done, _ = env.step(action)
            obs = encode(obs)
            frame = env.render_obs(color_last=True) * 255

            # frame[:, :, 1] = (frame[:, :, 1].astype(float) + reward * 100).clip(0, 255)
            frames.append(frame)

    utils.save_gif('{}/{}.mp4'.format(log_dir, total_timesteps),
                   [torch.tensor(frame.copy()).float()/255 for frame in frames],
                   color_last=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None)                         # Job name
    parser.add_argument("--policy_name", default="TD3")                 # Policy name
    parser.add_argument("--env_name", default="Reacher-v2")             # OpenAI gym environment name
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

    parser.add_argument("--source_env", default=None)                   # env name to take the decoder from
    parser.add_argument("--decoder", default=None, type=str)            # Name of saved decoder
    parser.add_argument("--dummy_decoder", action="store_true")         # use a dummy decoder that repeats actions
    parser.add_argument('--dummy_traj_len', type=int, default=1)        # traj_len of dummy decoder
    parser.add_argument("--replay_size", default=1e6, type=int)         # Size of replay buffer
    parser.add_argument("--render_freq", default=5e3, type=float)       # How often (time steps) we render
    
    parser.add_argument("--init", action="store_true")                  # use the initialization from DDPG for networks
    parser.add_argument("--stack", default=4, type=int)                 # frames to stack together as input
    parser.add_argument("--img_width", default=64, type=int)            # size of frames

    args = parser.parse_args()
    args.save_models = not args.no_save_models

    if args.name is None:
        args.name = "Pixel{}_{}_seed{}".format(args.env_name, args.policy_name, args.seed)

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

    # add a Monitor and log the command-line options
    log_dir = "results/{}/".format(args.name)
    os.makedirs(log_dir, exist_ok=True)
    env = PixelObservationWrapper(env, stack=args.stack, img_width=args.img_width)
    # env = bench.Monitor(env, log_dir, allow_early_resets=True)
    utils.write_options(args, log_dir)


    model_path = "../action-embedding/results/{}/{}/model_200.pt".format(args.source_env, args.decoder)
    print("Loading model from {}".format(model_path))
    model = torch.load(model_path).cuda().eval()

    state_dim = model.state_embed_size
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # # Initialize policy
    if args.policy_name == "TD3": policy = TD3.TD3(state_dim, action_dim, max_action)
    elif args.policy_name == "OurDDPG": policy = OurDDPG.DDPG(state_dim, action_dim, max_action)
    elif args.policy_name == "DDPG": policy = DDPG.DDPG(state_dim, action_dim, max_action)
    print(model)
    print(policy.actor)
    print(policy.critic)

    replay_buffer = utils.ReplayBuffer(max_size=args.replay_size)
    # replay_buffer = utils.ReplayDataset(max_size=args.replay_size)

    # Evaluate untrained policy
    evaluations = [(0, 0, evaluate_policy(policy))]

    total_timesteps = 0
    timesteps_since_eval = 0
    timesteps_since_render = 0
    episode_num = 0
    done = True
    start_time = time.time()
    # predictions = []
    # true = []



    while total_timesteps < args.max_timesteps:

        if done:
            print("Episode time: {:.3f}s".format(time.time() - start_time))

            if total_timesteps != 0:
                print("Total T: %d Episode Num: %d Episode T: %d Reward: %f" % (total_timesteps, episode_num, episode_timesteps, episode_reward))
                if args.policy_name == "TD3":
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq)
                else:
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

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
            obs = encode(obs)
            # predictions.append(obs)
            # true.append(env.unwrapped._get_obs())

            policy.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            start_time = time.time()

            # if total_timesteps > 0:
            #     import ipdb; ipdb.set_trace()
            #     pred_error = F.mse_loss(torch.from_numpy(np.stack(predictions)).float(), torch.from_numpy(np.stack(true)).float())
            #     print("encoder prediction error: {:.3f}".format(pred_error.item()))
            #     predictions = []
            #     true = []


        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        new_obs = encode(new_obs)
        # predictions.append(new_obs)
        # true.append(env.unwrapped._get_obs())

        done_bool = 0 if episode_timesteps + 1 == env_max_steps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

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
