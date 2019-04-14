import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import utils

from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

def build_conv(arch, img_width, spec_norm=False):
    def add_spec_norm(module):
        if spec_norm: return nn.utils.spectral_norm(module)
        else: return module

    if arch == "mine":
        # conv_output_dim = 576
        conv_layers = nn.ModuleList([
            add_spec_norm(nn.Conv2d(9, 32, 8, stride=2)),
            add_spec_norm(nn.Conv2d(32, 32, 4, stride=1)),
            add_spec_norm(nn.Conv2d(32, 1, 3)),
        ])

    elif arch == "mine_bn":
        # conv_output_dim = 576
        conv_layers = nn.ModuleList([
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 32, 8, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 4, stride=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 3),
        ])

    elif arch == "minev2":
        # conv_output_dim = 2704
        conv_layers = nn.ModuleList([
            nn.Conv2d(9, 32, 8, stride=1),
            nn.Conv2d(32, 32, 4, stride=1),
            nn.Conv2d(32, 1, 3),
        ])

    elif arch == "minev3":
        # conv_output_dim = 3025
        conv_layers = nn.ModuleList([
            nn.Conv2d(9, 32, 4, stride=1),
            nn.Conv2d(32, 32, 4, stride=1),
            nn.Conv2d(32, 1, 4),
        ])

    elif arch == "minev4":
        # conv_output_dim = 1849
        conv_layers = nn.ModuleList([
            nn.Conv2d(9, 32, 8, stride=1),
            nn.Conv2d(32, 32, 8, stride=1),
            nn.Conv2d(32, 1, 8),
        ])

    elif arch == "ilya":
        # conv_output_dim = 512
        conv_layers = nn.ModuleList([
            nn.Conv2d(9, 32, 8, stride=4),
            nn.Conv2d(32, 32, 4, stride=2),
            nn.Conv2d(32, 32, 3),
        ])

    elif arch == "ilya_bn":
        # conv_output_dim = 512
        conv_layers = nn.ModuleList([
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 32, 8, stride=4),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 4, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3),
        ])

    conv_output_dim = utils.prod(utils.conv_list_out_dim(conv_layers, img_width, img_width))
    return conv_layers, conv_output_dim


def ddpg_init(conv_layers, lin_layers):
    for layer in [*conv_layers, *lin_layers[:-1]]:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(layer.weight, -bound, bound)
            init.uniform_(layer.bias, -bound, bound)

    init.uniform_(lin_layers[-1].weight, -3e-4, 3e-4)
    init.uniform_(lin_layers[-1].bias, -3e-4, 3e-4)



class Actor(nn.Module):
    def __init__(self, action_dim, max_action, arch, initialize, img_width):
        super(Actor, self).__init__()
        self.conv_layers, self.conv_output_dim = build_conv(arch, img_width, spec_norm=True)

        if "bn" in arch:
            self.lin_layers = nn.ModuleList([
                nn.BatchNorm1d(self.conv_output_dim),
                nn.Linear(self.conv_output_dim, 200),
                nn.BatchNorm1d(200),
                nn.Linear(200, 200),
                nn.BatchNorm1d(200),
                nn.Linear(200, action_dim),
            ])
        else:
            self.lin_layers = nn.ModuleList([
                nn.utils.spectral_norm(nn.Linear(self.conv_output_dim, 200)),
                nn.utils.spectral_norm(nn.Linear(200, 200)),
                nn.Linear(200, action_dim),
            ])

        if init: ddpg_init(self.conv_layers, self.lin_layers)

        self.max_action = max_action

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        for layer in self.conv_layers:
            x = F.relu(layer(x))

        # import ipdb; ipdb.set_trace()
        # print("conv output dim: ", x.size())


        x = x.view(-1, self.conv_output_dim)

        for i, layer in enumerate(self.lin_layers):
            x = layer(x)
            if i < (len(self.lin_layers) - 1): x = F.relu(x)

        x = self.max_action * torch.tanh(x)
        return x


class Critic(nn.Module):
    def __init__(self, action_dim, arch, initialize, img_width):
        super(Critic, self).__init__()
        # self.q1_conv_layers, self.conv_output_dim = build_conv(arch, img_width)
        self.q1_conv_layers, self.conv_output_dim = build_conv(arch, img_width, spec_norm=True)
        self.q2_conv_layers, _ = build_conv(arch, img_width)


        self.q1_lin_layers = nn.ModuleList([
            nn.utils.spectral_norm(nn.Linear(self.conv_output_dim + action_dim, 200)),
            # nn.Linear(self.conv_output_dim + action_dim, 200),
            nn.utils.spectral_norm(nn.Linear(200, 200)),
            # nn.Linear(200, 200),
            nn.Linear(200, 1),
        ])

        self.q2_lin_layers = nn.ModuleList([
            nn.utils.spectral_norm(nn.Linear(self.conv_output_dim + action_dim, 200)),
            # nn.Linear(self.conv_output_dim + action_dim, 200),
            nn.utils.spectral_norm(nn.Linear(200, 200)),
            # nn.Linear(200, 200),
            nn.Linear(200, 1),
        ])

        if init:
            ddpg_init(self.q1_conv_layers, self.q1_lin_layers)
            ddpg_init(self.q2_conv_layers, self.q2_lin_layers)


    def forward(self, x, u):
        return self.Q1(x, u), self.Q2(x, u)


    def Q1(self, x, u):
        for layer in self.q1_conv_layers:
            x = F.relu(layer(x))

        # import ipdb; ipdb.set_trace()
        x = x.view(-1, self.conv_output_dim)
        x = torch.cat([x, u], dim=1)
        for i, layer in enumerate(self.q1_lin_layers):
            x = layer(x)
            if i < (len(self.q1_lin_layers) - 1): x = F.relu(x)
        
        return x

    def Q2(self, x, u):
        for layer in self.q2_conv_layers:
            x = F.relu(layer(x))

        x = x.view(-1, self.conv_output_dim)
        x = torch.cat([x, u], dim=1)
        for i, layer in enumerate(self.q2_lin_layers):
            x = layer(x)
            if i < (len(self.q2_lin_layers) - 1): x = F.relu(x)
        
        return x




class TD3Pixels(object):
    def __init__(self, state_dim, action_dim, max_action, arch="mine", initialize=True, img_width=128):
        self.actor = Actor(action_dim, max_action, arch, initialize, img_width).to(device)
        self.actor_target = Actor(action_dim, max_action, arch, initialize, img_width).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(action_dim, arch, initialize, img_width).to(device)
        self.critic_target = Critic(action_dim, arch, initialize, img_width).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        print(self.actor)
        print(self.critic)
        print("Actor params: ", sum([p.nelement() for p in self.actor.parameters()]))

        self.max_action = max_action
        # self.data_loader = None

    # @profile
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action = self.actor(state)
        action = action.cpu()
        action = action.data.numpy()
        action = action.flatten()
        return action

    def reset(self):
        pass

    def mode(self, mode):
        if mode == 'eval':
            self.actor.eval()
        elif mode == 'train':
            self.actor.train()
        else:
            assert False

    # @profile
    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.mode('train')


        loader = DataLoader(replay_buffer, batch_size, shuffle=True, num_workers=0)
        for it, batch in enumerate(loader):
            if it >= iterations: break
            state, next_state, action, reward, done = batch
            # import ipdb; ipdb.set_trace()
            
            state = state.to(device)
            next_state = next_state.to(device)
            action = action.to(device)
            reward = reward.float().unsqueeze(1).to(device)
            done = (1 - done).float().unsqueeze(1).to(device)


        # for it in range(iterations):

        #     # Sample replay buffer
        #     x, y, u, r, d = replay_buffer.sample(batch_size)
        #     state = torch.FloatTensor(x).to(device)
        #     action = torch.FloatTensor(u).to(device)
        #     next_state = torch.FloatTensor(y).to(device)
        #     done = torch.FloatTensor(1 - d).to(device)
        #     reward = torch.FloatTensor(r).to(device)


            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(action.size()).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()
            # import ipdb; ipdb.set_trace()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                # import ipdb; ipdb.set_trace()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        self.mode('eval')


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self, '%s/%s_all.pth' % (directory, filename))


    def load(self, filename, directory):
        if not torch.cuda.is_available():
            self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename), map_location='cpu'))
            self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename), map_location='cpu'))
        else:
            self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
            self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

def load(filename, directory):
    if not torch.cuda.is_available():
        return torch.load('%s/%s_all.pth' % (directory, filename), map_location='cpu')
    else:
        return torch.load('%s/%s_all.pth' % (directory, filename))
