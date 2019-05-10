import math
import time
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

class Lambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class AddCoords(nn.Module):
    def forward(self, x):
        img_width = x.size()[-1]
        bsize = x.size(0)
        coords = torch.linspace(-1, 1, img_width).type_as(x)
        x_coords = coords.reshape(1, 1, 1, img_width).repeat(bsize, 1, img_width, 1)
        y_coords = coords.reshape(1, 1, img_width, 1).repeat(bsize, 1, 1, img_width)
        result = torch.cat([x, x_coords, y_coords], dim=1)
        # import ipdb; ipdb.set_trace()
        return result

def build_conv(arch, img_width, stack=3):
    if arch == "dummy":
        return nn.ModuleList([]), img_width**2 * stack * 3

    elif arch == "lin1":
        size = img_width**2 * stack * 3
        conv_layers = nn.ModuleList([
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.Linear(size, size),
        ])
        return conv_layers, size

    elif arch == "lin2":
        size = img_width**2 * stack * 3
        conv_layers = nn.ModuleList([
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.Linear(size, size),
            nn.Linear(size, size),
        ])
        return conv_layers, size

    elif arch == "lin3":
        size = img_width**2 * stack * 3
        conv_layers = nn.ModuleList([
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.Linear(size, size),
            nn.Linear(size, size),
            nn.Linear(size, size),
        ])
        return conv_layers, size

    elif arch == "lin1_wide":
        size = img_width**2 * stack * 3
        conv_layers = nn.ModuleList([
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.Linear(size, 200),
        ])
        return conv_layers, 200

    elif arch == "lin2_wide":
        size = img_width**2 * stack * 3
        conv_layers = nn.ModuleList([
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.Linear(size, 200),
            nn.Linear(200, 200),
        ])
        return conv_layers, 200

    elif arch == "lin3_wide":
        size = img_width**2 * stack * 3
        conv_layers = nn.ModuleList([
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.Linear(size, 200),
            nn.Linear(200, 200),
            nn.Linear(200, 200),
        ])
        return conv_layers, 200

    elif arch == "lin1_extrawide":
        size = img_width**2 * stack * 3
        conv_layers = nn.ModuleList([
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.Linear(size, 3200),
        ])
        return conv_layers, 3200

    elif arch == "lin2_extrawide":
        size = img_width**2 * stack * 3
        conv_layers = nn.ModuleList([
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.Linear(size, 200),
            nn.Linear(200, 3200),
        ])
        return conv_layers, 3200

    elif arch == "lin3_extrawide":
        size = img_width**2 * stack * 3
        conv_layers = nn.ModuleList([
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.Linear(size, 200),
            nn.Linear(200, 200),
            nn.Linear(200, 3200),
        ])
        return conv_layers, 3200

    elif arch == "lin1_mediumwide":
        size = img_width**2 * stack * 3
        conv_layers = nn.ModuleList([
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.Linear(size, 500),
        ])
        return conv_layers, 500

    elif arch == "lin2_mediumwide":
        size = img_width**2 * stack * 3
        conv_layers = nn.ModuleList([
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.Linear(size, 200),
            nn.Linear(200, 500),
        ])
        return conv_layers, 500

    elif arch == "lin3_mediumwide":
        size = img_width**2 * stack * 3
        conv_layers = nn.ModuleList([
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.Linear(size, 200),
            nn.Linear(200, 200),
            nn.Linear(200, 500),
        ])
        return conv_layers, 500

    elif arch == "conv1_1":
        conv_layers = nn.ModuleList([
            nn.Conv2d(stack * 3, stack * 3, 1),
        ])

    elif arch == "conv1_1_wide":
        conv_layers = nn.ModuleList([
            nn.Conv2d(stack * 3, 200, 1),
        ])

    elif arch == "conv1_3":
        conv_layers = nn.ModuleList([
            nn.Conv2d(stack * 3, stack * 3, 3),
        ])

    elif arch == "conv2_1":
        conv_layers = nn.ModuleList([
            nn.Conv2d(stack * 3, stack * 3, 1),
            nn.Conv2d(stack * 3, stack * 3, 1),
        ])

    elif arch == "conv2_1_wide":
        conv_layers = nn.ModuleList([
            nn.Conv2d(stack * 3, 200, 1),
            nn.Conv2d(200, 200, 1),
        ])

    elif arch == "conv2_3":
        conv_layers = nn.ModuleList([
            nn.Conv2d(stack * 3, stack * 3, 3),
            nn.Conv2d(stack * 3, stack * 3, 3),
        ])

    elif arch == "bn1":
        conv_layers = nn.ModuleList([
            nn.BatchNorm2d(stack * 3),
        ])
        return conv_layers, img_width**2 * stack * 3

    elif arch == "conv1_stride":
        conv_layers = nn.ModuleList([
            nn.Conv2d(stack * 3, stack * 3, 3, stride=2),
        ])

    elif arch == "conv2_stride":
        conv_layers = nn.ModuleList([
            nn.Conv2d(stack * 3, stack * 3, 3, stride=2),
            nn.Conv2d(stack * 3, stack * 3, 3, stride=2),
        ])

    elif arch == "conv2_stride_grow":
        conv_layers = nn.ModuleList([
            nn.Conv2d(stack * 3, stack * 3 * 2, 3, stride=2),
            nn.Conv2d(stack * 3 * 2, stack * 3 * 4, 3, stride=2),
        ])

    elif arch == "mine":
        # conv_output_dim = 576
        conv_layers = nn.ModuleList([
            nn.Conv2d(stack * 3, 32, 8, stride=2),
            nn.Conv2d(32, 32, 4, stride=1),
            nn.Conv2d(32, 1, 3),
        ])

    elif arch == "mine_bn":
        # conv_output_dim = 576
        conv_layers = nn.ModuleList([
            nn.BatchNorm2d(stack * 3),
            nn.Conv2d(stack * 3, 32, 8, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 4, stride=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 3),
        ])

    elif arch == "minev2":
        # conv_output_dim = 2704
        conv_layers = nn.ModuleList([
            nn.Conv2d(stack * 3, 32, 8, stride=1),
            nn.Conv2d(32, 32, 4, stride=1),
            nn.Conv2d(32, 1, 3),
        ])

    elif arch == "minev2_bn":
        # conv_output_dim = 2704
        conv_layers = nn.ModuleList([
            nn.BatchNorm2d(stack * 3),
            nn.Conv2d(stack * 3, 32, 8, stride=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 4, stride=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 3),
        ])

    elif arch == "minev3":
        # conv_output_dim = 3025
        conv_layers = nn.ModuleList([
            nn.Conv2d(stack * 3, 32, 4, stride=1),
            nn.Conv2d(32, 32, 4, stride=1),
            nn.Conv2d(32, 1, 4),
        ])

    elif arch == "minev3_bn":
        # conv_output_dim = 3025
        conv_layers = nn.ModuleList([
            nn.BatchNorm2d(stack * 3),
            nn.Conv2d(stack * 3, 32, 4, stride=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 4, stride=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 4),
        ])

    elif arch == "minev4":
        # conv_output_dim = 1849
        conv_layers = nn.ModuleList([
            nn.Conv2d(stack * 3, 32, 8, stride=1),
            nn.Conv2d(32, 32, 8, stride=1),
            nn.Conv2d(32, 1, 8),
        ])

    elif arch == "minev4_bn":
        # conv_output_dim = 1849
        conv_layers = nn.ModuleList([
            nn.BatchNorm2d(stack * 3),
            nn.Conv2d(stack * 3, 32, 8, stride=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 8, stride=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 8),
        ])

    elif arch == "ilya":
        # conv_output_dim = 512
        conv_layers = nn.ModuleList([
            nn.Conv2d(stack * 3, 32, 8, stride=4),
            nn.Conv2d(32, 32, 4, stride=2),
            nn.Conv2d(32, 32, 3),
        ])

    elif arch == "ilya_bn":
        # conv_output_dim = 512
        conv_layers = nn.ModuleList([
            nn.BatchNorm2d(stack * 3),
            nn.Conv2d(stack * 3, 32, 8, stride=4),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 4, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3),
        ])

    elif arch == "dcgan_bn":
        conv_layers = nn.ModuleList([
            nn.Conv2d(stack * 3, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            # nn.Conv2d(256, 512, 4, stride=2, padding=1),
        ])

    elif arch == "dcgan_coord_bn":
        conv_layers = nn.ModuleList([
            AddCoords(),
            nn.Conv2d(stack * 3 + 2, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            # nn.Conv2d(256, 512, 4, stride=2, padding=1),
        ])

    elif arch == "dcgandeep_bn":
        conv_layers = nn.ModuleList([
            nn.Conv2d(stack * 3, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
        ])

    elif arch == "impala":
        conv_layers = nn.ModuleList([
            nn.Conv2d(stack * 3, 16, 8, stride=4),
            nn.Conv2d(16, 32, 4, stride=2),
        ])

    elif arch == "impala_bn":
        conv_layers = nn.ModuleList([
            nn.BatchNorm2d(stack * 3),
            nn.Conv2d(stack * 3, 16, 8, stride=4),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 4, stride=2),
        ])

    elif arch == "ilya_coord":
        # conv_output_dim = 512
        conv_layers = nn.ModuleList([
            AddCoords(),
            nn.Conv2d(stack * 3 + 2, 32, 8, stride=4),
            nn.Conv2d(32, 32, 4, stride=2),
            nn.Conv2d(32, 32, 3),
        ])

    elif arch == "ilya_coord_bn":
        # conv_output_dim = 512
        conv_layers = nn.ModuleList([
            AddCoords(),
            nn.BatchNorm2d(stack * 3 + 2),
            nn.Conv2d(stack * 3 + 2, 32, 8, stride=4),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 4, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3),
        ])

    conv_output_dim = utils.prod(utils.conv_list_out_dim(conv_layers, img_width, img_width))
    return conv_layers, conv_output_dim


def ddpg_init(conv_layers, lin_layers):
    print("doing ddpg_init")
    for layer in [*conv_layers, *lin_layers[:-1]]:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(layer.weight, -bound, bound)
            init.uniform_(layer.bias, -bound, bound)

    init.uniform_(lin_layers[-1].weight, -3e-4, 3e-4)
    init.uniform_(lin_layers[-1].bias, -3e-4, 3e-4)



class Actor(nn.Module):
    def __init__(self, action_dim, max_action, arch, initialize, img_width, stack):
        super(Actor, self).__init__()
        self.conv_layers, self.conv_output_dim = build_conv(arch, img_width, stack)

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
                nn.Linear(self.conv_output_dim, 200),
                nn.Linear(200, 200),
                nn.Linear(200, action_dim),
            ])

        if initialize: ddpg_init(self.conv_layers, self.lin_layers)

        self.max_action = max_action

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear): 
                x = F.relu(x)

        x = x.view(-1, self.conv_output_dim)

        for i, layer in enumerate(self.lin_layers):
            x = layer(x)
            if i < (len(self.lin_layers) - 1) and isinstance(layer, nn.Linear): 
                x = F.relu(x)

        x = self.max_action * torch.tanh(x)
        # if x.size(0) > 1: print("Action: {:.3f}".format(x.abs().mean().item()))
        return x


class Critic(nn.Module):
    def __init__(self, action_dim, arch, initialize, img_width, stack):
        super(Critic, self).__init__()
        self.q1_conv_layers, self.conv_output_dim = build_conv(arch, img_width, stack)
        self.q2_conv_layers, _ = build_conv(arch, img_width, stack)

        self.action_repeat = self.conv_output_dim // action_dim
        action_dim = action_dim * self.action_repeat

        self.q1_lin_layers = nn.ModuleList([
            nn.Linear(self.conv_output_dim + action_dim, 200),
            nn.Linear(200, 200),
            nn.Linear(200, 1),
        ])

        self.q2_lin_layers = nn.ModuleList([
            nn.Linear(self.conv_output_dim + action_dim, 200),
            nn.Linear(200, 200),
            nn.Linear(200, 1),
        ])

        if initialize:
            ddpg_init(self.q1_conv_layers, self.q1_lin_layers)
            ddpg_init(self.q2_conv_layers, self.q2_lin_layers)


    def forward(self, x, u):
        return self.Q1(x, u), self.Q2(x, u)


    def Q1(self, x, u):
        for layer in self.q1_conv_layers:
            x = layer(x)
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear): x = F.relu(x)

        x = x.view(-1, self.conv_output_dim)
        # print("Q feature: {:.3f}".format(x.abs().mean().item()))
        # x = torch.cat([x, u], dim=1)
        x = torch.cat([x, u.repeat([1, self.action_repeat])], dim=1)
        for i, layer in enumerate(self.q1_lin_layers):
            x = layer(x)
            if i < (len(self.q1_lin_layers) - 1) and isinstance(layer, nn.Linear): x = F.relu(x)
        
        return x

    def Q2(self, x, u):
        for layer in self.q2_conv_layers:
            x = layer(x)
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear): x = F.relu(x)

        x = x.view(-1, self.conv_output_dim)
        # x = torch.cat([x, u], dim=1)
        x = torch.cat([x, u.repeat([1, self.action_repeat])], dim=1)
        for i, layer in enumerate(self.q2_lin_layers):
            x = layer(x)
            if i < (len(self.q2_lin_layers) - 1) and isinstance(layer, nn.Linear): x = F.relu(x)
        
        return x


# def lr_lambda(epoch):
#     return max(1 - epoch / 10000, 0)

class TD3Pixels(object):
    def __init__(self, state_dim, action_dim, max_action, arch="mine", initialize=True, img_width=128, stack=4, ddpglr=False, lr_schedule=False):
        self.actor = Actor(action_dim, max_action, arch, initialize, img_width, stack).to(device)
        self.actor_target = Actor(action_dim, max_action, arch, initialize, img_width, stack).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        if ddpglr:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        else:
            # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-3)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(action_dim, arch, initialize, img_width, stack).to(device)
        self.critic_target = Critic(action_dim, arch, initialize, img_width, stack).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        if ddpglr:
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        else:
            # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-3)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.lr_schedule = lr_schedule
        if self.lr_schedule:
            # linear LR decay from 1 to 0 over the course of 10K episodes
            self.actor_lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda=self.lr_lambda)
            self.critic_lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda=self.lr_lambda)

        print(self.actor)
        print(self.critic)
        print("Actor params: ", sum([p.nelement() for p in self.actor.parameters()]))

        self.max_action = max_action
        # self.data_loader = None

    def lr_lambda(self, epoch):
        max_episodes = self.lr_schedule / 100
        # import ipdb; ipdb.set_trace()
        return max(1 - epoch / max_episodes, 0)

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
        # start = time.time()

        loader = DataLoader(replay_buffer, batch_size, shuffle=True, num_workers=0, pin_memory=True)
        it = 0
        while it < iterations:
            for batch in loader:
                it += 1
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

        if self.lr_schedule:
            self.actor_lr_schedule.step()
            self.critic_lr_schedule.step()
            # import ipdb; ipdb.set_trace()
        # end = time.time()
        # print("Training time: {:.3f}".format(end - start))
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
