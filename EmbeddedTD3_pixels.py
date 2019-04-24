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

def build_conv(arch, img_width, stack=3):
    if arch == "mine":
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

        if init: ddpg_init(self.conv_layers, self.lin_layers)

        # self.max_action = max_action * 2
        self.max_action = max_action


    def forward(self, x):
        for layer in self.conv_layers:
            x = F.relu(layer(x))

        x = x.view(-1, self.conv_output_dim)

        for i, layer in enumerate(self.lin_layers):
            x = layer(x)
            if i < (len(self.lin_layers) - 1): x = F.relu(x)

        x = self.max_action * torch.tanh(x)
        return x


class Critic(nn.Module):
    def __init__(self, action_dim, arch, initialize, img_width, stack):
        super(Critic, self).__init__()

        self.q1_conv_layers, self.conv_output_dim = build_conv(arch, img_width, stack)
        self.q2_conv_layers, _ = build_conv(arch, img_width, stack)


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

        if init:
            ddpg_init(self.q1_conv_layers, self.q1_lin_layers)
            ddpg_init(self.q2_conv_layers, self.q2_lin_layers)


    def forward(self, x, u, i):
        return self.Q1(x, u, i), self.Q2(x, u, i)


    def Q1(self, x, u, i):
        for layer in self.q1_conv_layers:
            x = F.relu(layer(x))

        # import ipdb; ipdb.set_trace()
        x = x.view(-1, self.conv_output_dim)
        x = torch.cat([x, u, i], dim=1)
        for i, layer in enumerate(self.q1_lin_layers):
            x = layer(x)
            if i < (len(self.q1_lin_layers) - 1): x = F.relu(x)
        
        return x

    def Q2(self, x, u, i):
        for layer in self.q2_conv_layers:
            x = F.relu(layer(x))

        x = x.view(-1, self.conv_output_dim)
        x = torch.cat([x, u, i], dim=1)
        for i, layer in enumerate(self.q2_lin_layers):
            x = layer(x)
            if i < (len(self.q2_lin_layers) - 1): x = F.relu(x)
        
        return x

# numpy_type_map = {
#     'float64': torch.DoubleTensor,
#     'float32': torch.FloatTensor,
#     'float16': torch.HalfTensor,
#     'int64': torch.LongTensor,
#     'int32': torch.IntTensor,
#     'int16': torch.ShortTensor,
#     'int8': torch.CharTensor,
#     'uint8': torch.ByteTensor,
# }
# import re
# from torch._six import container_abcs
# from torch._six import string_classes, int_classes, FileNotFoundError
# def default_collate(batch):
#     r"""Puts each data field into a tensor with outer dimension batch size"""

#     error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
#     elem_type = type(batch[0])
#     if isinstance(batch[0], torch.Tensor):
#         out = None
#         if _use_shared_memory:
#             # If we're in a background process, concatenate directly into a
#             # shared memory tensor to avoid an extra copy
#             numel = sum([x.numel() for x in batch])
#             storage = batch[0].storage()._new_shared(numel)
#             out = batch[0].new(storage)
#         return torch.stack(batch, 0, out=out)
#     if elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
#             and elem_type.__name__ != 'string_':
#         elem = batch[0]
#         if elem_type.__name__ == 'ndarray':
#             # array of string classes and object
#             if re.search('[SaUO]', elem.dtype.str) is not None:
#                 raise TypeError(error_msg.format(elem.dtype))

#             try:
#                 return torch.stack([torch.from_numpy(b) for b in batch], 0)
#             except:
#                 import ipdb; ipdb.set_trace()

#         if elem.shape == ():  # scalars
#             py_type = float if elem.dtype.name.startswith('float') else int
#             return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
#     elif isinstance(batch[0], int_classes):
#         return torch.LongTensor(batch)
#     elif isinstance(batch[0], float):
#         return torch.DoubleTensor(batch)
#     elif isinstance(batch[0], string_classes):
#         return batch
#     elif isinstance(batch[0], container_abcs.Mapping):
#         return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
#     elif isinstance(batch[0], container_abcs.Sequence):
#         transposed = zip(*batch)
#         return [default_collate(samples) for samples in transposed]

#     raise TypeError((error_msg.format(type(batch[0]))))


class EmbeddedTD3Pixels(object):
    def __init__(self, state_dim, action_dim, max_action, arch="mine", initialize=True, img_width=128, stack=4, decoder=None):
        self.decoder = decoder
        self.e_action_dim = decoder.embed_dim

        # set the maximum action in embedding space to the largest value the decoder saw during training
        self.max_e_action = self.decoder.max_embedding

        self.actor = Actor(self.e_action_dim, self.max_e_action, arch, initialize, img_width, stack).to(device)
        self.actor_target = Actor(self.e_action_dim, self.max_e_action, arch, initialize, img_width, stack).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # add an extra input to the critic for which timestep we're on
        self.critic = Critic(self.e_action_dim + 1, arch, initialize, img_width, stack).to(device)
        self.critic_target = Critic(self.e_action_dim + 1, arch, initialize, img_width, stack).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-2)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.pending_plan = torch.Tensor(0, 0, 0).to(device)
        self.current_e_action = None


    def select_action(self, state, expl_noise=None):
        if self.pending_plan.size(1) == 0:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            e_action = self.actor(state)
            if expl_noise is not None:
                # noise = torch.zeros(e_action.size()).normal_(0, expl_noise).to(device)
                noise = torch.from_numpy(np.random.normal(0, expl_noise, size=self.e_action_dim)).float().to(device)
                e_action = (e_action + noise).clamp(-self.max_e_action, self.max_e_action)

            self.pending_plan = self.decoder(e_action)
            self.current_e_action = e_action

        # next action is head of plan, new plan is tail of current plan
        action = self.pending_plan[:, 0].cpu().data.numpy().flatten()
        self.pending_plan = self.pending_plan[:, 1:]

        # ensure that the decoded action is legal in the environment
        action = action.clip(-self.max_action, self.max_action)

        plan_step = self.decoder.traj_len - self.pending_plan.size(1) - 1
        return action, self.current_e_action[0].detach().cpu().numpy(), plan_step


    def plan(self, state, target=False):
        actor = self.actor if not target else self.actor_target
        return self.decoder(actor(state))


    def reset(self):
        self.pending_plan = torch.Tensor(0, 0, 0).to(device)
        self.current_e_action = None


    def mode(self, mode):
        if mode == 'eval':
            self.actor.eval()
        elif mode == 'train':
            self.actor.train()
        else:
            assert False


    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.mode('train')

        # batch_size = min(batch_size, len(replay_buffer))
        traj_len = self.decoder.traj_len
        loader = DataLoader(replay_buffer, batch_size, shuffle=True, num_workers=0) #, collate_fn=default_collate)
        for it, batch in enumerate(loader):
            if it >= iterations: break
            state, next_state, action, e_action, plan_step, reward, done = batch
            batch_size = state.size(0)
            # import ipdb; ipdb.set_trace()
            
            state = state.to(device)
            next_state = next_state.to(device)
            action = action.to(device)
            e_action = e_action.to(device)
            plan_step = plan_step.float().to(device)
            reward = reward.float().to(device)
            done = (1 - done).float().to(device)

            noise = torch.FloatTensor(batch_size, self.e_action_dim).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)


            # remaining_plan_steps = k - i
            remaining_plan_steps = traj_len - plan_step[:, 0]

            # create a mask indicating whether the action at index was from the plan at index 0
            same_plan_mask = torch.zeros(plan_step.size()).to(device)
            for index in range(batch_size):
                # import ipdb; ipdb.set_trace()
                same_plan_mask[index][:int(remaining_plan_steps[index])] = 1

            # indicates whether the reward at time t is from the same episode as state[:, 0]
            same_episode_mask = torch.zeros(plan_step.size()).to(device)
            same_episode_mask[:, 0] = 1
            for t in range(1, traj_len):
                same_episode_mask[:, t] = same_episode_mask[:, t-1] * done[:, t-1]

            # \sum_{j=0}^{k-1} \gamma^j r_{t+j}
            discount_exponent = torch.linspace(0, traj_len-1, traj_len).repeat(state.size(0), 1).to(device)
            discount_factor = discount ** discount_exponent
            discounted_reward = reward * discount_factor
            # current_plan_reward = (discounted_reward * same_plan_mask).sum(1, keepdim=True)
            current_plan_reward = (discounted_reward * same_plan_mask * same_episode_mask).sum(1, keepdim=True)

            # find which state we next replan on
            # if there are 4 steps left in the plan, that means we replanned
            #   on the state we got to after the 4th action (action[3])
            # that is, we replan on next_state[3]
            next_plan_state = torch.Tensor(batch_size, *next_state.size()[2:]).to(device)
            for index in range(batch_size):
                # import ipdb; ipdb.set_trace()
                next_plan_state[index] = next_state[index, int(remaining_plan_steps[index]) - 1]

            # Select action according to policy and add clipped noise
            # note this is now an embedded action
            # noise = torch.FloatTensor(batch_size, self.e_action_dim).data.normal_(0, policy_noise).to(device)
            # noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_plan_state) + noise).clamp(-self.max_e_action, self.max_e_action)

            # make a new done mask that is 0 if the episode ended during the current plan
            done_mask = torch.zeros(batch_size, 1).to(device)
            for index in range(batch_size):
                done_mask[index] = done[index, :int(remaining_plan_steps[index])].prod()

            # Compute the target Q value
            # tell the target Q functions that we're on a new plan in this state
            # import ipdb; ipdb.set_trace()
            target_Q1, target_Q2 = self.critic_target(next_plan_state, next_action, torch.zeros(batch_size, 1).to(device))
            target_Q = torch.min(target_Q1, target_Q2)
            next_state_discount = discount ** remaining_plan_steps.unsqueeze(1)
            target_Q = current_plan_reward + (done_mask * next_state_discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state[:, 0], e_action[:, 0], plan_step[:, 0].unsqueeze(1))
            target_Q = target_Q.reshape(-1, 1)
            # import ipdb; ipdb.set_trace()
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                # If I was in this state, and I started following plan actor(state) right now, how would I do?
                # import ipdb; ipdb.set_trace()
                actor_loss = -self.critic.Q1(state[:, 0], self.actor(state[:, 0]), torch.zeros(batch_size, 1).to(device)).mean()
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
        torch.save(self.decoder.state_dict(), '%s/%s_decoder.pth' % (directory, filename))
        torch.save(self, '%s/%s_all.pth' % (directory, filename))


    def load(self, filename, directory):
        if not torch.cuda.is_available():
            self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename), map_location='cpu'))
            self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename), map_location='cpu'))
            self.decoder.load_state_dict(torch.load('%s/%s_decoder.pth' % (directory, filename), map_location='cpu'))
        else:
            self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
            self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
            self.decoder.load_state_dict(torch.load('%s/%s_decoder.pth' % (directory, filename)))

def load(filename, directory):
    if not torch.cuda.is_available():
        return torch.load('%s/%s_all.pth' % (directory, filename), map_location='cpu')
    else:
        return torch.load('%s/%s_all.pth' % (directory, filename))
