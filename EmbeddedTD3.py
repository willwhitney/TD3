import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action


    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Actor, self).__init__()

#         self.l1 = nn.Linear(state_dim, 400)
#         self.l2 = nn.Linear(400, 300)
#         self.l3 = nn.Linear(300, action_dim)

#         # self.max_action = max_action


#     def forward(self, x):
#         x = F.relu(self.l1(x))
#         x = F.relu(self.l2(x))
#         # x = self.max_action * torch.tanh(self.l3(x))
#         x = self.l3(x)
#         return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)


    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2


    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class EmbeddedTD3(object):
    def __init__(self, state_dim, action_dim, max_action, decoder):
        self.decoder = decoder
        self.e_action_dim = decoder.embed_dim

        self.actor = Actor(state_dim, self.e_action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, self.e_action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.pending_plan = torch.Tensor(0, 0, 0).to(device)


    def select_action(self, state):
        if self.pending_plan.size(1) == 0:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            e_action = self.actor(state)
            self.pending_plan = self.decoder(e_action)
        # import ipdb; ipdb.set_trace()
        action = self.pending_plan[:, 0].cpu().data.numpy().flatten()
        self.pending_plan = self.pending_plan[:, 1:]
        return action


    def plan(self, state, target=False):
        actor = self.actor if not target else self.actor_target
        return self.decoder(actor(state))


    def reset(self):
        self.pending_plan = torch.Tensor(0, 0, 0).to(device)


    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample_seq(batch_size, self.decoder.traj_len)

            state = torch.FloatTensor(x).to(device)         # [s_{t-k}, ..., s_t]
            action = torch.FloatTensor(u).to(device)        # [a_{t-k}, ..., a_t]
            next_state = torch.FloatTensor(y).to(device)    # [s_{t+1-k}, ..., s_{t+1}]
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # mask[i] represents whether next_state[i] is part of the same episode as reward[-1]
            # it also says whether state[i] is part of the same episode as state[-1]
            mask = torch.ones(done.size()).to(device)
            for i in range(-2, -self.decoder.traj_len-1, -1):
                mask[:, i] = mask[:, i+1] * done[:, i]
            # if mask.sum() < mask.nelement():
            #     print("Mask")
            #     print(mask)
            #     import ipdb; ipdb.set_trace()

            # Estimate the value of the next state by averaging over the different actions
            # the policy might take at the next state
            # i.e. averaging Q(s',a') over the timesteps that the plan (a') could have come from
            next_state_value = torch.zeros(state.size(0)).to(device)
            for i in range(self.decoder.traj_len):
                # next_action = D(pi_e(s_{t+1-i}))[i]
                next_action = self.plan(next_state[:, -i-1], target=True)[:, i]

                # Select action according to policy and add clipped noise
                # I think it's OK that the noise is inside this average;
                #   it will still bring down the estimated value of very sharp maxima
                noise = torch.zeros(next_action.size()).data.normal_(0, policy_noise).to(device)
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

                # Compute the target Q value
                next_state_V1_i, next_state_V2_i = self.critic_target(next_state[:, -1], next_action)
                next_state_V_i = torch.min(next_state_V1_i, next_state_V2_i).squeeze()
                next_state_value = next_state_value + next_state_V_i * mask[:, -i-1]

            # divide by the number of valid plans to get the expected value of the next state
            # (we masked out the invalid plans when totaling next_state_value)
            next_state_value = next_state_value / mask.sum(dim=1).squeeze()
            target_Q = reward[:, -1] + (done[:, -1] * discount * next_state_value).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state[:, -1], action[:, -1])
            target_Q = target_Q.reshape(-1, 1)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # import ipdb; ipdb.set_trace()

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                # import ipdb; ipdb.set_trace()
                actor_loss = torch.zeros(1).to(device)
                for i in range(self.decoder.traj_len):
                    decoded_action = self.plan(state[:, -i-1])[:, i]
                    # Compute actor loss
                    loss_i = -self.critic.Q1(state[:, -1], decoded_action)

                    # mask out the plans based on states that aren't from this episode
                    # and take the mean over the set of viable plans at this timestep
                    loss_i = (loss_i * mask[:, -i-1]).sum() / mask[:, -i-1].sum()
                    # import ipdb; ipdb.set_trace()
                    actor_loss = actor_loss + loss_i.mean()

                actor_loss = actor_loss / mask.sum()
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
