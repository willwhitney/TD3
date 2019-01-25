import numpy as np
import numpy.random as npr
import torch

class DummyDecoder:
    def __init__(self, embed_size, traj_len, action_space):
        self.traj_len = traj_len
        self.action_space = action_space
        self.embed_dim = embed_size

    def forward(self, embedding):
        # print("I'm a dummy")
        return embedding.unsqueeze(1).repeat(1, self.traj_len, 1)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)