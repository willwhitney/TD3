import numpy as np
import copy
import json

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        result = (np.array(x),
                  np.array(y),
                  np.array(u),
                  np.array(r).reshape(-1, 1),
                  np.array(d).reshape(-1, 1))
        return result

    def __len__(self):
        return len(self.storage)

    def sample_seq(self, batch_size, seq_len):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            transition_sequence = self.storage[i:i+seq_len])
            # take the sequence [(xyurd), (xyurd), (xyurd), (xyurd)]
            # and turn it into [(xxxx), (yyyy), (uuuu), (rrrr), (dddd)]
            X, Y, U, R, D = list(zip(*transition_sequence)
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        result = (np.array(x),
                  np.array(y),
                  np.array(u),
                  np.array(r).reshape(batch_size, seq_len, 1),
                  np.array(d).reshape(batch_size, seq_len, 1))
        return result


def serialize_opt(opt):
    # import ipdb; ipdb.set_trace()
    cleaned_opt = copy.deepcopy(vars(opt))
    return json.dumps(cleaned_opt, indent=4, sort_keys=True)

def write_options(opt, location):
    with open(location + "/opt.json", 'w') as f:
        serial_opt = serialize_opt(opt)
        print(serial_opt)
        f.write(serial_opt)
        f.flush()
