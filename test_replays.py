import numpy as np
import numpy.random as npr
import random
from torch.utils.data import DataLoader

import utils

def list_array_eq(l1, l2):
    # import ipdb; ipdb.set_trace()
    equalities = [np.allclose(a, b) for a,b in zip(l1, l2)]
    return all(equalities)


replay_size = 10
replay_buffer = utils.ReplayDataset(max_size=replay_size)
data_path = "results/test_replay"
disk_replay_buffer = utils.DiskReplayDataset(path=data_path, max_size=replay_size)

for _ in range(2):
    for _ in range(replay_size // 2):
        elem = [npr.rand(32, 32), npr.rand(32, 32), npr.rand(12), npr.randint(10), random.choice([0, 1])]
        replay_buffer.add(elem)
        disk_replay_buffer.add(elem)

    assert (len(disk_replay_buffer) == len(replay_buffer))
    for i in range(len(disk_replay_buffer)):
        if not list_array_eq(replay_buffer[i], disk_replay_buffer[i]):
            print("Error: elements are not the same.")
            import sys; sys.exit(1)

    loader = DataLoader(replay_buffer, 3, shuffle=False, num_workers=0, pin_memory=False)
    disk_loader = DataLoader(disk_replay_buffer, 3, shuffle=False, num_workers=0, pin_memory=False)

    for b1, b2 in zip(loader, disk_loader):
        if not list_array_eq(b1, b2):
            print("Error: batches are not the same.")
            import ipdb; ipdb.set_trace()
            import sys; sys.exit(1)

print("Success")