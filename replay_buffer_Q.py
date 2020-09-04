import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size, step):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._step = step

    def __len__(self):
        return len(self._storage)

    def add(self, rews_trajectory, states_trajectory ):
        """
        Add a single experience
        """
        data = (rews_trajectory, states_trajectory)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize


    def _encode_sample(self, idxes):
        rews_trajectorys, states_trajectorys = [], []
        for i in idxes:
            data = self._storage[i]
            rews_trajectory, states_trajectory = data
            rews_trajectorys.append(np.array(rews_trajectory, copy=False))
            states_trajectorys.append(np.array(states_trajectory, copy=False))
        return np.array(rews_trajectorys), np.array(states_trajectorys)


    def sample(self, batch_size):
        """
        Sample a batch of experiences.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

