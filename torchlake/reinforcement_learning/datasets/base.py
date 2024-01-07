import random
from collections import deque, namedtuple

Transition = namedtuple(
    "transition",
    ["state", "action", "reward", "next_state"],
)


class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, transition: Transition):
        """Saves a transition."""
        self.memory.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
