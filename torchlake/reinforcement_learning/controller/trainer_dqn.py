from itertools import count
import math

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from ..datasets.atari import AtariDataset
from ..datasets.base import Transition
from ..models.dqn.model import Dqn
from ..models.dqn.network import QNetwork


class DqnTrainer(object):
    def __init__(
        self,
        epoches: int,
        device: torch.device,
        n_actions: int,
        timeout: int = 500,
        gamma: int = 0.999,
        update_cycle: int = 10,
        epsilon_0: int = 9e-1,
        epsilon_t: int = 5e-2,
        epsilon_decay: int = 5e-2,
    ):
        self.epoches = epoches
        self.update_cycle = update_cycle
        self.gamma = gamma  # discount value
        self.epsilon_0 = epsilon_0  # random action probability at 0
        self.epsilon_t = epsilon_t  # random action probability at t
        self.epsilon_decay = epsilon_decay  # random action probability decay rate
        self.device = device
        self.policy_net = QNetwork(n_actions).to(device)
        self.timeout = timeout

    def get_epsilon(self, t: int) -> float:
        epsilon = self.epsilon_t + (self.epsilon_0 - self.epsilon_t) * math.exp(
            -1.0 * t / self.epsilon_decay
        )

        return epsilon

    def get_expected_q(self, model: nn.Module, batch, batch_size: int):
        reward_batch = torch.Tensor(batch.reward).to(self.device)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.BoolTensor(
            [item is not None for item in batch.next_state]
        ).to(self.device)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        next_state_values = torch.zeros((batch_size)).to(self.device)
        next_state_values[non_final_mask] = model(non_final_next_states).max(-1)[0]

        # Compute the expected Q values
        # E[Q] = r + {\gamma}Q'
        expected_q = (next_state_values * self.gamma) + reward_batch

        return expected_q

    def optimize(
        self,
        optimizer: Optimizer,
        criterion: nn.Module,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # Perform one step of the optimization (on the target network)
        optimizer.zero_grad()

        loss = criterion(pred, target.view(-1, 1))
        loss.backward()

        # clip grad for instability
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        optimizer.step()

        return loss

    def run(
        self,
        data: AtariDataset,
        model: Dqn,
        optimizer: Optimizer,
        criterion: nn.Module,
    ) -> tuple[list[float], list[int]]:
        episode_rewards = []
        training_loss = []

        for epoch in tqdm(range(self.epoches)):
            episode_reward = 0
            running_loss = 0
            state = data[0].unsqueeze(0).to(self.device)

            for t in count():
                action = model.get_action(state, self.get_epsilon(t))
                next_state, reward, done = data.get_next_step(action)
                episode_reward += reward

                if done or (t > self.timeout):
                    episode_rewards.append(episode_reward)
                    break

                next_state = data[t].unsqueeze(0).to(self.device)
                transition = Transition(state, action, reward, next_state)
                data.push_memory(transition)

                if len(data.replays) < data.batch_size:
                    continue

                transitions = data.replays.sample(data.batch_size)

                # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.
                batch = Transition(*zip(*transitions))

                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken in each state.
                states = torch.cat(batch.state)
                actions = torch.cat(batch.action)
                q = self.policy_net(states).gather(1, actions)

                y = self.get_expected_q(model, batch, data.batch_size)

                loss = self.optimize(optimizer, criterion, q, y)
                running_loss += loss.item()

                # Move to the next state
                state = next_state

                # Update the target network, copying all weights and biases in DQN
                if t % self.update_cycle == 0:
                    model.update_target_net(self.policy_net.state_dict())

            mean_loss = running_loss / (t or 1)
            training_loss.append(mean_loss)
            print("epoch", epoch, ":", mean_loss)

        data.close()

        return training_loss, episode_rewards
