"""Main module defining the autonomous RL optimizer."""
import copy

import gym
import numpy as np
import torch
from gym import spaces
from torch import optim


def make_observation(obj_value, obj_values, gradients, num_params, history_len):
    # Features is a matrix where the ith row is a concatenation of the difference
    # in the current objective value and that of the ith previous iterate as well
    # as the ith previous gradient.
    observation = np.zeros((history_len, 1 + num_params), dtype="float32")
    observation[: len(obj_values), 0] = (
        obj_value - torch.tensor(obj_values).detach().numpy()
    )
    for i, grad in enumerate(gradients):
        observation[i, 1:] = grad.detach().numpy()

    # Normalize and clip observation space
    observation /= 50
    return observation.clip(-1, 1)


class AutonomousOptimizer(optim.Optimizer):
    def __init__(self, params, policy, history_len=25):
        """
        Parameters:
            policy: Policy that takes in history of objective values and gradients
                as a feature vector - shape (history_len, num_parameters + 1),
                and outputs a vector to update parameters by of shape (num_parameters,).
            history_len: Number of previous iterations to keep objective value and
                gradient information for.

        """
        super().__init__(params, {})

        self.policy = policy
        self.history_len = history_len
        self.num_params = sum(
            p.numel() for group in self.param_groups for p in group["params"]
        )

        self.obj_values = []
        self.gradients = []

    @torch.no_grad()
    def step(self, closure):
        with torch.enable_grad():
            obj_value = closure()

        # Calculate the current gradient and flatten it
        current_grad = torch.cat(
            [p.grad.flatten() for group in self.param_groups for p in group["params"]]
        ).flatten()

        # Update history of objective values and gradients with current objective
        # value and gradient.
        if len(self.obj_values) >= self.history_len:
            self.obj_values.pop(-1)
            self.gradients.pop(-1)
        self.obj_values.insert(0, obj_value)
        self.gradients.insert(0, current_grad)

        # Run policy
        observation = make_observation(
            obj_value.item(),
            self.obj_values,
            self.gradients,
            self.num_params,
            self.history_len,
        )
        action, _states = self.policy.predict(observation, deterministic=True)

        # Update the parameters according to the policy
        param_counter = 0
        action = torch.from_numpy(action)
        for group in self.param_groups:
            for p in group["params"]:
                delta_p = action[param_counter : param_counter + p.numel()]
                p.add_(delta_p.reshape(p.shape))
                param_counter += p.numel()

        return obj_value


class Environment(gym.Env):
    """Optimization environment based on TF-Agents."""

    def __init__(
        self,
        dataset,
        num_steps,
        history_len,
    ):
        super().__init__()

        self.dataset = dataset
        self.num_steps = num_steps
        self.history_len = history_len

        self._setup_episode()
        self.num_params = sum(p.numel() for p in self.model.parameters())

        # Define action and observation space
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.num_params,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.history_len, 1 + self.num_params),
            dtype=np.float32,
        )

    def _setup_episode(self):
        res = np.random.choice(self.dataset)
        self.model = copy.deepcopy(res["model0"])
        self.obj_function = res["obj_function"]

        self.obj_values = []
        self.gradients = []
        self.current_step = 0

    def reset(self):
        self._setup_episode()
        return make_observation(
            None, self.obj_values, self.gradients, self.num_params, self.history_len
        )

    @torch.no_grad()
    def step(self, action):
        # Update the parameters according to the action
        action = torch.from_numpy(action)
        param_counter = 0
        for p in self.model.parameters():
            delta_p = action[param_counter : param_counter + p.numel()]
            p.add_(delta_p.reshape(p.shape))
            param_counter += p.numel()

        # Calculate the new objective value
        with torch.enable_grad():
            self.model.zero_grad()
            obj_value = self.obj_function(self.model)
            obj_value.backward()

        # Calculate the current gradient and flatten it
        current_grad = torch.cat(
            [p.grad.flatten() for p in self.model.parameters()]
        ).flatten()

        # Update history of objective values and gradients with current objective
        # value and gradient.
        if len(self.obj_values) >= self.history_len:
            self.obj_values.pop(-1)
            self.gradients.pop(-1)
        self.obj_values.insert(0, obj_value)
        self.gradients.insert(0, current_grad)

        # Return observation, reward, done, and empty info
        observation = make_observation(
            obj_value.item(),
            self.obj_values,
            self.gradients,
            self.num_params,
            self.history_len,
        )
        reward = -obj_value.item()
        done = self.current_step >= self.num_steps
        info = {}

        self.current_step += 1
        return observation, reward, done, info
