"""Main module defining the autonomous RL optimizer."""
import copy

import gym
import numpy as np
import torch
from gym import spaces
from stable_baselines.common.env_checker import check_env
from torch import optim


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

    def _make_observation(self, obj_value):
        # Features is a matrix where the ith row is a concatenation of the difference
        # in the current objective value and that of the ith previous iterate as well
        # as the ith previous gradient.
        observation = np.zeros((self.history_len, 1 + self.num_params), dtype="float32")
        observation[: len(self.obj_values), 0] = (
            obj_value - torch.tensor(self.obj_values).detach().numpy()
        )
        for i, grad in enumerate(self.gradients):
            observation[i, 1:] = grad.detach().numpy()

        # Clip observation to fit in observation space Box and return it
        return observation.clip(
            self.policy.observation_space.low, self.policy.observation_space.high
        )

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
        observation = self._make_observation(obj_value.item())
        action, _states = self.policy.predict(observation)

        # Update the parameters according to the policy
        param_counter = 0
        action = torch.from_numpy(action)
        for group in self.param_groups:
            for p in group["params"]:
                p.add_(action[param_counter : len(p)])

        return obj_value


class Environment(gym.Env):
    """Optimization environment based on TF-Agents."""

    def __init__(
        self,
        dataset,
        num_steps,
        history_len,
        observation_clip_threshold=100,
    ):
        super().__init__()

        self.dataset = dataset
        self.current_obj_function = 0
        self.num_steps = num_steps
        self.history_len = history_len

        self._setup_episode()
        self.num_params = sum(p.numel() for p in self.model.parameters())

        # Define action and observation space
        self.action_space = spaces.Box(
            low=-100, high=100, shape=(self.num_params,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-observation_clip_threshold,
            high=observation_clip_threshold,
            shape=(self.history_len, 1 + self.num_params),
            dtype=np.float32,
        )

        # Validate environment
        check_env(self)

    def _setup_episode(self):
        res = self.dataset[self.current_obj_function]
        self.model = copy.deepcopy(res["model0"])
        self.obj_function = res["obj_function"]
        self.current_obj_function = (self.current_obj_function + 1) % len(self.dataset)

        self.obj_values = []
        self.gradients = []
        self.current_step = 0

    def reset(self):
        self._setup_episode()
        return self._make_observation(None)

    @torch.no_grad()
    def step(self, action):
        # Update the parameters according to the action
        action = torch.from_numpy(action)
        param_counter = 0
        for p in self.model.parameters():
            p.add_(action[param_counter : len(p)])

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
        observation = self._make_observation(obj_value.item())
        reward = -obj_value.item()
        done = self.current_step >= self.num_steps
        info = {}

        self.current_step += 1
        return observation, reward, done, info

    def _make_observation(self, obj_value):
        # Features is a matrix where the ith row is a concatenation of the difference
        # in the current objective value and that of the ith previous iterate as well
        # as the ith previous gradient.
        observation = np.zeros((self.history_len, 1 + self.num_params), dtype="float32")
        observation[: len(self.obj_values), 0] = (
            obj_value - torch.tensor(self.obj_values).detach().numpy()
        )
        for i, grad in enumerate(self.gradients):
            observation[i, 1:] = grad.detach().numpy()

        # Clip observation to fit in observation space Box and return it
        return observation.clip(self.observation_space.low, self.observation_space.high)
