"""Main module defining the autonomous RL optimizer."""
import copy

import numpy as np
import tf_agents
import torch
from tf_agents.environments.utils import validate_py_environment
from tf_agents.trajectories import time_step as ts

'''class AutonomousOptimizer(optim.Optimizer):
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

        self.obj_values = []
        self.gradients = []

    @torch.no_grad()
    def step(self, current_obj_value):

        # Features is a matrix where the ith row is a concatenation of the difference
        # in the current objective value and that of the ith previous iterate as well
        # as the ith previous gradient.
        features = torch.zeros((self.history_len, 1 + self.num_params))
        features[: len(self.obj_values), 0] = current_obj_value - torch.tensor(
            self.obj_values
        )
        features[: len(self.gradients), 1:] = self.gradients

        # Update the parameters according to the policy
        parameter_update = self.policy(features)
        param_counter = 0
        for group in self.param_groups:
            for p in group["params"]:
                p.add_(parameter_update[param_counter : len(p)])

        return current_obj_value'''


class Environment(tf_agents.environments.py_environment.PyEnvironment):
    """Optimization environment based on TF-Agents."""

    def __init__(self, make_problem_instance, num_steps, history_len):
        self.make_problem_instance = make_problem_instance
        self.num_steps = num_steps
        self.history_len = history_len

        self._setup_episode()
        self.num_params = sum(
            len(p) for group in self.model.parameters() for p in group["parameters"]
        )

        self._action_spec = tf_agents.specs.ArraySpec(
            shape=(self.num_params,), dtype="float", name="action"
        )
        self._observation_spec = tf_agents.specs.ArraySpec(
            shape=(history_len, 1 + self.num_params), dtype="float", name="observation"
        )

        validate_py_environment(self, episodes=1)

    def _setup_episode(self):
        res = self.make_problem_instance()

        self.model = copy.deepcopy(res[0])
        self.loss_fn = res[1]

        self.obj_values = []
        self.gradients = []
        self.current_step = 0

    def _reset(self):
        self._setup_episode()
        return ts.restart(self._make_features(None))

    def action_spec(self):
        """Define agent actions."""
        return self._action_spec

    def observation_spec(self):
        """Define environment observations."""
        return self._observation_spec

    def _step(self, action):
        # Update the parameters according to the action
        param_counter = 0
        for param_group in self.model.parameters():
            for param in param_group["params"]:
                param.add_(action[param_counter : len(param)])

        # Calculate the new objective value
        self.model.zero_grad()
        obj_value = self.loss_fn(self.model)

        # Calculate the current gradient and flatten it
        obj_value.backward()
        current_grad = torch.tensor(
            [p.grad for group in self.model.parameters() for p in group["params"]]
        ).flatten()

        # Update history of objective values and gradients with current objective
        # value and gradient.
        if len(self.obj_values) >= self.history_len:
            self.obj_values.pop(-1)
            self.gradients.pop(-1)
        self.obj_values.insert(0, obj_value)
        self.gradients.insert(0, current_grad)

        # Return feature matrix (the observation) along with current
        # objective value (the reward).
        features = self._make_features(obj_value.item())
        if self.current_step >= self.num_steps:
            return ts.termination(features, obj_value.item())
        else:
            self.current_step += 1
            return ts.transition(features, obj_value.item())

    def _make_features(self, obj_value):
        # Features is a matrix where the ith row is a concatenation of the difference
        # in the current objective value and that of the ith previous iterate as well
        # as the ith previous gradient.
        features = np.zeros((self.history_len, 1 + self.num_params))
        features[: len(self.obj_values), 0] = (
            obj_value - torch.tensor(self.obj_values).detach().numpy()
        )
        for i, grad in enumerate(self.gradients):
            features[i, 1:] = torch.tensor(grad).detach().numpy()

        return features
