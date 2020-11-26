"""Main module defining the autonomous RL optimizer."""
import torch
import torch.optim as optim


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

        self.obj_values = []
        self.gradients = []

    @torch.no_grad()
    def step(self, current_obj_value):
        # Calculate the current gradient and flatten it.
        current_grad = torch.tensor(
            [p.grad for group in self.param_groups for p in group["params"]]
        ).flatten()

        # Update history of objective values and gradients with current objective
        # value and gradient.
        if len(self.obj_values) >= self.history_len:
            self.obj_values.pop(-1)
            self.gradients.pop(-1)
        self.obj_values.insert(0, current_obj_value)
        self.gradients.insert(0, current_grad)

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

        return current_obj_value

