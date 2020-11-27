import copy

import numpy as np
import scipy.linalg
import scipy.stats
import torch
import tqdm
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from torch import nn
from torch.nn import functional as F


class Variable(nn.Module):
    def __init__(self, x0):
        super().__init__()
        self.x = nn.Parameter(x0)


def convex_quadratic():
    """
    Generate a symmetric positive semidefinite matrix A with eigenvalues
    uniformly in [1e-3, 10].

    """
    num_vars = 2

    # First generate an orthogonal matrix (of eigenvectors)
    eig_vecs = torch.tensor(
        scipy.stats.ortho_group.rvs(dim=(num_vars)), dtype=torch.float
    )
    # Now generate eigenvalues
    eig_vals = torch.rand(num_vars) * 3 + 1

    A = eig_vecs @ torch.diag(eig_vals) @ eig_vecs.T
    b = torch.normal(0, 1 / num_vars, size=(num_vars,))

    x0 = torch.normal(0, 10 / np.sqrt(num_vars), size=(num_vars,))

    def quadratic(var):
        x = var.x
        return 0.5 * x.T @ A @ x + b.T @ x

    optimal_x = scipy.linalg.solve(A.numpy(), -b.numpy(), assume_a="pos")
    optimal_val = quadratic(Variable(torch.tensor(optimal_x))).item()

    return {
        "model0": Variable(x0),
        "obj_function": quadratic,
        "optimal_x": optimal_x,
        "optimal_val": optimal_val,
        "A": A.numpy(),
        "b": b.numpy(),
    }


def rosenbrock():
    num_vars = 10

    # Initialization strategy: x_i = -2 if i is even, x_i = +2 if i is odd
    x0 = torch.tensor([-1.5 if i % 2 == 0 else 1.5 for i in range(num_vars)])

    def rosen(x):
        return torch.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

    # Optimum at all x_i = 1, giving f(x) = 0
    optimal_x = np.ones(num_vars)
    optimal_val = 0

    return x0, rosen, optimal_x, optimal_val


def logistic_regression():
    num_vars = 3

    g0 = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.randn(num_vars),
        scale_tril=torch.tril(torch.randn((num_vars, num_vars))),
    )
    g1 = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.randn(num_vars),
        scale_tril=torch.tril(torch.randn((num_vars, num_vars))),
    )

    x = torch.cat([g0.sample((50,)), g1.sample((50,))])
    y = torch.cat([torch.zeros(50), torch.ones(50)])
    perm = torch.randperm(len(x))
    x = x[perm]
    y = y[perm]

    model0 = nn.Sequential(nn.Linear(num_vars, 1), nn.Sigmoid())

    def obj_function(model):
        y_hat = model(x).view(-1)
        return F.binary_cross_entropy(y_hat, y)

    return {"model0": model0, "obj_function": obj_function}


def mlp():
    model0 = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1), nn.Sigmoid())

    def obj_function(model):
        y_hat = model(x).view(-1)
        return F.binary_cross_entropy(y_hat, y)

    return {
        "model0": model0,
        "obj_function": obj_function,
    }


def minimize(obj_function, optimizer, model, iterations, verbose=False):
    trajectory = []
    values = []

    # Passed to optimizer. This setup is required to give the autonomous
    # optimizer access to the objective value and not just its gradients.
    def closure():
        optimizer.zero_grad()

        obj_value = obj_function(model)
        obj_value.backward()

        values.append(obj_value.item())
        return obj_value

    # Minimize
    for i in range(iterations):
        optimizer.step(closure)

        # Stop optimizing if we start getting nans as objective values
        if np.isnan(values[-1]) or np.isinf(values[-1]):
            break

        if verbose and i % (iterations // 10) == 0:
            print(i, values[-1].item())

    return np.array(values), np.array(trajectory)


def run_optimizer(make_optimizer, problem, iterations, hyperparams):
    # Initial solution
    model = copy.deepcopy(problem["model0"])
    obj_function = problem["obj_function"]

    # Define optimizer
    optimizer = make_optimizer(model.parameters(), **hyperparams)

    # Run
    vals, traj = minimize(obj_function, optimizer, model, iterations)
    return np.nan_to_num(vals, 1e6), traj


def make_experiment(tune_objective):
    def experiment(hyperparams):
        vals, traj = tune_objective(hyperparams)
        for obj_val in vals:
            tune.report(objective_value=obj_val)

    return experiment


def tune_algos(
    x0,
    obj_function,
    algo_iters,
    tune_iters,
    hyperparam_space,
    algos=["sgd", "momentum" "adam"],
):

    results = {}

    for algo in tqdm.tqdm(algos):

        if algo == "sgd":

            def run_sgd(hyperparams):
                return run_optimizer(
                    torch.optim.SGD, x0, obj_function, algo_iters, hyperparams
                )

            sgd_analysis = tune.run(
                make_experiment(run_sgd),
                config={"lr": hyperparam_space["lr"]},
                metric="objective_value",
                mode="min",
                search_alg=ConcurrencyLimiter(HyperOptSearch(), max_concurrent=3),
                num_samples=tune_iters,
                verbose=0,
            )
            sgd_hyperparams = sgd_analysis.get_best_config(
                metric="objective_value", mode="min"
            )

            results["sgd"] = {"analysis": sgd_analysis, "hyperparams": sgd_hyperparams}

        if algo == "momentum":

            def run_momentum(hyperparams):
                return run_optimizer(
                    torch.optim.SGD, x0, obj_function, algo_iters, hyperparams
                )

            momentum_analysis = tune.run(
                make_experiment(run_momentum),
                config={
                    "nesterov": True,
                    "lr": hyperparam_space["lr"],
                    "momentum": hyperparam_space["momentum"],
                },
                metric="objective_value",
                mode="min",
                search_alg=ConcurrencyLimiter(HyperOptSearch(), max_concurrent=3),
                num_samples=tune_iters,
                verbose=0,
            )
            momentum_hyperparams = momentum_analysis.get_best_config(
                metric="objective_value", mode="min"
            )

            results["momentum"] = {
                "analysis": momentum_analysis,
                "hyperparams": momentum_hyperparams,
            }

        if algo == "adam":

            def run_adam(hyperparams):
                return run_optimizer(
                    torch.optim.Adam, x0, obj_function, algo_iters, hyperparams
                )

            adam_analysis = tune.run(
                make_experiment(run_adam),
                config={"lr": hyperparam_space["lr"]},
                metric="objective_value",
                mode="min",
                search_alg=ConcurrencyLimiter(HyperOptSearch(), max_concurrent=3),
                num_samples=tune_iters,
                verbose=0,
            )
            adam_hyperparams = adam_analysis.get_best_config(
                metric="objective_value", mode="min"
            )

            results["adam"] = {
                "analysis": adam_analysis,
                "hyperparams": adam_hyperparams,
            }

    return results
