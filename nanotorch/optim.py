"""\
NanoTorch Optimization Algorithms API
=====================================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Saturday, March 16 2024
Last updated on: Saturday, March 16 2024

This module implements various optimization algorithms used for training
machine learning models. These algorithms adjust the parameters of a
model in order to minimize a loss function, a process essential to the
training of models in supervised learning tasks.

The module currently includes the following optimization algorithm:

    - SGD (Stochastic Gradient Descent): Implements the basic SGD
        optimization algorithm with optional momentum.

To use an optimizer, you need to instantiate it with the parameters
(typically from a model) that you wish to optimize. Then, within your
training loop, you would call the `step` method after computing the
gradients (e.g., via backpropagation) to update the parameters. It's
also good practice to reset gradients to zero between iterations using
the ``zero_grad`` method.

See https://github.com/xames3/nanotorch/ for more help.

:copyright: (c) 2024 Akshay Mestry (XAMES3). All rights reserved.
:license: MIT, see LICENSE for more details.
"""

from __future__ import annotations

from nanotorch._tensor import tensor


class SGD:
    """Implements stochastic gradient descent (SGD) optimization
    algorithm, optionally with momentum.

    SGD is a method that performs parameter updates based on the
    gradient of the loss function with respect to the parameter. If
    momentum is applied, the update vector will also consider the
    previous update step, thus accelerating the optimization in the
    direction of persistent reduction in the loss function over
    iterations.

    :param params: Iterable of parameters to optimize. Each parameter
                   should be an instance of a class with ``data`` and
                   ``grad`` attributes representing the parameter value
                   and the gradient of the loss function with respect to
                   this parameter, respectively.
    :param lr: Learning rate. This is a scalar used to determine the
               step size at each iteration while moving toward a minimum
               of a loss function, defaults to ``0.001``.
    :param momentum: Momentum factor. It is used to improve convergence
                     and reduce oscillations. Momentum accelerates the
                     SGD in the relevant direction and dampens
                     oscillations by combining the gradient with the
                     previous step's direction, defaults to ``0.0``,
                     which means no momentum is applied.
    """

    def __init__(
        self, params: list[tensor], lr: float = 0.001, momentum: float = 0.0
    ) -> None:
        """Initializes the SGD optimizer with given parameters, learning
        rate, and momentum.
        """
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.counter = 0

    def step(self) -> None:
        """Performs a single optimization step.

        This method updates each parameter based on its gradient and the
        learning rate. If momentum is used, the learning rate is
        adjusted based on the number of steps taken. This method should
        be called after calculating the gradients (e.g., via
        backpropagation).
        """
        if self.momentum:
            learning_rate = 1.0 - self.momentum * self.counter * self.lr
        else:
            learning_rate = self.lr
        for param in self.params:
            param.data -= learning_rate * param.grad
        self.counter += 1

    def zero_grad(self) -> None:
        """Resets the gradients of all optimized parameters to zero.

        This method should be called before a new optimization step or
        gradient calculation to avoid accumulating gradients from
        multiple forward passes.
        """
        for param in self.params:
            param.grad = 0.0
