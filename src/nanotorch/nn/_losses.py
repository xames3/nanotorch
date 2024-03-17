"""\
NanoTorch Losses API
====================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Saturday, March 09 2024
Last updated on: Saturday, March 16 2024

Loss Functions.

This module in NanoTorch is a comprehensive collection of loss
functions designed to facilitate the training of machine learning
models. It mirrors the functionality of PyTorch's loss modules,
providing a variety of loss computation mechanisms essential for
optimizing models across a wide array of tasks, including regression,
classification, and embedding learning. This module includes
implementations of popular loss functions such as Mean Squared Error
(MSE) Loss and Hinge Embedding Loss, among others, implemented from
scratch in Python.

Loss functions measure the discrepancy between the model predictions and
the actual target values. They are crucial for training neural networks,
guiding the optimization process by indicating how well the model is
performing. Below are descriptions of some key loss functions provided
in this module.

The ``MSELoss`` computes the mean squared error between the model's
predictions and the ground truth. It's widely used in regression tasks
where the goal is to minimize the average squared differences between
the estimated values and the actual value.

The ``HingeEmbeddingLoss`` is used for learning embeddings or distances
between pairs of inputs. It is typically used in tasks that involve
comparisons, such as determining whether two items are similar or
dissimilar. This loss function encourages correct predictions to be
closer to each other in the embedding space, while incorrect predictions
are pushed further apart.

The design of this module allows for easy extension, enabling users to
implement custom loss functions according to their specific requirements.
By inheriting from a base loss class, users can leverage shared
functionality and ensure compatibility with the NanoTorch training loop.
Future updates may include loss functions for specific applications such
as segmentation, object detection, and language modeling, as well as
improvements in efficiency and numerical stability.

The losses module is a fundamental part of the NanoTorch package,
providing essential tools for model optimization. With its range of
implemented loss functions and the ability to extend with custom losses,
it offers flexibility and power to the model training process.
The module's design follows closely with the intuitive and user-friendly
philosophy of NanoTorch, making advanced model training accessible to
both beginners and experienced practitioners in the field of machine
learning and deep learning.

See https://github.com/xames3/nanotorch/ for more help.

:copyright: (c) 2024 Akshay Mestry (XAMES3). All rights reserved.
:license: MIT, see LICENSE for more details.
"""

from __future__ import annotations

from nanotorch._tensor import tensor
from nanotorch.nn._layers import Loss
from nanotorch.nn.functional import hinge_embedding_loss
from nanotorch.nn.functional import mse_loss


class MSELoss(Loss):
    """Creates a criterion that measures the mean squared error (squared
    L2 norm) between each element in the input x and target y.
    """

    def forward(self, input: tensor, target: tensor) -> tensor:
        """Measures the element-wise mean squared error."""
        return mse_loss(input, target)

    __call__ = forward


class HingeEmbeddingLoss(Loss):
    """Creates a criterion that measures the hinge embedding loss
    between each element in the input x and target y.
    """

    def forward(self, input: tensor, target: tensor) -> tensor:
        """Measures the loss given an input tensor and a labels tensor."""
        return hinge_embedding_loss(input, target)

    __call__ = forward
