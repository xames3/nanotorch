.. Author: Akshay Mestry <xa@mes3.dev>
.. Created on: Saturday, December 02 2023
.. Last updated on: Tuesday, December 12 2023

NanoTorch
=========

Etymology: *nano* (Small) and *torch* (PyTorch)

Small-scale implementation of `PyTorch`_ from the ground up.

This project, a miniature implementation of the PyTorch library, is crafted
with the primary goal of elucidating the intricate workings of neural network
libraries. It serves as a pedagogical tool for those seeking to unravel the
mathematical complexities and the underlying architecture that powers such
sophisticated libraries.

**NOTE:** This project is based on the excellent work done by
`Andrej Karpathy`_ in his `micrograd`_ project.

Installation
------------

.. See more at: https://stackoverflow.com/a/15268990

Install the latest version of NanoTorch using `pip`_:

.. code-block:: bash

    pip install -U git+https://github.com/xames3/nanotorch.git#egg=nanotorch

Objective
---------

The cornerstone of this endeavor is to provide a hands-on learning experience
by replicating key components of PyTorch, thereby granting insights into its
functional mechanisms. This bespoke implementation focuses on the core aspects
of neural network computation, including tensor operations, automatic
differentiation, and basic neural network modules.

Features
--------

1. **Tensor Operations:** At the heart of this implementation lie tensor
operations, which are the building blocks of any neural network library. As of
now, our tensors support basic arithmetic functionalities found in PyTorch.

.. code:: python

    >>> a = nanotorch.tensor(2.0)
    >>> b = nanotorch.tensor(3.0)
    >>> a + b
    tensor(5.0)
    >>> a - 6
    tensor(-4.0)
    >>> c = a + b
    >>> c += 2 * a / b
    >>> c = c ** 3
    >>> nanotorch.arange(5)
    [tensor(0), tensor(1), tensor(2), tensor(3), tensor(4)]
    >>> nanotorch.arange(1, 4)
    [tensor(1), tensor(2), tensor(3)]
    >>> nanotorch.arange(1, 2.5, 0.5)
    [tensor(1), tensor(1.5), tensor(2.)]

2. **Automatic Differentiation:** A pivotal feature of this project is a
simplistic version of automatic differentiation, akin to PyTorch's
``autograd``. It allows for the computation of gradients automatically, which
is essential for training neural networks.

.. code:: python

    >>> c.backward()
    >>> print(a.grad)  # prints 200.55 as the gradient with respect to c i.e dc/da

3. **Neural Network Modules:** The implementation includes rudimentary neural
network modules such as linear layers and activation functions. These modules
can be composed to construct simple neural network architectures.

4. **Optimizers and Loss Functions:** Basic optimizers like SGD and common
loss functions are included to facilitate the training process of neural
networks.

Educational Value
-----------------

This project stands as a testament to the educational philosophy of learning
by doing. It is particularly beneficial for:

- Students and enthusiasts who aspire to gain a profound understanding of the
  inner workings of neural network libraries.

- Developers and researchers seeking to customize or extend the functionalities
  of existing deep learning libraries for their specific requirements.

Usage and Documentation
-----------------------

The codebase is structured to be intuitive and mirrors the design principles
of PyTorch to a significant extent. Comprehensive docstrings are provided for
each module and function, ensuring clarity and ease of understanding. Users
are encouraged to delve into the code, experiment with it, and modify it to
suit their learning curve.

Contributions and Feedback
--------------------------

Contributions to this project are warmly welcomed. Whether it's refining the
code, enhancing the documentation, or extending the current feature set, your
input is highly valued. Feedback, whether constructive criticism or 
commendation, is equally appreciated and will be instrumental in the evolution
of this educational tool.

Acknowledgments
---------------

This project is inspired by the remarkable work done by the `PyTorch
development team`_. It is a tribute to their contributions to the field of
machine learning and the open-source community at large.

Project Links
-------------

- Source Code: https://github.com/xames3/nanotorch
- Issue Tracker: https://github.com/xames3/nanotorch/issues

.. _Andrej Karpathy: https://github.com/karpathy
.. _PyTorch development team: https://github.com/pytorch/pytorch
.. _PyTorch: https://pytorch.org
.. _micrograd: https://github.com/karpathy/micrograd
.. _pip: https://pip.pypa.io/en/stable/getting-started/
