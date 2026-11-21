"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time

device = ndl.cpu()


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    # import gzip and struct and numpy
    import gzip
    import struct
    import numpy as np

    # open .gz files
    with gzip.open(image_filename) as f:
        image_data = f.read()

    with gzip.open(label_filename) as g:
        label_data = g.read()

    # extract header info and get image sizes
    _, num_images, image_size_row, image_size_column = struct.unpack(
        ">IIII", image_data[:16]
    )
    _, num_labels = struct.unpack(">II", label_data[:8])
    assert (
        num_labels == num_images
    ), f"number of images different than labels {num_images}, {num_labels}"
    total_pixel_size = image_size_row * image_size_column

    # use numpy to convert the bytes and store as normalized numpy array
    image_vector = np.frombuffer(image_data, dtype=np.uint8, offset=16) / 255.0
    image_array = image_vector.reshape((num_images, total_pixel_size)).astype(
        np.float32
    )

    # get label array
    label_array = np.frombuffer(label_data, dtype=np.uint8, offset=8)

    # tuple contents
    return (image_array, label_array)

    ### END YOUR CODE


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    # take the exp of logits
    exp_logits = ndl.exp(Z)

    # take the sum of the exp logits
    sum_exp_logits = ndl.summation(exp_logits, axes=1)

    # take the log of each sum
    log_of_sums = ndl.log(sum_exp_logits)

    # get correct logits
    correct_logits = ndl.summation(ndl.multiply(Z, y_one_hot), axes=1)

    # return average
    return ndl.summation(log_of_sums - correct_logits) / log_of_sums.shape[0]
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    # begin SGD loop
    for begin_index in range(0, X.shape[0], batch):
        end_index = min(begin_index + batch, X.shape[0])
        current_batch_size = end_index - begin_index

        # cast numpy array to tensor
        X_tensor = ndl.Tensor(X[begin_index:end_index])

        # get predictions
        Z1 = ndl.relu(ndl.matmul(X_tensor, W1))
        predictions = ndl.matmul(Z1, W2)

        # create one-hot input vector
        input_batch = y[begin_index:end_index]
        input_matrix = np.zeros(
            (current_batch_size, W2.shape[1]),
            dtype=np.float32,
        )
        input_matrix[np.arange(len(input_batch)), input_batch] = 1

        # cast to Tensor
        input_Tensor = ndl.Tensor(input_matrix)

        # get gradients
        loss = softmax_loss(predictions, input_Tensor)
        loss.backward()

        # update
        W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
        W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())

    return W1, W2
    ### END YOUR SOLUTION


### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
        model.train()

    parameter = next(iter(model.parameters()), None)
    correct = total_examples = 0
    total_loss = 0.0
    for images, labels in dataloader:
        if parameter is not None:
            images = ndl.Tensor(
                images,
                device=parameter.device,
                dtype=parameter.dtype,
                requires_grad=False,
            )
            labels = ndl.Tensor(labels, device=parameter.device, requires_grad=False)

        if opt is not None:
            opt.reset_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        if opt is not None:
            loss.backward()
            opt.step()

        batch_size = labels.shape[0]
        correct += int(
            np.count_nonzero(logits.numpy().argmax(axis=1) == labels.numpy())
        )
        total_loss += loss.numpy().item() * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("CIFAR-10 epoch requires at least one example")
    return correct / total_examples, total_loss / total_examples
    ### END YOUR SOLUTION


def train_cifar10(
    model,
    dataloader,
    n_epochs=1,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    loss_fn=nn.SoftmaxLoss,
):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if n_epochs < 1:
        raise ValueError("n_epochs must be positive")
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = loss_fn()
    for _ in range(n_epochs):
        metrics = epoch_general_cifar10(dataloader, model, loss_fn=criterion, opt=opt)
    return metrics
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    return epoch_general_cifar10(dataloader, model, loss_fn=loss_fn())
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(
    data,
    model,
    seq_len=40,
    loss_fn=nn.SoftmaxLoss(),
    opt=None,
    clip=None,
    device=None,
    dtype="float32",
):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    if len(data) < 2:
        raise ValueError("PTB epoch requires at least two time steps")
    if opt is None:
        model.eval()
    else:
        model.train()
    if device is None:
        parameter = next(iter(model.parameters()), None)
        if parameter is not None:
            device = parameter.device

    state = None
    correct = total_tokens = 0
    total_loss = 0.0
    for start in range(0, len(data) - 1, seq_len):
        tokens, targets = ndl.data.get_batch(data, start, seq_len, device=device, dtype=dtype)
        if opt is not None:
            opt.reset_grad()
        logits, state = model(tokens, state)
        loss = loss_fn(logits, targets)
        if opt is not None:
            loss.backward()
            if clip is not None:
                opt.clip_grad_norm(clip)
            opt.step()

        # Carry the values across chunks and end backpropagation at each boundary.
        if isinstance(state, tuple):
            state = tuple(part.detach() for part in state)
        else:
            state = state.detach()

        count = targets.shape[0]
        correct += int(np.count_nonzero(logits.numpy().argmax(axis=1) == targets.numpy()))
        total_loss += loss.numpy().item() * count
        total_tokens += count

    return correct / total_tokens, total_loss / total_tokens
    ### END YOUR SOLUTION


def train_ptb(
    model,
    data,
    seq_len=40,
    n_epochs=1,
    optimizer=ndl.optim.SGD,
    lr=4.0,
    weight_decay=0.0,
    loss_fn=nn.SoftmaxLoss,
    clip=None,
    device=None,
    dtype="float32",
):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if n_epochs < 1:
        raise ValueError("n_epochs must be positive")
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = loss_fn()
    for _ in range(n_epochs):
        metrics = epoch_general_ptb(
            data, model, seq_len=seq_len, loss_fn=criterion, opt=opt,
            clip=clip, device=device, dtype=dtype,
        )
    return metrics
    ### END YOUR SOLUTION


def evaluate_ptb(
    model, data, seq_len=40, loss_fn=nn.SoftmaxLoss, device=None, dtype="float32"
):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    return epoch_general_ptb(
        data, model, seq_len=seq_len, loss_fn=loss_fn(), device=device, dtype=dtype
    )
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
