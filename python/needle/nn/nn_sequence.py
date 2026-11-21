"""The module."""

from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Rescale numerator and denominator by the same factor to avoid overflow.
        negative = -x
        shift = ops.relu(negative).detach()
        numerator = ops.exp(-shift)
        return numerator / (numerator + ops.exp(negative - shift))
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError("nonlinearity must be 'tanh' or 'relu'")
        self.activation = ops.tanh if nonlinearity == "tanh" else ops.relu

        bound = hidden_size**-0.5
        options = dict(low=-bound, high=bound, device=device, dtype=dtype)
        self.W_ih = Parameter(init.rand(input_size, hidden_size, **options))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, **options))
        self.bias_ih = Parameter(init.rand(hidden_size, **options)) if bias else None
        self.bias_hh = Parameter(init.rand(hidden_size, **options)) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype)

        projected = X @ self.W_ih + h @ self.W_hh
        if self.bias_ih is not None:
            projected = projected + self.bias_ih.broadcast_to(projected.shape)
            projected = projected + self.bias_hh.broadcast_to(projected.shape)
        return self.activation(projected)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cells = [
            RNNCell(
                input_size if layer == 0 else hidden_size,
                hidden_size,
                bias=bias,
                nonlinearity=nonlinearity,
                device=device,
                dtype=dtype,
            )
            for layer in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h0 is None:
            initial_states = [None] * self.num_layers
        else:
            if h0.shape != (self.num_layers, X.shape[1], self.hidden_size):
                raise ValueError(
                    "h0 must have shape (num_layers, batch_size, hidden_size)"
                )
            initial_states = ops.split(h0, axis=0)

        steps = ops.split(X, axis=0)
        final_states = []
        for layer, cell in enumerate(self.rnn_cells):
            hidden = initial_states[layer]
            layer_steps = []
            for step in steps:
                hidden = cell(step, hidden)
                layer_steps.append(hidden)
            steps = layer_steps
            final_states.append(hidden)

        return ops.stack(steps, axis=0), ops.stack(final_states, axis=0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(
        self, input_size, hidden_size, bias=True, device=None, dtype="float32"
    ):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.sigmoid = Sigmoid()
        gate_size = 4 * hidden_size
        bound = hidden_size ** -0.5
        options = dict(low=-bound, high=bound, device=device, dtype=dtype)
        self.W_ih = Parameter(init.rand(input_size, gate_size, **options))
        self.W_hh = Parameter(init.rand(hidden_size, gate_size, **options))
        self.bias_ih = Parameter(init.rand(gate_size, **options)) if bias else None
        self.bias_hh = Parameter(init.rand(gate_size, **options)) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        if h is None:
            zero = init.zeros(batch_size, self.hidden_size, device=X.device, dtype=X.dtype)
            hidden, cell = zero, zero
        else:
            hidden, cell = h

        gates = X @ self.W_ih + hidden @ self.W_hh
        if self.bias_ih is not None:
            gates = gates + self.bias_ih.broadcast_to(gates.shape)
            gates = gates + self.bias_hh.broadcast_to(gates.shape)
        input_gate, forget_gate, candidate, output_gate = ops.split(
            gates.reshape((batch_size, 4, self.hidden_size)), axis=1
        )

        next_cell = self.sigmoid(forget_gate) * cell + self.sigmoid(input_gate) * ops.tanh(candidate)
        next_hidden = self.sigmoid(output_gate) * ops.tanh(next_cell)
        return next_hidden, next_cell
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = [
            LSTMCell(
                input_size if layer == 0 else hidden_size,
                hidden_size,
                bias=bias,
                device=device,
                dtype=dtype,
            )
            for layer in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            initial_states = [None] * self.num_layers
        else:
            h0, c0 = h
            state_shape = (self.num_layers, X.shape[1], self.hidden_size)
            if h0.shape != state_shape or c0.shape != state_shape:
                raise ValueError("h0 and c0 must have shape (num_layers, batch_size, hidden_size)")
            initial_states = list(zip(ops.split(h0, axis=0), ops.split(c0, axis=0)))

        steps = ops.split(X, axis=0)
        final_hidden, final_cell = [], []
        for layer, cell in enumerate(self.lstm_cells):
            state = initial_states[layer]
            layer_steps = []
            for step in steps:
                state = cell(step, state)
                layer_steps.append(state[0])
            steps = layer_steps
            hidden, cell_state = state
            final_hidden.append(hidden)
            final_cell.append(cell_state)

        return ops.stack(steps, axis=0), (
            ops.stack(final_hidden, axis=0),
            ops.stack(final_cell, axis=0),
        )
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(
            num_embeddings, embedding_dim, device=device, dtype=dtype
        ))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        indices = x.reshape((int(np.prod(x.shape)),))
        one_hot = init.one_hot(
            self.num_embeddings, indices, device=self.weight.device, dtype=self.weight.dtype
        )
        return (one_hot @ self.weight).reshape(x.shape + (self.embedding_dim,))
        ### END YOUR SOLUTION
