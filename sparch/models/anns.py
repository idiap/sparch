#
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the sparch package
#
"""
This is where the non-spiking Artificial Neural Network (ANN) baseline
is defined.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ANN(nn.Module):
    """
    A multi-layered Artificial Neural Network (ANN).

    It accepts input tensors formatted as (batch, time, feat). In the case of
    4d inputs like (batch, time, feat, channel) the input is flattened as
    (batch, time, feat*channel).

    The function returns the outputs of the last hidden or readout layer
    with shape (batch, time, feats) or (batch, feats) respectively.

    Arguments
    ---------
    input_shape : tuple
        Shape of an input example.
    layer_sizes : int list
        List of number of neurons in all hidden layers
    ann_type : str
        Type of neuron model, either 'MLP', 'RNN', 'LiGRU' or 'GRU'.
    dropout : float
        Dropout rate (must be between 0 and 1).
    normalization : str
        Type of normalization (batchnorm, layernorm). Every string different
        from batchnorm and layernorm will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
        Must be False with MLP ann type.
    use_readout_layer : bool
        If True, the final layer is a linear layer that outputs a cumulative
        sum of the sequence using a softmax function. The outputs have shape
        (batch, labels) with no time dimension. If False, the final layer
        is the same as the hidden layers and outputs sequences with shape
        (batch, time, labels).
    """

    def __init__(
        self,
        input_shape,
        layer_sizes,
        ann_type="MLP",
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        use_readout_layer=True,
    ):
        super().__init__()

        # Fixed parameters
        self.reshape = True if len(input_shape) > 3 else False
        self.input_size = float(torch.prod(torch.tensor(input_shape[2:])))
        self.batch_size = input_shape[0]
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.num_outputs = layer_sizes[-1]
        self.ann_type = ann_type
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.use_readout_layer = use_readout_layer
        self.is_snn = False

        if ann_type not in ["MLP", "RNN", "LiGRU", "GRU"]:
            raise ValueError(f"Invalid ann type {ann_type}")

        if bidirectional and ann_type == "MLP":
            raise ValueError("MLP cannot be bidirectional.")

        # Init trainable parameters
        self.ann = self._init_layers()

    def _init_layers(self):

        ann = nn.ModuleList([])
        input_size = self.input_size
        ann_class = self.ann_type + "Layer"

        if self.use_readout_layer:
            num_hidden_layers = self.num_layers - 1
        else:
            num_hidden_layers = self.num_layers

        # Hidden layers
        for i in range(num_hidden_layers):
            ann.append(
                globals()[ann_class](
                    input_size=input_size,
                    hidden_size=self.layer_sizes[i],
                    batch_size=self.batch_size,
                    dropout=self.dropout,
                    normalization=self.normalization,
                    use_bias=self.use_bias,
                    bidirectional=self.bidirectional,
                )
            )
            input_size = self.layer_sizes[i] * (1 + self.bidirectional)

        # Readout layer
        if self.use_readout_layer:
            ann.append(
                ReadoutLayerANN(
                    input_size=input_size,
                    output_size=self.layer_sizes[-1],
                    normalization=self.normalization,
                    use_bias=self.use_bias,
                )
            )

        return ann

    def forward(self, x):

        # Reshape input tensors to (batch, time, feats) for 4d inputs
        if self.reshape:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
            else:
                raise (NotImplementedError)

        # Process all layers
        for ann_lay in self.ann:
            x = ann_lay(x)

        return x, None  # so that same as SNN


class MLPLayer(nn.Module):
    """
    A single Multi-Layer-Perceptron layer without any recurrent connection
    (MLP). The activation function is a sigmoid.

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        Must be False. Only kept as an argument here for ANN class.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.batch_size = self.batch_size
        self.act_fct = nn.Sigmoid()

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply activation function and dropout
        y = self.drop(self.act_fct(Wx))

        return y


class RNNLayer(nn.Module):
    """
    A single recurrent layer without any gate (RNN). The activation function
    is a sigmoid.

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layer l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + bidirectional)
        self.act_fct = nn.Sigmoid()

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        nn.init.orthogonal_(self.V.weight)

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute recurrent dynamics
        y = self._rnn_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            y_f, y_b = y.chunk(2, dim=0)
            y_b = y_b.flip(1)
            y = torch.cat([y_f, y_b], dim=2)

        # Apply dropout
        y = self.drop(y)

        return y

    def _rnn_cell(self, Wx):

        # Initializations
        yt = torch.zeros(Wx.shape[0], Wx.shape[2]).to(Wx.device)
        y = []

        # Loop over time axis
        for t in range(Wx.shape[1]):
            yt = self.act_fct(Wx[:, t, :] + self.V(yt))
            y.append(yt)

        return torch.stack(y, dim=1)


class LiGRULayer(nn.Module):
    """
    A single layer of Light Gated Recurrent Units (LiGRU), introduced
    by Ravanelli et al. in https://arxiv.org/abs/1803.10225 (2018).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layer l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + bidirectional)
        self.act_fct = nn.ReLU()

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Wz = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.Vz = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        nn.init.orthogonal_(self.V.weight)
        nn.init.orthogonal_(self.Vz.weight)

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normz = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normz = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)
        Wzx = self.Wz(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

            _Wzx = self.normz(Wzx.reshape(Wzx.shape[0] * Wzx.shape[1], Wzx.shape[2]))
            Wzx = _Wzx.reshape(Wzx.shape[0], Wzx.shape[1], Wzx.shape[2])

        # Compute recurrent dynamics
        y = self._ligru_cell(Wx, Wzx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            y_f, y_b = y.chunk(2, dim=0)
            y_b = y_b.flip(1)
            y = torch.cat([y_f, y_b], dim=2)

        # Apply dropout
        y = self.drop(y)

        return y

    def _ligru_cell(self, Wx, Wzx):

        # Initializations
        yt = torch.zeros(Wx.shape[0], Wx.shape[2]).to(Wx.device)
        y = []

        # Loop over time axis
        for t in range(Wx.shape[1]):
            zt = torch.sigmoid(Wzx[:, t, :] + self.Vz(yt))
            ct = self.act_fct(Wx[:, t, :] + self.V(yt))
            yt = zt * yt + (1 - zt) * ct
            y.append(yt)

        return torch.stack(y, dim=1)


class GRULayer(nn.Module):
    """
    A single layer of Gated Recurrent Units (GRU), introduced by Cho et al.
    in https://arxiv.org/abs/1406.1078 (2014).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layer l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + bidirectional)
        self.act_fct = nn.Tanh()

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Wz = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.Vz = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Wr = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.Vr = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        nn.init.orthogonal_(self.V.weight)
        nn.init.orthogonal_(self.Vz.weight)
        nn.init.orthogonal_(self.Vr.weight)

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normz = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normr = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normz = nn.LayerNorm(self.hidden_size)
            self.normr = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)
        Wzx = self.Wz(x)
        Wrx = self.Wr(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

            _Wzx = self.normz(Wzx.reshape(Wzx.shape[0] * Wzx.shape[1], Wzx.shape[2]))
            Wzx = _Wzx.reshape(Wzx.shape[0], Wzx.shape[1], Wzx.shape[2])

            _Wrx = self.normr(Wrx.reshape(Wrx.shape[0] * Wrx.shape[1], Wrx.shape[2]))
            Wrx = _Wrx.reshape(Wrx.shape[0], Wrx.shape[1], Wrx.shape[2])

        # Compute recurrent dynamics
        y = self._gru_cell(Wx, Wzx, Wrx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            y_f, y_b = y.chunk(2, dim=0)
            y_b = y_b.flip(1)
            y = torch.cat([y_f, y_b], dim=2)

        # Apply dropout
        y = self.drop(y)

        return y

    def _gru_cell(self, Wx, Wzx, Wrx):

        # Initializations
        yt = torch.zeros(Wx.shape[0], Wx.shape[2]).to(Wx.device)
        y = []

        # Loop over time axis
        for t in range(Wx.shape[1]):
            zt = torch.sigmoid(Wzx[:, t, :] + self.Vz(yt))
            rt = torch.sigmoid(Wrx[:, t, :] + self.Vr(yt))
            ct = self.act_fct(Wx[:, t, :] + self.V(rt * yt))
            yt = zt * yt + (1 - zt) * ct
            y.append(yt)

        return torch.stack(y, dim=1)


class ReadoutLayerANN(nn.Module):
    """
    A readout layer that computes a cumulative sum over time using a softmax
    function, and then applies a linear layer to the sum. The input and output
    tensors therefore have shape (batch, time, feats) and (batch, labels).

    Arguments
    ---------
    input_size : int
        Feature dimensionality of the input tensors.
    output_size : int
        Number of output neurons.
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    """

    def __init__(
        self,
        input_size,
        output_size,
        normalization="batchnorm",
        use_bias=False,
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.normalization = normalization
        self.use_bias = use_bias

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.output_size, bias=use_bias)

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.output_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.output_size)
            self.normalize = True

    def forward(self, x):

        # Compute cumulative sum
        y = self._readout_cell(x)

        # Feed-forward affine transformations
        Wy = self.W(y)

        # Apply normalization
        if self.normalize:
            Wy = self.norm(Wy)

        return Wy

    def _readout_cell(self, x):

        # Cumulative sum to remove time dim
        y = 0
        for t in range(x.shape[1]):
            y += F.softmax(x[:, t, :], dim=-1)

        return y
