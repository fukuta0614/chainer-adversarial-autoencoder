# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import chainer
from chainer import cuda, Variable, function, FunctionSet, optimizers
from chainer import functions as F
from chainer import links as L


class AAE(chainer.Chain):
    def __init__(self, n_x, n_z, hidden_units_enc=(1000, 1000, 500), hidden_units_dec=(500, 1000, 1000)):
        layers = {}
        layer_units_enc = [(n_x, hidden_units_enc[0])]
        layer_units_enc += zip(hidden_units_enc[:-1], hidden_units_enc[1:])

        for i, (n_in, n_out) in enumerate(layer_units_enc):
            layers["enc_layer_{}".format(i)] = L.Linear(n_in, n_out)
            layers["enc_batchnorm_{}".format(i)] = L.BatchNormalization(n_out)

        layers["enc_layer_out"] = L.Linear(hidden_units_enc[-1], n_z)

        self.n_layers_enc = len(layer_units_enc) + 1

        layer_units_dec = [(n_z, hidden_units_dec[0])]
        layer_units_dec += zip(hidden_units_dec[:-1], hidden_units_dec[1:])

        for i, (n_in, n_out) in enumerate(layer_units_dec):
            layers["dec_layer_{}".format(i)] = L.Linear(n_in, n_out)
            layers["dec_batchnorm_{}".format(i)] = L.BatchNormalization(n_out)

        layers["dec_layer_out"] = L.Linear(hidden_units_dec[-1], n_x)

        self.n_layers_dec = len(layer_units_dec) + 1

        self.dropout = False
        super(AAE, self).__init__(**layers)

    def encode(self, x, test=False):

        activate = F.relu

        # Hidden
        h = x
        for i in range(self.n_layers_enc - 1):
            h = getattr(self, "enc_layer_%i" % i)(h)
            h = getattr(self, "enc_batchnorm_%i" % i)(h, test=test)

            h = activate(h)
            if self.dropout:
                h = F.dropout(h, train=not test)

        # Output
        output = getattr(self, "enc_layer_out")(h)

        return output

    def decode(self, x, test=False, sigmoid=True):

        activate = F.relu

        # Hidden
        h = x
        for i in range(self.n_layers_dec - 1):
            h = getattr(self, "dec_layer_%i" % i)(h)
            h = getattr(self, "dec_batchnorm_%i" % i)(h, test=test)

            h = activate(h)
            if self.dropout:
                h = F.dropout(h, train=not test)

        # Output
        output = getattr(self, "dec_layer_out")(h)
        if sigmoid:
            output = F.sigmoid(output)

        return output


class Discriminator(chainer.Chain):
    def __init__(self, n_z, hidden_units=(500, 250, 50)):

        layers = {}
        layer_units = [(n_z, hidden_units[0])]
        layer_units += zip(hidden_units[:-1], hidden_units[1:])

        for i, (n_in, n_out) in enumerate(layer_units):
            layers["layer_{}".format(i)] = L.Linear(n_in, n_out)
            layers["batchnorm_{}".format(i)] = L.BatchNormalization(n_out)

        layers["layer_out"] = L.Linear(hidden_units[-1], 2)

        self.n_layers = len(layer_units) + 1
        self.dropout = False
        super(Discriminator, self).__init__(**layers)

    def forward(self, x, test=False):

        activate = F.relu

        # Hidden
        h = x
        for i in range(self.n_layers - 1):
            h = getattr(self, "layer_%i" % i)(h)
            h = getattr(self, "batchnorm_%i" % i)(h, test=test)

            h = activate(h)
            if self.dropout:
                h = F.dropout(h, train=not test)

        # Output
        output = getattr(self, "layer_out")(h)

        return output

    def __call__(self, z, test=False):
        return self.forward(z, test=test)