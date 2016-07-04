# -*- coding: utf-8 -*-
import os, re, math, pylab
from math import *
import numpy as np

import matplotlib.patches as mpatches

import chainer

from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


def load_mnist():
    mnist = fetch_mldata('MNIST original')
    mnist_X, mnist_y = shuffle(mnist.data.astype('float32'), mnist.target.astype('int32'), random_state=1234)

    mnist_X /=  255.
    mnist_y = np.eye(10)[mnist_y].astype('int32')
    x_train, x_test, y_train, y_test = train_test_split(mnist_X, mnist_y, test_size=0.2, random_state=1234)

    return x_train, x_test, y_train, y_test


def sample_z_from_n_2d_gaussian_mixture(batchsize, z_dim, label_indices, n_labels=10, gpu=False):
    if z_dim % 2 != 0:
        raise Exception("z_dim must be a multiple of 2.")

    def sample(x, y, label, n_labels):
        shift = 2.0
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 0.5
    y_var = 0.18
    x = np.random.normal(0, x_var, (batchsize, z_dim / 2))
    y = np.random.normal(0, y_var, (batchsize, z_dim / 2))
    z = np.empty((batchsize, z_dim), dtype=np.float32)
    for batch in xrange(batchsize):
        for zi in xrange(z_dim / 2):
            z[batch, zi * 2:zi * 2 + 2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)

    z = chainer.Variable(z)
    if gpu:
        z.to_gpu()
    return z


def visualize_10_2d_gaussian_prior(n_z, y_label, visualization_dir=None):
    z_batch = sample_z_from_n_2d_gaussian_mixture(len(y_label), n_z, y_label, 10, False)
    z_batch = z_batch.data

    fig = pylab.gcf()
    fig.set_size_inches(15, 12)
    pylab.clf()
    colors = ["#2103c8", "#0e960e", "#e40402", "#05aaa8", "#ac02ab", "#aba808", "#151515", "#94a169", "#bec9cd",
              "#6a6551"]
    for n in xrange(z_batch.shape[0]):
        result = pylab.scatter(z_batch[n, 0], z_batch[n, 1], c=colors[y_label[n]], s=40, marker="o",
                               edgecolors='none')

    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    recs = []
    for i in range(0, len(colors)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=colors[i]))

    ax = pylab.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(recs, classes, loc="center left", bbox_to_anchor=(1.1, 0.5))
    pylab.xticks(pylab.arange(-4, 5))
    pylab.yticks(pylab.arange(-4, 5))
    pylab.xlabel("z1")
    pylab.ylabel("z2")
    if visualization_dir is not None:
        pylab.savefig("%s/10_2d-gaussian.png" % visualization_dir)
    pylab.show()


def visualize_labeled_z(xp, model, x, y_label, visualization_dir, epoch):
    x = chainer.Variable(xp.asarray(x))
    z_batch = model.encode(x, test=True)
    z_batch = z_batch.data.get()

    fig = pylab.gcf()
    fig.set_size_inches(8.0, 8.0)
    pylab.clf()
    colors = ["#2103c8", "#0e960e", "#e40402", "#05aaa8", "#ac02ab", "#aba808", "#151515", "#94a169", "#bec9cd",
              "#6a6551"]
    for n in xrange(z_batch.shape[0]):
        result = pylab.scatter(z_batch[n, 0], z_batch[n, 1], c=colors[y_label[n]], s=40, marker="o",
                               edgecolors='none')

    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    recs = []
    for i in range(0, len(colors)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=colors[i]))

    ax = pylab.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(recs, classes, loc="center left", bbox_to_anchor=(1.1, 0.5))
    pylab.xticks(pylab.arange(-4, 5))
    pylab.yticks(pylab.arange(-4, 5))
    pylab.xlabel("z1")
    pylab.ylabel("z2")
    pylab.savefig("{}/labeled_z_{}.png".format(visualization_dir, epoch))
    pylab.show()


def visualize_reconstruction(xp, model, x, visualization_dir, epoch):
    x_variable = chainer.Variable(xp.asarray(x))
    _x = model.decode(model.encode(x_variable), test=True)
    _x = _x.data.get()

    fig = pylab.gcf()
    fig.set_size_inches(8.0, 8.0)
    pylab.clf()
    pylab.gray()
    for m in range(50):
        i = m / 10
        j = m % 10
        pylab.subplot(10, 10, 20 * i + j + 1, xticks=[], yticks=[])
        pylab.imshow(x[m].reshape((28, 28)), interpolation="none")
        pylab.subplot(10, 10, 20 * i + j + 10 + 1, xticks=[], yticks=[])
        pylab.imshow(_x[m].reshape((28, 28)), interpolation="none")
        # pylab.imshow(np.clip((_x_batch.data[m] + 1.0) / 2.0, 0.0, 1.0).reshape(
        # (config.img_channel, config.img_width, config.img_width)), interpolation="none")
        pylab.axis("off")
    pylab.savefig("{}/reconstruction_{}.png".format(visualization_dir, epoch))
    pylab.show()
