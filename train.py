# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse

import matplotlib.pyplot as plt
import numpy as np

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers,optimizer
from chainer import serializers
from chainer import functions as F
import os, sys, time
from PIL import Image

import utils
from model import Discriminator, AAE

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=100, type=int,
                    help='number of epochs to learn')
parser.add_argument('--dimz', '-z', default=2, type=int,
                    help='dimention of encoded vector')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--model_dir", type=str, default="./models")
parser.add_argument("--visualization_dir", type=str, default="./visualization")
parser.add_argument("--load_epoch", type=int, default=0)
args = parser.parse_args()

batchsize = args.batchsize
n_epoch = args.epoch
n_z = args.dimz
if n_z % 2 != 0:
    raise Exception("The dimension of the latent code z must be a multiple of 2.")

print('GPU: {}'.format(args.gpu))
print('# dim z: {}'.format(args.dimz))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))

try:
    os.mkdir(args.model_dir)
except:
    pass

try:
    os.mkdir(args.visualization_dir)
except:
    pass

x_train, x_test, y_train, y_test = utils.load_mnist()
N = len(x_train)

model = AAE(784, n_z, hidden_units_enc=(1000, 1000, 500), hidden_units_dec=(500,1000,1000))
dis = Discriminator(n_z+10)

use_gpu = args.gpu >= 0
if use_gpu:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    dis.to_gpu()

xp = np if args.gpu < 0 else cuda.cupy

optimizer_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_aae = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_dis.setup(dis)
optimizer_aae.setup(model)
optimizer_dis.add_hook(optimizer.WeightDecay(0.0001))
optimizer_aae.add_hook(optimizer.WeightDecay(0.0001))

n_steps_to_optimize_dis = 1
total_time = 0
C = 10

for epoch in range(1, n_epoch+1):
    print('Epoch {}'.format(epoch))

    sum_loss_regularization = 0
    sum_loss_reconstruction = 0

    start_time = time.time()

    perm = np.random.permutation(N)

    for i in range(0, N, batchsize):
        x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
        y = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]].astype(np.float32)))
        y_label = np.argmax(y_train[perm[i:i + batchsize]], axis=1)

        # Reconstruction phase
        z_fake_batch = model.encode(x)
        _x = model.decode(z_fake_batch, sigmoid=False)

        loss_reconstruction = F.bernoulli_nll(x, _x) / batchsize

        # Adversarial phase
        z_real_batch = utils.sample_z_from_n_2d_gaussian_mixture(batchsize, n_z, y_label, 10, use_gpu)
        z_real_batch_with_label = F.concat((z_real_batch, y))
        p_real_batch = dis(z_real_batch_with_label)

        z_fake_batch_with_label = F.concat((z_fake_batch, y))
        p_fake_batch = dis(z_fake_batch_with_label)

        loss_dis_real = F.softmax_cross_entropy(p_real_batch, chainer.Variable(xp.zeros(batchsize, dtype=np.int32)))
        loss_dis_fake = F.softmax_cross_entropy(p_fake_batch, chainer.Variable(xp.ones(batchsize, dtype=np.int32)))
        loss_dis = loss_dis_fake + loss_dis_real

        loss_gen = F.softmax_cross_entropy(p_fake_batch, chainer.Variable(xp.zeros(batchsize, dtype=np.int32)))
        loss_aae = loss_reconstruction + C * loss_gen

        optimizer_dis.zero_grads()
        loss_dis.backward()
        optimizer_dis.update()

        optimizer_aae.zero_grads()
        loss_aae.backward()
        optimizer_aae.update()

        # sum_loss_regularization += float(loss_dis.data) * batchsize_loop
        sum_loss_regularization += float(C*loss_gen.data) * batchsize
        sum_loss_reconstruction += float(loss_reconstruction.data) * batchsize


    # Saving the models
    print(" reconstruction_loss", (sum_loss_reconstruction / N))
    print(" regularization_loss", (sum_loss_regularization / N))
    p_real_batch.to_cpu()
    p_real_batch = p_real_batch.data.transpose(1, 0)
    p_real_batch = np.exp(p_real_batch)
    sum_p_real_batch = p_real_batch[0] + p_real_batch[1]
    win_real = p_real_batch[0] / sum_p_real_batch
    print(" D(real_z)", win_real.mean())
    p_fake_batch.to_cpu()
    p_fake_batch = p_fake_batch.data.transpose(1, 0)
    p_fake_batch = np.exp(p_fake_batch)
    sum_p_fake_batch = p_fake_batch[0] + p_fake_batch[1]
    win_fake = p_fake_batch[0] / sum_p_fake_batch
    print(" D(enc_z) ", win_fake.mean())

    elapsed_time = time.time() - start_time
    # print "	time", elapsed_time
    total_time += elapsed_time
    print(" total_time", total_time)

    if epoch % 10 == 0:
        serializers.save_hdf5("%s/aae_epoch_%d.model" % (args.model_dir, epoch), model)
        serializers.save_hdf5("%s/dis_epoch_%d.model" % (args.model_dir, epoch), dis)
        utils.visualize_labeled_z(xp, model, x_test[:1000], np.argmax(y_test[:1000], axis=1), args.visualization_dir, epoch)
        utils.visualize_reconstruction(xp, model, x_test[:50], args.visualization_dir, epoch)
