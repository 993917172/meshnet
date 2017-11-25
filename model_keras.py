import keras
from keras import backend as K
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.convolutional import UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Activation
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

import pydot

import numpy as numpy
import meshnet

log_filepath = './meshnet01'
vertex_count = 53215
batch_size = 16
epochs = 10
is_training = True

# configs
nFeats = 256    # number of features in the hourglass
nStack = 2      # number of hourglasses to stack
nModules = 1    # number of residual modules at each location in the hourglass

inputRes = 256
outputRes = 64

def nnConv(input, nInputPlane, nOutputPlane, kW, kH, dW=1, dH=1, padW=0, padH=0, name=None):
    assert padW == padH
    assert kW == kH
    assert dW == dH
    assert input.shape[3] == nInputPlane
    if padW == 0:
        padding = 'VALID'
    else:
        assert padW == kW//2
        padding = 'SAME'
    return Conv2D(filters=nOutputPlane, kernel_size=(kW, kH), strides=(dW, dH), padding=padding, name=name)(input)

def nnMaxPooling(input, kW, kH, dW, dH):
    assert kW == kH
    return MaxPooling2D(pool_size=(kW, kH), strides=(dW, dH))(input)

def nnUpSamplingNearest(input, scale):
    # return tf.image.resize_nearest_neighbor(input, tf.shape(input)[1:3]*scale)
    return UpSampling2D(size=(scale, scale))(input)

def nnBatchNormalization(input):
  return BatchNormalization()(input)

def bn_relu(input):
    return Activation('relu')(nnBatchNormalization(input))

def Residual(input, numIn, numOut, name='residual_block'):
    def convBlock(input, numIn, numOut):
        norm_1 = bn_relu(input)
        conv_1 = nnConv(norm_1, numIn, numOut//2, 1,1)

        norm_2 = bn_relu(conv_1)
        conv_2 = nnConv(norm_2, numOut//2, numOut//2, 3,3, 1,1, 1,1)

        norm_3 = bn_relu(conv_2)
        conv_3 = nnConv(norm_3, numOut//2, numOut, 1,1)
        return conv_3

    def skipLayer(input, numIn, numOut):
        if numIn == numOut:
            return input
        else:
            # return nnBatchNormalization(nnConv(input, numIn, numOut, 1, 1))
            return nnConv(input, numIn, numOut, 1,1)
    with K.name_scope(name):
        return keras.layers.add([convBlock(input, numIn, numOut), skipLayer(input, numIn, numOut)])

def hourglass(inputs, n, f):
    # print(n, inputs.shape)
    up_1 = Residual(inputs, f, f)
    low_ = nnMaxPooling(inputs, 2,2, 2,2)
    low_1 = Residual(low_, f, f)
    if n > 1:
        low_2 = hourglass(low_1, n-1, f)
    else:
        low_2 = Residual(low_1, f, f)

    low_3 = Residual(low_2, f, f)
    up_2 = nnUpSamplingNearest(low_3, 2)
    return keras.layers.add([up_1, up_2])


def lin(input, numIn, numOut):
    return bn_relu(nnConv(input, numIn, numOut, 1,1, 1,1, 0,0))

def build_model():
    inputs = Input(shape=(256, 256, 3))
    conv1_ = nnConv(inputs, 3, 64, 7, 7, 2, 2, 3, 3)    #(1)
    conv1 = bn_relu(conv1_)                             #(2) (3)
    r1 = Residual(conv1, 64, 128)                       #(4)
    pool = nnMaxPooling(r1, 2, 2, 2, 2)                 #(5)
    r4 = Residual(pool, 128, 128)                       #(6)
    r5 = Residual(r4, 128, nFeats)                      #(7)

    # for _ in range(nStack):
    hg1 = hourglass(r5, 4, nFeats)
    # todo: drop out ?
    ll_ = Residual(hg1, nFeats, nFeats)
    ll = lin(ll_, nFeats, nFeats)
    sum1 = keras.layers.add([r5, ll])                   #(9)
    # print(sum1.shape)

    hg2 = hourglass(sum1, 4, nFeats)
    ll_2 = Residual(hg2, nFeats, nFeats)
    ll2 = lin(ll_2, nFeats, nFeats)
    sum2 = keras.layers.add([sum1, ll2])                #(11)

    print(sum2.shape)
    model = Model(inputs=inputs, outputs=[sum2])
    return model

model = build_model()
# print(model.summary())
keras.utils.plot_model(model, to_file='hourglass.png')