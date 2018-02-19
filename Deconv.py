from __future__ import print_function
import theano
import theano.tensor as T
import lasagne
from lasagne.nonlinearities import rectify
import numpy as np
from lasagne.layers import Deconv2DLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import InverseLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import get_output
from lasagne.objectives import squared_error

X_train = np.random.randint(low=10, high=20, size=(10,3,224,224))
y_train = np.random.randint(low=10, high=20, size=(10,21,224,224)) ###
X_valid = np.random.randint(low=10, high=20, size=(3,3,224,224))
y_valid = np.random.randint(low=10, high=20, size=(3,21,224,224))  ###

X_train = X_train.astype(float)
y_train = y_train.astype(float)
X_valid = X_valid.astype(float)
y_valid = y_valid.astype(float)


dataset = {
    'train': {'X': X_train, 'y': y_train},
    'valid': {'X': X_valid, 'y': y_valid}}
#input_shape = dataset['train']['X'][0].shape

input_var = T.tensor4('X')
target_var = T.tensor4('y')
l_in = InputLayer(shape=(None, 3, 224, 224),input_var=input_var)
#l_in = InputLayer(shape=(None, input_shape[0], input_shape[1], input_shape[2]))
l_conv1_1 = Conv2DLayer(l_in, num_filters=64, filter_size=(3, 3), stride=1, pad=1, flip_filters=False, nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_conv1_2 = Conv2DLayer(l_conv1_1, num_filters=64, filter_size=(3, 3), stride=1, pad=1, flip_filters=False,  nonlinearity=rectify, W=lasagne.init.Normal(0.01))
# Other arguments: Convolution type (full, same, or valid) and stride
l_pool1 = MaxPool2DLayer(l_conv1_2, pool_size=(2, 2), stride=2, pad=0)

l_conv2_1 = Conv2DLayer(l_pool1, num_filters=128, filter_size=(3, 3), stride=1, pad=1, flip_filters=False,  nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_conv2_2 = Conv2DLayer(l_conv2_1, num_filters=128, filter_size=(3, 3), stride=1, pad=1, flip_filters=False,  nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_pool2 = MaxPool2DLayer(l_conv2_2, pool_size=(2, 2), stride=2, pad=0)

l_conv3_1 = Conv2DLayer(l_pool2, num_filters=256, filter_size=(3, 3), stride=1, pad=1,  flip_filters=False,  nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_conv3_2 = Conv2DLayer(l_conv3_1, num_filters=256, filter_size=(3, 3), stride=1, pad=1, flip_filters=False,  nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_conv3_3 = Conv2DLayer(l_conv3_2, num_filters=256, filter_size=(3, 3), stride=1, pad=1,  flip_filters=False,  nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_pool3 = MaxPool2DLayer(l_conv3_3, pool_size=(2, 2), stride=2, pad=0)

l_conv4_1 = Conv2DLayer(l_pool3, num_filters=512, filter_size=(3, 3), stride=1, pad=1,  flip_filters=False,  nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_conv4_2 = Conv2DLayer(l_conv4_1, num_filters=512, filter_size=(3, 3), stride=1, pad=1, flip_filters=False, nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_conv4_3 = Conv2DLayer(l_conv4_2, num_filters=512, filter_size=(3, 3), stride=1, pad=1, flip_filters=False, nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_pool4 = MaxPool2DLayer(l_conv4_3, pool_size=(2, 2), stride=2, pad=0)

l_conv5_1 = Conv2DLayer(l_pool4, num_filters=512, filter_size=(3, 3), stride=1, pad=1, flip_filters=False, nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_conv5_2 = Conv2DLayer(l_conv5_1, num_filters=512, filter_size=(3, 3), stride=1, pad=1, flip_filters=False, nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_conv5_3 = Conv2DLayer(l_conv5_2, num_filters=512, filter_size=(3, 3), stride=1, pad=1, flip_filters=False, nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_pool5 = MaxPool2DLayer(l_conv5_3, pool_size=(2, 2), stride=2, pad=0)

l_fc6 = Conv2DLayer(l_pool5, num_filters=4096, filter_size=(7,7), stride=1, pad=0,  flip_filters=False, nonlinearity=rectify, W=lasagne.init.Normal(0.01)) ###
l_fc7 = Conv2DLayer(l_fc6, num_filters=4096, filter_size=(1,1), stride=1, pad=0, flip_filters=False, nonlinearity=rectify, W=lasagne.init.Normal(0.01))   ###
l_fc6_deconv = Deconv2DLayer(l_fc7, num_filters=512, filter_size=(7,7),stride=1, nonlinearity=rectify, W=lasagne.init.Normal(0.01))

l_unpool5 = InverseLayer(l_fc6_deconv, l_pool5)

l_deconv5_1 = Deconv2DLayer(l_unpool5, num_filters=512, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_deconv5_2 = Deconv2DLayer(l_deconv5_1, num_filters=512, filter_size=(3,3),stride=1, crop='same',  nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_deconv5_3 = Deconv2DLayer(l_deconv5_2, num_filters=512, filter_size=(3,3), stride=1, crop='same',  nonlinearity=rectify, W=lasagne.init.Normal(0.01))

l_unpool4 = InverseLayer(l_deconv5_3, l_pool4)

l_deconv4_1 = Deconv2DLayer(l_unpool4, num_filters=512, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_deconv4_2 = Deconv2DLayer(l_deconv4_1, num_filters=512, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_deconv4_3 = Deconv2DLayer(l_deconv4_2, num_filters=256, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01))

l_unpool3 = InverseLayer(l_deconv4_3, l_pool3)

l_deconv3_1 = Deconv2DLayer(l_unpool3, num_filters=256, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_deconv3_2 = Deconv2DLayer(l_deconv3_1, num_filters=256, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_deconv3_3 = Deconv2DLayer(l_deconv3_2, num_filters=128, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01))

l_unpool2 = InverseLayer(l_deconv3_3, l_pool2)

l_deconv2_1 = Deconv2DLayer(l_unpool2, num_filters=128, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_deconv2_2 = Deconv2DLayer(l_deconv2_1, num_filters=64, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01))

l_unpool1 = InverseLayer(l_deconv2_2, l_pool1)

l_deconv1_1 = Deconv2DLayer(l_unpool1, num_filters=64, filter_size=(3,3), stride=1, crop='same',  nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_deconv1_2 = Deconv2DLayer(l_deconv1_1, num_filters=64, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01))
l_output = Conv2DLayer(l_deconv1_2, num_filters=21, filter_size=(1, 1), pad=0) ###
#l_output = DenseLayer(l_hidden1_dropout, num_units=10, nonlinearity=softmax)
prediction = get_output(l_output)
# target_var = T.ftensor4('true_output') ###

loss = squared_error(prediction, target_var)
loss = loss.mean()
# loss.mean()
# loss_train = squared_error(lasagne.layers.get_output(l_output, deterministic=False), true_output).mean()
# loss_eval = squared_error(lasagne.layers.get_output(l_output, deterministic=True), true_output).mean()

all_params = lasagne.layers.get_all_params(l_output, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, all_params, learning_rate=0.001, momentum=0.985)

train_fn = theano.function([input_var, target_var], loss, updates=updates)
# get_output = theano.function([l_in.input_var], lasagne.layers.get_output(l_output, deterministic=True))

BATCH_SIZE = 10
N_EPOCHS = 5
batch_idx = 0
epoch = 0
input_batch = np.random.randint(low=10, high=20, size=(10, 3, 224, 224))
target_batch = np.random.randint(low=10, high=20, size=(10, 21, 224, 224))  ###

print("start training")
for epoch in range(N_EPOCHS):
    #print(net_output)
    loss = 0
    loss += train_fn(input_batch, target_batch)
    print("Epoch %d: Loss %g" % (epoch + 1, loss))

    # loss = train(dataset['train']['X'][batch_idx:batch_idx + BATCH_SIZE], dataset['train']['y'][batch_idx:batch_idx + BATCH_SIZE])


# true_output = T.ltensor4('true_output') ###
# loss = (T.sum(squared_error(net_output, true_output)))**2
# loss_train = (T.sum(squared_error(get_output(l_output, deterministic=False), true_output)))**2 ###
# loss_eval = (T.sum(squared_error(get_output(l_output, deterministic=True), true_output)))**2   ###
# all_params = lasagne.layers.get_all_params(l_output)
# updates = lasagne.updates.sgd(loss_train, all_params, learning_rate=0.001)
#
# train = theano.function([l_in.input_var, true_output], loss_train, updates=updates)
# get_output_function = theano.function([l_in.input_var], get_output(l_output, deterministic=True))
#
# BATCH_SIZE = 1
# N_EPOCHS = 10
# batch_idx = 1
# epoch = 0


# while epoch < N_EPOCHS:
#     loss = train(dataset['train']['X'][batch_idx:batch_idx + BATCH_SIZE],
#           dataset['train']['y'][batch_idx:batch_idx + BATCH_SIZE])
#     print (loss)
#     batch_idx += BATCH_SIZE
#     if batch_idx >= dataset['train']['X'].shape[0]:
#         batch_idx = 0
#         epoch += 1
#         val_output = get_output_function(dataset['valid']['X'])
#         val_predictions = val_output
#         accuracy = np.sum(val_predictions - dataset['valid']['y'])
#         print("Epoch {} validation accuracy: {}".format(epoch, accuracy))
