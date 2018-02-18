from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import lasagne
from lasagne.nonlinearities import rectify
from lasagne.layers import Deconv2DLayer
from lasagne.layers import InputLayer
from lasagne.layers import batch_norm
from lasagne.layers import InverseLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import Conv2DLayer
from lasagne.objectives import squared_error
from lasagne.objectives import categorical_crossentropy

def trans_shape(input):
    tmp = np.zeros((input.shape[0], 21, 224, 224));
    for x in range(input.shape[2]):
        for y in range(input.shape[3]):
            for n in range(input.shape[0]):
                #print [n, input[n,0,x,y], x,y]
                if input[n,0,x,y] ==255 or 244:
                    t=0
                else:
                    t=input[n,0,x,y]
                tmp[n, t, x, y] = 1
    return tmp

data_X_train = np.load('data_X_train_20.npy')
data_y_train = np.load('data_y_train_20.npy')
X_train = data_X_train[0:300,:,:,:]
y_train = trans_shape(data_y_train[0:300,:,:,:])
X_valid = data_X_train[300:360,:,:,:]
y_valid = data_y_train[300:360,:,:,:]
# X_train = np.random.randint(low=10, high=20, size=(20,3,224,224))
# y_train = np.random.randint(low=10, high=20, size=(20,21,224,224)) ###
# X_valid = np.random.randint(low=10, high=20, size=(5,3,224,224))
# y_valid = np.random.randint(low=10, high=20, size=(5,1,224,224))  ###
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_valid = X_valid.astype(np.float32)
y_valid = y_valid.astype(np.float32)


dataset = {
    'train': {'X': X_train, 'y': y_train},
    'valid': {'X': X_valid, 'y': y_valid}}
l_in = InputLayer(shape=(None, 3, 224, 224))
l_conv1_1 =  batch_norm(Conv2DLayer(l_in, num_filters=64, filter_size=(3, 3), stride=1, pad=1, flip_filters=False, nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_conv1_2 = batch_norm(Conv2DLayer(l_conv1_1, num_filters=64, filter_size=(3, 3), stride=1, pad=1, flip_filters=False,  nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_pool1 = MaxPool2DLayer(l_conv1_2, pool_size=(2, 2), stride=2, pad=0)

l_conv2_1 = batch_norm(Conv2DLayer(l_pool1, num_filters=128, filter_size=(3, 3), stride=1, pad=1, flip_filters=False,  nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_conv2_2 = batch_norm(Conv2DLayer(l_conv2_1, num_filters=128, filter_size=(3, 3), stride=1, pad=1, flip_filters=False,  nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_pool2 = MaxPool2DLayer(l_conv2_2, pool_size=(2, 2), stride=2, pad=0)

l_conv3_1 = batch_norm(Conv2DLayer(l_pool2, num_filters=256, filter_size=(3, 3), stride=1, pad=1,  flip_filters=False,  nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_conv3_2 = batch_norm(Conv2DLayer(l_conv3_1, num_filters=256, filter_size=(3, 3), stride=1, pad=1, flip_filters=False,  nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_conv3_3 = batch_norm(Conv2DLayer(l_conv3_2, num_filters=256, filter_size=(3, 3), stride=1, pad=1,  flip_filters=False,  nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_pool3 = MaxPool2DLayer(l_conv3_3, pool_size=(2, 2), stride=2, pad=0)

l_conv4_1 = batch_norm(Conv2DLayer(l_pool3, num_filters=512, filter_size=(3, 3), stride=1, pad=1,  flip_filters=False,  nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_conv4_2 = batch_norm(Conv2DLayer(l_conv4_1, num_filters=512, filter_size=(3, 3), stride=1, pad=1, flip_filters=False, nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_conv4_3 = batch_norm(Conv2DLayer(l_conv4_2, num_filters=512, filter_size=(3, 3), stride=1, pad=1, flip_filters=False, nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_pool4 = MaxPool2DLayer(l_conv4_3, pool_size=(2, 2), stride=2, pad=0)

l_conv5_1 = batch_norm(Conv2DLayer(l_pool4, num_filters=512, filter_size=(3, 3), stride=1, pad=1, flip_filters=False, nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_conv5_2 = batch_norm(Conv2DLayer(l_conv5_1, num_filters=512, filter_size=(3, 3), stride=1, pad=1, flip_filters=False, nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_conv5_3 = batch_norm(Conv2DLayer(l_conv5_2, num_filters=512, filter_size=(3, 3), stride=1, pad=1, flip_filters=False, nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_pool5 = MaxPool2DLayer(l_conv5_3, pool_size=(2, 2), stride=2, pad=0)

l_fc6 = batch_norm(Conv2DLayer(l_pool5, num_filters=4096, filter_size=(7,7), stride=1, pad=0,  nonlinearity=rectify, W=lasagne.init.Normal(0.01))) ###
l_fc7 = batch_norm(Conv2DLayer(l_fc6, num_filters=4096, filter_size=(1,1), stride=1, pad=0, nonlinearity=rectify, W=lasagne.init.Normal(0.01)))   ###
l_fc6_deconv = batch_norm(Deconv2DLayer(l_fc7, num_filters=512, filter_size=(7,7),stride=1, crop='valid', nonlinearity=rectify, W=lasagne.init.Normal(0.01)))

l_unpool5 = InverseLayer(l_fc6_deconv, l_pool5)

l_deconv5_1 = batch_norm(Deconv2DLayer(l_unpool5, num_filters=512, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_deconv5_2 = batch_norm(Deconv2DLayer(l_deconv5_1, num_filters=512, filter_size=(3,3),stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_deconv5_3 = batch_norm(Deconv2DLayer(l_deconv5_2, num_filters=512, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01)))

l_unpool4 = InverseLayer(l_deconv5_3, l_pool4)

l_deconv4_1 = batch_norm(Deconv2DLayer(l_unpool4, num_filters=512, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_deconv4_2 = batch_norm(Deconv2DLayer(l_deconv4_1, num_filters=512, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_deconv4_3 = batch_norm(Deconv2DLayer(l_deconv4_2, num_filters=256, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01)))

l_unpool3 = InverseLayer(l_deconv4_3, l_pool3)

l_deconv3_1 = batch_norm(Deconv2DLayer(l_unpool3, num_filters=256, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_deconv3_2 = batch_norm(Deconv2DLayer(l_deconv3_1, num_filters=256, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_deconv3_3 = batch_norm(Deconv2DLayer(l_deconv3_2, num_filters=128, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01)))

l_unpool2 = InverseLayer(l_deconv3_3, l_pool2)

l_deconv2_1 = batch_norm(Deconv2DLayer(l_unpool2, num_filters=128, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_deconv2_2 = batch_norm(Deconv2DLayer(l_deconv2_1, num_filters=64, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01)))

l_unpool1 = InverseLayer(l_deconv2_2, l_pool1)

l_deconv1_1 = batch_norm(Deconv2DLayer(l_unpool1, num_filters=64, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_deconv1_2 = batch_norm(Deconv2DLayer(l_deconv1_1, num_filters=64, filter_size=(3,3), stride=1, crop='same', nonlinearity=rectify, W=lasagne.init.Normal(0.01)))
l_output = batch_norm(Conv2DLayer(l_deconv1_2, num_filters=21, filter_size=(1, 1), pad=0,stride=1)) ###

# net_l_deconv2_2 = lasagne.layers.get_output(l_output);
# l_deconv2_2_func = theano.function([l_in.input_var], [net_l_deconv2_2]);
# l_deconv2_2_func_val = l_deconv2_2_func(X_train);
# print(l_deconv2_2_func_val[0].shape);
print('start training 1')
true_output = T.ftensor4('true_output') ###

loss = squared_error(lasagne.layers.get_output(l_output), true_output).mean()
loss_train = squared_error(lasagne.layers.get_output(l_output, deterministic=False), true_output).mean()
loss_eval = squared_error(lasagne.layers.get_output(l_output, deterministic=True), true_output).mean()

all_params = lasagne.layers.get_all_params(l_output, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss_train, all_params, learning_rate=0.001, momentum=0.985)

train = theano.function([l_in.input_var, true_output], loss_train, updates=updates)
get_output = theano.function([l_in.input_var], lasagne.layers.get_output(l_output, deterministic=True))

BATCH_SIZE = 10
N_EPOCHS = 5000
batch_idx = 0
epoch = 0
print("start training")
while epoch < N_EPOCHS:
    #print(net_output)
    loss = train(dataset['train']['X'][batch_idx:batch_idx + BATCH_SIZE], dataset['train']['y'][batch_idx:batch_idx + BATCH_SIZE])
    print(loss)
    batch_idx += BATCH_SIZE
    if batch_idx >= dataset['train']['X'].shape[0]:
        batch_idx = 0
        epoch += 1
        val_output = get_output(dataset['valid']['X'])
        val_predictions = np.argmax(val_output, axis=1)
        val_predictions = np.expand_dims(val_predictions, axis=1)
        print('the shape of prediction is:{}' , val_predictions.shape)
        s = (val_predictions.shape[0],val_predictions.shape[1],val_predictions.shape[2],val_predictions.shape[3])
        val_predictions_0 = np.zeros(s)
        accuracy = np.mean(val_predictions == dataset['valid']['y'])
        accuracy_0 = np.mean(val_predictions_0 == dataset['valid']['y'])

        print('prediction is :{}', val_predictions[0,0,100:110,100:110])
        print('the total sum of prediction is', np.sum(val_predictions))
        print('dataset for validation y is:{}'.format(dataset['valid']['y'][0,0,100:110,100:110]))
        print("Epoch {} validation accuracy: {}".format(epoch, accuracy))
        print("Epoch {} validation accuracy_0: {}" ,epoch, accuracy_0)
        print('this is loss', loss)
