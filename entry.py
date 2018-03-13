from __future__ import print_function

import logging
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn
import numpy as np
import json
import time


logging.basicConfig(level=logging.DEBUG)

# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #


def train(channel_input_dirs, hyperparameters, num_gpus, hosts, **kwargs):
    # SageMaker passes num_cpus, num_gpus and other args we can use to tailor training to
    # the current container environment, but here we just use simple cpu context.
    ctx = mx.cpu()

    # retrieve the hyperparameters we set in notebook (with some defaults)
    batch_size = hyperparameters.get('batch_size', 100)
    epochs = hyperparameters.get('epochs', 10)
    learning_rate = hyperparameters.get('learning_rate', 0.1)
    momentum = hyperparameters.get('momentum', 0.9)
    log_interval = hyperparameters.get('log_interval', 100)

    # load training and validation data
    # we use the gluon.data.vision.MNIST class because of its built in mnist pre-processing logic,
    # but point it at the location where SageMaker placed the data files, so it doesn't download them again.
    training_dir = channel_input_dirs['training']
    train_data = get_train_data(training_dir + '/train', batch_size)
    val_data = get_val_data(training_dir + '/test', batch_size)

    # define the network
    net = autoencoder()

    # Collect all parameters from net and its children, then initialize them.
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    # Trainer is for updating parameters with gradient.
    
    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync' if num_gpus > 0 else 'dist_sync'

    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': learning_rate, 'momentum': momentum}, kvstore=kvstore)
    loss = mx.gluon.loss.L2Loss()

    for epoch in range(epochs):
        # reset data iterator and metric at begining of epoch.
        for i, (img, _) in enumerate(train_data):
            # Copy data to ctx if necessary
            img = img.reshape((img.shape[0], -1)).as_in_context(ctx)
            # Start recording computation graph with record() section.
            # Recorded graphs can then be differentiated with backward.
            with autograd.record():
                output = net(img)
                L = loss(output, img)
            L.backward()
            # take a gradient step with batch_size equal to data.shape[0]
            trainer.step(img.shape[0])
            # update metric at last.


    return net


def save(model, model_dir):
    # save the model
    # y = model(mx.sym.var('data'))
    # y.save('%s/model.json' % model_dir)
    model.save_params('%s/model.params' % model_dir)


def define_network():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(128, activation='relu'))
        net.add(nn.Dense(64, activation='relu'))
        net.add(nn.Dense(10))
    return net

class autoencoder(mx.gluon.Block):
    def __init__(self):
        super(autoencoder, self).__init__()
        with self.name_scope():
            self.encoder = gluon.nn.Sequential('encoder_')
            with self.encoder.name_scope():
                self.encoder.add(gluon.nn.Dense(128, activation='relu'))
                self.encoder.add(gluon.nn.Dense(64, activation='relu'))
                self.encoder.add(gluon.nn.Dense(12, activation='relu'))
                self.encoder.add(gluon.nn.Dense(3))

            self.decoder = gluon.nn.Sequential('decoder_')
            with self.decoder.name_scope():
                self.decoder.add(gluon.nn.Dense(12, activation='relu'))
                self.decoder.add(gluon.nn.Dense(64, activation='relu'))
                self.decoder.add(gluon.nn.Dense(128, activation='relu'))
                self.decoder.add(gluon.nn.Dense(28 * 28, activation='tanh'))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def input_transformer(data, label):
    data = data.reshape((-1,)).astype(np.float32) / 255
    return data, label


def get_train_data(data_dir, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.MNIST(data_dir, train=True, transform=input_transformer),
        batch_size=batch_size, shuffle=True, last_batch='discard')


def get_val_data(data_dir, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.MNIST(data_dir, train=False, transform=input_transformer),
        batch_size=batch_size, shuffle=False)


def test(ctx, net, val_data):
    metric = mx.metric.Accuracy()
    for data, label in val_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        metric.update([label], [output])
    return metric.get()


# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.

    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    # symbol = mx.sym.load('%s/model.json' % model_dir)
    # outputs = mx.symbol.softmax(data=symbol, name='softmax_label')
    # inputs = mx.sym.var('data')
    # param_dict = gluon.ParameterDict('model_')
    # net = gluon.SymbolBlock(outputs, inputs, param_dict)
    net = autoencoder()
    net.load_params('%s/model.params' % model_dir, ctx=mx.cpu())
    return net.encoder


def transform_fn(model, input_data, content_type, accept):
    """
    Transform a request using the Gluon model. Called once per request.

    :param net: The Gluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    # we can use content types to vary input/output handling, but
    # here we just assume json for both
    parsed = json.loads(input_data)
    nda = mx.nd.array(parsed)
    output = model(nda)
    
    response_body = json.dumps(output.asnumpy().tolist()[0])
    return response_body, accept

