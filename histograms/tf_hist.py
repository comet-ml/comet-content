import comet_ml

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np

# initialize the number of epochs to train for, batch size, and
# initial learning rate
EPOCHS = 5
BS = 64
INIT_LR = 1e-3

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)


def get_comet_experiment():
    args = {"project_name": "histograms"}
    return comet_ml.Experiment(**args)


def build_model_graph():
    model = Sequential()
    model.add(Dense(128, activation="sigmoid", input_shape=(784,), name="dense1"))
    model.add(Dense(64, activation="sigmoid", name="dense2"))
    model.add(Dense(10, activation="softmax", name="output"))

    return model


def get_dataset():
    num_classes = 10

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


def log_histogram(experiment, gradmap, step, prefix=None):
    for k, v in gradmap.items():
        experiment.log_histogram_3d(v, name="%s/%s" % (prefix, k), step=step)


def log_weights(experiment, model, step):
    for tv in model.trainable_variables:
        experiment.log_histogram_3d(tv.numpy(), name="%s" % tv.name, step=step)

    return


def get_activations(activation_map, X, model):
    input = model.input
    outputs = [layer.output for layer in model.layers]
    functors = [K.function([input], [out]) for out in outputs]

    for func, layer in zip(functors, model.layers):
        layer_output = func([X])
        activation_map.setdefault(layer.name, 0)
        activation_map[layer.name] += np.array(layer_output)

    return activation_map


def get_gradients(gradmap, grads, model):
    for grad, param in zip(grads, model.trainable_variables):
        gradmap.setdefault(param.name, 0)
        gradmap[param.name] += grad

    return gradmap


def step(model, X, y, gradmap={}, activations={}):
    with tf.GradientTape() as tape:
        pred = model(X)
        loss = categorical_crossentropy(y, pred)

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    gradmap = get_gradients(gradmap, grads, model)
    activations = get_activations(activations, X, model)

    return loss.numpy().mean(), gradmap, activations


def train(model, X, y, epoch, steps_per_epoch, experiment):
    gradmap = {}
    activations = {}
    total_loss = 0
    with experiment.train():
        # show the current epoch number
        print("[INFO] starting epoch {}/{}...".format(epoch, EPOCHS), end="")
        for i in range(0, steps_per_epoch):
            start = i * BS
            end = start + BS
            curr_step = ((i + 1) * BS) * epoch

            loss, gradmap, activations = step(
                model, X[start:end], y[start:end], gradmap, activations
            )
            experiment.log_metric("batch_loss", loss, step=curr_step)

            total_loss += loss

        experiment.log_metric(
            "loss", total_loss / steps_per_epoch, step=epoch * steps_per_epoch
        )

    # scale gradients
    for k, v in gradmap.items():
        gradmap[k] = v / steps_per_epoch

    # scale activations
    for k, v in activations.items():
        activations[k] = v / steps_per_epoch

    log_weights(experiment, model, epoch * steps_per_epoch)
    log_histogram(experiment, gradmap, epoch * steps_per_epoch, prefix="gradient")
    log_histogram(experiment, activations, epoch * steps_per_epoch, prefix="activation")


def main():
    x_train, y_train, x_test, y_test = get_dataset()
    experiment = get_comet_experiment()
    model = build_model_graph()

    steps_per_epoch = int(x_train.shape[0] / BS)
    for epoch in range(1, EPOCHS + 1):
        train(model, x_train, y_train, epoch, steps_per_epoch, experiment)

    experiment.end()


if __name__ == "__main__":
    main()
