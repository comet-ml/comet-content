from comet_ml import Experiment

import os
import argparse
import pandas as pd

from keras.models import Model
from keras.layers import Input, Dense
from keras.models import load_model

from keras import regularizers

API_KEY = os.environ.get("COMET_API_KEY")
experiment = Experiment(api_key=API_KEY, project_name='home-credit')


def load_data(path):
    return pd.read_csv(path)


def build_model(input_dim, encoding_dim):
    input = Input(shape=(input_dim,), name='input')
    encode_1 = Dense(100, activation='relu')(input)
    encode_2 = Dense(80, activation='relu')(encode_1)
    encode_3 = Dense(50, activation='relu')(encode_2)
    code = Dense(
        encoding_dim,
        activity_regularizer=regularizers.l1(10e-5),
        activation='relu',
        name='encoded')(encode_3)
    decode_1 = Dense(50, activation='relu')(code)
    decode_2 = Dense(80, activation='relu')(decode_1)
    decode_3 = Dense(100, activation='relu')(decode_2)
    decoded = Dense(input_dim,
                    activation='sigmoid')(decode_3)

    return Model(input, decoded)


def extract_features(data_path, model_path, outfile):
    training_data = load_data(data_path)
    X = training_data.drop(labels=['SK_ID_CURR'], axis=1)

    model = load_model(model_path)
    encoder = Model(model.input, model.get_layer(name='encoded').output)

    features = encoder.predict(X)
    pd.DataFrame(features).to_csv(outfile)


def train_autofeature_model(data_path,
                            embedding_dimension,
                            batch_size):
    training_data = load_data(data_path)
    X = training_data.drop(labels=['SK_ID_CURR'], axis=1)

    experiment = Experiment(
        api_key=API_KEY, project_name="home-credit")
    experiment.set_name(
        'home-credit-autofeature-selection')

    model = build_model(X.shape[1], int(embedding_dimension))
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_logarithmic_error')

    model.fit(X, X,
              epochs=5,
              batch_size=int(batch_size))

    experiment.log_multiple_params(
        {"embedding_dimension": embedding_dimension,
         "batch_size": batch_size})

    model.save(
        'home-credit-encoder-{}-{}.hdf5'.format(
            embedding_dimension, batch_size))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--autofeature')
    parser.add_argument('--model_path')
    parser.add_argument('--embedding_dimension')
    parser.add_argument('--batch_size')
    parser.add_argument('--outfile')

    return parser.parse_args()


def main():
    args = get_args()

    if args.autofeature:
        train_autofeature_model(
            args.data, args.embedding_dimension, args.batch_size)

    else:
        extract_features(args.data, args.model_path, args.outfile)


if __name__ == '__main__':
    main()
