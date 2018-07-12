from comet_ml import Experiment
from comet_ml import Optimizer

import os
import argparse
import pandas as pd

from keras.models import Model

from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import GlobalAveragePooling1D

from keras.initializers import RandomUniform
from keras import optimizers

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import hamming_loss

API_KEY = os.environ.get('COMET_API_KEY')

# Converts Freebase MIDS to unique integer tokens


def tokenize_dataframe(df, entity_index, relation_index):
    entities = df[['entity_1', 'entity_2']].applymap(
        lambda x: entity_index.get(x))
    relationship = df['relationship'].map(lambda x: relation_index.get(x))

    return pd.concat([entities, relationship], axis=1)

# Filter out a subset of relationships and entities from Freebase 15K


def filter_df(df, n_entities, n_relationships):
    return df[
        (df['relationship'] < n_relationships) &
        (df['entity_1'] < n_entities) &
        (df['entity_2'] < n_entities)
    ]

# Create multilabel array for target classes


def create_labels(df, classes):
    s_keys = df['entity_1'].unique()
    r_keys = df['entity_2'].unique()

    df['1_N_labels'] = 'NaN'
    df['1_N_labels'].astype(object)

    sr_dict = {k: {r: set() for r in r_keys} for k in s_keys}
    for index, row in df.iterrows():
        sr_dict[row['entity_1']][(row['entity_2'])].add(row['relationship'])

    for index, row in df.iterrows():
        df.at[index, '1_N_labels'] = sr_dict.get(
            row['entity_1']).get(row['entity_2'])

    mlb = MultiLabelBinarizer(classes=classes)
    labels = mlb.fit_transform(df['1_N_labels'])

    return labels

# Load in Freebase training, validation, and test dataset, tokenize them and
# create target labels


def load_data(n_entities, n_relationships):
    dataset_path = './data/FB15k/{}'
    filenames = ['train.txt', 'valid.txt', 'test.txt']
    dataframes = [
        pd.read_table(
            dataset_path.format(name),
            header=None
        ) for name in filenames
    ]

    for df in dataframes:
        df.columns = ['entity_1', 'entity_2', 'relationship']

    entity_2_id = pd.read_table(
        dataset_path.format('entity2id.txt'), header=None)
    relation_2_id = pd.read_table(
        dataset_path.format('relation2id.txt'), header=None)

    entity_index = {v: k for k, v in entity_2_id.to_dict().get(0).items()}
    relation_index = {v: k for k, v in relation_2_id.to_dict().get(0).items()}

    tokenized_dataframes = [
        tokenize_dataframe(df, entity_index, relation_index) for df in dataframes
    ]

    filtered_dataframes = [
        filter_df(df, n_entities, n_relationships) for df in tokenized_dataframes
    ]

    labels = [
        create_labels(df, classes=list(
            range(n_relationships))) for df in filtered_dataframes
    ]

    entity_columns = ['entity_1', 'entity_2']
    data = {
        "train": (filtered_dataframes[0][entity_columns], labels[0]),
        "validation": (filtered_dataframes[1][entity_columns], labels[1]),
        "test": (filtered_dataframes[2][entity_columns], labels[2])
    }

    return data


def build_model(n_entities,
                n_relationships,
                embedding_dimension):
    main_input = Input(shape=(2,), dtype='int32', name='main_input')

    embedding = Embedding(name="embedding",
                          input_dim=n_entities,  # Number of entities being considered
                          input_length=2,
                          output_dim=embedding_dimension,
                          embeddings_initializer=RandomUniform(
                              minval=-0.05, maxval=0.05, seed=None))
    x = embedding(main_input)
    x = GlobalAveragePooling1D()(x)
    x = Dense(n_relationships)(x)  # Number of relationships being considered
    output = Activation('sigmoid')(x)

    model = Model(inputs=main_input, outputs=output)
    model.summary()

    return model


def train_with_optimizer(suggestion, experiment, args):
    experiment.log_multiple_params(suggestion)

    n_entities = args.n_entities
    n_relationships = args.n_relationships

    model = build_model(
        n_entities=n_entities,
        n_relationships=n_relationships,
        embedding_dimension=suggestion['embedding_dimension']
    )

    optimizer = optimizers.Adam(lr=suggestion['learning_rate'], decay=0.0)
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=['accuracy']
    )

    data = load_data(n_entities=n_entities, n_relationships=n_relationships)
    model.fit(
        data["train"][0],
        data["train"][1],
        verbose=1,
        epochs=suggestion['epochs'],
        batch_size=suggestion['batch_size'],
        shuffle=True,
        validation_data=data["validation"]
    )
    evaluation = model.evaluate(
        data["test"][0], data["test"][1], verbose=0)

    predictions = model.predict(data["test"][0])
    auc_score = roc_auc_score(
        data["test"][1], predictions, average='samples')
    auc_score_micro = roc_auc_score(
        data["test"][1], predictions, average='micro')

    metrics = {
        "evaluation_loss": evaluation[0],
        "evaluation_accuracy": evaluation[1],
        "auc_score": auc_score,
        "auc_score_micro": auc_score_micro
    }
    experiment.log_multiple_metrics(metrics)

    return metrics["evaluation_accuracy"]


def run_optimizer(args):
    optimizer = Optimizer(API_KEY)
    params = """
    epochs integer [5, 10] [5]
    batch_size integer [64, 256] [64]
    learning_rate real [0.0001, 0.01] [0.0001]
    embedding_dimension integer [25, 200] [25]
    """

    optimizer.set_params(params)
    # get_suggestion will raise when no new suggestion is available
    while True:
        # Get a suggestion
        suggestion = optimizer.get_suggestion()

        # Create a new experiment associated with the Optimizer
        experiment = Experiment(
            api_key=API_KEY, project_name="fasttext")

        score = train_with_optimizer(suggestion, experiment, args)
        # Report the score back
        suggestion.report_score("accuracy", score)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True, type=str)
    parser.add_argument('--epochs', '-e', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--embedding_dimension', default=100, type=int)
    parser.add_argument('--n_entities', default=15000, type=int)
    parser.add_argument('--n_relationships', default=60, type=int)
    parser.add_argument('--output', '-o', default='./checkpoints')
    parser.add_argument('--logdir', '-l', default='./logs')
    parser.add_argument('--use_checkpoint')
    parser.add_argument('--use_optimizer')

    return parser.parse_args()


def train(args):
    experiment = Experiment(
        api_key=API_KEY, project_name="fasttext")

    params = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "embedding_dimension": args.embedding_dimension
    }
    experiment.log_multiple_params(params)

    model_path = os.path.join(str(args.output), "model")
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    filepath = model_path + \
        "/weights-{epoch:02d}-{val_loss:.3f}-" + \
        args.id + ".hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        period=5
    )

    logdir = args.logdir
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    tensorboard = TensorBoard(
        log_dir=logdir,
        histogram_freq=0,
        write_grads=True,
        write_graph=False,
        write_images=False
    )

    n_entities = args.n_entities
    n_relationships = args.n_relationships

    model = build_model(
        n_entities=n_entities,
        n_relationships=n_relationships,
        embedding_dimension=args.embedding_dimension
    )

    optimizer = optimizers.Adam(lr=args.learning_rate, decay=0.0)
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=['accuracy']
    )

    data = load_data(n_entities=n_entities, n_relationships=n_relationships)
    model.fit(
        data["train"][0],
        data["train"][1],
        verbose=1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        validation_data=data["validation"],
        callbacks=[checkpoint, tensorboard]
    )
    evaluation = model.evaluate(
        data["test"][0], data["test"][1], verbose=0)

    predictions = model.predict(data["test"][0])
    auc_score = roc_auc_score(
        data["test"][1], predictions, average='samples')
    auc_score_micro = roc_auc_score(
        data["test"][1], predictions, average='micro')

    metrics = {
        "evaluation_loss": evaluation[0],
        "evaluation_accuracy": evaluation[1],
        "auc_score": auc_score,
        "auc_score_micro": auc_score_micro
    }
    experiment.log_multiple_metrics(metrics)


if __name__ == '__main__':
    args = get_args()

    if args.use_optimizer:
        run_optimizer(args)

    else:
        train(args)
