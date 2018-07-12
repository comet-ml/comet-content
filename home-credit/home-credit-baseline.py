from comet_ml import Experiment, Optimizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

import os
import argparse
import pandas as pd
import lightgbm as lgb


API_KEY = os.environ.get('COMET_API_KEY')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data', required=True)
    parser.add_argument('--validation_data', required=True)
    parser.add_argument('--model', required=True)

    return parser.parse_args()


def run_lightgbm(train_df, validation_df):
    train_data = lgb.Dataset(data=train_df.drop(columns=['TARGET']),
                             label=train_df['TARGET'])
    validation_data = lgb.Dataset(data=validation_df.drop(columns=['TARGET']),
                                  label=validation_df['TARGET'])
    num_round = 10

    params = """
    num_leaves integer [31, 51] [31]
    num_trees integer [50, 100] [50]
    """
    optimizer = Optimizer(API_KEY)
    optimizer.set_params(params)

    while True:
        suggestion = optimizer.get_suggestion()
        experiment = Experiment(
            api_key=API_KEY,
            project_name='home-credit')
        experiment.set_name('lightgbm')

        _param = {
            'num_leaves': suggestion['num_leaves'],
            'num_trees': suggestion['num_trees'],
            'objective': 'binary',
            'metric': 'auc'
        }

        experiment.log_multiple_params(_param)
        experiment.log_dataset_hash(
            pd.concat([train_df, validation_df], axis=0))
        bst = lgb.train(_param, train_data, num_round,
                        valid_sets=[validation_data])
        y_pred = bst.predict(validation_df.drop(columns=['TARGET']))

        auc_score = roc_auc_score(validation_df['TARGET'], y_pred)
        experiment.log_metric(name='auc_score', value=auc_score)
        suggestion.report_score("auc_score", auc_score)


def run_logistic_regression(train_df, validation_df):
    params = """
    C real [0.00001, 0.0001] [0.0001]
    """
    optimizer = Optimizer(API_KEY)
    optimizer.set_params(params)

    while True:
        suggestion = optimizer.get_suggestion()
        experiment = Experiment(
            api_key=API_KEY,
            project_name='home-credit')
        experiment.set_name('logreg')
        experiment.log_dataset_hash(
            pd.concat([train_df, validation_df], axis=0))
        experiment.log_parameter(name='C', value=suggestion['C'])

        logreg = LogisticRegression(C=suggestion['C'])
        logreg.fit(train_df.drop(columns=['TARGET']), train_df["TARGET"])

        y_pred = logreg.predict(validation_df.drop(columns=['TARGET']))
        auc_score = roc_auc_score(validation_df['TARGET'], y_pred)
        experiment.log_metric(name='auc_score', value=auc_score)
        suggestion.report_score("auc_score", auc_score)


def run_random_forest(train_df, validation_df):
    params = """
    n_estimators integer [100, 500] [100]
    """
    optimizer = Optimizer(API_KEY)
    optimizer.set_params(params)

    while True:
        suggestion = optimizer.get_suggestion()
        experiment = Experiment(
            api_key=API_KEY,
            project_name='home-credit')
        experiment.log_dataset_hash(
            pd.concat([train_df, validation_df], axis=0))
        experiment.set_name('rf')
        experiment.log_parameter(
            name='n_estimators', value=suggestion['n_estimators'])

        rf = RandomForestClassifier(n_estimators=suggestion['n_estimators'])
        rf.fit(train_df.drop(columns=['TARGET']), train_df["TARGET"])

        y_pred = rf.predict(validation_df.drop(columns=['TARGET']))
        auc_score = roc_auc_score(validation_df['TARGET'], y_pred)
        experiment.log_metric(name='auc_score', value=auc_score)
        suggestion.report_score("auc_score", auc_score)


def main():
    args = get_args()

    train_df = pd.read_csv(args.training_data, sep=',')
    valid_df = pd.read_csv(args.validation_data, sep=',')

    model = {
        'lightgbm': run_lightgbm,
        'logreg': run_logistic_regression,
        'rf': run_random_forest
    }.get(args.model)
    model(train_df, valid_df)


if __name__ == '__main__':
    main()
