#! /usr/bin/env python
"""
LIMEtree -- Tabular Experiments
===============================

This module implements tabular experiments executor for LIMEtree.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import functools
import joblib
import logging
import os.path
import pickle
import random
import sys

import sklearn.datasets as skl_data
import sklearn.model_selection as skl_tts
import sklearn.neural_network as skl_nn
import sklearn.linear_model as skl_lm
import sklearn.metrics as skl_metrics

import scripts.limetree as limetree

import multiprocessing as mp
from multiprocessing import Pool

PARALLELISE = False
ENABLE_LOGGING = False
SAMPLE_SIZE = 150
SAMPLE_STRATIFIED = True
BATCH_SIZE = 250
PICKLE_FILE = 'limetree_tabular_{:d}.pickle'
PICKLE_FILE_TEMP = 'limetree_tabular_temp_{:d}.pickle'


if ENABLE_LOGGING:
    # logging.basicConfig(level=logging.DEBUG)
    limetree.logger.setLevel(logging.DEBUG)


def process_parallel(X, Y, X_test, Y_test, clf, kernel_width=0.25):
    """Evaluates effectiveness of LIMEtree for a collection of instances."""
    processes = int(mp.cpu_count()/2) - 1
    with Pool(processes=processes) as pool:
        collector = dict()
        i_len = X_test.shape[0]
        _explain_tabular = functools.partial(
            limetree.explain_tabular_parallel,
            classifier=clf,
            data=X,
            data_labels=Y,
            batch_size=BATCH_SIZE,
            kernel_width=kernel_width)
        for imp in pool.imap_unordered(_explain_tabular, enumerate(zip(X_test, Y_test))):
            instance_id, top_pred, similarities, lime, limet = imp
            collector[instance_id] = (top_pred, similarities, lime, limet)
            i = len(collector.keys())
            limetree.logger.debug(f'Progress: {100*(i)/i_len:3.0f}% [{i} / {i_len}]')

    with open(PICKLE_FILE.format(SAMPLE_SIZE), 'wb') as f:
        pickle.dump(collector, f, protocol=pickle.HIGHEST_PROTOCOL)


def process_sequential(X, Y, X_test, Y_test, clf, kernel_width=0.25):
    """Evaluates effectiveness of LIMEtree for a collection of instances."""
    collector = dict()
    i_len = X_test.shape[0]

    for i, (x, y) in enumerate(zip(X_test, Y_test)):
        instance_id, top_pred, similarities, lime, limet = limetree.explain_tabular(
                x, i, clf, X, Y,
                random_seed=42, n_top_classes=3,
                samples_number=10000, batch_size=BATCH_SIZE,  # Processing
                kernel_width=kernel_width)                    # Similarity
        assert instance_id == i
        collector[instance_id] = (top_pred, similarities, lime, limet)
        limetree.logger.debug(f'Progress: {100*(i+1)/i_len:3.0f}% [{i+1} / {i_len}]')

        if not i%50:
            _temp_save_file = PICKLE_FILE_TEMP.format(SAMPLE_SIZE)
            limetree.logger.debug(f'Saving partial results to {_temp_save_file}')
            with open(_temp_save_file, 'wb') as f:
                pickle.dump(collector, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(PICKLE_FILE.format(SAMPLE_SIZE), 'wb') as f:
        pickle.dump(collector, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    """
    Runs various scripts.

    The first argument specifies the data set: wine or forest.

    The second argument specifies one of the following action:,

    exp_rand
        Runs tabular experiments with a random sample of instances.
        Execute with:
        `python experiments.py wine exp_rand
    exp
        Runs tabular experiments with all instances.
        Execute with:
        `python experiments.py wine exp
    models
        Train models.
        Execute with:
        `python experiments.py wine models`
    """
    if len(sys.argv) != 3:
        print('The script requires 2 arguments:')
        print('  1: Data set (wine or forest).')
        print('  2: Function (exp_rand, exp or models).')
        assert False
    if sys.argv[1].lower() not in ('wine', 'forest'):
        print('The first argument must specify one of the following data sets: '
              'wine or forest.')
        assert False
    if sys.argv[2].lower() not in ('exp_rand', 'exp', 'models'):
        print('The second argument must specify one of the following functions: '
              'exp_rand, exp or models.')
        assert False

    # data
    if sys.argv[1].lower() == 'wine':
        # *wine recognition* data set
        data_wine = skl_data.load_wine(return_X_y=True)[0]
        labels_wine = skl_data.load_wine(return_X_y=True)[1]

        wine_split = skl_tts.train_test_split(
            data_wine, labels_wine, train_size=0.8, random_state=42, stratify=labels_wine)
        data_wine_train, data_wine_test, labels_wine_train, labels_wine_test = wine_split

        clf_wine_name = 'clf_wine_lr.joblib'

        X, Y = data_wine, labels_wine
        X_train, Y_train = data_wine_train, labels_wine_train
        X_test, Y_test = data_wine_test, labels_wine_test

        clf_name = clf_wine_name

        kernel_width = 25
    elif sys.argv[1].lower() == 'forest':
        # *forest covertypes* data set
        data_forest = skl_data.fetch_covtype(return_X_y=True)[0]
        labels_forest = skl_data.fetch_covtype(return_X_y=True)[1]

        forest_split = skl_tts.train_test_split(
            data_forest, labels_forest, train_size=0.8, random_state=42, stratify=labels_forest)
        data_forest_train, data_forest_test, labels_forest_train, labels_forest_test = forest_split

        clf_forest_name = 'clf_forest_mlp.joblib'

        X, Y = data_forest, labels_forest
        X_train, Y_train = data_forest_train, labels_forest_train
        X_test, Y_test = data_forest_test, labels_forest_test

        clf_name = clf_forest_name

        kernel_width = 125
    else:
        assert False

    if sys.argv[2] == 'models':
        print('Train a model.')

        if os.path.exists(clf_name):
            print('The classifier has already been trained and is stored '
                  f'under the {clf_name} filename. '
                  'Please remove the file to train.')
            assert False

        if sys.argv[1].lower() == 'wine':
            clf_wine_lr = skl_lm.LogisticRegression(
                random_state=42,
                solver='lbfgs', multi_class='multinomial', max_iter=10000)
            clf = clf_wine_lr
        elif sys.argv[1].lower() == 'forest':
            clf_forest_mlp = skl_nn.MLPClassifier(
                random_state=42, verbose=True,
                hidden_layer_sizes=(100, 200, 100))
            clf = clf_forest_mlp
        else:
            assert False

        clf.fit(X_train, Y_train)

        Y_test_predicted = clf.predict(X_test)
        _bacc = skl_metrics.balanced_accuracy_score(Y_test, Y_test_predicted)
        print(f"Model's performance (balanced accuracy): {_bacc:0.3f}")

        joblib.dump(clf, clf_name)

        print(f'Model saved to: {clf_name}')
    else:
        if os.path.exists(clf_name):
            clf = joblib.load(clf_name)
        else:
            print('The classifier has NOT been trained '
                  f'(checked under the {clf_name} filename). '
                  'Please train the classifier first.')
            assert False
        
        Y_test_predicted = clf.predict(X_test)
        _bacc = skl_metrics.balanced_accuracy_score(Y_test, Y_test_predicted)
        limetree.logger.debug(f"Model's performance (balanced accuracy): {_bacc}")

        if sys.argv[2] == 'exp_rand':
            print('Running random tabular experiments.')

            if SAMPLE_SIZE > X_test.shape[0]:
                print(f'The data set has only {X_test.shape[0]} instances; '
                      f'the sample size ({SAMPLE_SIZE}) cannot be larger.')
                assert False

            if SAMPLE_STRATIFIED:
                limetree.logger.info('Using a stratified sample.')
                XY_sample = skl_tts.train_test_split(
                    X_test, Y_test, train_size=SAMPLE_SIZE, random_state=42, stratify=Y_test)
                X_sample, _, Y_sample, _ = XY_sample
            else:
                limetree.logger.info('Using a NON-stratified sample.')
                random.seed(a=42)
                idx = sorted(random.sample(range(X_test.shape[0]), SAMPLE_SIZE))
                X_sample = X_test[idx, :]
                Y_sample = Y_test[idx]
            assert SAMPLE_SIZE == X_sample.shape[0], f'{SAMPLE_SIZE} != {X_sample.shape[0]}'
            assert Y_sample.shape[0] == X_sample.shape[0], f'{Y_sample.shape[0]} != {X_sample.shape[0]}'
            print(f'Trying {SAMPLE_SIZE} instances.')

            if PARALLELISE:
                process_parallel(
                    X, Y, X_sample, Y_sample, clf, kernel_width=kernel_width)
            else:
                process_sequential(
                    X, Y, X_sample, Y_sample, clf, kernel_width=kernel_width)

        elif sys.argv[2] == 'exp':
            print('Running full tabular experiments.')

            SAMPLE_SIZE = X_test.shape[0]
            print(f'Trying {SAMPLE_SIZE} instances.')

            if PARALLELISE:
                process_parallel(
                    X, Y, X_test, Y_test, clf, kernel_width=kernel_width)
            else:
                process_sequential(
                    X, Y, X_test, Y_test, clf, kernel_width=kernel_width)

        else:
            print('Nothing to do.')
