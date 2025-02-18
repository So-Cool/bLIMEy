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
MEASURE_TIME = False
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
            kernel_width=kernel_width,
            measure_time=MEASURE_TIME)
        for imp in pool.imap_unordered(_explain_tabular, enumerate(zip(X_test, Y_test))):
            if MEASURE_TIME:
                instance_id, top_pred, similarities, lime, limet, lime_time, limet_time = imp
                collector[instance_id] = (top_pred, similarities, lime, limet, lime_time, limet_time)
            else:
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
        imp = limetree.explain_tabular(
                x, i, clf, X, Y,
                random_seed=42, n_top_classes=3,
                samples_number=10000, batch_size=BATCH_SIZE,  # Processing
                kernel_width=kernel_width,                    # Similarity
                measure_time=MEASURE_TIME)
        if MEASURE_TIME:
            instance_id, top_pred, similarities, lime, limet, lime_time, limet_time = imp
            collector[instance_id] = (top_pred, similarities, lime, limet, lime_time, limet_time)
        else:
            instance_id, top_pred, similarities, lime, limet = imp
            collector[instance_id] = (top_pred, similarities, lime, limet)
        assert instance_id == i
        limetree.logger.debug(f'Progress: {100*(i+1)/i_len:3.0f}% [{i+1} / {i_len}]')

        if not i%50:
            _temp_save_file = PICKLE_FILE_TEMP.format(SAMPLE_SIZE)
            limetree.logger.debug(f'Saving partial results to {_temp_save_file}')
            with open(_temp_save_file, 'wb') as f:
                pickle.dump(collector, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(PICKLE_FILE.format(SAMPLE_SIZE), 'wb') as f:
        pickle.dump(collector, f, protocol=pickle.HIGHEST_PROTOCOL)


def process_data(pickle_file):
    """Processes the data pickle file."""
    with open(pickle_file, 'rb') as f:
        collector = pickle.load(f)

    if isinstance(collector, list):
        collector = {i:j for i, j in enumerate(collector)}

    print(f'Number of processed instances: {len(collector.keys())}')
    top_classes, lime_scores, limet_scores = limetree.process_loss(
        collector, ignoreR=True, measure_time=MEASURE_TIME)
    lime_scores_summary = limetree.summarise_loss_lime(
        lime_scores, top_classes, ignoreR=True)
    limet_scores_summary = limetree.summarise_loss_limet(
        limet_scores, top_classes, ignoreR=True)

    # process execution time
    if MEASURE_TIME:
        times = limetree.compare_execution_time(
            collector, factor=1000)  # return milliseconds

    pickle_file_dir = os.path.dirname(pickle_file)
    pickle_file_base = f'processed_{os.path.basename(pickle_file)}'
    pickle_file_ = os.path.join(pickle_file_dir, pickle_file_base)

    with open(pickle_file_, 'wb') as f:
        if MEASURE_TIME:
            save = (lime_scores_summary, limet_scores_summary, times)
        else:
            save = (lime_scores_summary, limet_scores_summary)
        pickle.dump(save, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    """
    Runs various scripts.

    The first argument specifies the data set: wine or forest.

    The second argument specifies one of the following action:,

    exp_rand
        Runs tabular experiments with a random sample of instances.
        Execute with:
        `python experiments_tabular.py wine exp_rand
    exp
        Runs tabular experiments with all instances.
        Execute with:
        `python experiments_tabular.py wine exp
    models
        Train models.
        Execute with:
        `python experiments_tabular.py wine models`
    proc
        Processes the data for plotting.
        Execute with:
        `python experiments_tabular.py /path/to/a/pickle/file.pickle proc`
    """
    if len(sys.argv) != 3:
        print('The script requires 2 arguments:')
        print('  1: Data set (wine or forest); for `proc` this should be pickle file path.')
        print('  2: Function (exp_rand, exp, models or proc).')
        assert False
    if sys.argv[1].lower() not in ('wine', 'forest') and not sys.argv[1].lower().endswith('.pickle'):
        print('The first argument must specify one of the following data sets: '
              'wine or forest, or a pickle file path.')
        assert False
    if sys.argv[2].lower() not in ('exp_rand', 'exp', 'models', 'proc'):
        print('The second argument must specify one of the following functions: '
              'exp_rand, exp, models or proc.')
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
        assert os.path.isfile(sys.argv[1]), f'{sys.argv[1]} file does not exist.'

    if sys.argv[2] == 'models':
        print('Train a model.')

        if os.path.exists(clf_name):
            print('The classifier has already been trained and is stored '
                  f'under the {clf_name} filename. '
                  'Please remove the file to train.')
            clf = joblib.load(clf_name)
        else:
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

            joblib.dump(clf, clf_name)

            print(f'Model saved to: {clf_name}')

        Y_test_predicted = clf.predict(X_test)
        _bacc = skl_metrics.balanced_accuracy_score(Y_test, Y_test_predicted)
        print(f"Model's performance (balanced accuracy): {_bacc:0.3f}")
    elif sys.argv[2] == 'proc':
        print('Processing experiment data for plotting.')
        process_data(sys.argv[1])
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
