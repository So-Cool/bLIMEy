#! /usr/bin/env python
"""
LIMEtree -- Experiments
=======================

This module implements experiments executor for LIMEtree.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import functools
import logging
import os.path
import pickle
import sys

import scripts.helpers as helpers
import scripts.limetree as limetree

from torchvision.datasets import CIFAR10, CIFAR100

USE_GPU = False
ENABLE_LOGGING = False
SAMPLE_SIZE = 150
USE_RANDOM_TRAINING = False
if USE_RANDOM_TRAINING:
    PICKLE_FILE = 'limetree_{:d}_random.pickle'
    PICKLE_FILE_TEMP = 'limetree_temp_{:d}_random.pickle'
else:
    PICKLE_FILE = 'limetree_{:d}.pickle'
    PICKLE_FILE_TEMP = 'limetree_temp_{:d}.pickle'

if USE_GPU:
    import scripts.image_classifier as imgclf
else:
    import multiprocessing as mp
    from multiprocessing import Pool

if ENABLE_LOGGING:
    # logging.basicConfig(level=logging.DEBUG)
    limetree.logger.setLevel(logging.DEBUG)


def process_images_cpu(image_paths, cifar=None):
    """[CPU] Evaluates effectiveness of LIMEtree for a collection of images."""
    assert not USE_GPU
    processes = int(mp.cpu_count()/2) - 1
    with Pool(processes=processes) as pool:
        if cifar is None:
            collector = dict()
            _explain_image_exp = functools.partial(
                limetree.explain_image_exp, train_on_random=USE_RANDOM_TRAINING)
            for imp in pool.imap_unordered(_explain_image_exp, image_paths):
                img_path, top_pred, similarities, lime, limet = imp
                collector[img_path] = (top_pred, similarities, lime, limet)
        elif cifar == 10:
            collector = []
            _explain_image_exp = functools.partial(
                limetree.explain_cifar_exp,
                use_cifar100=False,
                train_on_random=USE_RANDOM_TRAINING)
            for imp in pool.imap_unordered(_explain_image_exp, image_paths):
                img_path, top_pred, similarities, lime, limet = imp
                collector.append((top_pred, similarities, lime, limet))
        elif cifar == 100:
            collector = []
            _explain_image_exp = functools.partial(
                limetree.explain_cifar_exp,
                use_cifar100=True,
                train_on_random=USE_RANDOM_TRAINING)
            for imp in pool.imap_unordered(_explain_image_exp, image_paths):
                img_path, top_pred, similarities, lime, limet = imp
                collector.append((top_pred, similarities, lime, limet))
        else:
            assert False

    with open(PICKLE_FILE.format(SAMPLE_SIZE), 'wb') as f:
        pickle.dump(collector, f, protocol=pickle.HIGHEST_PROTOCOL)


def process_images_gpu(image_paths):
    """[GPU] Evaluates effectiveness of LIMEtree for a collection of images."""
    assert USE_GPU
    clf = imgclf.ImageClassifier(use_gpu=USE_GPU)
    collector = dict()
    i_len = len(image_paths)
    for i, img in enumerate(image_paths):
        img_path, top_pred, similarities, lime, limet = limetree.explain_image(
            img, clf, random_seed=42, n_top_classes=3,
            batch_size=100,                             # Processing
            segmenter_type='slic',                      # Segmenter Type
            n_segments=13,                              # Slic Segmenter
            occlusion_colour='black',                   # Occluder
            generate_complete_sample=True,              # Sampler
            kernel_width=0.25,                          # Similarity
            train_on_random=USE_RANDOM_TRAINING)        # Training on random occlusion
        collector[img_path] = (top_pred, similarities, lime, limet)
        limetree.logger.debug(f'Progress: {100*i/i_len:3.0f} [{i} / {i_len}]')

        if not i%50:
            _temp_save_file = PICKLE_FILE_TEMP.format(SAMPLE_SIZE)
            limetree.logger.debug(f'Saving partial results to {_temp_save_file}')
            with open(_temp_save_file, 'wb') as f:
                pickle.dump(collector, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(PICKLE_FILE.format(SAMPLE_SIZE), 'wb') as f:
        pickle.dump(collector, f, protocol=pickle.HIGHEST_PROTOCOL)


def process_cifar_gpu(cifar_data, use_cifar100=False):
    """[GPU] Evaluates effectiveness of LIMEtree for CIFAR10 & CIFAR100."""
    assert USE_GPU

    if use_cifar100:
        assert isinstance(cifar_data, CIFAR100) or sys.argv[2] == 'img_rand'
        clf = imgclf.Cifar100Classifier(use_gpu=USE_GPU)
    else:
        assert isinstance(cifar_data, CIFAR10) or sys.argv[2] == 'img_rand'
        clf = imgclf.Cifar10Classifier(use_gpu=USE_GPU)

    collector = dict()
    i_len = len(cifar_data)
    for i, (img, _target) in enumerate(cifar_data):
        img_path, top_pred, similarities, lime, limet = limetree.explain_image(
            img, clf, random_seed=42, n_top_classes=3,
            batch_size=100,                             # Processing
            segmenter_type='slic',                      # Segmenter Type
            n_segments=13,                              # Slic Segmenter
            occlusion_colour='black',                   # Occluder
            generate_complete_sample=True,              # Sampler
            kernel_width=0.25,                          # Similarity
            train_on_random=USE_RANDOM_TRAINING)        # Training on random occlusion
        img_path = i
        collector[img_path] = (top_pred, similarities, lime, limet)
        limetree.logger.debug(f'Progress: {100*i/i_len:3.0f} [{i} / {i_len}]')

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

    print(f'Number of processed images: {len(collector.keys())}')
    top_classes, lime_scores, limet_scores = limetree.process_loss(collector)
    lime_scores_summary = limetree.summarise_loss_lime(lime_scores, top_classes)
    limet_scores_summary = limetree.summarise_loss_limet(limet_scores, top_classes)

    pickle_file_dir = os.path.dirname(pickle_file)
    pickle_file_base = f'processed_{os.path.basename(pickle_file)}'
    pickle_file_ = os.path.join(pickle_file_dir, pickle_file_base)

    with open(pickle_file_, 'wb') as f:
        pickle.dump((lime_scores_summary, limet_scores_summary),
                    f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    """
    Runs various scripts.

    The first argument specifies the data set: cifar10, cifar100 or imagenet.

    The second argument specifies one of the following actions,
    with the third argument providing a path to the data folder or
    results pickle file:

    img_rand
        Runs image experiments with a random sample of images.
        Execute with:
        `python experiments.py imagenet img_rand /path/to/a/folder/with/images`
    img
        Runs image experiments with all images.
        Execute with:
        `python experiments.py imagenet img /path/to/a/folder/with/images`
    proc
        Processes the data for plotting.
        Execute with:
        `python experiments.py imagenet proc /path/to/a/pickle/file.pickle`
    models
        Download CIFAR models (before running any experiments)
        to avoid processing errors.
        Execute with:
        `python experiments.py cifar10 models none`
    """
    if len(sys.argv) != 4:
        print('The script requires 3 arguments:')
        print('  1: Data set (cifar10, cifar100 or imagenet).')
        print('  2: Function (img_rand, img, proc or models).')
        print('  3: Path to data folder (/path/to/a/folder/with/images) or '
              '     results pickle file (/path/to/a/pickle/file.pickle).')
        assert False
    if sys.argv[1].lower() not in ('cifar10', 'cifar100', 'imagenet'):
        print('The first argument must specify one of the following data sets: '
              'cifar10, cifar100 or imagenet.')
        assert False
    if sys.argv[2].lower() not in ('img_rand', 'img', 'proc', 'models'):
        print('The second argument must specify one of the following functions: '
              'img_rand, img, proc or models.')
        assert False

    if sys.argv[1].lower() == 'imagenet' and sys.argv[2].lower() in ('img_rand', 'img'):
        data = None
        if USE_GPU:
            process_images = process_images_gpu
        else:
            process_images = process_images_cpu
    elif sys.argv[1].lower() == 'cifar10' and sys.argv[2].lower() in ('img_rand', 'img'):
        data = CIFAR10(sys.argv[3], train=False)
        if USE_GPU:
            process_images = lambda imp: process_cifar_gpu(imp, use_cifar100=False)
        else:
            process_images = lambda imp: process_images_cpu(imp, cifar=10)
    elif sys.argv[1].lower() == 'cifar100' and sys.argv[2].lower() in ('img_rand', 'img'):
        data = CIFAR100(sys.argv[3], train=False)
        if USE_GPU:
            process_images = lambda imp: process_cifar_gpu(imp, use_cifar100=True)
        else:
            process_images = lambda imp: process_images_cpu(imp, cifar=100)

    if len(sys.argv) == 4 and sys.argv[2] == 'img_rand':
        print('Running random image experiments.')
        if data is None:
            image_paths = helpers.select_images(
                sys.argv[3], sample_size=SAMPLE_SIZE, random_seed=42)
        else:
            image_paths = helpers.select_cifar(
                data, sample_size=SAMPLE_SIZE, random_seed=42)
        print(f'Trying {len(image_paths)} images.')
        process_images(image_paths)
    elif len(sys.argv) == 4 and sys.argv[2] == 'img':
        print('Running full image experiments.')
        if data is None:
            image_paths = helpers.select_images(
                sys.argv[3], sample_size=None, random_seed=None)
            SAMPLE_SIZE = len(image_paths)
        else:
            image_paths = data
            SAMPLE_SIZE = len(data)
        print(f'Trying {SAMPLE_SIZE} images.')
        process_images(image_paths)
    elif len(sys.argv) == 4 and sys.argv[2] == 'proc':
        print('Processing experiment data for plotting.')
        process_data(sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[2] == 'models':
        print('Download CIFAR models.')
        import scripts.image_classifier as imgclf
        _clf = imgclf.Cifar10Classifier(use_gpu=False)
        del _clf
        _clf = imgclf.Cifar100Classifier(use_gpu=False)
        del _clf
    else:
        print('Nothing to do.')
