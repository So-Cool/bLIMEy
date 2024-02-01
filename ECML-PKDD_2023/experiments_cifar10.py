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

from torchvision.datasets import CIFAR10

ENABLE_LOGGING = not False
SAMPLE_SIZE = 200
USE_RANDOM_TRAINING = False
if USE_RANDOM_TRAINING:
    PICKLE_FILE = 'limetree_{:d}_random.pickle'
else:
    PICKLE_FILE = 'limetree_{:d}.pickle'
PICKLE_FILE_TEMP = 'limetree_temp_{:d}.pickle'

import scripts.image_classifier_cifar10 as imgclf

if ENABLE_LOGGING:
    # logging.basicConfig(level=logging.DEBUG)
    limetree.logger.setLevel(logging.DEBUG)


def process_images_gpu(image_paths):
    """[GPU] Evaluates effectiveness of LIMEtree for a collection of images."""
    clf = imgclf.ImageClassifier(use_gpu=True)
    collector = dict()

    cifar10 = CIFAR10(image_paths, train=False)
    i_len = len(cifar10)

    for i, (img, target) in enumerate(cifar10):
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
        limetree.logger.debug(f'Progress: {100*i/i_len:3.0f} [{i}]')
        # TODO
        #if i > 10: break
        if not i%50:
            limetree.logger.debug(f'Dumping temp')
            with open(PICKLE_FILE_TEMP.format(SAMPLE_SIZE), 'wb') as f:
                pickle.dump(collector, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(PICKLE_FILE.format(SAMPLE_SIZE), 'wb') as f:
        pickle.dump(collector, f, protocol=pickle.HIGHEST_PROTOCOL)


def process_data(pickle_file):
    """Processes the data pickle file."""
    with open(pickle_file, 'rb') as f:
        collector = pickle.load(f)

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

    img_rand
        Runs image experiments with a random sample of images.
        Execute with:
        `python experiments.py img_rand /path/to/a/folder/with/images`
    img
        Runs image experiments with all images.
        Execute with:
        `python experiments.py img /path/to/a/folder/with/images`
    proc
        Processes the data for plotting.
        Execute with:
        `python experiments.py proc /path/to/a/pickle/file.pickle`
    """
    process_images = process_images_gpu

    if len(sys.argv) == 3 and sys.argv[1] == 'img_rand':
        print('Running random image experiments.')
        image_paths =  helpers.select_images(
            sys.argv[2], sample_size=SAMPLE_SIZE, random_seed=42)
        print(f'Trying {len(image_paths)} images.')
        process_images(image_paths)
    elif len(sys.argv) == 3 and sys.argv[1] == 'img':
        print('Running full image experiments.')
        process_images(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == 'proc':
        print('Processing experiment data for plotting.')
        process_data(sys.argv[2])
    else:
        print('Nothing to do.')
