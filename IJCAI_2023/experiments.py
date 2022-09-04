#! /usr/bin/env python
"""
LIMEtree -- Experiments
=======================

This module implements experiments executor for LIMEtree.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import logging
import pickle
import sys

import scripts.helpers as helpers
import scripts.limetree as limetree

USE_GPU = False
ENABLE_LOGGING = False
SAMPLE_SIZE = 150
PICKLE_FILE = 'limetree_{:d}.pickle'

if USE_GPU:
    import scripts.image_classifier as imgclf
else:
    import multiprocessing as mp
    from multiprocessing import Pool

if ENABLE_LOGGING:
    # logging.basicConfig(level=logging.DEBUG)
    limetree.logger.setLevel(logging.DEBUG)


def process_images_cpu(image_paths):
    """[CPU] Evaluates effectiveness of LIMEtree for a collection of images."""
    assert not USE_GPU
    collector = dict()
    processes = int(mp.cpu_count()/2) - 1
    with Pool(processes=processes) as pool:
        for imp in pool.imap_unordered(limetree.explain_image_exp, image_paths):
            img_path, top_pred, similarities, lime, limet = imp
            collector[img_path] = (top_pred, similarities, lime, limet)

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
            kernel_width=0.25)                          # Similarity
        collector[img_path] = (top_pred, similarities, lime, limet)
        limetree.logger.debug(f'Progress: {i/i_len:3.0d}')

    with open(PICKLE_FILE.format(SAMPLE_SIZE), 'wb') as f:
        pickle.dump(collector, f, protocol=pickle.HIGHEST_PROTOCOL)


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
    """
    if USE_GPU:
        process_images = process_images_gpu
    else:
        process_images = process_images_cpu

    if len(sys.argv) == 3 and sys.argv[1] == 'img_rand':
        print('Running random image experiments.')
        image_paths =  helpers.select_images(
            sys.argv[2], sample_size=SAMPLE_SIZE, random_seed=42)
        print(f'Trying {len(image_paths)} images.')
        process_images(image_paths)
    elif len(sys.argv) == 3 and sys.argv[1] == 'img':
        print('Running full image experiments.')
        image_paths =  helpers.select_images(
            sys.argv[2], sample_size=None, random_seed=None)
        SAMPLE_SIZE = len(image_paths)
        print(f'Trying {SAMPLE_SIZE} images.')
        process_images(image_paths)
    else:
        print('Nothing to do.')
