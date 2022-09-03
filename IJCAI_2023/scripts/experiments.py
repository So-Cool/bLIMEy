#! /usr/bin/env python
"""
LIMEtree -- Experiments
=======================

This module implements experiments executor for LIMEtree.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pickle
import sys

import helpers

USE_GPU = False
SAMPLE_SIZE = 150
PICKLE_FILE = 'limetree_{:d}.pickle'

if USE_GPU:
    import scripts.image_classifier as imgclf
else:
    import multiprocessing as mp
    from multiprocessing import Pool


def process_images_cpu(image_paths, segments_no):
    """[CPU] Gathers occlusion colour influence for a collection of image."""
    func = TODO

    exp = dict()
    processes = int(mp.cpu_count()/2) - 1
    with Pool(processes=processes) as pool:
        for imp in pool.imap_unordered(func, image_paths):
            image_path, image_prediction, image_occlusion = imp
            exp[image_path] = (image_prediction, image_occlusion)

    with open(PICKLE_FILE.format(segments_no), 'wb') as f:
        pickle.dump(exp, f, protocol=pickle.HIGHEST_PROTOCOL)


def process_images_gpu(image_paths, segments_no):
    """[GPU] Gathers occlusion colour influence for a collection of image."""
    clf = imgclf.ImageClassifier(use_gpu=True)
    exp = dict()
    for img in image_paths:
        image_path, image_prediction, image_occlusion = helpers.process_image(
            img, clf, segments_no=segments_no,
            sample_size=100, random_seed=42)

        exp[image_path] = (image_prediction, image_occlusion)

    with open(PICKLE_FILE.format(segments_no), 'wb') as f:
        pickle.dump(exp, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    """
    https://docs.python.org/3/library/multiprocessing.html#using-a-pool-of-workers
    https://stackoverflow.com/questions/29009790/python-how-to-do-multiprocessing-inside-of-a-class
    https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function
    https://scipy-cookbook.readthedocs.io/items/ParallelProgramming.html

    Runs various scripts.

    img_rand
        Runs image experiments with a random sample of images.
        Execute with:
        `python experiments.py img_rand /path/to/a/folder/with/images segments_no`
    img
        Runs image experiments with all images.
        Execute with:
        `python experiments.py img /path/to/a/folder/with/images segments_no`
    """
    if USE_GPU:
        process_images = process_images_gpu
    else:
        process_images = process_images_cpu

    if len(sys.argv) == 4 and sys.argv[1] == 'img_rand':
        print('Running random image experiments.')
        image_paths =  helpers.select_images(
            sys.argv[2], sample_size=SAMPLE_SIZE, random_seed=42)
        print(f'Trying {len(image_paths)} images.')
        process_images(image_paths, int(sys.argv[3]))
    elif len(sys.argv) == 4 and sys.argv[1] == 'img':
        print('Running full image experiments.')
        image_paths =  helpers.select_images(
            sys.argv[2], sample_size=None, random_seed=None)
        print(f'Trying {len(image_paths)} images.')
        process_images(image_paths, int(sys.argv[3]))
    elif len(sys.argv) == 2 and sys.argv[1] == 'tab':
        print('Execute the experiments_tabular.ipynb Jupyter Notebook to run '
              'the tabular experiments.')
    else:
        print('Nothing to do.')
