#! /usr/bin/env python
"""
Interpretable Representation -- Experiments
===========================================

This module implements experiments executor for studying interpretable
representations in XAI.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pickle
import sys

import multiprocessing as mp
from multiprocessing import Pool

import helpers

SAMPLE_SIZE = 150
PICKLE_FILE = 'intrep_{:d}.pickle'


def process_images(image_paths, segments_no):
    """Ghaters occlusion colour influence for a collection of image."""
    if segments_no == 5:
        func = helpers.exp_process_image_5
    elif segments_no == 10:
        func = helpers.exp_process_image_10
    elif segments_no == 15:
        func = helpers.exp_process_image_15
    elif segments_no == 20:
        func = helpers.exp_process_image_20
    elif segments_no == 30:
        func = helpers.exp_process_image_30
    elif segments_no == 40:
        func = helpers.exp_process_image_40
    else:
        raise ValueError('Unknown number of segments.')

    exp = dict()
    processes = int(mp.cpu_count()/2) - 1
    with Pool(processes=processes) as pool:
        for imp in pool.imap_unordered(func, image_paths):
            image_path, image_prediction, image_occlusion = imp
            exp[image_path] = (image_prediction, image_occlusion)

    with open(PICKLE_FILE.format(segments_no), 'wb') as f:
        pickle.dump(exp, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == 'img_rand':
        print('Running random image experiments.')
        image_paths =  helpers.select_images(
            sys.argv[2], sample_size=SAMPLE_SIZE, random_seed=42)
        process_images(image_paths, int(sys.argv[3]))
    elif len(sys.argv) == 4 and sys.argv[1] == 'img':
        print('Running full image experiments.')
        image_paths =  helpers.select_images(
            sys.argv[2], sample_size=None, random_seed=None)
        process_images(image_paths, int(sys.argv[3]))
    elif len(sys.argv) == 2 and sys.argv[1] == 'tab':
        print('Running tabular experiments.')
    else:
        print('Nothing to do.')
