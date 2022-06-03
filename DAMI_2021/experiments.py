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

import numpy as np

import helpers

USE_GPU = False
SAMPLE_SIZE = 150
PICKLE_FILE = 'intrep_{:d}.pickle'


if USE_GPU:
    # This module can be accessed here:
    # https://github.com/fat-forensics/resources/tree/master/surrogates_overview/scripts
    import scripts.image_classifier as imgclf

    # import torch.multiprocessing as mp
    # from torch.multiprocessing import Pool, set_start_method  # Process
    # try:
    #     set_start_method('spawn')
    # except RuntimeError:
    #     pass
else:
    import multiprocessing as mp
    from multiprocessing import Pool


def compress_img_exp(pickle_path):
    """
    Compresses experimental results of image data.

    See the `experiments_image.ipynb` Jupyter Notebook for more info.
    """
    if not pickle_path.endswith('.pickle'):
        raise RuntimeError('The file must have .pickle extension.')

    with open(pickle_path, 'rb') as f:
        exp_data = pickle.load(f)

    exp_data_compressed = {}
    for img_path in exp_data:
        if exp_data[img_path][1] is None:
            continue

        i = np.argmax(exp_data[img_path][0])

        by_colour = {}
        for colour in exp_data[img_path][1]:
            by_colour[colour] = []
            for j in exp_data[img_path][1][colour]:
                c1 = j[:, i]
                c0 = np.zeros(c1.shape)
                by_colour[colour].append(np.column_stack([c0, c1]))

        exp_data_compressed[img_path] = (
            np.array([0, exp_data[img_path][0][i]]),
            by_colour
        )

    pickle_path_ = pickle_path.replace('.pickle', '_compressed.pickle')
    with open(pickle_path_, 'wb') as f:
        pickle.dump(exp_data_compressed, f, protocol=pickle.HIGHEST_PROTOCOL)


def process_images_cpu(image_paths, segments_no):
    """[CPU] Gathers occlusion colour influence for a collection of image."""
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
    Runs various scripts.

    img_rand
        Runs image experiments with a random sample of images.
        Execute with:
        `python experiments.py img_rand /path/to/a/folder/with/images segments_no`
    img
        Runs image experiments with all images.
        Execute with:
        `python experiments.py img /path/to/a/folder/with/images segments_no`
    compress
        Compresses the size of the pickle file by only preserving the
        probability column that is argmax for non-occluded image.
        (See the `experiments_image.ipynb` Jupyter Notebook for more info.)
        Execute with:
        `python experiments.py compress experiment_results.pickle`
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
    elif len(sys.argv) == 3 and sys.argv[1] == 'compress':
        compress_img_exp(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == 'tab':
        print('Execute the experiments_tabular.ipynb Jupyter Notebook to run '
              'the tabular experiments.')
    else:
        print('Nothing to do.')
