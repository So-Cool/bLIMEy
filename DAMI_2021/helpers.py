"""
Interpretable Representation -- Helper Functions
================================================

This module implements helper functions for studying interpretable
representations in XAI.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import os
import random

import numpy as np

from PIL import Image

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('text', usetex=True)
plt.style.use('seaborn')  # 'classic'

import fatf

import fatf.utils.data.segmentation as fatf_segmentation
import fatf.utils.data.occlusion as fatf_occlusion

import scripts.image_classifier as imgclf

SEG_MSG = ('The segmentation array should encode unique segments with a '
           'continuous sequence of integers starting at 1.')


def plot_bar_exp(
        exp, savepath=None, onesided=True, feature_no=5, x_lim=None,
        fontsize=24, label_fmt=None
        ):
    """Plots feature importance/influence explanation."""
    if label_fmt is None:
        label_fmt = '\\(f_{{{:d}}}\\)'

    plt.figure(figsize=(6, 6))

    exp_ = []
    for i, v in enumerate(exp):
        c = 'red' if v < 0 else 'green'
        exp_.append((i + 1, v, c))
    exp_ = sorted(exp_, key=lambda i: abs(i[1]), reverse=False)

    exp_f, exp_v, exp_c = [], [], []
    for f, v, c in exp_[-feature_no:]:
        exp_f.append(f)
        exp_v.append(v)
        exp_c.append(c)
    exp_v_abs = [abs(v) for v in exp_v]

    if len(exp_v) < 4:
        bar_height = 0.45
    else:
        bar_height = 0.9
    loc = list(range(len(exp_v)))
    loc_name = [label_fmt.format(i) for i in exp_f]
    if onesided:
        loc_v = exp_v_abs
    else:
        loc_v = exp_v
    plt.barh(loc_name, loc_v, color=exp_c, height=bar_height)  # left=-0.4

    left_, right_ = plt.xlim()
    ratio_ = abs(right_ - left_)

    text_offset_ = 0.04 * ratio_
    for l, c, v in zip(loc, exp_c, exp_v):
        if onesided:
            v_ = abs(v)
        else:
            v_ = v
        if v_ > 0:
            v_ += text_offset_  # 0.02 # 0.01
        else:
            v_ = text_offset_  # 0.02 # 0.01
        plt.text(
            v_,
            l,  # - 0.06 # - 0.03
            '\\(\mathbf{{{:.2f}}}\\)'.format(v),
            color=c,
            fontweight='bold',
            fontsize=fontsize)  # 14

    plt.tight_layout()
    plt.gca().yaxis.grid(False)

    if x_lim is None:
        xlim_offset_ = 0.3 * ratio_  # 0.175
        # 1.15 * # 1.08 * # 1.20 * # 1.05 *
        # 1.3 * # 1.9 *
        x_lim = (left_, right_ + xlim_offset_)
    plt.xlim(x_lim)

    if len(exp_v) < 4:
        plt.ylim((-1, len(exp_v)))

    plt.tick_params(axis='x', labelsize=fontsize)  # 16
    plt.tick_params(axis='y', labelsize=fontsize)  # 16

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def select_images(img_dir, sample_size=150, random_seed=42):
    """Selects a random sample of image paths."""
    img_paths = []
    for file in os.listdir(img_dir):
        if file.lower().endswith('.jpeg'):
            img_paths.append(os.path.join(img_dir, file))

    if sample_size is not None:
        if random_seed is not None:
            random.seed(a=random_seed)
        img_paths = random.sample(img_paths, sample_size)

    return img_paths


def random_occlusion_matrix(instances, segments, occlusion_size):
    """Random occludes a given number of segments in each row."""
    mx = np.ones((instances, segments), dtype=np.int8)
    indices = np.arange(segments, dtype=np.int32)

    for i in range(instances):
        idx = np.random.choice(indices, size=occlusion_size, replace=False)
        mx[i, idx] = 0
    assert np.all(mx.sum(axis=1) == (segments - occlusion_size))

    return mx


def process_image(
        image_path, clf,
        segments_no=40, sample_size=20, occlusion_colours=None,
        random_seed=42):
    """Ghaters occlusion colour influence for a single image."""
    if occlusion_colours is None:
        occlusion_colours = [
            'mean', 'black', 'white', 'red', 'green', 'blue', 'pink', 'random',
            'random-patch', 'randomise', 'randomise-patch']

    img = np.asarray(Image.open(image_path))

    img_pred = clf.predict_proba([img])[0]

    # Define segmentation search range
    segments_search_range = (
        segments_no - (5 if segments_no > 5 else 2),
        segments_no + int(2.5 * segments_no) + 1
    )

    # Get the right number of segments
    for i in range(*segments_search_range):
        try:
            segmenter = fatf_segmentation.Slic(img, n_segments=i)
        except ValueError as e:
            if str(e) == SEG_MSG:
                continue
            else:
                raise e
        if segmenter.segments_number == segments_no:
            break
    else:
        # Could not get the right number of segments -- skipping the image
        print(f'Skipping image: {image_path}')
        return image_path, img_pred, None

    # Generate occluders
    occluders = {
        c: fatf_occlusion.Occlusion(img, segmenter.segments, colour=c)
        for c in occlusion_colours
    }

    predictions = {}
    for c in occluders:
        predictions[c] = []

    for occlusion_size in range(segments_no + 1):  # Occluded segments
        if random_seed is not None:
            fatf.setup_random_seed(42)
        occ_mx = random_occlusion_matrix(
            sample_size, segmenter.segments_number, occlusion_size)
        for colour, occluder in occluders.items():  # Colours
            print(image_path, occlusion_size, colour)
            ocd_imgs = occluder.occlude_segments_vectorised(occ_mx)
            ocd_proba = clf.predict_proba(ocd_imgs)
            predictions[colour].append(ocd_proba)

    return image_path, img_pred, predictions


def exp_process_image_5(img):
    """Get multiprocess experiment."""
    return process_image(
        img, imgclf.ImageClassifier(), segments_no=5,
        sample_size=100, random_seed=42)
def exp_process_image_10(img):
    """Get multiprocess experiment."""
    return process_image(
        img, imgclf.ImageClassifier(), segments_no=10,
        sample_size=100, random_seed=42)
def exp_process_image_15(img):
    """Get multiprocess experiment."""
    return process_image(
        img, imgclf.ImageClassifier(), segments_no=15,
        sample_size=100, random_seed=42)
def exp_process_image_20(img):
    """Get multiprocess experiment."""
    return process_image(
        img, imgclf.ImageClassifier(), segments_no=20,
        sample_size=100, random_seed=42)
def exp_process_image_30(img):
    """Get multiprocess experiment."""
    return process_image(
        img, imgclf.ImageClassifier(), segments_no=30,
        sample_size=100, random_seed=42)
def exp_process_image_40(img):
    """Get multiprocess experiment."""
    return process_image(
        img, imgclf.ImageClassifier(), segments_no=40,
        sample_size=100, random_seed=42)
