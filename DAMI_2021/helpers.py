"""
Interpretable Representation -- Helper Functions
================================================

This module implements helper functions for studying interpretable
representations in XAI.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import math
import os
import pickle
import random
import warnings

import numpy as np
import sklearn.tree as sk_tree

from PIL import Image

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('text', usetex=True)
plt.style.use('seaborn')  # 'classic'

import fatf

import fatf.utils.data.discretisation as fatf_discretisation
import fatf.utils.data.segmentation as fatf_segmentation
import fatf.utils.data.occlusion as fatf_occlusion
import fatf.utils.distances as fatf_dist

# This module can be accessed here:
# https://github.com/fat-forensics/resources/tree/master/surrogates_overview/scripts
import scripts.image_classifier as imgclf

# warnings.simplefilter('ignore', RuntimeWarning)

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


def _process_img_data(data_path):
    """Compresses experimental results of image data."""
    with open(data_path, 'rb') as f:
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

    data_path_ = data_path.replace('.pickle', '_compressed.pickle')
    with open(data_path_, 'wb') as f:
        pickle.dump(exp_data_compressed, f, protocol=pickle.HIGHEST_PROTOCOL)


def millify(number, textualise=False):
    """Translates a number into a numerical descriptor."""
    if textualise:
        nom = ['',' Thousand',' Million',' Billion',' Trillion']
    else:
        nom = ['','e+3','e+6','e+9','e+12', 'e+15', 'e+18', 'e+21', 'e+24']

    number = float(number)
    digits = int(math.floor(0 if number == 0 else math.log10(abs(number)) / 3))
    nom_idx = max(
        0,
        min(len(nom) - 1, digits)
    )

    return '{:.0f}{}'.format(number / 10**(3 * nom_idx), nom[nom_idx])


def gini(x):
    """
    Computes Gini coefficient from a numerical array.

    https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy

    Warning: This is a concise implementation, but it is O(n**2) in time and
    memory, where n = len(x).
    *Do not* pass in huge samples!
    """
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            category=RuntimeWarning,
            message='invalid value encountered in double_scalars')
        # Relative mean absolute difference
        r_mad = mad / np.mean(x)
    # Gini coefficient
    gini_ = 0.5 * r_mad

    if np.isnan(gini_):
        return 0
    else:
        return gini_


def mse(x):
    """Computes Mean Squared Error from a numerical array."""
    mean = np.mean(x)
    diff = x - mean
    diff_sq = np.square(diff)
    mse_ = np.mean(diff_sq)
    return mse_


def get_weighted_homogeneity(discretisation, discretisation_labels, metric):
    """Computes weighted homogeneity of a data set."""
    _dims = len(discretisation.shape)
    assert _dims in (1, 2)

    discretisation_unique = np.unique(discretisation, axis=0)
    # Count unique discrete encodings
    discretisation_unique_no = discretisation_unique.shape[0]

    # Get weighted homogeneity of the discretisation
    weighted_homogeneity = []
    total_count = 0
    for row in discretisation_unique:
        # Get discretised instances of this (`row`) particular encoding
        if _dims == 1:
            _comparison = (discretisation == row)
        else:
            _comparison = (discretisation == row).all(axis=1)
        matching_index = np.where(_comparison)[0]
        # Measure homogeneity of labels assigned to this particular encoding
        homogeneity = metric(discretisation_labels[matching_index])
        # Weight the homogeneity by the size of this particular encoding
        homogeneity_wghtd = homogeneity * matching_index.shape[0]

        total_count += matching_index.shape[0]
        weighted_homogeneity.append(homogeneity_wghtd)
    # Compute weighted average
    assert total_count == discretisation.shape[0]
    weighted_homogeneity = np.sum(weighted_homogeneity) / total_count
    return weighted_homogeneity, discretisation_unique_no


def get_lime(dataset, labels, distance_factor=0.3, classification=True):
    """Evaluates IR of LIME: GLOBAL quarile discretisation."""
    # Choose evaluation metric
    metric = gini if classification else mse

    # Discretise data
    dataset_q = fatf_discretisation.QuartileDiscretiser(dataset)
    dataset_discrete = dataset_q.discretise(dataset)

    # Get GLOBAL weighted homogeneity of the discretisation
    global_wghtd_homogeneity, dataset_discrete_unique_no = get_weighted_homogeneity(
        dataset_discrete, labels, metric)
    print('Global weighted homogeneity of discretisation: '
          f'{global_wghtd_homogeneity}.')
    print(f'Unique discrete points count: {dataset_discrete_unique_no}.')

    # Get distances between all the points in the data set
    dist_matrix = fatf_dist.euclidean_array_distance(dataset, dataset)
    # Get the distance radius for identifying close instances
    max_dist = dist_matrix.max()
    radius = max_dist * distance_factor

    # Assess LOCAL homogeneity around each instance in the data set
    local_wghtd_homogeneities = []
    dataset_binary_unique_nos = []
    for x_idx, x in enumerate(dataset):
        # Get indices of nearby instances
        dist = fatf_dist.euclidean_point_distance(x, dataset)
        val_ind = dist <= radius

        # Get nearby instances
        x_discrete = dataset_discrete[x_idx, :]
        val_data_discrete = dataset_discrete[val_ind]
        val_labels = labels[val_ind]

        # Binarise instances according to the selected instance
        dataset_binary = (val_data_discrete == x_discrete).astype(np.int8)

        # Get homogeneity of the binary representation (IR)
        local_wghtd_homogeneity, local_bin_unique = get_weighted_homogeneity(
            dataset_binary, val_labels, metric)
        local_wghtd_homogeneities.append(local_wghtd_homogeneity)
        dataset_binary_unique_nos.append(local_bin_unique)

    print('Local weighted homogeneity of binarisation: '
          f'{np.mean(local_wghtd_homogeneities)} +- '
          f'{np.std(local_wghtd_homogeneities)}.')
    print(f'Unique binary points count: {np.mean(dataset_binary_unique_nos)} '
          f'+- {np.std(dataset_binary_unique_nos)}.')

    return (global_wghtd_homogeneity,
            local_wghtd_homogeneities,
            dataset_discrete_unique_no,
            max(dataset_binary_unique_nos))


def get_tree_global(dataset, labels, leaves, classification=True,
                    report=True, random_seed=42,
                    validation_dataset=None, validation_labels=None):
    """Evaluates GLOBAL IR of a tree-based discretisation."""
    if validation_dataset is None:
        validation_dataset = dataset
    if validation_labels is None:
        validation_labels = labels

    # Choose evaluation metric
    metric = gini if classification else mse

    # Generate a range of leaf numbers
    assert (leaves % 8 == 0) and (leaves % 16 == 0)
    leaves_range = list(range(
        int(leaves / 8),
        leaves + 1,
        int(leaves / 16)
    ))[::-1]

    outcomes = {}
    for width in leaves_range:
        # Build a tree
        if classification:
            clf = sk_tree.DecisionTreeClassifier(
                max_leaf_nodes=width, random_state=random_seed)
        else:
            clf = sk_tree.DecisionTreeRegressor(
                max_leaf_nodes=width, random_state=random_seed)
        clf.fit(dataset, labels)

        # Get data assignment to leaves
        assignment = clf.apply(validation_dataset)

        global_wghtd_homogeneity, assignment_unique_no = get_weighted_homogeneity(
            assignment, validation_labels, metric)
        assert assignment_unique_no <= width

        if report:
            print('Global weighted homogeneity of tree-based discretisation '
                f'({width} leaves): {global_wghtd_homogeneity}.')

        outcomes[width] = global_wghtd_homogeneity
    return outcomes


def get_tree_local(
        dataset, labels, leaves, distance_factor=0.3, classification=True,
        random_seed=42):
    """Evaluates LOCAL IR of a tree-based discretisation."""
    # Choose evaluation metric
    metric = gini if classification else mse

    # Generate a range of leaf numbers
    assert (leaves % 8 == 0) and (leaves % 16 == 0)
    leaves_range = list(range(
        int(leaves / 8),
        leaves + 1,
        int(leaves / 16)
    ))[::-1]

    # Get distances between all the points in the data set
    dist_matrix = fatf_dist.euclidean_array_distance(dataset, dataset)
    # Get the distance radius for identifying close instances
    max_dist = dist_matrix.max()
    radius = max_dist * distance_factor

    outcomes = {}
    for width in leaves_range:
        outcomes[width] = list()

    # Assess LOCAL homogeneity around each instance in the data set
    for row in dataset:
        # Get indices of nearby instances
        dist = fatf_dist.euclidean_point_distance(row, dataset)
        val_ind = dist <= radius

        # Get nearby instances
        val_data_discrete = dataset[val_ind, :]
        val_labels = labels[val_ind]

        # Global within a local scope = local
        local_scope = get_tree_global(
            val_data_discrete, val_labels, leaves,
            classification=classification, report=False,
            random_seed=random_seed)
        assert np.array_equal(
            sorted(list(local_scope.keys()), reverse=True), leaves_range)
        for width in leaves_range:
            outcomes[width].append(local_scope[width])

    for width in leaves_range:
        print('Local weighted homogeneity of tree-based discretisation '
              f'({width} leaves): {np.mean(outcomes[width])} '
              f'+- {np.std(outcomes[width])}.')

    return outcomes
