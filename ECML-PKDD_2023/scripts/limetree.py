"""
LIMEtree Functions
==================

This module implements helper functions for the LIMEtree explainer.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import fatf
import itertools
import logging
import matplotlib
import scipy
import sklearn.linear_model
import sklearn.tree
import warnings

import fatf.utils.data.augmentation as fatf_augmentation
import fatf.utils.data.segmentation as fatf_segmentation
import fatf.utils.data.occlusion as fatf_occlusion
import fatf.utils.kernels as fatf_kernels
import fatf.utils.models.processing as fatf_processing
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import numpy as np

from PIL import Image
from sklearn.tree import _tree

import scripts.image_classifier as imgclf

__all__ = ['imshow', 'visualise_img',
           'tree_to_code', 'rules_dict2array', 'rules_dict2list',
           'tree_get_explanation', 'filter_explanations',
           'explain_image', 'explain_image_exp',
           'lime_loss', 'limet_loss', 'compute_loss',
           'process_loss', 'summarise_loss_lime', 'summarise_loss_limet',
           'plot_loss_summary']

# Set up logging; enable logging of level INFO and higher
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
_logger_handler = logging.StreamHandler()  # pylint: disable=invalid-name
_logger_formatter = logging.Formatter(  # pylint: disable=invalid-name
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%y-%b-%d %H:%M:%S')
_logger_handler.setFormatter(_logger_formatter)
logger.addHandler(_logger_handler)

plt.style.use('seaborn')  # 'classic'

matplotlib.rc('text', usetex=True)
# matplotlib.rc('font', family='sans-serif', sans-serif=['Helvetica'])
# matplotlib.rc('font', family='serif', serif=['Palatino'])

SEG_MSG = ('Slic segmenter failed. '
           'Try upgrading to scikit-image 0.19.2 or higher.')


def imshow(img):
    plt.grid(None)
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def visualise_img(explanation, segmenter, top_features=None):
    """Visualise explanation -- image"""
    highlight_mask = []
    for i, v in explanation:
        c = (255, 0, 0) if v < 0 else (0, 255, 0)
        c_ = 'red' if v < 0 else 'green'
        highlight_mask.append((i, v, c, c_))
    highlight_mask = sorted(
        highlight_mask, key=lambda i: abs(i[1]), reverse=True)

    seg, col = [], []
    for s, _, c, _ in highlight_mask[:top_features]:
        seg.append(int(s))
        col.append(c)

    _explanation = segmenter.highlight_segments(seg, colour=col)
    _explanation_ = segmenter.mark_boundaries(
        image=_explanation, colour=(255, 255, 0))
    
    return _explanation_


def tree_to_code(
        tree,
        feature_number=None, feature_names=None,
        include_split_nodes=True):
    tree_ = tree.tree_
    rules = dict()

    if feature_number is None and feature_names is None:
        raise RuntimeError(
            'Either the number of features or feature names must be provided.')
    elif feature_names is not None:
        pass
    else:
        feature_names = list(range(feature_number))
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else 'undefined!'
        for i in tree_.feature
    ]

    def recurse(node, depth, collector):
        local_collector = dict(collector)
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            assert threshold == 0.5, 'The logic works only for trees on binary data'

            # current
            if include_split_nodes:
                rules[node] = dict(collector)

            # left
            # print('{}if {} <= {}:'.format(indent, name, threshold))
            local_collector[name] = 0
            recurse(tree_.children_left[node], depth + 1, local_collector)

            # right
            # print('{}else:  # if {} > {}'.format(indent, name, threshold))
            local_collector[name] = 1
            recurse(tree_.children_right[node], depth + 1, local_collector)
        else:
            # print('{}return {}'.format(indent, tree_.value[node]))
            local_collector['prediction'] = tree_.value[node]
            rules[node] = local_collector

    recurse(0, 1, {})
    
    return rules


def rules_dict2array(rules_dict, features_number):
    dict_array = dict()
    for node, path in rules_dict.items():
        point = np.ones(features_number, dtype=np.int16)  # argmax

        for feature_id, feature_value in path.items():
            if feature_id == 'prediction': continue
            point[feature_id] = feature_value
        
        dict_array[node] = point
        
    return dict_array


def rules_dict2list(rules_dict, features_number, start_at_one=True):
    start_at_one = 1 if start_at_one else 0
    dict_array = dict()

    for node, path in rules_dict.items():
        point = {'off': [], 'on': [], 'none':[]}

        remainder = list(range(features_number))
        for feature_id, feature_value in path.items():
            if feature_id == 'prediction': continue
            if feature_value:
                point['on'].append(feature_id + start_at_one)
            else:
                point['off'].append(feature_id + start_at_one)
            remainder.remove(feature_id)
        point['none'] = [i + start_at_one for i in remainder]
        if 'prediction' in path: point['prediction'] = path['prediction']

        dict_array[node] = point

    return dict_array


def tree_get_explanation(tree, class_id, feature_number,
                         include_split_nodes=True, start_at_one=True,
                         max_pred=True, max_occlusion=False, discard_others=False):
    if max_occlusion: max_pred = False
    if discard_others: max_pred = False

    start_at_one = 1 if start_at_one else 0
    tree_ = tree.tree_
    rules = dict()

    feature_names = list(range(feature_number))
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else 'undefined!'
        for i in tree_.feature
    ]

    def recurse(node, depth, collector):
        local_collector = dict(collector)
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            assert threshold == 0.5, 'The logic works only for trees on binary data'

            # current
            if include_split_nodes:
                rules[node] = dict(collector)
                rules[node]['prediction'] = tree_.value[node][:, 0]

            # left
            local_collector[name] = 0
            recurse(tree_.children_left[node], depth + 1, local_collector)

            # right
            local_collector[name] = 1
            recurse(tree_.children_right[node], depth + 1, local_collector)
        else:
            local_collector['prediction'] = tree_.value[node][:, 0]
            rules[node] = local_collector

    recurse(0, 1, {})

    if max_pred:
        # Find maximum prediction
        summary = [(node, spec['prediction'][class_id])
                   for node, spec in rules.items()]
        summary = sorted(summary, key=lambda x: x[1], reverse=True)
        repr = dict(rules[summary[0][0]])

        prediction = repr['prediction'][class_id]
        del repr['prediction']

        representation = {'off': [], 'on': [], 'none':[]}
        # Fill in the gaps
        remainder = list(range(feature_number))
        for feature_id, feature_value in repr.items():
            if feature_value:
                representation['on'].append(feature_id + start_at_one)
            else:
                representation['off'].append(feature_id + start_at_one)
            remainder.remove(feature_id)
        representation['none'] = [i + start_at_one for i in remainder]
    else:
        argmax_nodes =[]
        for node, spec in rules.items():
            if discard_others:
                full_vector = spec['prediction']
            else:
                other = 1 - spec['prediction'].sum()
                if other < 0: warnings.warn("Probs don't add up!", RuntimeWarning)
                full_vector = np.concatenate([spec['prediction'], [other]])
            if np.argmax(full_vector) == class_id:
                argmax_nodes.append(node)

        prediction, representation = [], []
        for i in argmax_nodes:
            prediction.append(rules[i]['prediction'][class_id])
            repr = dict(rules[i])
            del repr['prediction']

            repr_ = {'off': [], 'on': [], 'none':[]}
            # Fill in the gaps
            remainder = list(range(feature_number))
            for feature_id, feature_value in repr.items():
                if feature_value:
                    repr_['on'].append(feature_id + start_at_one)
                else:
                    repr_['off'].append(feature_id + start_at_one)
                remainder.remove(feature_id)
            repr_['none'] = [i + start_at_one for i in remainder]

            representation.append(repr_)

    if max_occlusion:
        by_count = {}
        for pred, rep in zip(prediction, representation):
            count = len(rep['off'])
            if count in by_count:
                by_count[count].append((pred, rep))
            else:
                by_count[count] = [(pred, rep)]
        max_count = max(by_count.keys())

        prediction = [i[0] for i in by_count[max_count]]
        representation = [i[1] for i in by_count[max_count]]

    return prediction, representation


def filter_explanations(instance_spec, on=None, off=None, use_none=False):
    on = set() if on is None else set(on)
    off = set() if off is None else set(off)

    assert on.difference(off) == on, "There mustn't be any overlap between on and off."

    idxs = []
    for i, spec in enumerate(instance_spec):
        if use_none:
            on_ = spec['on'] + spec['none']
            off_ = spec['off'] + spec['none']
        else:
            on_ = spec['on']
            off_ = spec['off']

        if off.issubset(off_) and on.issubset(on_):
            idxs.append(i)

    return idxs


def explain_image(image_path, classifier,
                  random_seed=42, n_top_classes=3,            # General
                  samples_number=150, batch_size=50,          # Processing
                  segmenter_type='slic',                      # Segmenter Type
                  ratio=0.5, kernel_size=5, max_dist=10,      # QS Segmenter
                  n_segments=13,                              # Slic Segmenter
                  occlusion_colour='black',                   # Occluder
                  generate_complete_sample=True,              # Sampler
                  kernel_width=0.25                           # Similarity
                  ):
    logger.debug(f'Image: {image_path}')
    img = np.asarray(Image.open(image_path))

    assert segmenter_type in ('slic', 'quick-shift'), 'Unknown segmenter.'
    logger.debug(f'Segmenter in use: {segmenter_type}')

    img_pred = classifier.predict_proba([img])[0]
    top_three_classes = np.flip(np.argsort(img_pred))[:n_top_classes]
    logger.debug(f'Top n classes: {n_top_classes}')

    # Get segmenter
    fatf.setup_random_seed(random_seed)
    if segmenter_type == 'slic':
        assert n_segments >= 2, 'You need at least two segments.'
        while True:
            try:
                segmenter = fatf_segmentation.Slic(img, n_segments=n_segments)
            except RuntimeError as e:
                if str(e) == SEG_MSG:
                    if n_segments < 2:
                        logger.debug('Could not segment the image.')
                        return image_path, top_three_classes, None, None, None
                    n_segments -= 1
                    continue
                else:
                    raise e
            if segmenter.segments_number < 2 or n_segments < 2:
                logger.debug('Could not segment the image.')
                return image_path, top_three_classes, None, None, None
            elif segmenter.segments_number > n_segments:
                n_segments -= 1
            else:
                break
    else:
        segmenter = fatf_segmentation.QuickShift(
            img, ratio=ratio, kernel_size=kernel_size, max_dist=max_dist)
    logger.debug(f'Segments number: {segmenter.segments_number}')
    # Get occluder
    occluder = fatf_occlusion.Occlusion(
        img, segmenter.segments, colour=occlusion_colour)
    occluderR = fatf_occlusion.Occlusion(
        img, segmenter.segments, colour='randomise-patch')

    # Generate training sample
    exhaustive_no = 2 ** segmenter.segments_number
    if exhaustive_no <= samples_number or generate_complete_sample:
        sampled_data = np.array(list(itertools.product(
            [0, 1], repeat=segmenter.segments_number)))
    else:
        fatf.setup_random_seed(random_seed)
        sampled_data = fatf_augmentation.random_binary_sampler(
            segmenter.segments_number, samples_number)
    logger.debug(f'Sampled data shape: {sampled_data.shape}')

    # Predict sample -- black
    iter_ = fatf_processing.batch_data(
        sampled_data,
        batch_size=batch_size,
        transformation_fn=occluder.occlude_segments_vectorised)
    sampled_data_probabilities = []
    for batch in iter_:
        batch_predictions = classifier.predict_proba(batch)
        sampled_data_probabilities.append(batch_predictions)
    sampled_data_probabilities = np.vstack(sampled_data_probabilities)

    # Predict sample -- random
    iter_ = fatf_processing.batch_data(
        sampled_data,
        batch_size=batch_size,
        transformation_fn=occluderR.occlude_segments_vectorised)
    sampled_data_probabilitiesR = []
    fatf.setup_random_seed(random_seed)
    for batch in iter_:
        batch_predictions = classifier.predict_proba(batch)
        sampled_data_probabilitiesR.append(batch_predictions)
    sampled_data_probabilitiesR = np.vstack(sampled_data_probabilitiesR)

    # Get distances to the sampled data
    one_sample = np.ones(shape=(1, sampled_data.shape[1]), dtype=np.int8)
    distances = scipy.spatial.distance.cdist(one_sample, sampled_data, 'cosine').flatten()
    # Cosine distance between [1, 1, 1] and [0, 0, 0] is nan so we replace it with 1
    np.nan_to_num(distances, nan=1.0, copy=False)
    assert not np.isnan(distances).any(), 'Do not expect any nans.'

    # Transform distance into similarity
    similarities = fatf_kernels.exponential_kernel(
                distances, width=kernel_width)

    # LIME -- explain each class with a ridge regression
    lime_dict = {}
    for idx in top_three_classes:
        class_probs = sampled_data_probabilities[:, idx]
        class_probsR = sampled_data_probabilitiesR[:, idx]

        clf_weighted = sklearn.linear_model.Ridge(
            alpha=1, fit_intercept=True, random_state=random_seed)
        clf_weighted.fit(sampled_data, class_probs, sample_weight=similarities)

        preds = clf_weighted.predict(sampled_data)
        diffs_weighted = class_probs - preds
        diffs_weightedR = class_probsR - preds

        clf = sklearn.linear_model.Ridge(
            alpha=1, fit_intercept=True, random_state=random_seed)
        clf.fit(sampled_data, class_probs)

        preds = clf.predict(sampled_data)
        diffs = class_probs - preds
        diffsR = class_probsR - preds

        lime_dict[idx] = dict(diffs=diffs, diffsR=diffsR,
                              diffs_weighted=diffs_weighted,
                              diffs_weightedR=diffs_weightedR)

    # LIMEtree -- explain each class with a multi-output regression tree
    lime_tree_dict = {}
    for depth_bound in range(2, segmenter.segments_number + 1):
        lime_tree_dict[depth_bound] = {}

        for classes_no in range(1, int(top_three_classes.shape[0]) + 1):
            class_ids = top_three_classes[:classes_no]
            if classes_no == 1:
                class_probs =  sampled_data_probabilities[:, class_ids[0]]
                class_probsR = sampled_data_probabilities[:, class_ids[0]]
            else:
                class_probs = sampled_data_probabilities[:, class_ids]
                class_probsR = sampled_data_probabilities[:, class_ids]
           
            # LIMEtree
            tree = sklearn.tree.DecisionTreeRegressor(
                random_state=random_seed, max_depth=depth_bound)
            tree.fit(sampled_data, class_probs)
            
            pred = tree.predict(sampled_data)
            diffs = class_probs - pred
            diffsR = class_probsR - pred

            tree_weighted = sklearn.tree.DecisionTreeRegressor(
                random_state=random_seed, max_depth=depth_bound)
            tree_weighted.fit(
                sampled_data, class_probs, sample_weight=similarities)

            pred = tree_weighted.predict(sampled_data)
            diffs_weighted = class_probs - pred
            diffs_weightedR = class_probsR - pred

            # LIMEtree -- overridden
            t2c = tree_to_code(tree, segmenter.segments_number,
                               include_split_nodes=False)
            t2a = rules_dict2array(t2c, segmenter.segments_number)
            # Map each node id to a prediction from the black box
            node_ids = sorted(t2a.keys())
            node_imgs = np.asarray([t2a[i] for i in node_ids])
            iter_ = fatf_processing.batch_data(
                node_imgs,
                batch_size=batch_size,
                transformation_fn=occluder.occlude_segments_vectorised)
            node_probs = []
            for batch in iter_:
                batch_predictions = classifier.predict_proba(batch)
                node_probs.append(batch_predictions)
            node_probs = np.vstack(node_probs)
            # Construct the tree
            fixed_tree = {}
            for i, node_id in enumerate(node_ids):
                # Assume shortest explanations are best
                if classes_no == 1:
                    fixed_tree[node_id] = node_probs[i, class_ids[0]]
                else:
                    fixed_tree[node_id] = node_probs[i, class_ids]
            # Get the overriden predictions
            pred = np.zeros_like(pred)
            pred_leaf_id = tree.apply(sampled_data)
            for i, p in enumerate(pred_leaf_id):
                pred[i] = fixed_tree[p]
            # Get the residuals
            diffs_fixed = class_probs - pred
            diffs_fixedR = class_probsR - pred

            t2c = tree_to_code(tree_weighted, segmenter.segments_number,
                               include_split_nodes=False)
            t2a = rules_dict2array(t2c, segmenter.segments_number)
            # Map each node id to a prediction from the black box
            node_ids = sorted(t2a.keys())
            node_imgs = np.asarray([t2a[i] for i in node_ids])
            iter_ = fatf_processing.batch_data(
                node_imgs,
                batch_size=batch_size,
                transformation_fn=occluder.occlude_segments_vectorised)
            node_probs = []
            for batch in iter_:
                batch_predictions = classifier.predict_proba(batch)
                node_probs.append(batch_predictions)
            node_probs = np.vstack(node_probs)
            # Construct the tree
            fixed_tree = {}
            for i, node_id in enumerate(node_ids):
                # Assume shortest explanations are best
                if classes_no == 1:
                    fixed_tree[node_id] = node_probs[i, class_ids[0]]
                else:
                    fixed_tree[node_id] = node_probs[i, class_ids]
            # Get the overriden predictions
            pred = np.zeros_like(pred)
            pred_leaf_id = tree_weighted.apply(sampled_data)
            for i, p in enumerate(pred_leaf_id):
                pred[i] = fixed_tree[p]
            # Get the residuals
            diffs_fixed_weighted = class_probs - pred
            diffs_fixed_weightedR = class_probsR - pred

            lime_tree_dict[depth_bound][classes_no] = dict(
                cls_id=class_ids,
                diffs=diffs, diffsR=diffsR,
                diffs_weighted=diffs_weighted, diffs_weightedR=diffs_weightedR,
                diffs_fixed=diffs_fixed, diffs_fixedR=diffs_fixedR,
                diffs_fixed_weighted=diffs_fixed_weighted,
                diffs_fixed_weightedR=diffs_fixed_weightedR)

    return image_path, top_three_classes, similarities, lime_dict, lime_tree_dict


def explain_image_exp(
        image_path, use_gpu=False,
        random_seed=42, n_top_classes=3,            # General
        samples_number=150, batch_size=50,          # Processing
        segmenter_type='slic',                      # Segmenter Type
        ratio=0.5, kernel_size=5, max_dist=10,      # QS Segmenter
        n_segments=13,                              # Slic Segmenter
        occlusion_colour='black',                   # Occluder
        generate_complete_sample=True,              # Sampler
        kernel_width=0.25                           # Similarity
        ):
    clf = imgclf.ImageClassifier(use_gpu=use_gpu)
    return explain_image(
        image_path, clf,
        random_seed=random_seed, n_top_classes=n_top_classes,
        samples_number=samples_number, batch_size=batch_size,
        segmenter_type=segmenter_type,
        ratio=ratio, kernel_size=kernel_size, max_dist=max_dist,
        n_segments=n_segments,
        occlusion_colour=occlusion_colour,
        generate_complete_sample=generate_complete_sample,
        kernel_width=kernel_width)


def lime_loss(residuals, weights=None):
    """Individual loss (LIME)."""
    if weights is None:
        weights = np.ones(residuals.shape[0], dtype=np.int8)

    assert len(residuals.shape) == 1, 'Expect one dimensional vector'
    metric = np.power(residuals, 2)
    # mse = metric.mean()
    metric_weighted = metric * weights
    mse = metric_weighted.sum() / weights.sum()

    return mse


def limet_loss(residuals, weights=None):
    """Cumulative loss (LIMEtree)."""
    if len(residuals.shape) == 1:
        logger.debug('Reshaping for LIMEtree loss (got 1D).')
        residuals_ = np.array([residuals])
    else:
        residuals_ = residuals
    if weights is None:
        weights = np.ones(residuals_.shape[1], dtype=np.int8)
    classes_no = residuals_.shape[0]
    assert classes_no > 0, 'Must be at least one class.'

    # This is for cummulative: 1, 1--2, 1--3
    metric = np.power(residuals_, 2)
    metric_class_sum = metric.sum(axis=0)
    metric_class_sum_weighted = metric_class_sum * weights
    if classes_no != 1: metric_class_sum_weighted *= 0.5
    mse = metric_class_sum_weighted.sum() / weights.sum()

    return mse


def compute_loss(pred_idxs, similarities, diff, diff_type):
    assert diff_type in ('lime', 'limet'), 'Only lime and limet are supported.'
    lime_measurements = {'diffs', 'diffsR', 'diffs_weighted', 'diffs_weightedR'}
    limet_measurements = lime_measurements.union({
        'cls_id',
        'diffs_fixed', 'diffs_fixedR',
        'diffs_fixed_weighted', 'diffs_fixed_weightedR'})

    loss_collector = dict()

    def collect_residuals_lime(diff_, idxs_, type_):
        assert type_ in lime_measurements, 'Must be a known type.'
        collector_ = [diff_[i][type_] for i in idxs_]
        collector_ = np.vstack(collector_)
        return collector_

    if diff_type == 'lime':
        for i, idx in enumerate(pred_idxs):
            assert not set(diff[idx].keys()).difference(lime_measurements)

            # Individual loss
            mse = lime_loss(diff[idx]['diffs'])
            mseR = lime_loss(diff[idx]['diffsR'])

            wmse = lime_loss(diff[idx]['diffs_weighted'], weights=similarities)
            wmseR = lime_loss(diff[idx]['diffs_weightedR'], weights=similarities)

            # Cumulative loss
            lt_mse = limet_loss(collect_residuals_lime(
                diff, pred_idxs[:i+1], 'diffs'))
            lt_mseR = limet_loss(collect_residuals_lime(
                diff, pred_idxs[:i+1], 'diffsR'))

            lt_wmse = limet_loss(
                collect_residuals_lime(diff, pred_idxs[:i+1], 'diffs_weighted'),
                weights=similarities)
            lt_wmseR = limet_loss(
                collect_residuals_lime(diff, pred_idxs[:i+1], 'diffs_weightedR'),
                weights=similarities)
 
            loss_collector[idx] = dict(
                mse=mse, mseR=mseR, wmse=wmse, wmseR=wmseR,  # LIME loss
                lt_mse=lt_mse, lt_mseR=lt_mseR,              # LIMEtree loss
                lt_wmse=lt_wmse, lt_wmseR=lt_wmseR)
    else:
        for depth, depth_dict in diff.items():
            loss_collector[depth] = dict()
            for classes_no, diff_ in depth_dict.items():
                assert not set(diff_.keys()).difference(limet_measurements)

                # Individual loss
                mse = lime_loss(diff_['diffs'].T.flatten())
                mseR = lime_loss(diff_['diffsR'].T.flatten())

                mseF = lime_loss(diff_['diffs_fixed'].T.flatten())
                mseFR = lime_loss(diff_['diffs_fixedR'].T.flatten())

                similarities_ = np.tile(similarities, classes_no)
                wmse = lime_loss(
                    diff_['diffs_weighted'].T.flatten(),
                    weights=similarities_)
                wmseR = lime_loss(
                    diff_['diffs_weightedR'].T.flatten(),
                    weights=similarities_)

                wmseF = lime_loss(
                    diff_['diffs_fixed_weighted'].T.flatten(),
                    weights=similarities_)
                wmseFR = lime_loss(
                    diff_['diffs_fixed_weightedR'].T.flatten(),
                    weights=similarities_)

                # Cumulative loss
                lt_mse = limet_loss(diff_['diffs'].T)
                lt_mseR = limet_loss(diff_['diffsR'].T)

                lt_mseF = limet_loss(diff_['diffs_fixed'].T)
                lt_mseFR = limet_loss(diff_['diffs_fixedR'].T)

                lt_wmse = limet_loss(
                    diff_['diffs_weighted'].T, weights=similarities)
                lt_wmseR = limet_loss(
                    diff_['diffs_weightedR'].T, weights=similarities)

                lt_wmseF = limet_loss(
                    diff_['diffs_fixed_weighted'].T, weights=similarities)
                lt_wmseFR = limet_loss(
                    diff_['diffs_fixed_weightedR'].T, weights=similarities)

                loss_collector[depth][classes_no] = dict(
                    mse=mse, mseR=mseR,                    # LIMEtree loss
                    mseF=mseF, mseFR=mseFR,
                    wmse= wmse, wmseR=wmseR,
                    wmseF=wmseF, wmseFR=wmseFR,
                    lt_mse=lt_mse, lt_mseR=lt_mseR,        # LIMEtree loss
                    lt_mseF=lt_mseF, lt_mseFR=lt_mseFR,
                    lt_wmse= lt_wmse, lt_wmseR=lt_wmseR,
                    lt_wmseF=lt_wmseF, lt_wmseFR=lt_wmseFR)

    return loss_collector


def process_loss(loss_collector):
    lime_scores, limet_scores, top_classes = [], [], []
    imgs_sorted = sorted(loss_collector.keys())

    for img in imgs_sorted:
        top_pred, similarities, lime, limet = loss_collector[img]

        if similarities is None:
            logger.debug(f'Image not processed: {img}')
            continue

        top_classes.append(top_pred)
        loss = compute_loss(top_pred, similarities, lime, 'lime')
        lime_scores.append(loss)
        loss = compute_loss(top_pred, similarities, limet, 'limet')
        limet_scores.append(loss)

    return top_classes, lime_scores, limet_scores


def summarise_loss_lime(lime_loss, top_classes):
    score_types = ['mse', 'mseR', 'wmse', 'wmseR',
                   'lt_mse', 'lt_mseR', 'lt_wmse', 'lt_wmseR']

    lime_loss_summary = {}
    for score_type in score_types:
        lime_loss_summary[score_type] = dict()
        for run_classes, run in zip(top_classes, lime_loss):
            for i, run_classes in enumerate(run_classes):
                i_ = i + 1
                score_ = run[run_classes][score_type]
                if i_ in lime_loss_summary[score_type]:
                    lime_loss_summary[score_type][i_].append(score_)
                else:
                    lime_loss_summary[score_type][i_] = [score_]

    lime_loss_summary_ = {}
    for score_type in score_types:
        lime_loss_summary_[score_type] = {}
        for cls_, scores in lime_loss_summary[score_type].items():
            lime_loss_summary_[score_type][cls_] = (
                np.mean(scores), np.var(scores))

    return lime_loss_summary_


def summarise_loss_limet(limet_loss, top_classes, rounding=2):
    score_types = ['mse', 'mseR', 'wmse', 'wmseR',
                   'lt_mse', 'lt_mseR', 'lt_wmse', 'lt_wmseR',
                   'mseF', 'mseFR', 'wmseF', 'wmseFR',
                   'lt_mseF', 'lt_mseFR', 'lt_wmseF', 'lt_wmseFR']

    assert top_classes, 'Must not be empty.'
    classes_n = top_classes[0].shape[0]
    for tc in top_classes:
        assert classes_n == tc.shape[0]

    limet_loss_summary = {}
    for score_type in score_types:
        limet_loss_summary[score_type] = dict()
        for classes_no in range(1, classes_n+1):
            limet_loss_summary[score_type][classes_no] = dict()

    for score_type in score_types:
        for run in limet_loss:
            depths = sorted(run.keys())
            depths_max = max(depths)
            for classes_no in range(1, classes_n+1):
                for depth in depths:
                    depth_ratio = round(depth/depths_max, rounding)
                    if depth_ratio in limet_loss_summary[score_type][classes_no]:
                        limet_loss_summary[score_type][classes_no][depth_ratio].append(
                            run[depth][classes_no][score_type])
                    else:
                        limet_loss_summary[score_type][classes_no][depth_ratio] = [
                            run[depth][classes_no][score_type]]

    limet_loss_summary_ = {}
    for score_type in score_types:
        limet_loss_summary_[score_type] = {}
        for classes_no in range(1, classes_n+1):
            limet_loss_summary_[score_type][classes_no] = {}
            for depth_ratio, scores in limet_loss_summary[score_type][classes_no].items():
                limet_loss_summary_[score_type][classes_no][depth_ratio] = (
                    np.mean(scores), np.var(scores))

    return limet_loss_summary_


def plot_loss_summary(lime_scores_summary, limet_scores_summary, class_id,
                      use_limet_loss=False, use_weighted=True, use_random=False,
                      fontsize=16):
    if use_limet_loss:
        if use_weighted:
            if use_random:
                stub = 'limet_weighted_random'
                lime_loss_key, limetf_loss_key = 'lt_wmseR', 'lt_wmseFR'
                limet_loss_key = lime_loss_key
            else:
                stub = 'limet_weighted_Xrandom'
                lime_loss_key, limetf_loss_key = 'lt_wmse', 'lt_wmseF'
                limet_loss_key = lime_loss_key
        else:
            if use_random:
                stub = 'limet_Xweighted_random'
                lime_loss_key, limetf_loss_key = 'lt_mseR', 'lt_mseFR'
                limet_loss_key = lime_loss_key
            else:
                stub = 'limet_Xweighted_Xrandom'
                lime_loss_key, limetf_loss_key = 'lt_mse', 'lt_mseF'
                limet_loss_key = lime_loss_key
    else:
        if use_weighted:
            if use_random:
                stub = 'lime_weighted_random'
                lime_loss_key, limetf_loss_key = 'wmseR', 'wmseFR'
                limet_loss_key = lime_loss_key
            else:
                stub = 'lime_weighted_Xrandom'
                lime_loss_key, limetf_loss_key = 'wmse', 'wmseF'
                limet_loss_key = lime_loss_key
        else:
            if use_random:
                stub = 'lime_Xweighted_random'
                lime_loss_key, limetf_loss_key = 'mseR', 'mseFR'
                limet_loss_key = lime_loss_key
            else:
                stub = 'lime_Xweighted_Xrandom'
                lime_loss_key, limetf_loss_key = 'mse', 'mseF'
                limet_loss_key = lime_loss_key

    lime_mean, lime_var = lime_scores_summary[lime_loss_key][class_id]
    limet = limet_scores_summary[limet_loss_key][class_id]
    limetf = limet_scores_summary[limetf_loss_key][class_id]

    plt.figure(figsize=(8, 6))
    cc = plt.get_cmap('tab10')  # Set3
    colours = [plt_colors.rgb2hex(cc(i)) for i in range(cc.N)]

    # LIMEtree loss
    y_, err_ = [], []
    x_ = sorted(limet.keys())
    for x in x_:
        y_.append(limet[x][0])
        err_.append(limet[x][1])
    plt.errorbar(x_, y_, yerr=err_, label='\\textbf{LIMEt}',
                 solid_capstyle='projecting', capsize=5, color=colours[0])

    # (Fixed) LIMEtree loss
    y_, err_ = [], []
    x_ = sorted(limetf.keys())
    for x in x_:
        y_.append(limetf[x][0])
        err_.append(limetf[x][1])
    plt.errorbar(x_, y_, yerr=err_, label='\\(\\underline{\\textbf{LIMEt}}\\)',
                 solid_capstyle='projecting', capsize=5, color=colours[1])

    # LIME loss
    lime_range = [x_[0], x_[-1]]
    lime_plt = plt.plot(
        lime_range, 2*[lime_mean], label='\\textbf{LIME}', color=colours[2])
    # lime_c = lime_plt[-1].get_color()
    plt.fill_between(
        lime_range, 2*[lime_mean-lime_var], 2*[lime_mean+lime_var],
        alpha=0.3, color=colours[2])

    plt.tick_params(axis='x', labelsize=fontsize)
    plt.tick_params(axis='y', labelsize=fontsize)
    # plt.xlim(-0.1, 1.1)
    # plt.ylim(-.1, .1)

    plt.legend(loc='upper right', fontsize=fontsize,
               frameon=True, framealpha=.75)  # title='Method:', facecolor='white'
    plt.tight_layout()

    plt.savefig(f'_figures/loss-cls{class_id}-{stub}.pdf',
                dpi=300, bbox_inches='tight', pad_inches=0)
