"""
LIMEtree Functions
==================

This module implements helper functions for the LIMEtree explainer.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD


import matplotlib
import warnings

import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import _tree

__all__ = ['imshow', 'visualise_img',
           'tree_to_code', 'rules_dict2array', 'rules_dict2list',
           'tree_get_explanation', 'filter_explanations']


matplotlib.rc('text', usetex=True)
plt.style.use('seaborn')  # 'classic'


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
