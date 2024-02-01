"""
Image Classifier
================

This module implements an image classifier based on resnet56 for cifar10 in
PyTorch.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import numpy as np

import logging
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if CUDA_AVAILABLE else 'cpu')

CIFAR10_LABELS = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


def _get_preprocess_transform():
    normalize = transforms.Normalize(
        mean=[0.5070, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
    transf = transforms.Compose([transforms.ToTensor(), normalize])

    return transf


class ImageClassifier(object):
    """Image classifier based on PyTorch."""

    def __init__(self, model='resnet56', use_gpu=False):
        """Initialises the image classifier."""
        assert model in ('resnet56')

        # Get class labels
        self.class_idx = CIFAR10_LABELS

        # Get the model
        clf = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)

        if use_gpu:
            if CUDA_AVAILABLE:
                clf = clf.to(DEVICE)
                # clf.cuda()
                predict_proba = self._predict_proba_gpu
            else:
                logger.warning('GPU was requested but it is not available. '
                               'Using CPU instead.')
                predict_proba = self._predict_proba_cpu
        else:
            predict_proba = self._predict_proba_cpu
        self.predict_proba = predict_proba

        self.clf = clf
        self.clf.eval()

        # Get transformation
        self.preprocess_transform = _get_preprocess_transform()

    def fit(self, _X, _y):
        """Fits the image classifier -- a dummy method."""
        return

    def _predict_proba_cpu(self, X):
        """[CPU] Predicts probabilities of the collection of images."""
        X_ = [self.preprocess_transform(x) for x in X]
        tensor = torch.stack(X_, dim=0)
        prediction = F.softmax(self.clf(tensor), dim=1)
        return prediction.detach().cpu().numpy()

    def _predict_proba_gpu(self, X):
        """[GPU] Predicts probabilities of the collection of images."""
        X_ = [self.preprocess_transform(x) for x in X]
        tensor = torch.stack(X_, dim=0)
        tensor = tensor.to(DEVICE)
        prediction = F.softmax(self.clf(tensor), dim=1)
        return prediction.detach().cpu().numpy()

    def predict(self, X, labels=False):
        """Predicts class indices of the collection of images."""
        prediction = self.predict_proba(X)
        prediction_idxs = np.argmax(prediction, axis=1)
        if labels:
            classes = [self.class_idx[i] for i in prediction_idxs]
        else:
            classes = prediction_idxs
        return classes

    def proba2prediction(self, Y):
        """Converts predicted probabilities to class indices."""
        classes = np.argmax(Y, axis=1)
        return classes

    def prediction2label(self, Y):
        """Converts class indices to label names."""
        labels = [self.class_idx[y] for y in Y]
        return labels

    def proba2label(self, Y, labels_no=5):
        """Converts class probabilities to label names."""
        ordered_classes = np.flip(np.argsort(Y, axis=1))
        ordered_classes_top = ordered_classes[:, :labels_no]
        labels_top = np.vectorize(self.class_idx.get)(ordered_classes_top)
        return labels_top

    def proba2tuple(self, Y, labels_no=5):
        """Converts class probabilities to (label name, probability) tuples."""
        ordered_classes = np.flip(np.argsort(Y, axis=1))

        ordered_classes_top = ordered_classes[:, :labels_no]
        labels_top = np.vectorize(self.class_idx.get)(ordered_classes_top)

        tuples = []
        for idx in range(Y.shape[0]):
            tuples_ = []
            for cls, lab in zip(ordered_classes_top[idx], labels_top[idx]):
                tuples_.append((lab, Y[idx, cls], cls))
            tuples.append(tuples_)
        return tuples
