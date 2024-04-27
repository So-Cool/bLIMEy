[![new BSD](https://img.shields.io/github/license/So-Cool/bLIMEy.svg)](https://github.com/So-Cool/bLIMEy/blob/master/LICENCE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/So-Cool/bLIMEy/master?filepath=ECML-PKDD_2023)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2005.01427-violet)][doi]

# LIMEtree: Consistent and Faithful Surrogate Explanations of Multiple Classes #

This directory contains a collection of Jupyter Notebooks that can be used to
reproduce the explanations, experiments and plots reported in the
"*LIMEtree: Consistent and Faithful Surrogate Explanations of Multiple Classes*"
paper.

The manuscript is available on [arXiv].

A collection of related resources -- illustrating how to build custom surrogates
for tabular and image data -- is available as
[part of the FAT Forensics documentation][doc] and
[standalone interactive presentations][events].

To run the notebooks you need to install a collection of Python packages listed
in the `requirements.txt` file (`pip install -r requirements.txt`).
Alternatively the notebooks can be executed on Binder by following the Binder
button included at the top of this page.

Additionally, running some of the experiments requires downloading auxiliary
FAT Forensics scripts:

```bash
mkdir -p scripts

# helpers.py
cp ../DAMI_2024/helpers.py scripts/

# __init__.py
wget https://raw.githubusercontent.com/fat-forensics/resources/master/surrogates_overview/scripts/__init__.py -O scripts/__init__.py
# image_classifier.py
wget https://raw.githubusercontent.com/fat-forensics/resources/master/surrogates_overview/scripts/image_classifier.py -O scripts/image_classifier.py
# imagenet_label_map.py
wget https://raw.githubusercontent.com/fat-forensics/resources/master/surrogates_overview/scripts/imagenet_label_map.py -O scripts/imagenet_label_map.py
# cifar_label_map.py
wget https://raw.githubusercontent.com/fat-forensics/resources/master/surrogates_overview/scripts/cifar_label_map.py -O scripts/cifar_label_map.py
```

## Abstract ##

Explainable machine learning provides tools to better understand predictive
models and their decisions, but many such methods are limited to producing
insights with respect to a single class.
When generating explanations for several classes, reasoning over them to
obtain a complete view may be difficult since they can present competing or
contradictory evidence.
To address this issue we introduce a novel paradigm of
*multi-class explanations*.
We outline the theory behind such techniques and propose a local surrogate
model based on multi-output regression trees -- called `LIMEtree` -- which
offers *faithful* and *consistent* explanations of multiple classes for
individual predictions while being post-hoc, model-agnostic and data-universal.
In addition to strong fidelity guarantees, our implementation supports
(interactive) *customisation* of the explanatory insights and delivers a range
of diverse explanation types, including counterfactual statements favoured in
the literature.
We evaluate our algorithm with a collection of quantitative experiments, a
qualitative analysis based on explainability desiderata and a preliminary user
study on an image classification task, comparing it to LIME.
Our contributions demonstrate the benefits of multi-class explanations and
wide-ranging advantages of our method across a diverse set scenarios.

## BibTeX ##

```
@article{sokol2020limetree,
  title={{LIMEtree}: {Consistent} and Faithful Surrogate Explanations of
         Multiple Classes},
  author={Sokol, Kacper and Flach, Peter},
  journal={arXiv preprint arXiv:2005.01427},
  url={https://arxiv.org/abs/2005.01427},
  year={2020}
}
```

[arXiv]: https://arxiv.org/abs/2005.01427
[doc]: https://fat-forensics.org/how_to/index.html#transparency-how-to
[events]: https://events.fat-forensics.org
[doi]: https://doi.org/10.48550/arXiv.2005.01427
