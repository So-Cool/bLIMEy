[![new BSD](https://img.shields.io/github/license/So-Cool/bLIMEy.svg)](https://github.com/So-Cool/bLIMEy/blob/master/LICENCE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/So-Cool/bLIMEy/master?filepath=IJCAI_2023)

# Consistent Explanations of Multiple Classes with LIMEtree #

This directory contains a collection of Jupyter Notebooks that can be used to
reproduce the explanations, experiments and plots reported in the
"*Consistent Explanations of Multiple Classes with LIMEtree*" paper.

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

# __init__.py
wget https://raw.githubusercontent.com/fat-forensics/resources/master/surrogates_overview/scripts/__init__.py -O scripts/__init__.py
# image_classifier.py
wget https://raw.githubusercontent.com/fat-forensics/resources/master/surrogates_overview/scripts/image_classifier.py -O scripts/image_classifier.py
# imagenet_label_map.py
wget https://raw.githubusercontent.com/fat-forensics/resources/master/surrogates_overview/scripts/imagenet_label_map.py -O scripts/imagenet_label_map.py
```

## Abstract ##

Explainable machine learning provides numerous tools to better understand
predictive models and their decisions, however many of these methods are
limited to producing explanations of a single class.
While such insights can be generated for different classes, reasoning over
them to obtain a complete view my be difficult or even impossible when they
present similar, competing or contradictory evidence.
To address this shortcoming we introduce a novel paradigm of
*multi-class explanations*.
We outline the theory behind such techniques and propose a local surrogate
based on multi-output regression trees -- called LIMEtree -- which offers
*faithful* and *consistent* explanations of multiple classes for individual
predictions while being post-hoc, model-agnostic and data-universal.
In addition to strong fidelity guarantees, our implementation supports
(interactive) *customisation* of the explanatory insights and delivers a
range of diverse explanation types, including counterfactual statements
praised in the literature.
We evaluate our algorithm with a collection of quantitative experiments and
a preliminary user study on an image classification task, comparing it
against LIME.
Our analysis demonstrates the benefits of multi-class explanations and
superiority of our method across a wide array of scenarios.

## BibTeX ##

```
@article{sokol2020limetree,
  title={Consistent Explanations of Multiple Classes with {LIMEtree}},
  author={Sokol, Kacper and Flach, Peter},
  journal={arXiv preprint arXiv:2005.01427},
  url={https://arxiv.org/abs/2005.01427},
  year={2020}
}
```

[arXiv]: https://arxiv.org/abs/2005.01427
[doc]: https://fat-forensics.org/how_to/index.html#transparency-how-to
[events]: https://events.fat-forensics.org
