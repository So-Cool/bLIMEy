[![new BSD](https://img.shields.io/github/license/So-Cool/bLIMEy.svg)](https://github.com/So-Cool/bLIMEy/blob/master/LICENCE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/So-Cool/bLIMEy/master?filepath=DAMI_2024)
[![DOI](https://img.shields.io/badge/DOI-10.1007/s10618--024--01010--5-violet)][Springer]

# Interpretable Representations in Explainable AI: From Theory to Practice #

This directory contains a collection of Jupyter Notebooks that can be used to
reproduce the plots and experiments reported in the
"*Interpretable Representations in Explainable AI: From Theory to Practice*" paper
published in the *Special Issue on Explainable and Interpretable Machine Learning and Data Mining*
of the Springer *Data Mining and Knowledge Discovery* journal.

The manuscript is available on [Springer] and [arXiv].

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

Interpretable representations are the backbone of many explainers that target
black-box predictive systems based on artificial intelligence and machine
learning algorithms.
They translate the low-level data representation necessary for good predictive
performance into high-level human-intelligible concepts used to convey the
explanatory insights.
Notably, the explanation type and its cognitive complexity are directly
controlled by the interpretable representation, tweaking which allows to target
a particular audience and use case.
However, many explainers built upon interpretable representations overlook
their merit and fall back on default solutions that often carry implicit
assumptions, thereby degrading the explanatory power and reliability of such
techniques.
To address this problem, we study properties of interpretable representations
that encode presence and absence of human-comprehensible concepts.
We demonstrate how they are operationalised for tabular, image and text data;
discuss their assumptions, strengths and weaknesses; identify their core
building blocks; and scrutinise their configuration and parameterisation.
In particular, this in-depth analysis allows us to pinpoint their explanatory
properties, desiderata and scope for (malicious) manipulation in the context of
tabular data where a linear model is used to quantify the influence of
interpretable concepts on a black-box prediction.
Our findings lead to a range of recommendations for designing trustworthy
interpretable representations;
specifically, the benefits of class-aware (supervised) discretisation of
tabular data, e.g., with decision trees, and sensitivity of image interpretable
representations to segmentation granularity and occlusion colour.

## BibTeX ##
```
@article{sokol2024interpretable,
  title={Interpretable Representations in Explainable {AI}:
         {From} Theory to Practice},
  author={Sokol, Kacper and Flach, Peter},
  journal={Data Mining and Knowledge Discovery},
  publisher={Springer},
  doi={10.1007/s10618-024-01010-5},
  year={2024}
}
```

[arXiv]: https://arxiv.org/abs/2008.07007
[Springer]: https://doi.org/10.1007/s10618-024-01010-5
[doc]: https://fat-forensics.org/how_to/index.html#transparency-how-to
[events]: https://events.fat-forensics.org
