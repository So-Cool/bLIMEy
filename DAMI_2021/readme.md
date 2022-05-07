[![new BSD](https://img.shields.io/github/license/So-Cool/bLIMEy.svg)](https://github.com/So-Cool/bLIMEy/blob/master/LICENCE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/So-Cool/bLIMEy/master?filepath=DAMI_2021)

# Towards Faithful and Meaningful Interpretable Representations #

This directory contains a collection of Jupyter Notebooks that can be used to
reproduce the plots and experiments reported in the
"*Towards Faithful and Meaningful Interpretable Representations*" paper.

The manuscript is available on [arXiv].

A collection of related resources -- illustrating how to build custom surrogates
for tabular and image data -- is available as
[part of the FAT Forensics documentation][doc] and
[standalone interactive presentations][events].

To run the notebooks you need to install a collection of Python packages listed
in the `requirements.txt` file (`pip install -r requirements.txt`).
Alternatively the notebooks can be executed on Binder by following the Binder
button included at the top of this page.

## Abstract ##

Interpretable representations are the backbone of many black-box explainers.
They translate the low-level data representation necessary for good predictive
performance into high-level human-intelligible concepts used to convey the
explanation.
Notably, the explanation type and its cognitive complexity are directly
controlled by the interpretable representation, allowing to target a particular
audience and use case.
However, many explainers that rely on interpretable representations overlook
their merit and fall back on default solutions, which may introduce implicit
assumptions, thereby degrading the explanatory power of such techniques.
To address this problem, we study properties of interpretable representations
that encode presence and absence of human-comprehensible concepts.
We show how they are operationalised for tabular, image and text data,
discussing their strengths and weaknesses.
This allows us to analyse their explanatory properties in the context of tabular
data, where a linear model is used to quantify the importance of interpretable
concepts.
Our findings show benefits of tree-based interpretable representations, and
sensitivity of images to segmentation granularity and occlusion colour.

## BibTeX ##
```
@article{sokol2020towards,
  title={{T}owards {F}aithful and {M}eaningful {I}nterpretable
         {R}epresentations},
  author={Sokol, Kacper and Flach, Peter},
  journal={arXiv preprint arXiv:2008.07007},
  url={https://arxiv.org/abs/2008.07007},
  year={2020}
}
```

[arXiv]: https://arxiv.org/abs/2008.07007
[doc]: https://fat-forensics.org/how_to/index.html#transparency-how-to
[events]: https://events.fat-forensics.org
