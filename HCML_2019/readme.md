[![new BSD](https://img.shields.io/github/license/So-Cool/bLIMEy.svg)](https://github.com/So-Cool/bLIMEy/blob/master/LICENCE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/So-Cool/bLIMEy/master?filepath=HCML_2019)

# bLIMEy: Surrogate Prediction Explanations Beyond LIME #

This directory contains a Jupyter Notebook that can be used to reproduce the
results included in the Appendix of the
"*bLIMEy: Surrogate Prediction Explanations Beyond LIME*" paper published at
the *2019 Workshop on Human-Centric Machine Learning* ([HCML 2019]) held during
the 33rd Conference on Neural Information Processing Systems (NeurIPS 2019),
Vancouver, Canada.

The manuscript is available on [arXiv].

A related *how-to* guide illustrating how to build custom surrogates for
tabular data is [a part of the FAT Forensics documentation].

To run the notebook (`bLIMEy.ipynb`) you need to install
`fat-forensics>=0.0.2`, `matplotlib` and `scikit-learn`. Additionally, the
`watermark` package is required for watermarking the notebook.
You can install these dependencies using the `requirements.txt` file
(included in this directory) by executing `pip install -r requirements.txt`.
Alternatively you can run it on Binder by following the Binder link above
(click on the *Binder* button).

## Abstract ##

Surrogate explainers of black-box machine learning predictions are of paramount
importance in the field of eXplainable Artificial Intelligence since they can
be applied to any type of data (images, text and tabular), are model-agnostic
and are post-hoc (i.e., can be retrofitted). The Local Interpretable
Model-agnostic Explanations (LIME) algorithm is often mistakenly unified with a
more general framework of surrogate explainers, which may lead to a belief that
it is the solution to surrogate explainability. In this paper we empower the
community to "build LIME yourself" (bLIMEy) by proposing a principled
algorithmic framework for building custom local surrogate explainers of
black-box model predictions, including LIME itself. To this end, we demonstrate
how to decompose the surrogate explainers family into algorithmically
independent and interoperable modules and discuss the influence of these
component choices on the functional capabilities of the resulting explainer,
using the example of LIME.

## BibTeX ##
```
@article{sokol2019blimey,
  title={b{LIME}y: {S}urrogate {P}rediction {E}xplanations {B}eyond {LIME}},
  author={Sokol, Kacper and Hepburn, Alexander and Santos-Rodriguez, Raul
          and Flach, Peter},
  journal={2019 Workshop on Human-Centric Machine Learning (HCML 2019) at the
           33rd Conference on Neural Information Processing Systems
           (NeurIPS 2019), Vancouver, Canada},
  note={arXiv preprint arXiv:1910.13016},
  url={https://arxiv.org/abs/1910.13016},
  year={2019}
}
```

[HCML 2019]: https://sites.google.com/view/hcml-2019
[arXiv]: https://arxiv.org/abs/1910.13016
[a part of the FAT Forensics documentation]: https://fat-forensics.org/how_to/transparency/tabular-surrogates.html
