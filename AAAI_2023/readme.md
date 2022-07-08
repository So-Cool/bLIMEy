[![new BSD](https://img.shields.io/github/license/So-Cool/bLIMEy.svg)](https://github.com/So-Cool/bLIMEy/blob/master/LICENCE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/So-Cool/bLIMEy/master?filepath=AAAI_2023)

# LIMEtree: Customisable, Faithful and Consistent Multi-class Explanations #

This directory contains a collection of Jupyter Notebooks that can be used to
reproduce the plots and experiments reported in the
"*LIMEtree: Customisable, Faithful and Consistent Multi-class Explanations*" paper.

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

TODO

## BibTeX ##

```
@article{sokol2020limetree,
  title={{LIMEtree}: {C}ustomisable, Faithful and Consistent
         Multi-class Explanations},
  author={Sokol, Kacper and Flach, Peter},
  journal={arXiv preprint arXiv:2005.01427},
  url={https://arxiv.org/abs/2005.01427},
  year={2020}
}
```

[arXiv]: https://arxiv.org/abs/2005.01427
[doc]: https://fat-forensics.org/how_to/index.html#transparency-how-to
[events]: https://events.fat-forensics.org
