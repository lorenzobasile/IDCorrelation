# Intrinsic Dimension Correlation: uncovering nonlinear connections in multimodal representations

As a first step, `requirements.txt` have to be installed, and N24News and MS-COCO 2014 have to be downloaded from their official sources in a `data` folder, and extracted.

Then, representations for all models can be computed by running the `extract_representations.py` script, specifying a `model` and a `dataset`. After that, $I_d$Corr can be computed using `correlation.py`, while baselines are computed by running `baselines.py`. In both cases, a `dataset` parameter can be specified.

On ImageNet, we also compute correlation between coarsely aligned representations. The code for this experiment is found in `coarse.py`.

Finally, the example on MNIST (section 4.1) can be reproduced running `mnist.py`.
