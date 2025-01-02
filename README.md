# Overview
This repository contains the code to regenerate the visualizations of the paper [*Great Minds Think Alike: The Universal Convergence Trend of Input Salience*](https://openreview.net/pdf?id=7PORYhql4V) accepted at NeurIPS 2024.


# Contents

**Libraries**
For the implementation of the code, please have
<pre>
  numpy==1.19
  torch==1.10
  torchvision=0.11
</pre>


**Tutorials**
Due to the limited file size, we include pre-trained CNNSmall models on CIFAR-10, with 
k = 8, 10, 12, 14,
    16, 20, 24, 28,
    32, 40, 48, 56,
    64, 80, 96, 112,
    128
They are used for a simple demonstration of the results shown in the manuscript. Please refer to the **Training** session for the comprehensive re-implementation of the results. All the models are saved in the folder models_example

In order to visualize the saliency similarity, please run
<pre>
    python saliency_similarity.py
</pre>
This will generate a upper-triangular similarity map of $\rho$. Due to the limited file size, we are unable to upload multiple models of the same structure (i.e. the diagonal results). But it can be obtained through train.py.

In order to re-implement the black-box attack, please run
<pre>
    python black_box_attack.py
</pre>
This will generate a black-box attack map of $\alpha$. Since this map is non-symmetric, it's not upper-triangular as $\rho$. However, we do not have the diagonal results for the same reason. Please use train.py to obtain models for the results.



**Training**

Note that the training may take a very long time. Please revise GPU_index and perform parallel computation accordingly.
In order to train all the single models, please run
<pre>
    chmod +x train_all_single_models.sh
    ./train_all_single_models.sh
</pre>
This will train models with varying k values (widths) and fixed seeds.


In order to train all the models for approximating the population mean direction, please run
<pre>
    chmod +x train_population.sh
    ./train_population.sh
</pre>
This will train 100 models with different seeds of widths k=10,20,40 for CNNSmall, CNNLarge, ResNetSmall, ResNetLarge, and CIFAR-10/100.
