[![Project page](https://img.shields.io/badge/project-page-red.svg)](https:/peterhuistyping.github.io/FreNBRDF/) [![arXiv](https://img.shields.io/badge/arXiv-2507.00476-orange.svg)](https://arxiv.org/abs/2507.00476)  [![MERL dataset](https://img.shields.io/badge/MERL-dataset-green.svg)](https://cdfg.csail.mit.edu/wojciech/brdfdatabase)

<!-- [![Python](https://img.shields.io/badge/Python3-Pytorch-blue.svg)] -->

<!-- (https://www.python.org/) -->

<!-- [![Base model weights](https://img.shields.io/badge/model-weight-yellow.svg)](https://huggingface.co/Peter2023HuggingFace/NeuMaDiff)   -->

# FreNBRDF: A Frequency-Rectified Neural Material Representation

<p align="center"><a href="https://chenliang-zhou.github.io">Chenliang Zhou</a>†, <a href="https://peterhuistyping.github.io/">Zheyuan Hu</a>†, <a href="https://www.cl.cam.ac.uk/~aco41/">Cengiz Öztireli</a></p>

<p align="center">Department of Computer Science and Technology<br>University of Cambridge</p>

<p align="center"><small>† denotes equal contribution.</small></p>

<p align="center">
    <a href="https:/peterhuistyping.github.io/FreNBRDF/">[Project page]</a>  
    <a href="https://arxiv.org/abs/2507.00476">[Paper]</a>
    <!-- <a href="https://huggingface.co/Peter2023HuggingFace/NeuMaDiff">[Base model weights]</a> -->
    <a href="https://cdfg.csail.mit.edu/wojciech/brdfdatabase">[MERL dataset]</a>
</p>


![teaser](./docs/img/teaser.jpg)

Overview of our FreNBRDF architecture.

# Abstract

Accurate material modeling is crucial for achieving photorealistic rendering, bridging the gap between computer-generated imagery and real-world photographs. While traditional approaches rely on tabulated BRDF data, recent work has shifted towards implicit neural representations, which offer compact and flexible frameworks for a range of tasks. However, their behavior in the frequency domain remains poorly understood.

To address this, we introduce *FreNBRDF*, a frequency-rectified neural material representation. By leveraging spherical harmonics, we integrate frequency-domain considerations into neural BRDF modeling. We propose a novel *frequency-rectified loss*, derived from a frequency analysis of neural materials, and incorporate it into a generalizable and adaptive reconstruction and editing pipeline. This framework enhances fidelity, adaptability, and efficiency.

Extensive experiments demonstrate that FreNBRDF improves the accuracy and robustness of material appearance reconstruction and editing compared to state-of-the-art baselines, enabling more structured and interpretable downstream tasks and applications.

# Citation

Please feel free to contact us if you have any questions or suggestions.

If you found the paper or code useful, please consider citing,

```
@misc{zhou2025FreNBRDF,
      title={FreNBRDF: A Frequency-Rectified Neural Material Representation}, 
      author={Chenliang Zhou and Zheyuan Hu and Cengiz Oztireli},
      year={2025},
      eprint={2507.00476},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2507.00476}, 
}
```
