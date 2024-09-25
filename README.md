# Unimotion: Unifying 3D Human Motion Synthesis and Understanding

<img src='https://github.com/Coral79/Unimotion/blob/main/assets/teaser.png' width=1200> 

**Unimotion: Unifying 3D Human Motion Synthesis and Understanding** <br>
[Chuqiao Li](https://virtualhumans.mpi-inf.mpg.de/people/Li.html), [Julian Chibane](https://virtualhumans.mpi-inf.mpg.de/people/Chibane.html), [Yannan He](https://virtualhumans.mpi-inf.mpg.de/people/He.html), [Naama Pearl](https://naamapearl.github.io/), [Andreas Geiger](https://www.cvlibs.net/), [Gerard Pons-Moll](https://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html) <br>
[[Project Page]](https://coral79.github.io/uni-motion/) [[Paper]](http://arxiv.org/abs/2409.15904)

Arxiv, 2024

## News :triangular_flag_on_post:
- [2024/06/14] Unimotion paper is available on [ArXiv](http://arxiv.org/abs/2409.15904).
- [2024/06/14] Code and pre-trained weights will be released soon.

## Key Insight
- Alignment between frame-level text and motion enables the explainability of the motion generation!
- Separate diffusion process for aligned motion and text enbales multi-directional inference!
- Our model allows **Multiple Novel Applications**:
  - **Hierarchical Control**: Allowing users to specify motion at different levels of detail
  - **Motion Text Generation**: Obtaining motion text descriptions for existing MoCap data or youtube videos
  - **Motion Editing**: Allowing for editability, generating motion from text and editing the motion via text edits

![](https://github.com/Coral79/Unimotion/blob/main/assets/unimotion_overview.png)

## Citation

When using the code/figures/data/etc., please cite our work
```
@article{li2024unimotion,
  author    = {Li, Chuqiao and Chibane, Julian and He, Yannan and Pearl, Naama and Geiger, Andreas and Pons-Moll, Gerard},
  title     = {Unimotion: Unifying 3D Human Motion Synthesis and Understanding},
  journal   = {Arxiv},
  year      = {2024},
}
```
