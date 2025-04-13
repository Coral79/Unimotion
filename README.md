# Unimotion: Unifying 3D Human Motion Synthesis and Understanding

<img src='https://github.com/Coral79/Unimotion/blob/main/assets/teaser.png' width=1200> 

**Unimotion: Unifying 3D Human Motion Synthesis and Understanding** <br>
[Chuqiao Li](https://virtualhumans.mpi-inf.mpg.de/people/Li.html), [Julian Chibane](https://virtualhumans.mpi-inf.mpg.de/people/Chibane.html), [Yannan He](https://virtualhumans.mpi-inf.mpg.de/people/He.html), [Naama Pearl](https://naamapearl.github.io/), [Andreas Geiger](https://www.cvlibs.net/), [Gerard Pons-Moll](https://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html) <br>
[[Project Page]](https://coral79.github.io/uni-motion/) [[Paper]](http://arxiv.org/abs/2409.15904)

3DV(Oral), 2025

## News :triangular_flag_on_post:
- [2024/09/30] Unimotion paper is available on [ArXiv](http://arxiv.org/abs/2409.15904).
- [2025/13/04] Code and pre-trained released.

## Key Insight
- Alignment between frame-level text and motion enables the **temproal semantic awareness** of the motion generation!
- Separate diffusion process for aligned motion and text enables **multi-directional inference**!
- Our model allows **Multiple Novel Applications**:
  - **Hierarchical Control**: Allowing users to specify motion at different levels of detail
  - **Motion Text Generation**: Obtaining motion text descriptions for existing MoCap data or YouTube videos
  - **Motion Editing**: Allowing for editability, generating motion from text, and editing the motion via text edits

## Install Environment

Install ffmpeg (if not already installed):

```shell
sudo apt update
sudo apt install ffmpeg
```
For windows use [this](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/) instead.

Setup conda env:
```shell
conda env create -f environment.yml
conda activate unimotion
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```
Download dependencies:

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```
## Data Preparation

Download the data:

**HumanML3D** (Sequence-level motion and text) - Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git), then run the following command:

```shell
cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D
```

**BABEL** Frame-level text Embeddings

You can download the preprocessed CLIP text embeddings (derived from BABEL annotations) with:

```bash
bash prepare/download_clip_embeddings.sh
```
These processed embeddings are all you need for training, sampling, and evaluation.

If you'd like to inspect the ground-truth frame-level motion-text alignments yourself, please refer to the instructions in this [repo](https://github.com/Mathux/AMASS-Annotation-Unifier) to download text labels and unify annotations accross different datasets.

<details>
  <summary><b>Directory Structure</b></summary>

After running the download scripts, your directory structure should look like this:
```
Unimotion/
├── dataset/
    └── HumanML3D/
        ├── clip_encoder.py
        ├── clip_enc_single/
        ├── examples_editing.txt
        ├── Mean_seg_pca_51.npy
        ├── pca/
        ├── README.md
        ├── Std_seg_pca_51.npy
        ├── test_ft.txt
        ├── test_ft_no_overlap.txt
        ├── texts/
        ├── train_ft.txt
        ├── val_ft.txt
        └── val_ft_no_overlap.txt
```

</details>

## Download Pretrained Models
Download the model then unzip and place them in `./save/`. 

```bash
bash prepare/download_checkpoints.sh
```


## Sampling

<details>
  <summary><b>Frame-Level Text to Motion</b></summary>


### Generate from your frame-level text file

```shell
python -m sample.generate \
--model_path ./save/unimotion_pca_51_humanml_trans_enc_512/model000400000.pt \
--sample_condition t2m \
--input_gt_local_txt ./assets/walk_sit.csv \
--guidance_param 0
```

### Generate from test set frame-level prompts

```shell
python -m sample.generate \
--model_path ./save/unimotion_pca_51_humanml_trans_enc_512/model000400000.pt \
--sample_condition t2m \
--num_samples 10 \
--guidance_param 0
```
</details>

<details>
  <summary><b>Hierarchical Text to Motion (frame-level + sequence-level)</b></summary>

### Generate from your text file (frame-level + squence-level)

```shell
python -m sample.generate \
--model_path ./save/unimotion_pca_51_humanml_trans_enc_512/model000400000.pt \
--sample_condition t2m \
--input_gt_local_txt ./assets/walk_sit.csv \
--input_text ./assets/wave_hands.txt
```

### Generate from test set prompts (frame-level + squence-level)

```shell
python -m sample.generate \
--model_path ./save/unimotion_pca_51_humanml_trans_enc_512/model000400000.pt \
--sample_condition t2m \
--num_samples 10 
```

</details>


<details>
  <summary><b>Squence-Level Text to Motion</b></summary>


### Generate from your sequence-level text file

```shell
python -m sample.generate \
--model_path ./save/unimotion_pca_51_humanml_trans_enc_512/model000400000.pt \
--sample_condition m+t \
--input_text ./assets/demos.txt 
```

### Generate from test set sequence-level prompts 

```shell
python -m sample.generate \
--model_path ./save/unimotion_pca_51_humanml_trans_enc_512/model000400000.pt \
--sample_condition m+t \
--num_samples 10 
```

### Generate a single sequence-level prompt

```shell
python -m sample.generate \
--model_path ./save/unimotion_pca_51_humanml_trans_enc_512/model000400000.pt \
--sample_condition m+t \
--text_prompt "the person paces back and forth."
```
</details>

<details>
  <summary><b>Motion to Text</b></summary>


### Generate from your motion file

demo_youtube.npy is a human pose estimation from youtube video, feel free to use avaliable methods and be creative with video selection

```shell
python -m sample.generate \
--model_path ./save/unimotion_pca_51_humanml_trans_enc_512/model000400000.pt \
--sample_condition m2t \
--input_motion_path ./assets/demo_youtube.npy
```

### Generate from test set motions

```shell
python -m sample.generate \
--model_path ./save/unimotion_pca_51_humanml_trans_enc_512/model000400000.pt \
--sample_condition m2t \
--num_samples 10 
```
</details>

<details>
  <summary><b>Motion Editing</b></summary>


### Edit from your motion file

This Example replace the walk forward from frame 83-135 to jog forward, you could also create this motion from any previous text to motion sampling and then conduct the edit.

```shell
python -m sample.edit \
--model_path ./save/unimotion_pca_51_humanml_trans_enc_512/model000400000.pt \
--edit_mode in_between \
--input_gt_local_txt ./assets/motion_edited.csv \
--input_motion_path ./assets/example_motion.npy \
--sample_condition t2m \
--guidance_param 0 \
--prefix_end 83 \
--suffix_start 135 \
--input_idx 8 \
--show_input
```

</details>


## Training

```shell
python -m train.train_unimotion \
--save_dir save/new_unimotion_pca_51_humanml_trans_enc_512 \
--eval_during_training \
--save_results
```

## Evaluation
Comming soon


## Citation

When using the code/figures/data/etc., please cite our work
```
@article{li2024unimotion,
  author    = {Li, Chuqiao and Chibane, Julian and He, Yannan and Pearl, Naama and Geiger, Andreas and Pons-Moll, Gerard},
  title     = {Unimotion: Unifying 3D Human Motion Synthesis and Understanding},
  journal   = {arXiv preprint arXiv:2409.15904},
  year      = {2024},
}
```