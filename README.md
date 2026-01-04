# Efficient Alignment of Unconditioned Action Prior for Language-conditioned Pick and Place in Clutter

> **Note:** This is a modernized fork of the [original repository](https://github.com/xukechun/Action-Prior-Alignment) with updated dependencies, Python 3.10-3.12 support, and streamlined installation via `uv`.

[Paper](https://arxiv.org/abs/2503.09423) | [Video](https://www.bilibili.com/video/BV1dPX4YzEzk/?spm_id_from=333.1391.0.0) | [Original Repo](https://github.com/xukechun/Action-Prior-Alignment)

We study the task of language-conditioned pick and place in clutter, where a robot should grasp a target object in open clutter and move it to a specified place. Some approaches learn end-to-end policies with features from vision foundation models, requiring large datasets. Others combine foundation models in a zero-shot setting, suffering from cascading errors. In addition, they primarily leverage vision and language foundation models, focusing less on action priors. In this paper, we aim to develop an effective policy by integrating foundation priors from vision, language, and action. We propose A2, an action prior alignment method that aligns unconditioned action priors with 3D vision-language priors by learning one attention layer. The alignment formulation enables our policy to train with less data and preserve zero-shot generalization capabilities. We show that a shared policy for both pick and place actions enhances the performance for each task, and introduce a policy adaptation scheme to accommodate the multi-modal nature of actions. Extensive experiments in simulation and the real-world show that our policy achieves higher task success rates with fewer steps for both pick and place tasks in clutter, effectively generalizing to unseen objects and language instructions.

![system overview](images/system.png)

#### Contact

- Original authors: kcxu@zju.edu.cn
- This fork: [@grach0v](https://github.com/grach0v)

## Setup

### Quick Installation (uv, Python 3.10-3.12)

**Prerequisites:** Ubuntu 20.04+, [uv](https://docs.astral.sh/uv/) installed, CUDA toolkit (tested with CUDA 12.4).

```bash
git clone git@github.com:grach0v/Action-Prior-Alignment.git
cd Action-Prior-Alignment

# Install everything with uv
uv sync

# Build CUDA extensions (clones GraspNet, patches files, builds extensions)
uv run python scripts/setup/setup_cuda_extensions.py
```

That's it! The setup script will:
- Clone the GraspNet baseline repository
- Apply patches for modern PyTorch compatibility
- Build the pointnet2 and knn CUDA extensions
- Download the GraspNet checkpoint

**For different CUDA versions:** Edit `pyproject.toml` and change `cu124` to your version (e.g., `cu118`, `cu121`).

**For LeRobot integration:** Install with the lerobot extras:
```bash
uv sync --extra lerobot
```

### Troubleshooting

- **`ModuleNotFoundError: No module named 'torch'`** during extension build: The setup script uses `--no-build-isolation` to handle this automatically.
- **`CUDA error: no kernel image is available`**: Your PyTorch CUDA version doesn't match your GPU. Check `torch.version.cuda` and reinstall with matching wheels.
- **`THC/THC.h: No such file or directory`**: Run the setup script again - it patches these deprecated headers.
- **`ModuleNotFoundError: No module named 'helpers'`**: Use `uv run python -m ...` instead of running files directly.

### Installation (legacy conda)

<details>
<summary>Click to expand conda instructions</summary>

For the original conda-based installation with CUDA 11.3 / PyTorch 1.10.1, see the [original repository](https://github.com/xukechun/Action-Prior-Alignment).

**Pre-packed environment:** A conda environment via `conda-pack` is available [here](https://huggingface.co/datasets/KechunXu1/A2_Dataset/blob/main/vilg3d.tar.gz) (CUDA 11.3 only).

</details>

### Assets
We provide the processed object models in this [link](https://drive.google.com/drive/folders/1WxKDFXJktoqiP0jmkDZrMCcNNBx5u-YM?usp=drive_link). Please download the file and unzip it in the `assets` folder.

### Data and Pre-trained Models
We provide our training data in this [link](https://huggingface.co/datasets/KechunXu1/A2_Dataset). Please download the file and unzip it in the `data` folder. 

We provide our testing cases in this [link](https://drive.google.com/drive/folders/1OuTua-69NEeV7RYIi9nzR1jmdZEugB68?usp=sharing). Please download the file and unzip it in the `testing_cases` folder. 

We provide our pre-trained models in this [link](https://drive.google.com/drive/folders/1uoDGIgkcSi8okcr8qjKOaF57TyRaHRd_?usp=sharing). Please download the file and unzip it in the `logs` folder.

### Data Collection
- For pick data
```
bash scripts/data_collection/collect_data_grasp.sh
```
- For place data
```
bash scripts/data_collection/collect_data_place.sh
```
- For unified pick+place data (recreates `data/a2_pp_data.npy` and saves per-episode frames)
```
bash scripts/data_collection/collect_data_pp.sh
```
Frame dumps go to `data/a2_pp_frames` (disable with `--no_record`).

## Training

- Unified training for pick and place
```
bash scripts/train/train_clutter_gp_unified.sh
```
- Adaptation for place
```
bash scripts/train/train_clutter_gp_adaptive.sh
```


## Evaluation
To test the pre-trained model, simply change the location of `--model_path`:

- Pick
```
bash scripts/test/test_grasp.sh
```
- Place
```
bash scripts/test/test_place.sh
```
- Pick and place
```
bash scripts/test/test_pickplace.sh
```

## Citation

If you find this work useful, please consider citing:

```
@article{xu2025efficient,
  title={Efficient Alignment of Unconditioned Action Prior for Language-conditioned Pick and Place in Clutter},
  author={Xu, Kechun and Xia, Xunlong and Wang, Kaixuan and Yang, Yifei and Mao, Yunxuan and Deng, Bing and Xiong, Rong and Wang, Yue},
  journal={arXiv preprint arXiv:2503.09423},
  year={2025}
}
```
