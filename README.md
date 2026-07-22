# Next-Scale Autoregressive Models for Text-to-Motion Generation

<a href="https://arxiv.org/abs/2604.03799"><img src="https://img.shields.io/badge/arXiv-2604.03799-b31b1b.svg" alt="arXiv"></a>
<a href="https://zhiwei-zzz.github.io/MoScale"><img src="https://img.shields.io/badge/Project-Website-green" alt="Project Page"></a>


## 🛠️ Installation

### 1. Clone the repository

```bash
git clone git@github.com:zhiwei-zzz/MoScale.git
cd MoScale
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate moscale
```

## 📦 Dataset

This project uses [HumanML3D](https://github.com/EricGuo5513/HumanML3D). Follow the instructions in that repository to prepare the dataset, then set `data.root_dir` in all `config/*.yaml` files to your local HumanML3D path.

The dataset directory should have the following structure:

```
dataset/
└── HumanML3D/
    ├── new_joint_vecs/   # 263-dim motion features (.npy per clip)
    ├── new_joints/       # 3D joint positions (.npy per clip)
    ├── texts/            # text annotations
    ├── train.txt
    ├── val.txt
    ├── test.txt
    ├── Mean.npy
    └── ...
```

Set `data.root_dir` in all `config/*.yaml` files to the `HumanML3D/` subdirectory, e.g. `/your/path/dataset/HumanML3D`.

### 📥 Checkpoints

Download the motion evaluator, GloVe word vectors, and checkpoints:

```bash
bash prepare/download_evaluators.sh
bash prepare/download_glove.sh
bash prepare/download_model.sh
```

## 📊 Evaluation

### Evaluate HRVQVAE reconstruction quality

```bash
python eval_hrvqvae.py
```

Configuration: `config/eval_hrvqvae.yaml`. Set `data.root_dir` to your dataset path.

### Evaluate MoScale generation quality

```bash
python eval_moscale.py
```

Configuration: `config/eval_moscale.yaml`. Set `data.root_dir` to your dataset path.

## 🚀 Training

Training is a two-stage process. Update `data.root_dir` in the config files before running.

### Stage 1: Train HRVQVAE tokenizer

```bash
python train_hrvqvae.py
```

Configuration: `config/train_hrvqvae.yaml`


### Stage 2: Train MoScale Transformer

Set `vq_name` and `vq_ckpt` in `config/train_moscale.yaml` to point to your trained HRVQVAE checkpoint, then run:

```bash
python train_moscale.py
```

Configuration: `config/train_moscale.yaml`


## ✏️ Motion Editing

Regenerate a specified region of a source motion conditioned on a new text prompt:

```bash
python edt_t2m.py
```

Configuration: `config/edit.yaml`. Set `source_motion` to your source motion `.npy` file, `text_prompt` to the desired description, and `mask_edit_section` to the `"start, end"` fraction of the motion to edit (e.g. `["0.0, 0.6"]`).


## 📝 Todo

- [x] Release inference code
- [x] Release checkpoints
- [x] Release training code
- [x] Release editing code


## 💡 Idea Note (2026-07-16): Exposure-Bias-Aware Flow Decoder Training

Status: proposal / not yet implemented. Written down while comparing this project's flow decoder (`model/flow_decoder/`, `trainers/flow_decoder_trainer.py`, `config/train_flow_decoder.yaml`) against DisCoRD's RF decoder design.

**Problem.** The flow decoder is currently trained only on latents `z` derived from *ground-truth* tokens (encode GT motion → HRVQVAE codes → z → train `flow(gt_motion, z)`). At real inference, Stage-2 (the MoScale Transformer) sometimes predicts tokens that are positionally right but semantically different from GT (it captures context more than the exact code identity). Decoding those inference-time token sequences therefore drifts from GT motion — the decoder never saw this kind of error during training (classic **exposure bias** / train-inference mismatch, not something "consistency flow matching" addresses — that trick only enforces self-consistency along a single ODE trajectory for few-step sampling, unrelated to this).

**Proposed directions:**

1. **Text-conditioned decoder.** Feed the caption embedding into the flow decoder alongside `z` (both backbones already support a conditioning hook — `unet1d_backbone.py` / `dit_backbone.py` — so this is a moderate extension, not a redesign). Intuition: when `z` is corrupted by a token-prediction error, text can act as an anchor toward the intended motion. Caveat: `z` already carries most of the content, so the marginal value of text needs to be ablated (`z` only vs `z + text`).

2. **Train on actual/simulated inference-time tokens, not just GT tokens.** Run Stage-2 (MoScale Transformer) to get predicted token sequences → predicted `z'` → supervise `flow(gt_motion, z')` toward the *same* GT motion. This is standard **scheduled sampling / exposure-bias mitigation** (Bengio et al. 2015 family), not "consistency" in the flow-matching sense. Practical notes:
   - Generating predicted tokens on-the-fly every training step is expensive (Stage-2 decoding is iterative); precompute/cache predicted-`z` offline per (motion, caption) and mix with GT-`z` pairs (curriculum ratio, not 100% predicted from the start).
   - Risk: forcing convergence to the exact GT motion for every corrupted `z'` may over-constrain if some prediction "errors" are actually plausible alternative motions rather than pure noise.

3. **Multi-caption invariance.** HumanML3D provides multiple rephrased captions per motion. Idea: predicted tokens from caption A, B, C (same underlying motion, different rephrasings) will differ slightly from each other and from GT — train the decoder so all three map back to the *same* GT motion. This is a conditional-invariance / robustness objective (closer to denoising-autoencoder style training than to Consistency Flow Matching, which is about trajectory self-consistency, not multi-condition-to-same-target). Since one GT motion already has several caption/token variants available, this needs no new data collection, only precomputing predicted tokens per caption.

Net effect of (2)+(3): the decoder should learn to extract the invariant "identity" signal from `z` and text while ignoring the token-level noise introduced by Stage-2's imperfect prediction — directly targeting the FID gap between reconstruction-from-GT-tokens and generation-from-predicted-tokens.

## 🙏 Acknowledgements

The code is built upon open-source projects including [MoMask++](https://github.com/snap-research/SnapMoGen) and [VAR](https://github.com/FoundationVision/VAR). We thank the authors for their helpful code.

## 📜 Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{Zheng_2026_moscale,
    author    = {Zheng, Zhiwei and Jin, Shibo and Liu, Lingjie and Zhao, Mingmin},
    title     = {Next-Scale Autoregressive Models for Text-to-Motion Generation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {16376-16386}
}
```
