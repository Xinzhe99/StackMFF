<div align="center">

# <img src="assets/stackmff_logo.svg" alt="StackMFF" height="320" style="vertical-align: middle;"/> StackMFF

**StackMFF: End-to-end Multi-Focus Image Stack Fusion Network**

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![GitHub](https://img.shields.io/badge/GitHub-StackMFF-black.svg)](https://github.com/Xinzhe99/StackMFF)

</div>

## 📢 News

> [!NOTE]
> 🎉 **2025.11**: We reorganized the project code, which has subtle differences from the original implementation in the paper.

> 🎉 **2025.4**: The paper has been published by Applied Intelligence! [Paper Link](https://link.springer.com/article/10.1007/s10489-025-06383-8).

</div>

##  Table of Contents

- [Overview](#-overview)
- [Highlights](#-highlights)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Downloads](#-downloads)
- [Usage](#-usage)
- [Citation](#-citation)

</div>

## 📖 Overview

Existing end-to-end multi-focus image fusion (MFF) networks demonstrate excellent performance when fusing image pairs. However, when image stacks are processed, the necessity for iterative fusion leads to error accumulation, resulting in various types and degrees of image degradation, which ultimately limits the algorithms’ practical applications. To address this challenge and expand the application scenarios of multi-focus fusion algorithms, we propose a relatively simple yet effective approach: utilizing 3D convolutional neural networks to directly model and fuse entire multi-focus image stacks in an end-to-end manner. To obtain large-scale training data, we developed a refocusing pipeline based on monocular depth estimation technology that can synthesize a multi-focus image stack from any all-in-focus image. Furthermore, we extended the attention mechanisms commonly used in image pair fusion networks from two dimensions to three dimensions and proposed a comprehensive loss function group, effectively enhancing the fusion quality. Extensive experimental results demonstrate that the proposed method achieves state-of-the-art performance in both fusion quality and processing speed while avoiding image degradation issues, establishing a simple yet powerful baseline for the multi-focus image stack fusion task.


## ✨ Highlights

🌟 Proposes the first network specifically designed for multi-focus image stack fusion.

🔑 Introduces a novel pipeline for synthesizing image stacks based on depth estimation.

🎯 Establishes a benchmark for the multi-focus image stack fusion task.

🛠️ Releases a multi-focus image stack fusion toolbox containing 12 algorithms.

🏆 Achieves state-of-the-art fusion performance and processing speed.

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Xinzhe99/StackMFF.git
cd StackMFF
```

2. Create and activate a virtual environment (recommended):
```bash
conda create -n stackmff python=3.8
conda activate stackmff
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📥 Downloads

| Resource | Link | Code | Description |
|----------|------|------|-------------|
| 🗂️ **Training Datasets** | [![GitHub](https://img.shields.io/badge/GitHub-2196F3?style=flat-square)](https://github.com/cvdfoundation/open-images-dataset#download-full-dataset-with-google-storage-transfer) | `-` | Complete training datasets |
| 🗂️ **Test Datasets** | [![Download](https://img.shields.io/badge/Download-4CAF50?style=flat-square)](https://pan.baidu.com/s/1vnEciGFDDjDybmoxNSAVSA) | `cite` | Complete evaluation datasets |
| 📊 **Benchmark Results** | [![Download](https://img.shields.io/badge/Download-FF9800?style=flat-square)](https://pan.baidu.com/s/1Q93goQxa0tdXne1UQxA8-Q?pwd=cite) | `cite` | Fusion results from all methods |
| 🧰 **Fusion Toolbox** | [![GitHub](https://img.shields.io/badge/GitHub-2196F3?style=flat-square)](https://github.com/Xinzhe99/Toolbox-for-Multi-focus-Image-Stack-Fusion) | - | Iterative fusion implementations |
| 🧰 **Metric3D** | [![GitHub](https://img.shields.io/badge/GitHub-2196F3?style=flat-square)](https://github.com/YvanYin/Metric3D) | - | Metric3D |

## 💻 Usage

The pre-trained model weights file `checkpoint.pth` should be placed in the `weights` directory.

## ✈️Inference
### If you want to inference datasets, run:
```
python predict_dataset.py --model_path checkpoint/checkpoint.pth --stack_basedir_path data/Datasets_StackMFF/4D-Light-Field/image stack
```

### Predict Single Stack
```
python predict.py --stack_path path/to/image/stack
```

### Predict Dataset
```bash
python predict_dataset.py --stack_basedir_path path/to/dataset
```

### Training
1. Download the validation set of the original dataset [Open Images V7](https://github.com/cvdfoundation/open-images-dataset#download-full-dataset-with-google-storage-transfer) used to make the training dataset, and put all images to `data/OpenImagesV7`.
2. Use Metric3D to get depth maps.
3. Use make_dataset.py to make training datasets.
4. Run:
```bash
python train.py --datapath path/to/training_datasets --exp_name stackmff_training --datapath data/OpenImagesV7
```

## 📚 Citation

If you use this project in your research, please cite our papers:

<details>
<summary>📋 BibTeX</summary>

```bibtex
@article{xie2025stackmff,
  title={StackMFF: end-to-end multi-focus image stack fusion network},
  author={Xie, Xinzhe and Qingyan, Jiang and Chen, Dong and Guo, Buyu and Li, Peiliang and Zhou, Sangjun},
  journal={Applied Intelligence},
  volume={55},
  number={6},
  pages={503},
  year={2025},
  publisher={Springer}
}
```

</details>

## 🙏 Acknowledgments

We sincerely thank all the reviewers and the editors for their responsible efforts and valuable feedback, which have greatly improved the quality of this study!

⭐ If you find this project helpful, please give it a star!
</div>

