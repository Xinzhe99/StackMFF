<div align="center">

# <img src="assets/stackmff.svg" alt="StackMFF" height="320" style="vertical-align: middle;"/>

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![GitHub](https://img.shields.io/badge/GitHub-StackMFF-black.svg)](https://github.com/Xinzhe99/StackMFF)
[![Paper](https://img.shields.io/badge/Paper-Springer-brightgreen.svg)](https://link.springer.com/article/10.1007/s10489-025-06383-8)

*Official PyTorch implementation for StackMFF: end-to-end multi-focus image stack fusion network*

</div>

## 📢 News

> [!NOTE]
> 🎉 **2026.01**: This work has been included in [StackMFF-Series](https://github.com/Xinzhe99/StackMFF-Series)

> 🎉 **2026.01**: StackMFF V4 has been submitted and is under review! [Project Link](https://github.com/Xinzhe99/StackMFF-V4)

> 🎉 **2025.11**: StackMFF V3 has been submitted and is under review! [Project Link](https://github.com/Xinzhe99/StackMFF-V3)

> 🎉 **2025.11**: We reorganized the project code, which has subtle differences from the original implementation described in the paper.

> 🎉 **2025.9**: StackMFF V2 has been accepted by EAAI! [Project Link](https://github.com/Xinzhe99/StackMFF-V2)

> 🎉 **2025.4**: The paper has been published by Applied Intelligence! [Paper Link](https://link.springer.com/article/10.1007/s10489-025-06383-8).

## Table of Contents

- [Installation](#-installation)
- [Downloads](#-downloads)
- [Usage](#-usage)
- [Citation](#-citation)

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
| 🗂️ **Training Datasets** | [![GitHub](https://img.shields.io/badge/GitHub-2196F3?style=flat-square)](https://github.com/cvdfoundation/open-images-dataset#download-full-dataset-with-google-storage-transfer) | `-` | Complete training datasets for model training |
| 🗂️ **Test Datasets** | [![Download](https://img.shields.io/badge/Download-4CAF50?style=flat-square)](https://pan.baidu.com/s/1vnEciGFDDjDybmoxNSAVSA) | `cite` | Complete evaluation datasets for testing |
| 📊 **Benchmark Results** | [![Download](https://img.shields.io/badge/Download-FF9800?style=flat-square)](https://pan.baidu.com/s/1Q93goQxa0tdXne1UQxA8-Q?pwd=cite) | `cite` | Benchmark results from all compared methods |
| 🧰 **Fusion Toolbox** | [![GitHub](https://img.shields.io/badge/GitHub-2196F3?style=flat-square)](https://github.com/Xinzhe99/Toolbox-for-Multi-focus-Image-Stack-Fusion) | - | Implementations of iterative fusion methods |
| 🧰 **Metric3D** | [![GitHub](https://img.shields.io/badge/GitHub-2196F3?style=flat-square)](https://github.com/YvanYin/Metric3D) | - | Depth estimation tool |

## 💻 Usage

The pre-trained model weights file `checkpoint.pth` should be placed in the `weights` directory to run the model.

## ✈️ Inference
### To inference datasets, run:
```
python predict_dataset.py --stack_basedir_path "path/to/dataset"
```

### Predict Single Stack
```
python predict.py --stack_path "path/to/image/stack"
```

### Predict Dataset
```bash
python predict_dataset.py --stack_basedir_path "path/to/dataset"
```

### Training
1. Download the validation set of the original dataset [Open Images V7](https://github.com/cvdfoundation/open-images-dataset#download-full-dataset-with-google-storage-transfer) used to make the training dataset, and put all images in `data/OpenImagesV7`.
2. Use Metric3D to get depth maps.
3. Use make_dataset.py to make training datasets.
4. Run:
```bash
python train.py --datapath "path/to/training_datasets" --exp_name stackmff_training
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

We sincerely thank all the reviewers and editors for their responsible efforts and valuable feedback, which have greatly improved the quality of this study!

⭐ If you find this project helpful, please give it a star!
