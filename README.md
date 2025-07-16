<div align="center">
  <img src="logo.png" alt="LymphNet Logo" width="150"/>
</div>


# LymphNet

Deep learning pipeline for capturing morphometric immune features in lymph nodes of breast cancer patients. This project implements a segmentation pipeline based on Fully Convolutional Networks (FCNs) for automated analysis of lymph node histopathology.

## 🎯 Overview

LymphNet is designed to automatically segment and analyze lymph node structures, particularly focusing on:
- **Germinal Centers**: Detection and quantification of germinal center regions
- **Sinus Structures**: Segmentation of sinus areas within lymph nodes
- **Multi-class Segmentation**: Support for binary and multi-class segmentation tasks

## 🏗️ Architecture

The pipeline consists of several key components:

- **Data Processing**: TFRecord-based data loading and preprocessing
- **Model Architecture**: Multiple U-Net variants (Attention U-Net, ResU-Net, Mobile U-Net, etc.)
- **Training**: Distributed multi-GPU training with custom loss functions
- **Inference**: Patch-based prediction for whole slide images
- **Post-analysis**: Quantification and measurement of segmented features

## 📋 Requirements

### System Requirements
- Python 3.7+
- TensorFlow 2.x
- CUDA-compatible GPU (for training)

### Key Dependencies
- TensorFlow 2.x
- OpenSlide
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- scikit-image
- scikit-learn

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/lymphNet.git
   cd lymphNet
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
   ```

## 📖 Usage

### Training

1. **Prepare your data** in TFRecord format
2. **Configure parameters** in `config/config_germinal.yaml`
3. **Run training**:
   ```bash
   python src/main.py \
     --record_path /path/to/tfrecords \
     --record_dir your_dataset \
     --save_path /path/to/output \
     --test_path /path/to/test/images \
     --checkpoint_path /path/to/checkpoints \
     --config_file config/config_germinal.yaml \
     --model_name attention
   ```

### Prediction

```bash
python src/predict.py \
  --model_path /path/to/trained/model \
  --test_path /path/to/test/images \
  --save_path /path/to/predictions
```

### Data Preprocessing

Generate patches from whole slide images:
```bash
python src/tiler/generate_patches.py \
  --wsi_path /path/to/wsi \
  --annotations_path /path/to/annotations \
  --save_path /path/to/patches
```

## 🏛️ Model Architectures

The project supports multiple U-Net variants:

- **U-Net**: Standard U-Net architecture
- **Attention U-Net**: U-Net with attention mechanisms
- **ResU-Net**: Residual U-Net with skip connections
- **Mobile U-Net**: Lightweight U-Net based on MobileNetV2
- **Multi-scale U-Net**: Multi-scale feature extraction
- **DeepLabV3+**: Advanced semantic segmentation model

## 📊 Configuration

Configuration files are stored in the `config/` directory:

- `config_germinal.yaml`: Germinal center segmentation
- `config_pextract_g.json`: Patch extraction for germinal centers
- `config_pextract_s.json`: Patch extraction for sinuses
- `config_tf.json`: TFRecord generation settings

## 📁 Project Structure

```
lymphNet/
├── src/
│   ├── main.py                 # Main training script
│   ├── distributed_train.py    # Multi-GPU training
│   ├── predict.py              # Inference script
│   ├── models/                 # Model architectures
│   ├── data/                   # Data loading utilities
│   ├── tiler/                  # WSI patching utilities
│   ├── preprocessing/          # Data preprocessing
│   ├── postanalysis/           # Quantification tools
│   └── utilities/              # Helper functions
├── config/                     # Configuration files
├── bash/                       # Shell scripts
└── README.md                   # This file
```

## 🔬 Research Applications

This pipeline has been developed for:
- Automated lymph node analysis in breast cancer pathology
- Quantification of immune cell distributions
- Morphometric feature extraction
- Digital pathology workflow automation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👨‍💻 Contact

**Gregory Verghese** - gregory.verghese@gmail.com

Project Link: [https://github.com/yourusername/lymphNet](https://github.com/yourusername/lymphNet)

## 🚧 TODO

- [ ] Improve sinus and germinal predictions
- [ ] Implement weakly supervised approaches
- [ ] Add weighted map for sinuses (uncertain/fuzzy annotations)
- [ ] Transfer learning for U-Net
- [ ] Random conditional forest integration
- [ ] Add comprehensive unit tests
- [ ] Improve documentation and examples
