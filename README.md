# PyTorch Poker Cards Classifier CNN

A deep learning project implementing a Convolutional Neural Network (CNN) for classifying poker playing cards using PyTorch. The model is trained to recognize and classify standard deck playing cards across 53 classes including all suits, ranks, and joker cards.

## Overview

This project utilizes deep learning techniques to build an accurate playing card classifier. The implementation leverages PyTorch and CNNs to perform image classification on poker card images, achieving high accuracy through careful model architecture design and training optimization.

## Features

- Custom CNN architecture optimized for playing card classification
- Support for 53 card classes (52 standard cards plus joker)
- Data preprocessing and augmentation pipeline
- Training and evaluation scripts in Jupyter notebook format
- Early stopping implementation to prevent overfitting

## Project Structure

```
pytorch-poker-cards-classifier-cnn/
├── dataset/                      # Training and validation datasets
├── card-classifier.ipynb         # Main notebook with model implementation
└── .gitignore                    # Git ignore configuration
```

## Prerequisites

- Python 3.10 or higher
- pip package manager
- CUDA-capable GPU (optional, but recommended for training)

## Requirements

- PyTorch
- torchvision
- NumPy
- Matplotlib
- JupyterLab
- PIL/Pillow

## Installation

1. Clone the repository:

```bash
git clone https://github.com/bacadlo/pytorch-poker-cards-classifier-cnn.git
cd pytorch-poker-cards-classifier-cnn
```

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. Install required dependencies:

```bash
pip install torch torchvision numpy matplotlib jupyterlab pillow
```

For GPU support with CUDA, install PyTorch with CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

1. Launch JupyterLab:

```bash
jupyter lab
```

## Usage

### Training the Model

Open and run the notebook in JupyterLab:

```bash
jupyter lab
```

Navigate to `card-classifier.ipynb` and run the cells. The notebook contains the complete pipeline including:

- Data loading and preprocessing
- Model architecture definition
- Training loop with validation
- Model evaluation and metrics

## Model Architecture

The CNN architecture consists of:

- Multiple convolutional layers with ReLU activation
- Max pooling layers for dimensionality reduction
- Fully connected layers for classification
- Output layer with 53 neurons corresponding to card classes

The model is trained using:

- Adam optimizer
- Cross-Entropy loss function
- Early stopping mechanism
- Validation monitoring for performance tracking

## Dataset

The dataset should be organized in a standard format with separate directories for training and validation:

```
dataset/
├── train/
│   ├── ace_of_spades/
│   ├── two_of_hearts/
│   └── ...
└── val/
    ├── ace_of_spades/
    ├── two_of_hearts/
    └── ...
```

## Performance

The model is optimized to achieve high classification accuracy on poker card images through:

- Careful hyperparameter tuning
- Data augmentation techniques
- Regularization methods
- Validation-based model selection

## Contributing

Contributions are welcome. Please feel free to submit issues or pull requests for improvements, bug fixes, or new features.

## License

This project is open source and available under the MIT License.

## References

This project was built using the following resources:

- [Train Your First PyTorch Model - Card Classifier by Rob Mulla](https://www.kaggle.com/code/robikscube/train-your-first-pytorch-model-card-classifier) - Tutorial and training approach
- [Cards Image Dataset Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification) - Dataset source
- [PyTorch CNN Playing Cards Classifier by hiroonwijekoon](https://github.com/hiroonwijekoon/pytorch-cnn-playing-cards-classifier) - Reference implementation

## Acknowledgments

- PyTorch team for the deep learning framework
- Rob Mulla for the comprehensive PyTorch tutorial
- Kaggle community for the playing cards dataset
- Reference implementations that guided this project’s development

## Contact

For questions or feedback, please open an issue in the repository.