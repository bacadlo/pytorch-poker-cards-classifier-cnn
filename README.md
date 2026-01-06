# PyTorch Poker Cards Classifier CNN

A deep learning project implementing a Convolutional Neural Network (CNN) for classifying poker playing cards using PyTorch. The model is trained to recognize and classify standard deck playing cards across 53 classes including all suits, ranks, and joker cards.

## Overview

This project utilizes deep learning techniques to build an accurate playing card classifier. The implementation leverages PyTorch and CNNs to perform image classification on poker card images, achieving high accuracy through careful model architecture design and training optimization.

## Features

- Custom CNN architecture optimized for playing card classification
- Support for 53 card classes (52 standard cards plus joker)
- Pre-trained model weights included for immediate inference
- Data preprocessing and augmentation pipeline
- Training and evaluation scripts in Jupyter notebook format
- Early stopping implementation to prevent overfitting

## Project Structure

```
pytorch-poker-cards-classifier-cnn/
├── dataset/                      # Training and validation datasets
├── card-classifier.ipynb         # Main notebook with model implementation
├── best_card_classifier.pth      # Pre-trained model weights
└── .gitignore                    # Git ignore configuration
```

## Requirements

- Python 3.6 or higher
- PyTorch
- torchvision
- NumPy
- Matplotlib
- Jupyter Notebook
- PIL/Pillow

## Installation

1. Clone the repository:

```bash
git clone https://github.com/bacadlo/pytorch-poker-cards-classifier-cnn.git
cd pytorch-poker-cards-classifier-cnn
```

1. Install required dependencies:

```bash
pip install torch torchvision numpy matplotlib jupyter pillow
```

## Usage

### Training the Model

Open and run the Jupyter notebook:

```bash
jupyter notebook card-classifier.ipynb
```

The notebook contains the complete pipeline including:

- Data loading and preprocessing
- Model architecture definition
- Training loop with validation
- Model evaluation and metrics

### Using Pre-trained Model

The repository includes pre-trained weights (`best_card_classifier.pth`) for immediate inference:

```python
import torch
from model import CardClassifierCNN  # Assuming model class is defined

# Load the model
model = CardClassifierCNN(num_classes=53)
model.load_state_dict(torch.load('best_card_classifier.pth'))
model.eval()

# Perform inference on new images
# ... your inference code here
```

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

## Acknowledgments

- PyTorch team for the deep learning framework
- Playing card dataset contributors
- CNN architecture inspirations from computer vision research

## Contact

For questions or feedback, please open an issue in the repository.