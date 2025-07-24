# Fashion MNIST Classification

A neural network project for classifying fashion items using the Fashion MNIST dataset with a simple feedforward neural network implemented from scratch using NumPy.

## 📋 Project Overview

This project implements a 2-layer feedforward neural network from scratch using NumPy to classify 10 different types of fashion items from the Fashion MNIST dataset. The implementation includes custom forward propagation, backpropagation, and visualization functions.

## 📊 Dataset Requirements

This project expects the Fashion MNIST dataset in CSV format:
- **File**: `fashion-mnist_train.csv`
- **Format**: First column contains labels (0-9), remaining 784 columns contain pixel values
- **Training samples**: 59,000 (after 1,000 set aside for dev/test)
- **Dev/Test samples**: 1,000

Download from: [Fashion MNIST CSV](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

### Class Labels
| Label | Description |
|-------|-------------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy
pip install matplotlib
pip install pandas
pip install scikit-learn
```

### Installation
```bash
git clone https://github.com/yourusername/fashion-mnist-classification
cd fashion-mnist-classification
pip install -r requirements.txt
```

## 📁 Project Structure
```
fashion-mnist-classification/
│
├── fashion-mnist_train.csv      # Dataset file (download required)
├── fashion_mnist.py             # Main implementation file
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔧 Usage

### Quick Start
```python
python fashion_mnist.py
```

Make sure you have `fashion-mnist_train.csv` in the same directory.

### Code Structure
The main functions include:

- `initparams()`: Initialize weights and biases
- `forward_prop()`: Forward propagation through the network
- `backward_prop()`: Compute gradients via backpropagation
- `update_params()`: Update weights using gradient descent
- `training()`: Main training loop with evaluation
- `visualize_predictions()`: Display sample predictions and confusion matrix

## 🏗️ Model Architecture

The Neural Network consists of:

```
Input Layer: 784 neurons (28×28 flattened pixels)
    ↓
Hidden Layer: 10 neurons + ReLU activation
    ↓  
Output Layer: 10 neurons + Softmax activation
```

### Implementation Details
- **Weight Initialization**: Random normal (mean=0, std=0.01)
- **Optimizer**: Gradient Descent
- **Learning Rate**: 0.3
- **Epochs**: 1,000
- **Loss Function**: Cross Entropy (via one-hot encoding)
- **Data Preprocessing**: Pixel normalization (÷255)
- **Total Parameters**: 7,950 + 110 = 8,060

## 📊 Results

### Output Features
- **Sample Predictions**: Visual display of 8 test images with actual vs predicted labels
- **Confusion Matrix**: Heatmap showing classification performance across all classes
- **Accuracy Metrics**: Final accuracy percentage on dev/test set (1,000 samples)

### Typical Performance
- **Test Accuracy**: ~85-90% (depending on random initialization)
- **Training Samples**: 59,000
- **Test Samples**: 1,000

### Visualization
The program automatically displays:
1. Grid of sample predictions (2×4 layout)
2. Confusion matrix with color-coded performance
3. Accuracy statistics in terminal output

## 🔍 Key Features

### Custom Implementation
- **From-scratch neural network**: No high-level ML frameworks
- **Manual backpropagation**: Hand-coded gradient computation
- **Vectorized operations**: Efficient NumPy matrix operations
- **Custom visualization**: Matplotlib-based prediction display

### Data Processing
- **Automatic normalization**: Pixel values scaled to [0,1]
- **One-hot encoding**: Labels converted for cross-entropy loss
- **Train/dev split**: 59k/1k split from original training data
