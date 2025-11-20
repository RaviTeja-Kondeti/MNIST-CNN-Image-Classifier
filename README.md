# MNIST CNN Image Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

A deep learning project implementing a Convolutional Neural Network (CNN) for multi-class image classification on the MNIST handwritten digits dataset using TensorFlow and Keras.

## 📊 Project Overview

This project demonstrates the implementation of a CNN architecture for recognizing handwritten digits (0-9) from the MNIST dataset. The model achieves high accuracy through carefully designed convolutional layers, pooling operations, and regularization techniques.

### Key Features

- ✅ **Multi-class Classification**: Classifies handwritten digits into 10 classes (0-9)
- ✅ **Deep Learning Architecture**: Custom CNN with multiple convolutional and pooling layers
- ✅ **High Performance**: Achieves excellent accuracy on test data
- ✅ **Well-Documented Code**: Clear explanations and comments throughout
- ✅ **Visualization**: Training metrics and model performance visualization

## 💻 Technologies Used

- **Python 3.8+**
- **TensorFlow 2.x**
- **Keras** (High-level Neural Networks API)
- **NumPy** (Numerical computing)
- **Matplotlib** (Data visualization)
- **Jupyter Notebook** (Interactive development)

## 💾 Dataset

The MNIST database contains:
- **Training Set**: 60,000 grayscale images (28x28 pixels)
- **Test Set**: 10,000 grayscale images (28x28 pixels)
- **Classes**: 10 categories (digits 0-9)
- **Image Format**: 28x28 pixel grayscale images

## 🏛️ Model Architecture

The CNN architecture includes:

1. **Input Layer**: 28x28 grayscale images
2. **Convolutional Layers**: Multiple Conv2D layers with ReLU activation
3. **Pooling Layers**: MaxPooling2D for dimensionality reduction
4. **Dropout Layers**: Regularization to prevent overfitting
5. **Flatten Layer**: Converts 2D matrices to 1D vector
6. **Dense Layers**: Fully connected layers for classification
7. **Output Layer**: 10 neurons with Softmax activation for multi-class classification

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.8
tensorflow >= 2.0
keras >= 2.0
numpy
matplotlib
jupyter
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/RaviTeja-Kondeti/MNIST-CNN-Image-Classifier.git
cd MNIST-CNN-Image-Classifier
```

2. Install required packages:
```bash
pip install tensorflow keras numpy matplotlib jupyter
```

3. Run the Jupyter Notebook:
```bash
jupyter notebook
```

## 📚 Usage

1. Open the notebook in Jupyter
2. Run cells sequentially to:
   - Load and preprocess the MNIST dataset
   - Build the CNN model architecture
   - Train the model on training data
   - Evaluate performance on test data
   - Visualize results and predictions

## 📈 Results

The model demonstrates:
- **High Accuracy**: Achieves excellent classification accuracy on test set
- **Fast Training**: Efficient training with optimized architecture
- **Robust Performance**: Consistent results across different runs
- **Low Loss**: Minimized loss through proper regularization

## 📝 Model Training

The model is trained with:
- **Optimizer**: Adam optimizer for efficient gradient descent
- **Loss Function**: Categorical crossentropy for multi-class classification
- **Metrics**: Accuracy tracking during training
- **Epochs**: Multiple epochs with validation monitoring
- **Batch Size**: Optimized batch size for efficient training

## 🔍 Project Structure

```
MNIST-CNN-Image-Classifier/
│
├── 23302_RaviTejaKondetiAssignment2_github.ipynb  # Main notebook
├── README.md                                      # Project documentation
└── requirements.txt                               # Python dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**RaviTeja Kondeti**

- GitHub: [@RaviTeja-Kondeti](https://github.com/RaviTeja-Kondeti)

## 🚀 Acknowledgments

- MNIST Database creators for the dataset
- TensorFlow and Keras teams for the excellent deep learning frameworks
- The open-source community for continuous support and inspiration

---

⭐ If you found this project helpful, please consider giving it a star!
