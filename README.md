Project Overview

This project demonstrates the application of deep learning for image classification using PyTorch. The model is trained to recognize handwritten digits (0-9) from 28x28 pixel grayscale images.
Key Features

High Accuracy: 98.94% on test dataset (9,894/10,000 correct predictions)
Efficient Architecture: CNN with only 421,642 parameters
Fast Training: Converges in just 3 epochs
Comprehensive Visualization: Includes confusion matrices, prediction samples, and error analysis

 Results
MetricValueTest Accuracy98.94%Training Samples60,000Test Samples10,000Epochs3Training Time~3 minutes (CPU)
Per-Digit Accuracy
All digits achieve 98-99% individual accuracy, with consistent performance across the entire dataset.
 Model Architecture
MNISTNet(
  (conv1): Conv2d(1, 32, kernel_size=3, padding=1)
  (conv2): Conv2d(32, 64, kernel_size=3, padding=1)
  (pool): MaxPool2d(kernel_size=2, stride=2)
  (fc1): Linear(3136, 128)
  (fc2): Linear(128, 10)
  (dropout): Dropout(p=0.25)
)
Architecture Details:

2 Convolutional layers with ReLU activation
Max pooling for spatial dimension reduction
2 Fully connected layers
Dropout (25%) for regularization
Total parameters: 421,642

🚀 Getting Started
Prerequisites
bashPython 3.8+
PyTorch
torchvision
numpy
matplotlib
Installation

Clone the repository:

bashgit clone https://github.com/yourusername/mnist-digit-recognition.git
cd mnist-digit-recognition

Install dependencies:

bashpip install torch torchvision numpy matplotlib

Run the notebook:

bashjupyter notebook MNIST_Digit_Recognition_Samson.ipynb
📁 Project Structure
mnist-digit-recognition/
│
├── MNIST_Digit_Recognition_Samson.ipynb   # Main notebook
├── README.md                               # Project documentation
├── requirements.txt                        # Python dependencies
└── images/                                 # Sample outputs and visualizations
    ├── predictions.png
    ├── confusion_matrix.png
    └── error_analysis.png
💻 Usage
Training the Model
python# Load data
train_loader, test_loader = load_mnist_data()

# Initialize model
model = MNISTNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(3):
    train(model, train_loader, optimizer, epoch)
    test(model, test_loader)
Making Predictions
python# Load trained model
model.load_state_dict(torch.load('mnist_model.pth'))

# Predict on new image
prediction = model(image)
digit = prediction.argmax(dim=1)
📈 Training Process
The model shows rapid convergence:

Epoch 1: 98.79% test accuracy
Epoch 2: 98.97% test accuracy
Epoch 3: 98.94% test accuracy (final)

Loss decreases consistently from 2.3 to 0.02 over training batches.
🔍 Error Analysis
The 106 misclassified images reveal interesting patterns:

Most errors occur on genuinely ambiguous handwriting
Common confusions: 3↔9, 4↔9, 5↔3
Even human annotators might struggle with these cases

🎓 What I Learned

CNNs are highly effective for spatial pattern recognition
Simple architectures can achieve excellent results on structured data
Proper normalization and data preprocessing are crucial
Dropout prevents overfitting in small networks
Most classification errors occur on ambiguous samples

Future Improvements

 Data augmentation (rotation, scaling, shearing)
 Deeper network architecture (ResNet-style)
 Ensemble methods for improved accuracy
 Transfer learning from larger pre-trained models
 Real-time digit recognition from camera input
 Deploy as web application using Flask/FastAPI

 Technologies Used

PyTorch: Deep learning framework
torchvision: Dataset loading and transformations
NumPy: Numerical computations
Matplotlib: Data visualization
scikit-learn: Confusion matrix generation
Google Colab: Cloud development environment

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
👤 Author
Samson Oluwadare


Acknowledgments

MNIST dataset creators: Yann LeCun, Corinna Cortes, and Christopher Burges
PyTorch team for the excellent framework
Google Colab for free GPU resources

