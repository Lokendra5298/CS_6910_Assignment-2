# CS_6910_Assignment2
## PART-A
### Introduction
This repository contains code for a Convolutional Neural Network (CNN) implemented in PyTorch for image classification tasks. The CNN is designed to classify images from the iNaturalist dataset into one of 10 categories. This README provides an overview of the code structure, how to run the experiments, and the functionality provided.

### Code Structure
The repository contains the following files:
1. `cnn_model.py`: Defines the CNN architecture (`CNN` class) and functions for training and testing the model.
2. `hyperparameter_tuning.py`: Performs hyperparameter tuning using the Weights and Biases (WandB) platform.
3. `visualization.py`: Contains functions for visualizing model filters.

### Setup and Dependencies
Before running the code, ensure you have Python installed along with the necessary dependencies. You can install the dependencies using the following command:
```
pip install -r requirements.txt
```

### Usage
1. **Training and Evaluation**:
   - Run the `cnn_model.py` script to train and evaluate the CNN model. This script trains the model on the training set and evaluates its performance on the validation set.
   - Example command:
     ```
     python cnn_model.py
     ```
   - **Accuracy on Best Model**: 34.07%
   - **Best Validation Accuracy**: 34.07%
   - **Best Training Accuracy**: 32.95%
   - **Best Test Accuracy**: 25.16%

2. **Hyperparameter Tuning**:
   - Hyperparameter tuning is performed using the `hyperparameter_tuning.py` script with the help of the WandB platform.
   - Hyperparameters tuned:
     - Filter sizes
     - Activation functions
     - Number of dense layers
     - Batch normalization
     - Filter organization
     - Dropout rate
     - Data augmentation
   - The script utilizes a sweep configuration to search for optimal hyperparameters efficiently.


### Results
- **Accuracy on Best Model**: 34.07%
- **Best Validation Accuracy**: 34.07%
- **Best Training Accuracy**: 32.95%
- **Best Test Accuracy**: 25.16%

### License
This project is licensed under the MIT License - see the `LICENSE` file for details.

### Acknowledgments
- The code utilizes the PyTorch deep learning framework.
- Weights and Biases (WandB) platform is used for hyperparameter tuning and experiment tracking. Sweep configuration is employed to efficiently search for optimal hyperparameters.
