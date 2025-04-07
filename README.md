# Recurrent Neural Networks (RNN) from Scratch - Weather Prediction

This Jupyter notebook demonstrates building a Recurrent Neural Network (RNN) from scratch to predict the maximum temperature (`tmax_tomorrow`) using weather data. The implementation includes forward/backward propagation, gradient updates, and training loops without relying on deep learning frameworks like TensorFlow or PyTorch.

## Overview
- **Objective**: Predict the next day's maximum temperature (`tmax_tomorrow`) using historical weather data.
- **Dataset**: Clean weather dataset from Kaggle, containing daily records of `tmax`, `tmin`, `rain`, and `tmax_tomorrow`.
- **Key Steps**:
  - Manual implementation of RNN forward and backward passes.
  - Custom loss calculation (Mean Squared Error) and gradient updates.
  - Data scaling and sequential splitting into train/validation/test sets.
  - Training loop with periodic validation loss evaluation.

## Dataset
- **Source**: [Kaggle: Clean Weather Dataset](https://www.kaggle.com/datasets/sahilchambyal/clean-weather)
- **Features**:
  - `tmax`: Maximum temperature (°F).
  - `tmin`: Minimum temperature (°F).
  - `rain`: Precipitation (inches).
  - `tmax_tomorrow`: Target variable (next day's maximum temperature).

## Implementation Details
### Model Architecture
- **Input Layer**: 3 units (for `tmax`, `tmin`, `rain`).
- **RNN Layer**: 4 hidden units with `tanh` activation.
- **Output Layer**: 1 unit (predicted `tmax_tomorrow`).

### Key Functions
- **`init_params`**: Initializes RNN weights and biases using Xavier initialization.
- **`forward`**: Computes hidden states and predictions for a sequence.
- **`backward`**: Performs backpropagation through time (BPTT) to update weights.

### Training
- **Epochs**: 250.
- **Learning Rate**: `1e-5`.
- **Sequence Length**: 7 days (sliding window).
- **Loss Function**: Mean Squared Error (MSE).
