# WLSTM
A PyTorch implementation of a single-layer LSTM model that predicts the next resource block usage in a 4×50 grid, based on the previous 50 steps of trajectory data, applicable for V2X communication in SPS.
## Overview
This project trains an LSTM-based neural network to predict the next (x, y) position on a 4×50 grid given the previous 50 positions. The model treats the task as a **200-class classification problem** (4×50 grid cells). The goal is to predict which cell in the grid the next position will fall into, based on a sequence of prior positions.

## Dataset

The dataset consists of pairs of trajectory coordinates, where:
- Input: Sequences of length 50, each step containing 2 coordinates [x, y] (0-based indices).
- Output: The class ID for the next position, calculated as `class_id = y * grid_x + x` (range 0-199).
- Format: The dataset is a CSV file, where odd rows contain X coordinates and even rows contain Y coordinates.
  
Example data structure:
| Row | Values           |
| --- | ---------------- |
| 1 (X1)  | 2, 1, 3, ...      |
| 2 (Y1)  | 10, 11, 12, ...   |

## Installation

### 1. Clone the repository:
git clone https://github.com/yyymk/WLSTM.git

### 2. Install dependencies:
pip install -r requirements.txt

torch
numpy
pandas
scikit-learn
matplotlib

## Usage
1. Prepare your dataset in CSV format similar to RP_power_data.csv (included in the repo).
2. Run the training script:
   python3 WLSTM_train.py
3. After training, the script will output accuracy and a plot showing predicted vs true positions.

## Command-line Arguments
| Argument        | Default | Description                    |
| --------------- | ------- | ------------------------------ |
| `--data`        | None    | Path to your dataset CSV file  |
| `--epochs`      | 50      | Number of training epochs      |
| `--batch_size`  | 32      | Batch size                     |
| `--hidden_size` | 128     | Number of hidden units in LSTM |

## Model Architecture
- The model is a single-layer LSTM followed by a fully connected layer for classification:
- Input: Sequences of 50 time steps, each with 2 features [x, y].
- LSTM Layer: Hidden size of 128, number of layers is 1.
- Fully Connected Layer: Projects LSTM output to **200 classes** (grid positions).
- Output: 200 class probabilities for the next position.
- Loss: CrossEntropyLoss for multi-class classification.

## Trained Model（WLSTM）
- Dataset Size: The model was trained on **5940** sequences of trajectory data.
- Data Split: The dataset was split into training and testing sets with a ratio of **80% training and 20% testing**.
- Training Parameters:

    Epochs: 50

    Batch Size: 32

    Hidden Size: 128 (number of units in the LSTM layer)

    Optimizer: Adam optimizer with a learning rate of 0.001.

    Loss Function: CrossEntropyLoss for multi-class classification.

- The model achieved an accuracy of 23.57% on the test set after training (adjust with your actual accuracy).

## License
This project is licensed under the MIT License.
