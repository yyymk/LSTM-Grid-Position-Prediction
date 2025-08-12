# WLSTM
A PyTorch implementation of a single-layer Long Short-Term Memory (LSTM) model that predicts the next Transport Block (TB) usage in a given New Radio Vehicle-to-Everything (NR V2X) resource pool is named 'Walter - LSTM (WLSTM)'. 

WLSTM is applicable for Semi-Persistent Scheduling (SPS) based on the multiple steps of  regular historical data.

## Notations

- The slot number in a selection window for Vulnerable Road User (VRU): $\tau$.


- The independent subchannels number in the resource pool: $N$.


- The number of historical data: $t$.


- X-coordinates of the TB usage in the $t$-th selection window corresponding to the subchannel number: $x_t\in[1,2,\cdots,N]$.


- Y-coordinates of the TB usage in the $t$-th selection window corresponding to the  slot number: $y_t\in[1,2,\cdots,\tau]$.


- The position of the TB usage in the $t$-th selection window: $[x_{t}, y_{t}]$.
## Overview
This project trains an LSTM-based model to predict $[x_{t+1}, y_{t+1}]$ given the rule of SPS algorithm and the historical data $[x_{1}, y_{1}],[x_{2}, y_{2}],\cdots,[x_{t}, y_{t}]$. 

The model treats the task as a $(N\times\tau)$-class classification problem, and each class corresponds to one TB in the selection window. 

## Dataset

- Input: Sequences length of $t$ steps, and each step containing two-dimensional data: $[x_{1}, y_{1}],[x_{2}, y_{2}],\cdots,[x_{t}, y_{t}]$.
- Output: $C_{t+1}$ for the $(t+1)$-th selection window: $C_{t+1}=(x_{t+1}-1)\times \tau+y_{t+1}-1$, where $C_{t+1}\in[0,1,2,\cdots,(N\times\tau-1)]$.
- Format: The input dataset is a CSV file, where odd rows contain X-coordinates ($x_1, x_2,\cdots,x_t$ values corresponding to the subchannel number) and even rows contain Y-coordinates ($y_1,y_2,\cdots,y_t$ values corresponding to the  slot number).
  
Example data structure:
| Row Numbuer | Values           |
| --- | ---------------- |
| 1 (X1)  | 1, 1, 3, ...      |
| 2 (Y1)  | 11, 11, 16, ...   |

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
3. After training, the script will output $C_{t+1}$ for the $(t+1)$-th selection window. According to the following equations, the position $[x_{t+1}, y_{t+1}]$ of the TB usage in the $(t+1)$-th selection window can be calculated.
$$
C_{t+1}=(x_{t+1}-1)\times \tau+y_{t+1}-1,\\
x_{t+1}\in[1,2,\cdots,N],\\
y_{t+1}\in[1,2,\cdots,\tau].
$$

5. By comparing the actual position with the predicted one, the accuracy can be obtained.

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

- In our demo with default parameter settings ($\tau=50, N=4$, and $t=50$), the model achieved an accuracy of $23.57%$ on the test set after training.

## License
This project is licensed under the MIT License.
