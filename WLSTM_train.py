import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. 数据预处理
grid_x, grid_y = 4, 50
seq_len = 50
file_path = './RP_power_data.csv'
data = pd.read_csv(file_path, header=None)
x_coords = data.iloc[0::2].to_numpy(dtype=np.int64)  
y_coords = data.iloc[1::2].to_numpy(dtype=np.int64)  
x_hist = x_coords[:, :seq_len] - 1     
y_hist = y_coords[:, :seq_len] - 1
x_next = x_coords[:, seq_len] - 1      
y_next = y_coords[:, seq_len] - 1
assert x_hist.min() >= 0 and x_hist.max() < grid_x
assert y_hist.min() >= 0 and y_hist.max() < grid_y
assert x_next.min() >= 0 and x_next.max() < grid_x
assert y_next.min() >= 0 and y_next.max() < grid_y
X = np.stack([x_hist, y_hist], axis=-1).astype(np.float32)
y = (y_next * grid_x + x_next).astype(np.int64)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train_tensor = torch.from_numpy(X_train)          
y_train_tensor = torch.from_numpy(y_train)          
X_test_tensor  = torch.from_numpy(X_test)
y_test_tensor  = torch.from_numpy(y_test)

# 2. 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_classes=200):  
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :]) 
        return output
    
# 3. 训练模型
model = LSTMModel(input_size=2, hidden_size=128, num_classes=200)  
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
batch_size = 32

for epoch in range(epochs):
    model.train()
    perm = torch.randperm(X_train_tensor.size(0))  
    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = perm[i:i + batch_size]
        batch_x = X_train_tensor[indices]
        batch_y = y_train_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "./model.pth")

# 4. 评估模型
model.eval()
with torch.no_grad():
    test_output = model(X_test_tensor)
    _, predicted = torch.max(test_output, 1)
    accuracy = (predicted == y_test_tensor).float().mean()
    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")

# 5. 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(y_test[:100], label="True Position IDs")
plt.plot(predicted[:100], label="Predicted Position IDs", linestyle='dashed')
plt.legend()
plt.title("True vs Predicted Position IDs (First 100 Points)")
plt.show()
