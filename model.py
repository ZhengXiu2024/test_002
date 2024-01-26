from data import input_train_final, concatenated_train
from data import input_test_final, concatenated_test
# 相关分析
correlations = input_train_final.corrwith(concatenated_train.iloc[:, 0])
print(correlations)
# 建模
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 假设你有两个DataFrame：dataframe1 和 dataframe2
# dataframe1 包含输入变量，dataframe2 包含标签

# 数据预处理
# 假设你的输入变量和标签是列，你需要将它们转换成NumPy数组
X = input_train_final.values
y = concatenated_train.values

# 切分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.145, random_state=42)

# 对输入数据进行标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为PyTorch的Tensor
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test)



# 构建模型
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h_n = self.gru(x)
        output = self.fc(h_n[-1])
        return output

# 定义模型和优化器
input_size = X_train_tensor.shape[2]
hidden_size = 50
output_size = 1

model = GRUModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 10
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

# 评估模型
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs.squeeze(), y_test_tensor)

    # 计算额外的指标
    mae = nn.L1Loss()(test_outputs.squeeze(), y_test_tensor)
    mape = torch.mean(torch.abs((y_test_tensor - test_outputs.squeeze()) / y_test_tensor)) * 100
    rmse = torch.sqrt(test_loss)
    re = mae / torch.mean(y_test_tensor)

    print(f'Mean Squared Error on Test Set: {test_loss.item()}')
    print(f'Mean Absolute Error on Test Set: {mae.item()}')
    print(f'Mean Absolute Percentage Error on Test Set: {mape.item()}%')
    print(f'Root Mean Squared Error on Test Set: {rmse.item()}')
    print(f'Relative Error on Test Set: {re.item()}')

