from datetime import date
import os
import pandas as pd
from taosrest import RestClient
import warnings
import numpy as np

warnings.simplefilter("ignore", UserWarning)

# 气象数据
client = RestClient("http://192.168.0.203:6041", database='weather_data_db')
res = client.sql("select ts,avg(dswrfsfc),avg(rh2m),avg(tmp2m),avg(ws100m) from w_45d_1h_latest where ts >= "
                 "'2023-01-02 00:00:00.000' and ts < '2024-01-01 00:00:00.000' group by ts order by ts")
# print(res)
columns = [col[0] for col in res['column_meta']]
df = pd.DataFrame(res['data'], columns=columns)
# print(df)
# 异常值检测和处理
print('Nan:', np.isnan(df).values.any())
df.set_index('ts', inplace=True)


def filter_function(group):
    return len(group) == 24


filtered_df = df.groupby(df.index.date).filter(filter_function)
# print(filtered_df)
print('Nan:', np.isnan(filtered_df).values.any())
# nan_rows = filtered_df[filtered_df.isnull().any(axis=1)]
# print(nan_rows)
# print(filtered_df.columns)
selected_data = filtered_df['2023-02-01 00:00:00.000':'2023-02-28 23:00:00.000']
avg_tmp2m = selected_data['avg(tmp2m)'].mean()
filtered_df['avg(tmp2m)'] = filtered_df['avg(tmp2m)'].fillna(avg_tmp2m)
# print(filtered_df)
print('Nan:', np.isnan(filtered_df).values.any())

# 日期数据量化
filtered_df['Quantized_Date'] = 0
filtered_df.loc[filtered_df.index.weekday < 5, 'Quantized_Date'] = 1
filtered_df.loc[filtered_df.index.weekday >= 5, 'Quantized_Date'] = 2
holidays = [
    date(2023, 1, 2),  # 元旦
    date(2023, 1, 21),  # 春节
    date(2023, 1, 22),  # 春节
    date(2023, 1, 23),  # 春节
    date(2023, 1, 24),  # 春节
    date(2023, 1, 25),  # 春节
    date(2023, 1, 26),  # 春节
    date(2023, 1, 27),  # 春节
    date(2023, 4, 5),  # 清明节
    date(2023, 4, 29),  # 劳动节
    date(2023, 4, 30),  # 劳动节
    date(2023, 5, 1),  # 劳动节
    date(2023, 5, 2),  # 劳动节
    date(2023, 5, 3),  # 劳动节
    date(2023, 6, 22),  # 端午节
    date(2023, 6, 23),  # 端午节
    date(2023, 6, 24),  # 端午节
    date(2023, 9, 29),  # 中秋节、国庆节
    date(2023, 9, 30),  # 中秋节、国庆节
    date(2023, 10, 1),  # 中秋节、国庆节
    date(2023, 10, 2),  # 中秋节、国庆节
    date(2023, 10, 3),  # 中秋节、国庆节
    date(2023, 10, 4),  # 中秋节、国庆节
    date(2023, 10, 5),  # 中秋节、国庆节
    date(2023, 10, 6),  # 中秋节、国庆节
    date(2023, 12, 30),  # 元旦
    date(2023, 12, 31),  # 元旦
]
# holidays_index = pd.to_datetime(holidays)
idx = filtered_df.index
filtered_df['date'] = idx.date
filtered_df.loc[filtered_df['date'].isin(holidays), 'Quantized_Date'] = 3
# print(filtered_df)

# 划分训练集和测试集
condition_1 = filtered_df['date'] < pd.to_datetime('2023-10-16').date()
condition_2 = ~condition_1
input_train = filtered_df[condition_1]
input_test = filtered_df[condition_2]
# print("训练集输入:")
# print(input_train)
# print("测试集输入:")
# print(input_test)
# 训练集插值
new_index1 = pd.date_range(start=input_train.index.min(), end=input_train.index.max(), freq='15T')
input_train_final = input_train.reindex(new_index1, method='ffill')
last_index1 = new_index1[-1]
last_data1 = input_train_final.loc[last_index1]
input_train_final.loc[last_index1 + pd.DateOffset(minutes=15)] = last_data1
input_train_final.loc[last_index1 + pd.DateOffset(minutes=30)] = last_data1
input_train_final.loc[last_index1 + pd.DateOffset(minutes=45)] = last_data1
print(input_train_final)
# 1h测试集插值
new_index2 = pd.date_range(start=input_test.index.min(), end=input_test.index.max(), freq='15T')
input_test_final1 = input_test.reindex(new_index2, method='ffill')
last_index2 = new_index2[-1]
last_data2 = input_test_final1.loc[last_index2]
input_test_final1.loc[last_index2 + pd.DateOffset(minutes=15)] = last_data2
input_test_final1.loc[last_index2 + pd.DateOffset(minutes=30)] = last_data2
input_test_final1.loc[last_index2 + pd.DateOffset(minutes=45)] = last_data2
print(input_test_final1)

# 15m测试集气象数据获取和处理
res1 = client.sql("select ts,avg(dswrfsfc),avg(rh2m),avg(tmp2m),avg(ws100m) from w_15d_15min_latest where ts >= "
                  "'2023-11-13 00:00:00.000' and ts < '2024-01-01 00:00:00.000' group by ts order by ts")
# print(res)
columns1 = [col[0] for col in res['column_meta']]
df1 = pd.DataFrame(res1['data'], columns=columns1)
print(df1)

input_test_final1_reset = input_test_final1.reset_index(drop=True)
last_column = input_test_final1_reset.iloc[:, -2]
input_test_final = pd.concat([df1, last_column], axis=1)
print(input_test_final)

# 负荷数据处理
# def process_excel_file(file_path, idx):
#     try:
#         load = pd.read_excel(file_path)
#         load['日期时间'] = load['日期'].astype(str) + ' ' + load['时间'].astype(str)
#         load['日期时间'] = load['日期时间'].str.replace('24:00:00', '23:59:59')
#         # 尝试传递format参数
#         load['日期时间'] = pd.to_datetime(load['日期时间'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
#         load = load.drop(['日期', '时间'], axis=1)
#         total_load_aggregated = load.groupby('日期时间').sum()
#         output_file_name = f'aggregated_{idx}_{os.path.basename(file_path)}'
#         output_file_path = os.path.join(folder_path, output_file_name)
#         total_load_aggregated.to_excel(output_file_path)
#         print(f'处理完成：{os.path.basename(file_path)}，保存到：{output_file_path}')
#     except Exception as e:
#         print(f'处理文件时出现错误：{os.path.basename(file_path)}，错误信息：{str(e)}')
#
#
# folder_path = r'E:\业务\3.负荷预测\数据\代理用户23年历史用电量数据'
# excel_files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]
# for idx, file in enumerate(excel_files, start=1):
#     file_path = os.path.join(folder_path, file)
#     process_excel_file(file_path, idx)

# 数据拼接
folder_path = r'E:\业务\3.负荷预测\数据\代理用户23年历史用电量数据'
excel_files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]
# 分别筛选出11月和12月的文件和其他文件
nov_dec_files = [file for file in excel_files if
                 file.startswith('aggregated_11_2023年11月') or file.startswith('aggregated_12_2023年12月')]
other_files = [file for file in excel_files if file not in nov_dec_files]
load_test = []
for file in sorted(nov_dec_files):
    file_path = os.path.join(folder_path, file)
    df = pd.read_excel(file_path)
    load_test.append(df)
concatenated_test = pd.concat(load_test)
load_train = []
for file in sorted(other_files):
    file_path = os.path.join(folder_path, file)
    df = pd.read_excel(file_path)
    load_train.append(df)
concatenated_train = pd.concat(load_train)
# print("测试集负荷数据:")
# print(concatenated_test)
# print("训练集负荷数据:")
# print(concatenated_train)
concatenated_train = concatenated_train.sort_values(by='日期时间')
# print(concatenated_train)

# 相关分析
input_train_final = input_train_final.drop(columns=['date'])
input_train_final = input_train_final.reset_index(drop=True)
correlations = input_train_final.corrwith(concatenated_train.iloc[:, -1])
print(correlations)

# 建立模型
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# input_test_final1 = input_test_final1.drop(columns=['date'])
# input_test_final1 = input_test_final1.reset_index(drop=True)
# X = pd.concat([input_train_final, input_test_final1], axis=0)
# y = pd.concat([concatenated_train, concatenated_test], axis=0)
# 数据标准化, 划分训练集和测试集
concatenated_train_last_column = concatenated_train.iloc[:, -1].reset_index(drop=True)
input_train_final2 = pd.concat([input_train_final, concatenated_train_last_column], axis=1)
# print(input_train_final2.columns)
print(input_train_final2.shape)


def create_rolling_window_data(data, window_size, target_size):
    X, y = [], []
    for i in range(len(data) - window_size - target_size + 1):
        window = data.iloc[i:i+window_size]
        target = data.iloc[i+window_size:i+window_size+target_size]['历史用电量']
        X.append(window.values)
        y.append(target.values)
    return np.array(X), np.array(y)


window_size = 30  # 30天的滚动窗口
target_size = 3  # 预测未来5天的负荷
X, y = create_rolling_window_data(input_train_final2, window_size, target_size)
print(X.shape)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_flattened = X.reshape(X.shape[0], -1)
X_normalized = scaler_X.fit_transform(X_flattened)
y_normalized = scaler_y.fit_transform(y[:, -1].reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.145, random_state=42)
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# 构建模型
class MultiLayerGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(MultiLayerGRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, :])
        return out


# 初始化模型和优化器
input_size = X_train.shape[1]
hidden_size = 50  # 隐藏单元数
output_size = target_size  # 输出尺寸
num_layers = 4  # GRU的层数，可以根据需要调整
model = MultiLayerGRUModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 训练模型
epochs = 50
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

# 在测试集上评估模型
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs.squeeze(), y_test_tensor)

    # 进行逆标准化
    test_outputs_unscaled = scaler_y.inverse_transform(test_outputs.numpy().reshape(-1, target_size))
    y_test_unscaled = scaler_y.inverse_transform(y_test_tensor.numpy())

    # 打印评估指标
    mae = nn.L1Loss()(test_outputs.squeeze(), y_test_tensor)
    mape = torch.mean(torch.abs((y_test_tensor - test_outputs.squeeze()) / y_test_tensor)) * 100
    rmse = torch.sqrt(test_loss)
    re = mae / torch.mean(y_test_tensor)

    print(f'Mean Squared Error on Test Set: {test_loss.item()}')
    print(f'Mean Absolute Error on Test Set: {mae.item()}')
    print(f'Root Mean Squared Error on Test Set: {rmse.item()}')
    print(f'Relative Error on Test Set: {re.item()}')

    # 计算每隔96个值的MAPE
    interval = 96
    mape_values = []
    for i in range(0, len(test_outputs_unscaled), interval):
        end_idx = min(i + interval, len(test_outputs_unscaled))
        true_vals = y_test_unscaled[i:end_idx]
        pred_vals = test_outputs_unscaled[i:end_idx]
        mape_i = np.mean(np.abs((true_vals - pred_vals) / true_vals)) * 100
        mape_values.append(mape_i)

    # 打印每隔96个值的MAPE
    for i, mape_i in enumerate(mape_values):
        print(f'MAPE for interval {i + 1}: {mape_i}%')

print("Zero values in predicted values:", torch.sum(test_outputs.squeeze() == 0).item())
print("Zero values in true values:", torch.sum(y_test_tensor == 0).item())
