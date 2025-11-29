import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

#子任务一:数据准备与预处理
def convert_raw_to_flow_data(raw_file_path="cleaned_data.csv"):
    """
    把原始出租车接单数据转换为连续的10分钟间隔交通流量数据
    :param raw_file_path: 原始数据文件路径
    :return: 连续时序的流量数据（time_interval, traffic_count）
    """
    # 1. 读取原始数据并解析时间
    print(f"正在读取原始数据：{raw_file_path}")
    data = pd.read_csv(raw_file_path, parse_dates=["lpep_pickup_datetime"])
    print(
        f"原始数据形状：{data.shape}，时间范围：{data['lpep_pickup_datetime'].min()} 至 {data['lpep_pickup_datetime'].max()}")

    # 2. 按10分钟间隔统计流量（接单量=流量）
    data["time_interval"] = data["lpep_pickup_datetime"].dt.floor("10T")  # 时间向下取整到10分钟间隔
    flow_data = data.groupby("time_interval").size().reset_index(name="traffic_count")  # 统计每组流量
    print(f"10分钟间隔统计完成，有效数据组数：{flow_data.shape[0]}")

    # 3. 补全缺失的10分钟间隔（确保时序连续，缺失值填0）
    full_time_index = pd.date_range(
        start=flow_data["time_interval"].min(),
        end=flow_data["time_interval"].max(),
        freq="10T"  # 频率：10分钟
    )
    full_flow_data = pd.DataFrame({"time_interval": full_time_index})
    full_flow_data = full_flow_data.merge(flow_data, on="time_interval", how="left")
    full_flow_data["traffic_count"] = full_flow_data["traffic_count"].fillna(0).astype(int)  # 缺失流量填0并转为整数
    print(f"补全后连续时序数据形状：{full_flow_data.shape}，无时间断层")

    return full_flow_data


# 执行原始数据转换
# 生成连续的10分钟流量数据
full_flow_data = convert_raw_to_flow_data("cleaned_data.csv")

# ===================== 数据集划分（保持时序性，不打乱） =====================
train_data, test_data = train_test_split(full_flow_data, test_size=0.2, shuffle=False)
print(f"\n数据集划分完成：")
print(f"训练集：{train_data.shape} 条记录（{train_data['time_interval'].min()} 至 {train_data['time_interval'].max()}）")
print(f"测试集：{test_data.shape} 条记录（{test_data['time_interval'].min()} 至 {test_data['time_interval'].max()}）")

# 保存划分后的数据集（供后续模型训练使用）
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)
print("训练集已保存为：train_data.csv")
print("测试集已保存为：test_data.csv")

#可视化：完整流量趋势图
plt.figure(figsize=(16, 8))
plt.plot(
    full_flow_data["time_interval"],
    full_flow_data["traffic_count"],
    color='#2E86AB',
    linewidth=1.2,
    alpha=0.8,
    label='10分钟交通流量'
)
plt.title('交通流量时间序列趋势图（原始数据转换后）', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('时间', fontsize=12)
plt.ylabel('交通流量计数（接单量）', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('traffic_flow_trend.png', dpi=300, bbox_inches='tight')
plt.close()
print("流量趋势图已保存为：traffic_flow_trend.png")


# 特征工程（为模型训练准备特征）
def create_time_features(df):
    """创建时间相关特征，适配后续模型训练"""
    df_features = df.copy()
    df_features['hour'] = df_features['time_interval'].dt.hour  # 小时（0-23）
    df_features['minute'] = df_features['time_interval'].dt.minute  # 分钟（0-59）
    df_features['dayofweek'] = df_features['time_interval'].dt.dayofweek  # 星期几（0-6）
    df_features['dayofmonth'] = df_features['time_interval'].dt.day  # 每月日期（1-31）
    df_features['is_weekend'] = df_features['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)  # 是否周末（0/1）
    return df_features


# 为训练集和测试集添加特征
train_features = create_time_features(train_data)
test_features = create_time_features(test_data)

# 分离特征（X）和目标变量（y）
X_train = train_features.drop(['time_interval', 'traffic_count'], axis=1)
y_train = train_features['traffic_count']
X_test = test_features.drop(['time_interval', 'traffic_count'], axis=1)
y_test = test_features['traffic_count']

print(f"\n特征工程完成，准备进入模型训练：")
print(f"训练集特征形状：{X_train.shape}，目标变量形状：{y_train.shape}")
print(f"测试集特征形状：{X_test.shape}，目标变量形状：{y_test.shape}")
print(f"特征列表：{list(X_train.columns)}")


# 子任务2：线性回归模型的构建与评估
print("=== 子任务2：线性回归模型的构建与评估 ===")

# 2.1 训练线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 2.2 在测试集上进行预测
y_pred_lr = lr_model.predict(X_test)

# 2.3 计算模型性能指标
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"线性回归模型性能指标:")
print(f"- 均方误差 (MSE): {mse_lr:.4f}")
print(f"- 决定系数 (R²): {r2_lr:.4f}")

# 2.4 查看特征重要性（系数）
feature_importance_lr = pd.DataFrame({
    '特征': X_train.columns,
    '系数': lr_model.coef_
}).sort_values(by='系数', ascending=False)

print(f"\n线性回归模型特征系数:")
print(feature_importance_lr)

# 2.5 绘制实际值与预测值的对比图
plt.figure(figsize=(16, 8))

# 采样
sample_size = min(893, len(y_test))
time_index = y_test.index[:sample_size]

plt.plot(time_index, y_test.values[:sample_size],
         color='#A23B72', linewidth=2, label='实际流量', alpha=0.8)
plt.plot(time_index, y_pred_lr[:sample_size],
         color='#F18F01', linewidth=2, linestyle='--', label='预测流量', alpha=0.8)

plt.title('线性回归模型：交通流量实际值与预测值对比',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('时间', fontsize=12)
plt.ylabel('交通流量计数', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('linear_regression_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n子任务2完成：线性回归模型训练、评估和可视化已完成，图片已保存")


# 子任务3：决策树模型的构建与评估
print("=== 子任务3：决策树模型的构建与评估 ===")

# 3.1 训练决策树回归模型
dt_model = DecisionTreeRegressor(random_state=42, max_depth=8, min_samples_split=20)
dt_model.fit(X_train, y_train)

# 3.2 在测试集上进行预测
y_pred_dt = dt_model.predict(X_test)

# 3.3 计算模型性能指标
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print(f"决策树回归模型性能指标:")
print(f"- 均方误差 (MSE): {mse_dt:.4f}")
print(f"- 决定系数 (R²): {r2_dt:.4f}")

# 3.4 查看特征重要性
feature_importance_dt = pd.DataFrame({
    '特征': X_train.columns,
    '重要性': dt_model.feature_importances_
}).sort_values(by='重要性', ascending=False)

print(f"\n决策树模型特征重要性:")
print(feature_importance_dt)

# 3.5 绘制实际值与预测值的对比图
plt.figure(figsize=(16, 8))

# 采样所有893个数据点
sample_size = min(893, len(y_test))
time_index = y_test.index[:sample_size]

plt.plot(time_index, y_test.values[:sample_size],
         color='#A23B72', linewidth=2, label='实际流量', alpha=0.8)
plt.plot(time_index, y_pred_dt[:sample_size],
         color='#C73E1D', linewidth=2, linestyle='--', label='预测流量', alpha=0.8)

plt.title('决策树回归模型：交通流量实际值与预测值对比',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('时间', fontsize=12)
plt.ylabel('交通流量计数', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('decision_tree_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 3.6 比较线性回归和决策树模型的性能
print(f"\n=== 模型性能对比 ===")
comparison_df = pd.DataFrame({
    '模型': ['线性回归', '决策树回归'],
    '均方误差 (MSE)': [mse_lr, mse_dt],
    '决定系数 (R²)': [r2_lr, r2_dt]
})

print(comparison_df.to_string(index=False))

# 绘制模型性能对比图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# MSE对比
models = comparison_df['模型']
mse_values = comparison_df['均方误差 (MSE)']
bars1 = ax1.bar(models, mse_values, color=['#F18F01', '#C73E1D'], alpha=0.7)
ax1.set_title('模型MSE对比（值越小越好）', fontsize=14, fontweight='bold')
ax1.set_ylabel('均方误差 (MSE)')
# 在柱状图上添加数值标签
for bar, value in zip(bars1, mse_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
             f'{value:.2f}', ha='center', va='bottom')

# R²对比
r2_values = comparison_df['决定系数 (R²)']
bars2 = ax2.bar(models, r2_values, color=['#F18F01', '#C73E1D'], alpha=0.7)
ax2.set_title('模型R²对比（值越接近1越好）', fontsize=14, fontweight='bold')
ax2.set_ylabel('决定系数 (R²)')
ax2.set_ylim(0, 1)
# 在柱状图上添加数值标签
for bar, value in zip(bars2, r2_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{value:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n子任务3完成：决策树模型训练、评估、可视化及模型对比已完成，图片已保存")
print("\n=== 所有实验任务完成 ===")
print("生成的文件列表:")
print("1. traffic_flow_trend.png - 一个月流量数据折线图")
print("2. linear_regression_comparison.png - 线性回归实际值vs预测值")
print("3. decision_tree_comparison.png - 决策树实际值vs预测值")
print("4. model_comparison.png - 模型性能对比图")