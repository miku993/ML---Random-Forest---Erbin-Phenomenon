import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 设置 matplotlib 字体和解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体，适用于中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取 Excel 文件数据
df = pd.read_excel("D:/大学自学学习资料/数学建模/2024比赛/Code/随机森林回归/data/data.xlsx")

# 删除多余的索引列
df = df.reset_index(drop=True)

# 将年份列中的中文字符“年”去掉，并转换为数值类型
df['年份'] = df['年份'].apply(lambda x: int(x[:-1]) if isinstance(x, str) else np.nan)

# 用线性插值方法填充缺失值
df['商品零售价格指数'] = df['商品零售价格指数'].interpolate(method='linear')
df['失业率%'] = df['失业率%'].interpolate(method='linear')
df['环境质量：可吸入颗粒物年均值（毫克/立方米）'] = df['环境质量：可吸入颗粒物年均值（毫克/立方米）'].interpolate(method='linear')

# 导出预处理后的数据到新的Excel文件
preprocessed_file_path = "D:/大学自学学习资料/数学建模/2024比赛/Code/随机森林回归/data/preprocessed_data.xlsx"
df.to_excel(preprocessed_file_path, index=False)
print(f"预处理后的数据已保存至 {preprocessed_file_path}")

# 保留用于分析的特征和目标变量
features = [
    '接待旅游者人数（万人次）',
    '搜索量',
    '城镇居民人均年消费支出（元）',
    '省外游客占比%',
    '商品零售价格指数',
    '失业率%',
    '环境质量：可吸入颗粒物年均值（毫克/立方米）'
]

# 选择旅游总收入作为目标变量 y，其余作为特征 X
X = df[features]
y = df['旅游总收入（亿元）']

# 数据预处理 - 标准化特征
scaler = StandardScaler()  # 实例化标准化器
X_scaled = scaler.fit_transform(X)  # 标准化特征数据

# 将标准化后的数据转换为 DataFrame 并保留列名
X_scaled = pd.DataFrame(X_scaled, columns=features)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)  # 按 7:3 分割训练集和测试集

# 建立模型
rf = RandomForestRegressor(n_estimators=80, max_depth=20, bootstrap=True, random_state=6)  # 实例化随机森林回归模型
rf.fit(X_train, y_train)  # 训练模型

# 预测测试集结果
y_pred = rf.predict(X_test)  # 用模型预测测试集

# 未来5年预测
future_years = np.arange(2024, 2029)  # 定义未来5年的年份（2024-2028）

# 用回归模型对未来特征进行变化并预测
last_known_data = X.iloc[-5:]  # 获取已知数据中的最后5条记录
future_features_scaled = scaler.transform(last_known_data)  # 标准化未来特征
future_features_scaled = pd.DataFrame(future_features_scaled, columns=features)  # 保留特征名称
future_predictions = rf.predict(future_features_scaled)  # 预测未来特征对应的目标变量

# 将预测结果和未来5年的预测数据合并到 DataFrame 中
prediction_df = pd.DataFrame({
    '年份': np.append(df['年份'].values, future_years),  # 合并现有年份和未来年份
    '旅游总收入（亿元）': np.append(y.values, future_predictions.flatten())  # 合并现有旅游收入和未来预测值
})

# 导出预测的数据到新的Excel文件
predict_file_path = "D:/大学自学学习资料/数学建模/2024比赛/Code/随机森林回归/data/predict_data.xlsx"
prediction_df.to_excel(predict_file_path, index=False)
print(f"预测的数据已保存至 {predict_file_path}")

# 模型评估
mse = mean_squared_error(y_test, y_pred)  # 计算均方误差
rmse = np.sqrt(mse)  # 计算均方根误差
mae = mean_absolute_error(y_test, y_pred)  # 计算平均绝对误差
r2 = r2_score(y_test, y_pred)  # 计算 R^2 得分

# 输出结果
print(f"均方根误差 (RMSE): {rmse:.2f}")
print(f"平均绝对误差 (MAE): {mae:.2f}")
print(f"R^2 得分: {r2:.4f}")

# 绘制模型拟合效果图
plt.figure(figsize=(12, 8))  # 设置图表大小
sorted_indices = np.argsort(df.loc[y_test.index, '年份'])  # 获取按年份排序的索引
x_labels = [f"样本{i+1}({year}年)" for i, year in enumerate(df.loc[y_test.index, '年份'].values[sorted_indices])]  # 创建 x 轴标签
x_ticks = np.arange(len(x_labels))  # 创建 x 轴刻度
plt.plot(x_ticks, y_test.values[sorted_indices], label='实际值', color='skyblue', marker='o')  # 绘制实际值，浅蓝色线
plt.plot(x_ticks, y_pred[sorted_indices], label='预测值', color='orange', marker='x')  # 绘制预测值，橙色线
plt.title('模型拟合效果图', fontsize=18)  # 设置图表标题
plt.xlabel('样本', fontsize=14)  # 设置 x 轴标签
plt.ylabel('旅游总收入（亿元）', fontsize=14)  # 设置 y 轴标签
plt.xticks(ticks=x_ticks, labels=x_labels, fontsize=12, rotation=45)  # 设置 x 轴刻度和标签
plt.yticks(fontsize=12)  # 设置 y 轴刻度
plt.legend(fontsize=14)  # 显示图例
plt.grid(True, linestyle='--', alpha=0.6)  # 设置网格线
plt.tight_layout()  # 调整图表布局
plt.show()  # 显示图表

# 绘制所有特征与旅游总收入的相关性
plt.figure(figsize=(12, 6))  # 设置图表大小
corr_matrix = df.corr()  # 计算数据的相关性矩阵
feature_names = features  # 特征名称列表
corr_with_target = corr_matrix.loc[feature_names, '旅游总收入（亿元）']  # 获取特征与目标变量的相关性
print("特征与旅游总收入的相关系数：")
print(corr_with_target)
max_corr = max(corr_with_target)  # 获取最大相关性值
colors = ['gold' if v == max_corr else 'cadetblue' for v in corr_with_target]  # 设置条形图颜色，最大相关性为金色，其余为青色
plt.bar(feature_names, corr_with_target, color=colors)  # 绘制条形图
plt.title('特征与旅游总收入的相关性', fontsize=18)  # 设置图表标题
plt.xlabel('特征', fontsize=14)  # 设置 x 轴标签
plt.ylabel('相关系数', fontsize=14)  # 设置 y 轴标签
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.6)  # 添加水平线
plt.xticks(rotation=45, ha='right', fontsize=12)  # 设置 x 轴刻度和标签
plt.yticks(fontsize=12)  # 设置 y 轴刻度
plt.tight_layout()  # 调整图表布局
plt.show()  # 显示图表

# 去除2020年到2022年的数据，确保这些年份不出现在实际数据点中
missing_years = [2020, 2021, 2022]  # 省略的年份
# 获取2019年和2023年的实际值
value_2019 = df.loc[df['年份'] == 2019, '旅游总收入（亿元）'].values[0]  # 获取2019年的实际值
value_2023 = df.loc[df['年份'] == 2023, '旅游总收入（亿元）'].values[0]  # 获取2023年的实际值
# 绘制2008年到2028年的目标变量走势
years = np.append(df['年份'].tolist(), future_years)  # 合并现有年份和未来年份
revenues = np.append(df['旅游总收入（亿元）'].tolist(), future_predictions)  # 合并现有旅游收入和未来预测值
# 去除2020年到2022年的数据点
actual_years = [year for year in df['年份'] if year not in missing_years]  # 省略的年份实际年份列表
actual_revenue = [revenue for year, revenue in zip(df['年份'], df['旅游总收入（亿元）']) if year not in missing_years]  # 省略年份的实际收入列表
plt.figure(figsize=(14, 8))  # 设置图表大小
# 绘制2008年至2018年及2023年的实际值
actual_years_exclude_missing = [year for year in actual_years if year <= 2019 or year >= 2023]  # 省略年份实际年份列表，只包含2019年之前和2023年之后的年份
actual_revenue_exclude_missing = [revenue for year, revenue in zip(actual_years, actual_revenue) if year <= 2019 or year >= 2023]  # 省略年份实际收入列表，只包含2019年之前和2023年之后的收入
plt.plot(actual_years_exclude_missing, actual_revenue_exclude_missing, label='实际值', marker='o', color='skyblue')  # 绘制实际值，浅蓝色线
# 用黑色实线连接2019年和2023年
plt.plot([2019, 2023], [value_2019, value_2023], color='black', alpha=0.7, linewidth=2, label='省略线')  # 绘制省略线，黑色
# 在2023年至未来几年用橙色实线绘制预测值
future_years_with_2023 = np.insert(future_years, 0, 2023)  # 在未来年份数组的开头插入2023年
future_predictions_with_2023 = np.insert(future_predictions, 0, value_2023)  # 在未来预测值数组的开头插入2023年的实际值
plt.plot(future_years_with_2023, future_predictions_with_2023, label='预测值', marker='x', color='orange')  # 绘制预测值，橙色线
plt.annotate('因疫情省略', xy=(2021, (value_2019 + value_2023) / 2), fontsize=12, color='black', ha='center')  # 添加注释“因疫情省略”
# 设置图表标题和轴标签
plt.title('2008年到2028年旅游总收入走势', fontsize=18)  # 设置图表标题
plt.xlabel('年份', fontsize=14)  # 设置 x 轴标签
plt.ylabel('旅游总收入（亿元）', fontsize=14)  # 设置 y 轴标签
# 设置 x 轴刻度和标签
plt.xticks(np.arange(2008, 2029, 1), rotation=45)  # 设置 x 轴刻度
# 设置 y 轴刻度
plt.yticks(fontsize=12)  # 设置 y 轴刻度
# 显示图例
plt.legend(fontsize=14)  # 设置图例
# 设置网格线
plt.grid(True, linestyle='--', alpha=0.6)  # 设置网格线
# 调整图表布局
plt.tight_layout()  # 调整图表布局
# 显示图表
plt.show()  # 显示图表