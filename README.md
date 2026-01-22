# 特征选择方法对比实验 - 运行指南

## 📋 项目简介

本项目实现了基于周志华《机器学习》第11章的三种特征选择方法对比实验：
- **Relief算法**（过滤式方法）
- **LVW算法**（包裹式方法）
- **Lasso正则化**（嵌入式方法）

## 🚀 快速开始

### 1. 环境要求

- Python 3.7 或更高版本
- pip（Python包管理器）

### 2. 安装依赖

在项目目录下运行：

```bash
pip3 install -r requirements.txt
```

或者手动安装：

```bash
pip3 install numpy pandas scikit-learn scipy
```

**注意**：如果使用XGBoost（当前版本不需要），可能需要额外安装：
```bash
brew install libomp  # macOS
pip3 install xgboost
```

### 3. 运行实验

#### 方法一：直接运行Python脚本

```bash
python3 experiment.py
```

#### 方法二：使用可执行权限（如果已设置）

```bash
chmod +x experiment.py
./experiment.py
```

### 4. 查看结果

运行完成后，会生成以下文件：

- **`results.json`** - 详细的实验结果数据（JSON格式）
- **`results_table.tex`** - LaTeX格式的结果表格

## 📊 运行输出示例

```
============================================================
特征选择方法对比实验
基于周志华《机器学习》第11章
============================================================
数据集: Breast Cancer
样本数: 569, 特征数: 30

运行基准测试（全部特征）...
  准确率: 0.9561 ± 0.0123
  时间: 0.7378秒

运行Relief算法...
  准确率: 0.9526 ± 0.0172
  选择的特征数: 10
  时间: 0.0345秒

运行LVW算法...
  准确率: 0.9473 ± 0.0166
  选择的特征数: 26
  时间: 1.4281秒

运行Lasso算法...
  准确率: 0.9578 ± 0.0102
  选择的特征数: 24
  时间: 0.1210秒

进行特征稳定性分析...

结果表格已保存到 results_table.tex

============================================================
实验完成！结果已保存到 results.json 和 results_table.tex
============================================================
```

## 🔧 常见问题

### Q1: 提示 "ModuleNotFoundError: No module named 'sklearn'"

**解决方案**：
```bash
pip3 install scikit-learn
```

### Q2: 提示 "ModuleNotFoundError: No module named 'scipy'"

**解决方案**：
```bash
pip3 install scipy
```

### Q3: LVW算法运行很慢

**原因**：LVW算法需要多次训练分类器，计算成本高。

**解决方案**：可以在 `experiment.py` 中修改 `T` 参数（最大迭代次数）：
```python
lvw_selected, lvw_score = lvw_selection(X, y, T=50)  # 减少迭代次数
```

### Q4: 每次运行结果略有不同

**原因**：这是正常的，因为：
- Relief算法有随机采样
- LVW算法使用随机搜索
- 交叉验证的随机性

**解决方案**：脚本已设置随机种子 `np.random.seed(42)`，但LVW的随机搜索仍会导致结果差异。

### Q5: 如何修改实验参数？

编辑 `experiment.py` 文件中的参数：

```python
# Relief算法参数
relief_weights = relief_selection(X, y, m=100, k=5)  # m: 采样次数, k: 最近邻数
n_features_to_select = 10  # 选择特征数

# LVW算法参数
lvw_selected, lvw_score = lvw_selection(X, y, T=100)  # T: 最大迭代次数

# Lasso算法参数（在lasso_selection函数中）
lasso = LassoCV(cv=5, random_state=42, max_iter=2000)  # cv: 交叉验证折数
```

## 📝 生成LaTeX论文

如果需要编译完整的LaTeX论文：

```bash
# 编译LaTeX（需要安装XeLaTeX）
xelatex main.tex
xelatex main.tex  # 运行两次以解决交叉引用
```

生成的PDF文件：`main.pdf`

## 🎯 实验流程说明

1. **加载数据**：自动加载Breast Cancer数据集
2. **基准测试**：使用全部30个特征训练分类器
3. **Relief算法**：计算特征权重，选择前10个特征
4. **LVW算法**：随机搜索最优特征子集
5. **Lasso算法**：通过L1正则化自动选择特征
6. **性能评估**：使用5折交叉验证评估各方法
7. **稳定性分析**：多次运行分析特征选择的一致性

## 📚 相关文件

- `experiment.py` - 主实验脚本
- `main.tex` - LaTeX论文主文件
- `results.json` - 实验结果数据
- `results_table.tex` - 结果表格（LaTeX格式）
- `requirements.txt` - Python依赖列表

## 💡 提示

- 实验运行时间约1-2分钟（取决于LVW的迭代次数）
- 所有结果会自动保存，可以多次运行对比
- 建议在运行前确保有足够的磁盘空间（结果文件很小）
