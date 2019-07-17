# numpy-ml
Ever wish you had an inefficient but somewhat legible collection of machine
learning algorithms implemented exclusively in numpy? No? 

## Models
This repo includes code for the following models:

1. **Gaussian mixture model**
    - EM training

2. **Hidden Markov model**
    - Viterbi decoding
    - Likelihood computation
    - MLE parameter estimation via Baum-Welch/forward-backward algorithm

3. **Latent Dirichlet allocation** (topic model)
    - Standard model with MLE parameter estimation via variational EM
    - Smoothed model with MAP parameter estimation via MCMC 

4. **Neural networks** 
    * Layers / Layer-wise ops
        - Add
        - Flatten
        - Multiply
        - Softmax
        - Fully-connected/Dense
        - Sparse evolutionary connections
        - LSTM 
        - Elman-style RNN 
        - Max + average pooling 
        - Dot-product attention
        - Restricted Boltzmann machine (w. CD-n training)
        - 2D deconvolution (w. padding and stride)
        - 2D convolution (w. padding, dilation, and stride)
        - 1D convolution (w. padding, dilation, stride, and causality)
    * Modules
        - Bidirectional LSTM 
        - ResNet-style residual blocks (identity and convolution)
        - WaveNet-style residual blocks with dilated causal convolutions
        - Transformer-style multi-headed scaled dot product attention
    * Regularizers
        - Dropout 
    * Normalization
        - Batch normalization (spatial and temporal)
        - Layer normalization (spatial and temporal)
    * Optimizers
        - SGD w/ momentum 
        - AdaGrad 
        - RMSProp 
        - Adam
    * Learning Rate Schedulers
        - Constant
        - Exponential
        - Noam/Transformer 
        - Dlib scheduler
    * Weight Initializers
        - Glorot/Xavier uniform and normal
        - He/Kaiming uniform and normal
        - Standard and truncated normal
    * Losses
        - Cross entropy
        - Squared error
        - Bernoulli VAE loss
        - Wasserstein loss with gradient penalty
    * Activations
        - ReLU
        - Tanh
        - Affine
        - Sigmoid
        - Leaky ReLU
        - ELU
        - SELU
        - Exponential
        - Hard Sigmoid
        - Softplus
    * Models
        - Bernoulli variational autoencoder
        - Wasserstein GAN with gradient penalty
    * Utilities
        - `col2im` (MATLAB port)
        - `im2col` (MATLAB port)
        - `conv1D`
        - `conv2D`
        - `deconv2D`
        - `minibatch`

5. **Tree-based models**
    - Decision trees (CART)
    - [Bagging] Random forests 
    - [Boosting] Gradient-boosted decision trees

6. **Linear models**
    - Ridge regression
    - Logistic regression
    - Ordinary least squares 
    - Bayesian linear regression w/ conjugate priors
        - Unknown mean, known variance (Gaussian prior)
        - Unknown mean, unknown variance (Normal-Gamma / Normal-Inverse-Wishart prior)

7. **n-Gram sequence models**
    - Maximum likelihood scores
    - Additive/Lidstone smoothing
    - Simple Good-Turing smoothing

8. **Reinforcement learning models**
    - Cross-entropy method agent
    - First visit on-policy Monte Carlo agent
    - Weighted incremental importance sampling Monte Carlo agent
    - Expected SARSA agent
    - TD-0 Q-learning agent
    - Dyna-Q / Dyna-Q+ with prioritized sweeping

9. **Nonparameteric models**
    - Nadaraya-Watson kernel regression
    - k-Nearest neighbors classification and regression
    - Gaussian process regression

10. **Preprocessing**
    - Discrete Fourier transform (1D signals)
    - Discrete cosine transform (type-II) (1D signals)
    - Bilinear interpolation (2D signals)
    - Nearest neighbor interpolation (1D and 2D signals)
    - Autocorrelation (1D signals)
    - Signal windowing
    - Text tokenization
    - Feature hashing
    - Feature standardization
    - One-hot encoding / decoding
    - Huffman coding / decoding
    - Term frequency-inverse document frequency encoding
    - MFCC encoding

11. **Utilities**
    - Similarity kernels
    - Distance metrics
    - Priority queues
    - Ball tree data structure

## Contributing

Am I missing your favorite model? Is there something that could be cleaner /
less confusing? Did I mess something up? Submit a PR! The only requirement is
that your models are written with just the [Python standard
library](https://docs.python.org/3/library/) and [numpy](https://www.numpy.org/). The
[SciPy library](https://scipy.github.io/devdocs/) is also permitted under special
circumstances ;)

See full contributing guidelines [here](./CONTRIBUTING.md). 




1. 高斯混合模型

EM 训练
2. 隐马尔可夫模型

维特比解码
似然计算
通过 Baum-Welch/forward-backward 算法进行 MLE 参数估计
3. 隐狄利克雷分配模型（主题模型）

用变分 EM 进行 MLE 参数估计的标准模型
用 MCMC 进行 MAP 参数估计的平滑模型
4. 神经网络

4.1 层/层级运算

Add
Flatten
Multiply
Softmax
全连接/Dense
稀疏进化连接
LSTM
Elman 风格的 RNN
最大+平均池化
点积注意力
受限玻尔兹曼机 (w. CD-n training)
2D 转置卷积 (w. padding 和 stride)
2D 卷积 (w. padding、dilation 和 stride)
1D 卷积 (w. padding、dilation、stride 和 causality)
4.2 模块

双向 LSTM
ResNet 风格的残差块（恒等变换和卷积）
WaveNet 风格的残差块（带有扩张因果卷积）
Transformer 风格的多头缩放点积注意力
4.3 正则化项

Dropout
归一化
批归一化（时间上和空间上）
层归一化（时间上和空间上）
4.4 优化器

SGD w/ 动量
AdaGrad
RMSProp
Adam
4.5 学习率调度器

常数
指数
Noam/Transformer
Dlib 调度器
4.6 权重初始化器

Glorot/Xavier uniform 和 normal
He/Kaiming uniform 和 normal
标准和截断正态分布初始化
4.7 损失

交叉熵
平方差
Bernoulli VAE 损失
带有梯度惩罚的 Wasserstein 损失
4.8 激活函数

ReLU
Tanh
Affine
Sigmoid
Leaky ReLU
4.9 模型

Bernoulli 变分自编码器
带有梯度惩罚的 Wasserstein GAN
4.10 神经网络工具

col2im (MATLAB 端口)
im2col (MATLAB 端口)
conv1D
conv2D
deconv2D
minibatch
5. 基于树的模型

决策树 (CART)
[Bagging] 随机森林
[Boosting] 梯度提升决策树
6. 线性模型

岭回归
Logistic 回归
最小二乘法
贝叶斯线性回归 w/共轭先验
7.n 元序列模型

最大似然得分
Additive/Lidstone 平滑
简单 Good-Turing 平滑
8. 强化学习模型

使用交叉熵方法的智能体
首次访问 on-policy 蒙特卡罗智能体
加权增量重要采样蒙特卡罗智能体
Expected SARSA 智能体
TD-0 Q-learning 智能体
Dyna-Q / Dyna-Q+ 优先扫描
9. 非参数模型

Nadaraya-Watson 核回归
k 最近邻分类与回归
10. 预处理

离散傅立叶变换 (1D 信号)
双线性插值 (2D 信号)
最近邻插值 (1D 和 2D 信号)
自相关 (1D 信号）
信号窗口
文本分词
特征哈希
特征标准化
One-hot 编码/解码
Huffman 编码/解码
词频逆文档频率编码
11. 工具

相似度核
距离度量
优先级队列
Ball tree 数据结构
