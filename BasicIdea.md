# 基本思路
（一）Conv​​ ：算子融合（Conv-BN-ReLU）+ 内核级优化 预期优化效率：15%-30% 建议方法（ai提供）：消除中间存储+加速矩阵乘法

​​BatchNormalization​​：算子融合（与Conv/LeakyReLU）+ 内核级优化 预期优化效率：10%-20% 建议方法（ai提供）：减少中间输出存储+加速统计量计算

（二）Attention​​：矩阵分块（或FlashAttention）+ 内核级优化 预期优化效率：2-4倍（主办方说可以优化到8倍及以上）建议方法（ai提供）：减少内存占用+加速QKᵀ/WV计算

（三）​​LeakyReLU​​：内核级优化（向量化）+ 算子融合 预期优化效率：5%-15% 建议方法（ai提供）：向量化并行计算+减少数据搬运

​​GroupNormalization​​：内核级优化（分组向量化）+ 算子融合 预期优化效率：8%-20% 建议方法（ai提供）：分组内并行统计量计算+与Conv融合减少存储

建议分为三部分，其中Attention难度最高
# 优化策略
（吴）整理了一下这几个方法的相关博客，仅供参考，如有更好的欢迎提供

算子融合：https://blog.csdn.net/AggressiveYu/article/details/149442537?ops_request_misc=&request_id=&biz_id=102&utm_term=%E7%AE%97%E5%AD%90%E8%9E%8D%E5%90%88%E2%80%8B%E2%80%8B&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-149442537.142^v102^control&spm=1018.2226.3001.4187

矩阵分块：暂未找到合适博客

内核级优化：暂未找到合适博客


## 算子融合
（吴）
## 内核级优化
（胡）

# 优化方法概览

## 1. 分层图优化架构（Hierarchical Graph Optimization）

ONNX Runtime采用多级优化策略，执行独立于提供程序的优化，然后根据可用的执行提供程序将图分割成子图集合，每个子图分配给一个执行提供程序。

### 1.1 基础图变换（Basic Graph Transformations）
- **Dead Code Elimination（DCE）**：移除计算图中未被使用的节点和边，减少计算复杂度
- **Common Subexpression Elimination（CSE）**：识别并合并重复的计算子表达式
- **Constant Propagation**：在编译时计算常量表达式，将结果直接嵌入图中
- **Node Fusion Optimization**：将相邻的兼容算子融合为单一操作，减少内存访问和kernel启动开销

### 1.2 高级图重构（Advanced Graph Restructuring）
- **Layout Transformation**：根据硬件特性调整张量的内存布局（NCHW vs NHWC）
- **Reshape Propagation**：通过图传播reshape操作，消除冗余的形状变换
- **Batch Dimension Optimization**：针对批处理维度进行特殊优化

**相关研究**：
- 论文："Graph Optimization Techniques in Deep Learning Inference Frameworks" 
- URL: https://arxiv.org/abs/2208.07100

## 2. 内存管理与分配策略（Memory Management Strategies）

### 2.1 Arena-based内存分配
执行提供程序定义其内存分配器，为自定义加速器和运行时提供正确的抽象和运行时支持。

- **Memory Arena Pooling**：预分配大块内存池，避免频繁的malloc/free操作
- **Memory Reuse Optimization**：通过生命周期分析实现张量内存的高效重用
- **Peak Memory Reduction**：使用梯度检查点和内存交换技术降低峰值内存使用

### 2.2 张量生命周期管理
- **Live Variable Analysis**：分析张量的生命周期，优化内存分配时机
- **In-place Operations**：尽可能使用原地操作减少内存拷贝
- **Memory Layout Optimization**：根据访问模式优化数据在内存中的排列

**技术论文**：
- "Memory Optimizer for ONNX Runtime Training"
- URL: https://github.com/microsoft/onnxruntime/blob/main/docs/Memory_Optimizer.md

## 3. 执行提供程序架构（Execution Provider Architecture）

### 3.1 异构计算调度
ONNX Runtime必须能够在涉及多个执行提供程序的异构环境中执行单个模型。

- **Graph Partitioning Algorithm**：基于算子支持度和性能模型的智能图分割
- **Cross-EP Memory Management**：管理不同执行提供程序间的内存传输
- **Dynamic Load Balancing**：根据运行时性能动态调整算子分配

### 3.2 硬件特化优化
- **CUDA Provider优化**：
  - cuBLAS/cuDNN库集成
  - CUDA Graph优化
  - Multi-Stream执行
- **TensorRT Provider优化**：
  - Dynamic Shape处理
  - Mixed Precision优化
  - Plugin接口扩展

## 4. 量化与精度优化（Quantization and Precision Optimization）

### 4.1 静态量化策略
- **Calibration Dataset Optimization**：校准数据集的选择和采样策略
- **Per-Channel vs Per-Tensor Quantization**：根据权重分布选择最优量化粒度
- **Quantization-Aware Training Integration**：与训练框架的QAT集成

### 4.2 动态量化技术
- **Runtime Quantization Parameter Estimation**：运行时动态估计量化参数
- **Selective Layer Quantization**：基于敏感性分析的选择性量化
- **Mixed Precision Strategies**：FP16/INT8混合精度优化

**重要研究**：
- "Selective Quantization Tuning for ONNX Models"
- URL: https://arxiv.org/html/2507.12196v1

## 5. 算子级性能优化（Operator-Level Performance Optimization）

### 5.1 内核优化策略
- **Vectorization**：利用SIMD指令集进行向量化计算
- **Cache-aware Algorithms**：考虑缓存层次结构的算法设计
- **Register Blocking**：优化寄存器使用模式

### 5.2 自动调优框架
- **Auto-tuning Infrastructure**：自动搜索最优kernel参数
- **Performance Model Guided Search**：基于性能模型的搜索空间缩减
- **Cross-Platform Optimization**：跨平台性能参数迁移

## 6. 动态形状处理（Dynamic Shape Handling）

### 6.1 形状推断优化
- **Symbolic Shape Inference**：符号化形状推断技术
- **Shape Specialization**：针对常见形状模式的特化优化
- **Dynamic Kernel Selection**：根据运行时形状选择最优kernel

### 6.2 内存预分配策略
- **Shape Profile-based Allocation**：基于形状profile的内存预分配
- **Growth Strategy Optimization**：动态内存增长策略优化

## 7. 编译时与运行时协同优化

### 7.1 JIT编译优化
- **Lazy Evaluation**：延迟计算策略
- **Code Generation Optimization**：针对特定硬件的代码生成
- **Runtime Specialization**：运行时特化优化

### 7.2 Profile-Guided Optimization（PGO）
- **Runtime Profiling Integration**：集成运行时性能分析
- **Adaptive Optimization**：基于运行时反馈的自适应优化

**核心技术文献**：
- "Extending the ONNX Runtime Framework for the Processing-in-Memory Execution"
- URL: https://www.researchgate.net/publication/359884564_Extending_the_ONNX_Runtime_Framework_for_the_Processing-in-Memory_Execution

## 8. 分布式与并行优化

### 8.1 模型并行策略
- **Pipeline Parallelism**：流水线并行执行
- **Data Parallelism**：数据并行处理
- **Hybrid Parallelism**：混合并行策略

### 8.2 通信优化
- **Gradient Compression**：梯度压缩技术
- **All-Reduce Optimization**：高效的All-Reduce算法实现

这些优化策略在ONNX Runtime中形成了一个完整的优化流水线，基于使用场景要求，延迟、吞吐量、内存利用率和模型/应用程序大小是性能测量的常见维度。每种策略都有其适用的场景和性能权衡，需要根据具体的部署环境和性能要求进行选择和调优。
