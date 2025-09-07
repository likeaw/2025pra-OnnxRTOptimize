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
## （李）
