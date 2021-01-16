# DiffPool
## Introduction
This is a reproduction of *Hierarchical graph representation learning with differentiable pooling*.

由于我并未看懂原文中作者的架构，所以我就自己搭了一个架构，并使用了一层的Diff_Pool来对图进行坍缩。
## Requirements
* pytorch 1.7
* torch_geometric 1.6.3
## Experiment
模型在D&D数据集上进行训练和测试。

学习曲线如下所示：
![learning_cure](img/loss.svg)

精度曲线如下所示：
![accuracy](img/accuracy.svg)

模型在测试集上准确率如下所示：
![test_accuracy](img/test_accuracy.JPG)

模型在验证集上的最高准确率达到85%。
## Runing the code
训练模型`python main.py`