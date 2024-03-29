# Deep-Learning-Method-For-Stock-Prediction
## Introduction
 由于股票市场在社会经济中的重要性，股票预测是当前的研究热点。然而，现有研究无法克服股票数据的时间序列和非线性问题，导致预测结果不尽如人意。我们首次提出在双向长短期记忆网络(BiLSTM)和卷积神经网络(CNN)的混合股票预测模型中分别引入CA、SE以及CBAM注意力模块，并进行了比较研究。为了验证模型的性能，使用包含网易财经同行业 50 只股票的七年数据。使用均方误差（MSE）和 R2 作为模型的性能评价指标。主要结果总结如下。(1）引入注意力机制的模型（BiLSTM-CNN-CA、BiLSTM-CNN-SE 和 BiLSTM-CNN-CBAM）比 BiLSTM-CNN 性能更好。(2) BiLSTM-CNN-CA的MSE和R2分别为0.072和0.920，优于其他三个模型。

 下图为BiLSTM-CNN-CA网络结构
 <p align="center">
 <img width="75%" src="./BiLSTM-CNN-CA.png" />
</p>

 ## Results
 
 BiLSTM-CNN、BiLSTM-CNN-CA、BiLSTM-CNN-SE 和 BiLSTM-CNN-CBAM指标结果
  <p align="center">
 <img width="75%" src="./result1.png" />
</p>
BiLSTM-CNN、BiLSTM-CNN-CA、BiLSTM-CNN-SE 和 BiLSTM-CNN-CBAM预测结果比较
  <p align="center">
 <img width="75%" src="./result3.png" />
</p>

## Note
在此声明，本项目为作者2023年浙江大学软件学院金融夏令营暑期项目，代码及其图片均不存在抄袭
