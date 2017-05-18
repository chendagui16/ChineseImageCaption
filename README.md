# 中文的Image Caption - 模式识别大作业
## 安装，配置
* 安装tensorflow
* 安装keras
* 修改configuration.py 文件, 改变你的数据位置
* 修改CaptionModel.py 文件, 更改网络模型和参数
* 通过train.py 或者 train.ipynb (jupyter 推荐使用这种方式, 方便记录) 训练
* 选择checkpoint文件中的精度较高的模型, 修改test\_beam\_search.ipynb或者test\_greedy.ipynb文件进行测试

## beam_search
目前**beam search** 已经写好，只需要调用即可，完整的测试案例参考test\_beam\_search.ipynb文件
## 结构目录
* 在 configuration.py 中记载了数据库文件的位置，将其中的workspace 选项进行更改
* utils.py 中设定了如何将数据文件进行预处理的过程，这里一般不用更改（注意到我们这里把每个caption都pad成一样长，不够长的caption，末位用空字符填充, 所以空字符表示停止符)
* CaptionModel.py中是表示模型的设置，其中的函数设置为
    * \_\_init\_\_(): 构造函数，设定模型的参数，可以自行更改, 包括batch\_size, epochs的数量等等, 也可自行添加
    * generator(): 数据生成，设定如何按照batchsize生成样本，如何随机采样，如何打乱，一般不用更改
    * build_train_model(): 建立训练模型，通过修改这里，可以更新成你的模型
    * train(): 训练模型，一般不用更改, 这里设定为每次epoch结束后, 若loss比以前小, 就会保存模型. 且在5个epoch中, loss没有下降的话, 会自动衰减学习率.
    * build\_inference\_model(): 建立推理模型，这里的修改要非常小心，需要针对train\_model进行修改，理由后面说明
    * inference(): 对测试样本进行推断，为了保证在一个epoch里面将所有的测试样本都推断一遍，需要将 self.inference\_batch\_size 设置成能整除测试样本数量1000的数

## 一些说明
RNN模型在训练和推断时是不一样的, 所以搭建的模型也不一样, 训练的时候是已知图像和上一个单词, 来生成下一个单词的分布, 然而在推断的时候, 我们没有caption的输入, 所以这里使用的方法是, 每次推断一个字符, 把输出的最可能的字符作为下一个时刻的输入. (文献中说采用beam search效果更好, 但是我目前没有时间写, 后续可以完善)

因此在推断的时候, 需要修改网络的一些结构
* 把网络的参数冻结, 不用训练
* 把RNN改成一步迭代, 我们迭代一步, 就找一个最好的输出作为下一个输入
* 把RNN改成Stateful, 即每次迭代一步后, 网络中的状态保留, 知道这句话生成完毕, 然后清零

## 我的模型
目前我只使用了非常简单的模型, 只使用了fc2的特征, 用的RNN是GRU, 为了加快速度, 会先用一个全连接层把fc2层embed到一个较低的维度(128)

另外, 我的输入是将图片embed后作为RNN的第一个输入, 然后将文字embed后按序输入.

## task
目前有以下的工作可能需要完成, 大家自行领取
* 尝试不同的网络模型进行训练, 并调参数. 尽量得到好的val_acc(训练的时候会打印). 然后在写相应的inference model
* 尝试用不同的图像特征进行训练, 并采用不同的网络, 希望能利用到卷积层的特征
* 尝试用beam search的方法进行推断(这个任务我先领了, 如果有问题可以协商)
* 统计结果, 撰写报告
