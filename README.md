# tensorflow_project
All code of tensorflow


-[x] fasttext

1. 调参 tricks
>
   - Ngram特征的长度，根据文本长度，适当提高特征的长度，有助于提高模型的准确率
   - 隐藏层的维度可以适当增大
   - 剔除数量较少的类别和特征

2. 优缺点
> 优点
   1. 与实现相同精度的其他方法相比，速度快
   2. 句子向量（被监督）容易计算
   3. 与gensim相比，fastText在小数据集上的运行效果更好
   4. 语义性能上，fastText在语法表现和FAIR语言表现都优于gensim

> 缺点
   1. 不是nlp独立库， 需要另一个库进行预处理步骤
 