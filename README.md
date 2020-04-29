# deep_learning_project
深度学习工程项目

整个工程提供了从数据, 配置文件, 数据预处理, 特征抽取, 模型, 模型训练, 模型评估等相关代码,
主要应用于各种任务(包括NLP任务,爬虫任务,图像任务,视频任务,推荐系统等)


主要包括
- [x] 基础学习

主要包括数据结构, TensorFlow的一些基本过程


>- 基础内容的学习模块:`ai_learning`模块

    - data_structure 数据结构,包括一些搜索,排序算法
    - TensorFlow基础知识



- [x] 文件目录

用于存放数据处理,训练等需要的数据文件的目录
>- 配置文件目录: `config`
    
    - 模型训练的基本参数配置, 以及一些具体任务的参数配置

>- 数据目录: `data`

    
    - 通用数据目录(common)
        ├── font
        │   字体
        ├── pos_dict
        │   词性标注
        ├── sensitive_words
        │   常用敏感词词典
        ├── stopwords
        │   常用停用词词典
        └── test_sample
            用于测试模型结构的数据用例

    - 项目数据目录(corpus)
        原始数据目录存放路径
    - 模型目录(model)
        个人训练的模型结果存放路径
    - 预训练模型目录(pretrained_model)
        主要包括bert,ERNIE,vgg等
    - 预训练Embedding目录(pretrained_embedding)
        主要包括一些预训练的词向量,搜狗,知乎,腾讯等预训练的词向量

- [x] 预训练

主要提供整理常用的一些预训练embedding的模型方法,
包括fasttext, gensim的word2vec, glove.
>- embedding预训练模块:`embedding`模块

- [x] 评估

主要基于sklearn的评估方法
>- 模型评估模块: `evaluate`模块

- [x] 特征抽取

主要提供的是tfidf的方法
>- 特征抽取模块: `feature_extractor`模块
- [x] 模型

主要是三种框架的模型
>- Keras实现模型模块: `model_kears`模块

    - 主要包括svm, cnn, rnn, cnn_attention,
    - rnn_attention, han, bert等

>- 普通模型封装模块: `model_normal`模块
    
    - 主要包括fasttext

>- PyTorch实现模型模块: `model_pytorch`模块

    - 主要包括fasttext, cnn, rnn, rcnn,
    - attention, transformer, rnn_attention, dpcnn,
    - hmm, bilstm_crf,
    - bert, ERNIE, bert_cnn, bert_rnn, bert_rcnn, bert_dpcnn等
    
>- TensorFlow实现模型模块: `model_tensorflow`模块

    - bert_model(google的原生bert)
    - textsum_model(文本摘要)
    - fasttext, textcnn, textrnn, textrcnn bilstm, char_cnn, char_rnn,
    - transformer, bilstm_attention, 
    - ner_bilstm, ner_bilstm_crf, ner_lstm,
    - vgg_text_generator等模型 

- [x] 预处理

主要提供了一些用于预处理的工具函数
>- 预处理模块: `preprocess`模块

- [x] 配置

主要是工程的配置,定义一些数据的路径,日志路径等
>- 工程配置模块: `setting`模块

- [x] 工具

主要提供日志模块
>- 工具模块: `utils`模块

- [x] 测试

用于一些函数方法的测试用例
>- 测试模块: `tests`模块

- [x] 任务

实际应用的任务,包括NLP, 图像, 视频, 推荐系统等
>- 爬虫任务: `spider_tasks`
    
    - 爬虫项目(在另一个项目里,未整理) 

>- NLP任务: `nlp_tasks`

    - NLP项目
        ├── chatbot
        |    聊天机器人
        ├── information_extraction
        │    信息抽取任务
        ├── knowledge_graph
        │    知识图谱任务
        ├── new_words_mininag
        |    新词挖掘任务
        ├── pretrain
        │    预训练任务
        ├── sensitive_words
        |     敏感词任务
        ├── sentiment_analysis
        │     情感分析任务
        ├── sequence_labeling
        │     序列标注任务
        ├── text_classification
        │     文本分类任务
        ├── text_generation
        │     文本生成任务
        └── textsum
              文本摘要任务
    
>- 图像任务: `image_tasks`
    
    - 图像项目
     ├── mnist_digit_recognition
     │     数字识别
     └── vgg16_classification
           图片分类任务

>- 视频任务: `video_tasks`
    
    - 视频理解任务
      视频分类

>- 推荐系统: `recommender_system`
    
    - 推荐系统
    ├── music_recommender
    │     网易云音乐推荐系统
    └── news_recommender
          黑马头条新闻推荐系统


主要做NLP方向, 其他任务更新比较少, 代码很多地方在更改, 一直陆续更新.






