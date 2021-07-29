# 社交媒体抑郁检测

Awesome Multimodal Depression Detection with Wudao Pretrained Language Model for Weibo Users

### requirements

- python >= 3.7.0
- torch >= 1.5.0
- torchvision == 0.8.2
- python_opencv >= 4.5.1
- sklearn >= 0.24.1
- numpy >= 1.20.1
- jieba >= 0.42.1

### usage

```bash
bash run_experiment.sh 
[--dataset dataset_name]
[--download-picture]
[--overwrite-cache]
[--model model_name]
[--hidden-size hidden_size]
[--lr lr]
[--max-epoch max_epoch]
[--random-seed random_seed]
[--batch-size batch_size]
[--device device_id]
[--logging-steps logging_steps]
[--eval-steps eval_steps]
```

### Datasets

*TODO：数据集情况*

#### weibo2012

|                           | weibo2012 | WU3D    |
| ------------------------- | --------- | ------- |
| depressed users count     | 1233      | 10000   |
| non-depressed users count | 1233      | 10000   |
| images count              | 33166     | 601777  |
| microblogs count          | 161842    | 1184918 |
| train                     | 1972      | 8000    |
| eval                      | 247       | 1000    |
| test                      | 247       | 1000    |

#### WU3D

### Features

|     Category     |                           Feature                            | Dim  |
| :--------------: | :----------------------------------------------------------: | :--: |
| visual features  |                 saturation (mean, varience)                  |  2   |
|                  |                  brightness(mean, varience)                  |  2   |
|                  |               warm color (hue $\in$[30, 110])                |  1   |
|                  |                 cold color(saturation < 0.7)                 |  1   |
|                  |                       five color theme                       |  15  |
| textual features |          emotional words count(positive, negative)           |  2   |
|                  |         emoticon count(positive, neutral, negative)          |  3   |
|                  |                 pronoun count(i, you, they)                  |  3   |
|                  |                 punctuation count(!, ?, ...)                 |  3   |
|                  | topic related words count(biology, body, health, death, society, family, friends, money, work and leisure) |  10  |
| profile features |                            gender                            |  1   |
|                  |                        length of name                        |  1   |
|                  |      followers count(followers, bi-followers, friends)       |  3   |
| posting features |                     all microblogs count                     |  1   |
|                  |                   original microblog ratio                   |  1   |
|                  |                  microblog with image ratio                  |  1   |
|                  |                       tweet time count                       |  24  |
|                  |                 late post ratio(23:00~6:00)                  |  1   |
|                  |                   post frequency(per week)                   |  1   |
|                  |                     post time deviation                      |  1   |



### Models

*TODO：模型简介*

#### 1. FeatMLP

见`models/featMLP.py`，将特征工程的结果送入MLP中计算结果。

#### 2. ResNet

见`baseline/Resnet`，将微博图片放入resnet用于分类。

### Experiments

*实验结果与分析*

#### weibo2012

|           Model            |    Modality     |  Test F1  | Test precision | Test Recall | Test Acc  |
| :------------------------: | :-------------: | :-------: | :------------: | :---------: | :-------: |
|            SVM             |      Feat       |   55.06   |     56.12      |    44.71    |   55.06   |
|          FeatMLP           |      Feat       |   70.80   |     55.56      |  **97.56**  |   59.91   |
|           ResNet           |      Image      |   64.96   |     53.40      |    82.92    |   55.47   |
|       GRU+Att_Concat       |      Text       |   77.42   |     76.80      |    78.05    |   77.32   |
|     GRU+Att_Microblog      |      Text       |   80.62   |   **77.03**    |    84.55    |   79.75   |
|  FusionNet_XLNet_TextCNN   |    Feat+Text    |   79.68   |     76.69      |    82.92    |   78.94   |
|    FusionNet_XLNet_LSTM    |    Feat+Text    |   78.83   |     71.52      |    87.80    |   76.52   |
|  FusionNet_XLNet_GRU+Att   |    Feat+Text    | **82.22** |     75.51      |    90.24    |   80.56   |
| FusionNet_word2vec_TextCNN |    Feat+Text    |   73.64   |     70.37      |    77.23    |   72.47   |
|  FusionNet_word2vec_LSTM   |    Feat+Text    |   71.67   |     73.50      |    69.92    |   72.47   |
|   FusionNet_word2vec_GRU   |    Feat+Text    |   74.22   |     71.43      |    77.24    |   73.28   |
| FusionNet_word2vec_GRU+Att |    Feat+Text    |   76.09   |     68.63      |    85.36    |   73.28   |
|            PMFN            | Feat+Text+Image |   81.67   |     76.97      |    86.99    | **80.57** |

#### WU3D

|           Model           | Modality  | Test F1 | Test precision | Test Recall | Test Acc |
| :-----------------------: | :-------: | :-----: | :------------: | :---------: | :------: |
|            SVM            |   Feat    |  76.19  |     85.68      |    68.59    |  77.75   |
|          FeatMLP          |   Feat    |  91.16  |     94.06      |    88.44    |  91.10   |
| FusionNet_word2vec_GRUAtt | Feat+text |  92.77  |     95.09      |    90.57    |  91.70   |
|  FusionNet_XLNet_GRUAtt   | Feat+text |  94.16  |     96.18      |    92.19    |  94.05   |

#### Ablation Studies

#### Can pretrain-models overtake pre-defined features?



#### Seq len

Model：GRU+Att_Concat

Dataset: weibo2012

Dropout: 0.1

| max_seq_len | Batch size | Training acc | Test acc | time cost |
| :---------: | :--------: | :----------: | :------: | :-------: |
|     64      |    128     |    99.80     |  65.59   |    58     |
|     128     |     64     |    99.84     |  60.72   |    150    |
|     256     |     32     |    99.80     |  68.42   |    640    |
|     512     |     16     |    99.80     |  75.30   |   1742    |
|    1024     |     4      |    84.05     |  77.32   |   10046   |

