# distillation-arcface
工程复现2019年CVPR论文[Learning Metrics from Teachers: Compact Networks for Image Embedding](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yu_Learning_Metrics_From_Teachers_Compact_Networks_for_Image_Embedding_CVPR_2019_paper.pdf)。

原文代码基于PyTorch，本文基于MXNet，在[Arcface](https://github.com/deepinsight/insightface)基础上进行蒸馏，Teacher模型为Arcface loss训练的resnet100，测试集700+id 20000+pic，并取得比直接训练更佳的效果，实现了度量学习蒸馏的工程价值。

ACC(FPR<0.001）：
| loss | resnet18 | resnet34 | resnet50 | resnet100 |
| :-----| ----: | :----: | :----: | :----: |
| Arcface loss | 0.7261 | 0.7942 | 0.8478 | 0.9006 |
| Relative loss | 0.7788 |  0.8027 | - | - |
| Relative loss + Arcface loss| 0.8247 | 0.8408 | - | - |