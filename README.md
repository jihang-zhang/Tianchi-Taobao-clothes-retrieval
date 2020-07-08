# 天池淘宝直播商品识别大赛 复赛第7名方案

## 1、模型思路
```
 · 先检测再根据检测框进行图像检索
 · 目标检测模型 (Detectron2)： 以VoVNet-v2-99-FPN为骨干网络的Faster-RCNN。输入数据为train part 1-5
    · 模型细节及训练技巧: 多尺度训练，cosine-annealing lr scheduler with warmup，hflip augmentation
 · 图像检索模型： 以EfficientNet-b5/b6/b7为骨干网络，ArcFace为损失函数的检索模型。输入数据为 train+valid 中每个instance检测框的crop。拥有相同instance_id的为同一identity
    · 模型细节及训练技巧: 
      (a) 将宽高比相似的检测框crop合并成一个batch进行多尺度训练
      (b) 大量的data augmentation以减少不同服装颜色，观察角度，图片及视频清晰度对检索效果产生的影响
      (c) Consine-annealing lr scheduler with warmup
      (d) Cross-batch memory以变相增加batch size
      (f) GeM-pooling
      (g) Mixed-precision training
      (h) 每个视频抽取第[80, 120, 160, ..., 360]帧进行训练（很多视频的前80帧并未含有目标服饰）
      (i) Head为global pooling -> linear(512) -> BN -> ArcFace
 · 后处理及推理
    · 用目标检测模型生成图片及视频的检测框。
      (a) 只对视频第[80, 100, 120, 140, ..., 380]共16帧进行推理
      (b) 图片检测框的阈值为0.9，视频检测框的阈值为0.95
      (c) 推理时视频短边放大到 608 pixels
    · 对每个检测框crop用检测网络输出 512维 的feature embedding
    · 将四个检测模型（两个efficientnet-b5，一个b6，一个b7）所生成的feature进行l2-normalize -> concatenate -> l2-normalize处理，生成2048维的feature
    · 将所有视频检测框的feature当作Gallary set，将所有图片检测框的feature当作Query set，通过以下步骤找到每个图片文件夹下所对应的视频item_id
      (a) 假设某一图片文件夹下共生成N个检测框。对每一个检测框，找到其对应欧氏距离最近的视频检测框所对应的视频帧及item_id
      (b) 从N个备选中，只保留欧氏距离前28%的视频item_id
      (c) 我们统计出每个剩余备选视频item_id的个数，然后只保留50%出现次数较多的item_id
      (d) 假设最终剩下M个备选视频item_id。如果M=0，则该图片文件夹没有对应视频item_id； 如果M=1，则将此唯一item_id作为该图片文件夹对应视频item_id； 如果M>1，则将欧氏距离最近的视频item_id当作该文件夹对应的视频item_id
      (e) 找到该文件夹内拥有置信度最高的检测框的图片。此图片及对应检测框分别为img_name以及item_box
      (f) 将step (a) 中所检索到的视频帧作为frame_index； 该视频帧内置信度最高的检测框为frame_box
    · 因为我们把图片当作query，视频当作gallary，所以会出现同一视频对应多个图片文件夹的情况。对此我们只保留欧氏距离最近的匹配对
    · 再一次移除欧式距离超过1.12的匹配对
    · 以上后处理及推理过程的细节请参考./metric_sub/src_infer/run.py
```

## 2、路径信息
```
.
├── bbox_sub                                 # 目标检测模型
|   ├── config                               # Detectron2 模型配置文件夹
|   ├── pretrained                           # 预训练模型
|   ├── src_infer                            # 推理源代码
|   ├── src_train                            # 训练源代码
|   ├── Dockerfile
|   └── run.sh                               # 执行文件
|
|
├── metric_sub                               # 图像检索模型及最终结果推理
|   ├── pretrained                           # 预训练模型
|   ├── src_infer                            # 推理源代码
|   ├── src_train                            # 训练源代码
|   ├── Dockerfile
|   ├── result.json                          # 最终提交结果
|   └── run.sh                               # 执行文件
|
|
└── readme.txt
```

## 3、代码运行
```
 · 下载在ImageNet-1k预训练的[VoVNetV2-99](https://dl.dropbox.com/s/1mlv31coewx8trd/vovnet99_ese_detectron2.pth)模型权重
 · 将vovnet99_ese_detectron2.pth放置到 ./bbox_sub/pretrained 文件夹下
 · 先build并提交bbox_sub，再build并提交metric_sub
   · 由于比赛提供的训练环境没有网络连接，所以无法直接下载EfficientNet-Noisy-Student的预训练权重。请在构建模型后手动导入预训练权重，保存为
     ./metric_sub/pretrained/effnet-b5-imagenet-pretrained.pt
     ./metric_sub/pretrained/effnet-b6-imagenet-pretrained.pt
     ./metric_sub/pretrained/effnet-b7-imagenet-pretrained.pt
 · 代码为清理过后的代码，与实际训练和推理过程有所差别，因此运行时间可能会超过5天的时间限制
```

## 4、环境
```
 · 参考两个Dockerfile。主要区别为bbox_sub内用的PyTorch 1.4，而metric_sub中用的为PyTorch 1.6 (nightly)
```