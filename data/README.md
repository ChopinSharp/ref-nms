# Data Preparation
Download pretrained GloVe word vector: [glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip). Unzip it to current directory.

Follow these [descriptions](https://github.com/lichengunc/refer/tree/master/data) to setup referring expression data. Name `$DATA_PATH` as `data/refer`.

Follow the instructions in [MAttNet](https://github.com/lichengunc/MAttNet) until the 2-nd step of the "Training" section. This will extract and save the ResNet features needed for training and evaluation. Then link these features to current directory:
```
ln -s $MATTNET_ROOT_DIR/cache/feats/<dataset>_<split_by>/mrcn/res101_coco_minus_refer_notime data/head_feats/<dataset>_<split_by>
```

Download from [Google Drive](https://drive.google.com/drive/folders/1VnPhOBjuv6XoFj1BhzoJRHySEIJ0w3tM?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1G4k7APKSUs-_5StXoYaNrA) (extraction code: 5a9r). Extract `data` directory and move the contents to current directory.