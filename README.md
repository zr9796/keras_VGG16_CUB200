# keras_VGG16_CUB200

基于keras实现的VGG16，在CUB-200-2011分类数据集上进行微调，记录一下自己总结的学习成果

## 训练要点
### 1. 先对全连接层训练，再对整个网络训练
因为载入了预训练的imagenet模型，微调的话仅仅是把VGG16的尾部换成输出200类，其他几乎一致，所以7*7*512之前的层是有参数的，之后是无参数的，如果放在一块拟合显然不太适合。所以先冻结前面imagenet预训练的参数，仅训练4096-4096-200的全连接层部分。
### 2. dropout防止过拟合
如果直接训练的话，由于这个数据集的训练：测试几乎为1：1，所以训练集拟合速度飞快，很容易出现测试集loss还很高acc很低的情况下，训练集已经100%的准确率了。所以必须加入dropout，控制参数在训练中随机失活，这个方法直观来说就是压低了训练集的拟合速度，这样可以给拟合测试集提供更多的空间
### 3. 对图片加入随机变换防止过拟合
仅仅按照上述两个方法，我最高拟合的测试集准确率只有57，训练集的拟合速度还是过于快，因此仍然存在过拟合，其实归根结底还是因为这个数据集1：1的坑爹分布——训练集不够全面。所以这里在训练集生成器中加入几个参数，来提供对图像进行随机变换的操作，这种方法对小数据集非常有效，相当于一个图顶多个用。在加入了这个调整以后我的训练结果就飞速上升了。
train_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

## 我的训练步骤
通用参数：
batch_size = 64
step_per_epoch = TRAIN_NUM    #每迭代一次遍历所有训练集图像
SGD(lr=1e-4,momentum=0.9,decay=0.0005)
### 1. epoch = 30 keep_prob = 1 只训练全连接层，冻结前面的参数
### 2. epoch = 35 keep_prob = 0.5 解冻，训练所有参数
### 3. epoch = 20 keep_prob = 0.7 训练所有参数
### 4. epoch = 20 keep_prob = 0.9 训练所有参数

## 我的最终结果：
test_acc：69.99

Top-1 cls error: 30.01
Top-5 cls error:  8.13
