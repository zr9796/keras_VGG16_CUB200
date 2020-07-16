import os
import numpy as np
from config import *
from utils import *
from keras import Sequential
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (
    ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, ModelCheckpoint, TensorBoard)
import datetime

def form_dataset():
    if not os.path.exists('/data/private/hzr/dataset/CUB_200_2011/CUB_200_2011/train/'):
        os.makedirs('/data/private/hzr/dataset/CUB_200_2011/CUB_200_2011/train/')
    if not os.path.exists('/data/private/hzr/dataset/CUB_200_2011/CUB_200_2011/test/'):
        os.makedirs('/data/private/hzr/dataset/CUB_200_2011/CUB_200_2011/test/')

    f_image = open(IMAGE_TXT, 'r')
    f_class = open(CLASS_TXT, "r")
    f_image_class = open(IMIAGE_CLASS_TXT, "r")
    f_train_test = open(TRAIN_TEST_TXT, 'r')

    images = f_image.read().split("\n") 
    class_list = f_class.read().split("\n")     
    image_class_list = f_image_class.read().split("\n") 
    train_test = f_train_test.read().split("\n") 

    train_list = []
    test_list = []
    for file_ in train_test[:-1]:
        _, flag = file_.split()
        if flag == '0':
            test_list.append(_)
        elif flag == '1':
            train_list.append(_)
    f_image.close()
    f_class.close()
    f_image_class.close()
    f_train_test.close()

    # train_X = []
    # train_y = []
    # test_X = []
    # test_y = []

    for num in train_list:
        _, path = images[int(num)-1].split()
        image_path = IMAGE_PATH+'/'+path

        image_shape, img = load_image(image_path)
        _, class_index = image_class_list[int(num)-1].split()
        _, class_label = class_list[int(class_index)-1].split()

        # train_X.append(img)
        # train_y.append(class_index)

        if not os.path.exists('/data/private/hzr/dataset/CUB_200_2011/CUB_200_2011/train/' + class_label + '/'):
            os.makedirs('/data/private/hzr/dataset/CUB_200_2011/CUB_200_2011/train/' + class_label + '/')
        cv2.imwrite('/data/private/hzr/dataset/CUB_200_2011/CUB_200_2011/train/' + class_label + '/' + num.zfill(5) + ".jpg", img)

    for num in test_list:
        _, path = images[int(num)-1].split()
        image_path = IMAGE_PATH+'/'+path

        image_shape, img = load_image(image_path)
        _, class_index = image_class_list[int(num)-1].split()
        _, class_label = class_list[int(class_index)-1].split()

        # test_X.append(img)
        # test_y.append(class_index)

        if not os.path.exists('/data/private/hzr/dataset/CUB_200_2011/CUB_200_2011/test/' + class_label + '/'):
            os.makedirs('/data/private/hzr/dataset/CUB_200_2011/CUB_200_2011/test/' + class_label + '/')
        cv2.imwrite('/data/private/hzr/dataset/CUB_200_2011/CUB_200_2011/test/' + class_label + '/' + num.zfill(5) + ".jpg", img)

    # return test_X, test_y

def vgg16_model(classes_num = 1000, weights_path = None, keep_prob = 0.5, include_top = True):
    base_model=VGG16(weights='imagenet',include_top=False, input_shape=(224,224,3))
    # for layer in base_model.layers:
    #     layer.trainable = False

    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(4096, activation='relu'))
    top_model.add(Dropout(rate = 1 - keep_prob)) # rate = 1 - keep_prob
    top_model.add(Dense(4096, activation='relu'))
    top_model.add(Dropout(rate = 1 - keep_prob))
    top_model.add(Dense(classes_num, activation='softmax'))

    my_VGG16_model=Model(inputs=base_model.input, outputs=top_model(base_model.output))

    if os.path.exists(weights_path):
        my_VGG16_model.load_weights(weights_path)
        print("loading weights --!")
    
    if include_top is False:
        return base_model
    # top_model.summary()
    # my_VGG16_model.summary()
    return my_VGG16_model

def train_CUB(model, batch_size = 64, epochs = 20, train_num = TRAIN_NUM, test_num = TEST_NUM):
        
    
    train_data_dir = '/data/private/hzr/dataset/CUB_200_2011/CUB_200_2011/train/'
    validation_data_dir = '/data/private/hzr/dataset/CUB_200_2011/CUB_200_2011/test/'

    #这是训练集的生成器
    train_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    ## 训练图片生成器
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,#训练样本地址
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical') #多分类

    test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
    
    ##验证集的生成器
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,#验证样本地址
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False) #多分类

    
    path = '/home/hzr/projects/CAM-RL-detector/model/hisVGG16_base_5/'
    if not os.path.exists(path):
        os.makedirs(path)
    # train
    print("----------------start training ----------------")
    callback_lists = [
                EarlyStopping(monitor = 'acc',  # 监控模型的验证精度
                                patience = 2,),   # 如果精度在多于一轮的时间（即两轮）内不再改善，就中断训练
    
                # ModelCheckpoint用于在每轮过后保存当前权重
                ModelCheckpoint(filepath = path + 'CUB_VGG_weights.h5', # 目标文件的保存路径
                                                # 这两个参数的意思是，如果val_loss没有改善，那么不需要覆盖模型文件，
                                                # 这就可以始终保存在训练过程中见到的最佳模型
                                                monitor = 'val_acc', save_best_only = True,),
                TensorBoard(log_dir= path + 'logs', write_graph=True, write_images=True, update_freq='epoch')
                            ]
    model.compile(loss='categorical_crossentropy',optimizer = SGD(lr=1e-6,momentum=0.9,decay=0.0005), metrics = ['accuracy'])
    history = model.fit_generator(train_generator, 
                                    steps_per_epoch=len(train_generator),
                                    epochs=epochs,
                                    validation_data=validation_generator,
                                    # validation_steps=200,
                                    callbacks=callback_lists)
    # model.save("/home/hzr/model/CAM-RL-detector/model/hisVGG16_base/CUB_VGG_weights_" + str(history.history['val_acc']) + ".h5")


if __name__ == "__main__":

    # get CUB dataset
    # test_X, test_y = form_dataset()
    # form_dataset()

    # train
    model = vgg16_model(classes_num = 200, weights_path = "/home/hzr/projects/CAM-RL-detector/model/hisVGG16_base_4/CUB_VGG_weights.h5", keep_prob=0.9)
    train_CUB(model)

    # test
    # score = model.evaluate(test_X, test_y, verbose=1)
    # print("Large CNN Error: %.2f%%" %(100-score[1]*100))