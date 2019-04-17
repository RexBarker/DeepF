#######################################################################
# PrepareFiles.py
#  - Purpose:  to sort files into various locations for the model 
#              operation
#######################################################################

import threading
import shutil
import os
from pathlib import Path
import re
import cv2
import json
from datetime import datetime

# will use them for creating custom directory iterator
import numpy as np
from six.moves import range

# model training
from keras.models import Model
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, \
                            EarlyStopping, TensorBoard, Callback, CSVLogger
from keras import backend as K

# regular expression for splitting by whitespace
splitter = re.compile("\s+")

# remove old stopping requests
if os.path.exists('./STOP'): os.remove('./STOP')

def process_folders(partitionFile):
    # Read the relevant annotation file and preprocess it
    # Assumed that the annotation files are under '<project folder>/data/anno' path
    with open(partitionFile, 'r') as eval_partition_file:
        list_eval_partition = [line.rstrip('\n') for line in eval_partition_file][2:]
        list_eval_partition = [splitter.split(line) for line in list_eval_partition]
        list_all = [(v[0][4:], v[0].split('/')[1].split('_')[-1], v[1]) for v in list_eval_partition]
    
    return list_all

def create_dict_bboxes(list_all, split='train'):
    lst = [(line[0], line[1], line[3], line[2]) for line in list_all if line[2] == split]
    lst = [("".join(line[3] + '/' + line[1] + line[0][3:]), line[1], line[2]) for line in lst]
    lst_shape = [cv2.imread('../fashion_data/' + line[0]).shape for line in lst]
    lst = [(line[0], line[1], (round(line[2][0] / shape[1], 2), round(line[2][1] / shape[0], 2), round(line[2][2] / shape[1], 2), round(line[2][3] / shape[0], 2))) for line, shape in zip(lst, lst_shape)]
    dict_ = {"/".join(line[0].split('/')[1:]): {'x1': line[2][0], 'y1': line[2][1], 'x2': line[2][2], 'y2': line[2][3]} for line in lst}
    return dict_

def get_dict_bboxes(base_path):
    with open(base_path.joinpath('../data/Anno/list_category_img.txt'), 'r') as category_img_file, \
            open(base_path.joinpath('../data/Anno/list_eval_partition.txt'), 'r') as eval_partition_file, \
            open(base_path.joinpath('../data/Anno/list_bbox.txt'), 'r') as bbox_file:
        list_category_img = [line.rstrip('\n') for line in category_img_file][2:]
        list_eval_partition = [line.rstrip('\n') for line in eval_partition_file][2:]
        list_bbox = [line.rstrip('\n') for line in bbox_file][2:]

        list_category_img = [splitter.split(line) for line in list_category_img]
        list_eval_partition = [splitter.split(line) for line in list_eval_partition]
        list_bbox = [splitter.split(line) for line in list_bbox]

        list_all = [(k[0], k[0].split('/')[1].split('_')[-1], v[1], (int(b[1]), int(b[2]), int(b[3]), int(b[4])))
                    for k, v, b in zip(list_category_img, list_eval_partition, list_bbox)]

        list_all.sort(key=lambda x: x[1])

        dict_train = create_dict_bboxes(list_all,split='train')
        dict_val = create_dict_bboxes(list_all, split='val')
        dict_test = create_dict_bboxes(list_all, split='test')

        return dict_train, dict_val, dict_test

#not used
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


class DirectoryIteratorWithBoundingBoxes(DirectoryIterator):
    def __init__(self, directory, image_data_generator, bounding_boxes: dict = None, target_size=(256, 256),
                 color_mode: str = 'rgb', classes=None, class_mode: str = 'categorical', batch_size: int = 32,
                 shuffle: bool = True, seed=None, data_format=None, save_to_dir=None,
                 save_prefix: str = '', save_format: str = 'jpeg', follow_links: bool = False):
        super().__init__(directory, image_data_generator, target_size, color_mode, classes, class_mode, batch_size,
                         shuffle, seed, data_format, save_to_dir, save_prefix, save_format, follow_links)
        self.bounding_boxes = bounding_boxes

    def next(self):
        """
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        locations = np.zeros((len(batch_x),) + (4,), dtype=K.floatx())

        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = image.load_img(os.path.join(self.directory, fname),
                                 grayscale=grayscale,
                                 target_size=self.target_size)
            x = image.img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

            if self.bounding_boxes is not None:
                bounding_box = self.bounding_boxes[fname]
                locations[i] = np.asarray(
                    [bounding_box['x1'], bounding_box['y1'], bounding_box['x2'], bounding_box['y2']],
                    dtype=K.floatx())
        # optionally save augmented images to disk for debugging purposes
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), 9), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x

        if self.bounding_boxes is not None:
            return batch_x, [batch_y, locations]
        else:
            return batch_x, batch_y

#----------------------------------------------------------------------
class TerminateOnRequest(Callback):
    """ Custom callback to stop training on request, based on the 
        existance of a file in the current directory: 
        \'STOP\'
        training execution will stop at the end of the epoch.
        To stop, execute >touch STOP in the current running directory
    """
    def on_epoch_end(self, epoch, logs=None):
        if os.path.exists('./STOP'):
            print(f'Terminating on request, epoch={epoch}')
            self.stopped_epoch = epoch
            self.model.stop_training = True


#---------------------------------------------------------------------#

if __name__ == '__main__':
    # prepare path information
    base_path = Path.cwd()
    source_path = base_path.joinpath('../fashion_data_upper/Img/')
    work_path = base_path.joinpath('./dataset/')
    partitionFile = base_path.joinpath('../data/Eval/list_eval_partition.txt')
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    modelname = f'model_{time_stamp}.h5'

    # get the processing data
    list_all = process_folders(partitionFile)

    # import and reset model
    model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    for layer in model_resnet.layers[:-12]:
        # 6 - 12 - 18 have been tried. 12 is the best.
        layer.trainable = False
    
    # category classification in branch
    x = model_resnet.output
    x = Dense(512, activation='elu', kernel_regularizer=l2(0.001))(x)
    y = Dense(9, activation='softmax', name='img')(x)  # hard coded to number of classes we have

    # build bbox detection into model
    x_bbox = model_resnet.output
    x_bbox = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x_bbox)
    x_bbox = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x_bbox)
    bbox = Dense(4, kernel_initializer='normal', name='bbox')(x_bbox)

    # final model
    final_model = Model(inputs=model_resnet.input, outputs=[y, bbox])

    # optimization
    opt = SGD(lr=0.0001, momentum=0.9, nesterov=True)

    # compile
    final_model.compile(optimizer=opt,
                    loss={'img': 'categorical_crossentropy',
                          'bbox': 'mean_squared_error'},
                    metrics={'img': ['accuracy', 'top_k_categorical_accuracy'], # default: top-5
                             'bbox': ['mse']})

    train_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)
    test_datagen = ImageDataGenerator()

    # current test
    if os.path.exists('dict_test.json') and  \
       os.path.exists('dict_train.json') and \
       os.path.exists('dict_val.json'):
        with open('dict_test.json','r') as fp: dict_test = json.load(fp)
        with open('dict_train.json','r') as fp: dict_train = json.load(fp)
        with open('dict_val.json','r') as fp: dict_val = json.load(fp)
    
    else:
        dict_train, dict_val, dict_test = get_dict_bboxes(base_path)
        with open('dict_test.json','w') as fp: json.dump(dict_test,fp)
        with open('dict_train.json','w') as fp: json.dump(dict_train,fp)
        with open('dict_val.json','w') as fp: json.dump(dict_val,fp)

    train_iterator = DirectoryIteratorWithBoundingBoxes("../fashion_data_upper/train", train_datagen, bounding_boxes=dict_train, target_size=(200, 200))
    test_iterator = DirectoryIteratorWithBoundingBoxes("../fashion_data_upper/test", test_datagen, bounding_boxes=dict_test,target_size=(200, 200))



    # save results
    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               patience=12,
                               factor=0.5,
                               verbose=1)

    tensorboard = TensorBoard(log_dir='./logs')

    early_stopper = EarlyStopping(monitor='val_loss',
                              patience=30,
                              verbose=1)

    checkpoint = ModelCheckpoint(modelname)

    csvlogger  = CSVLogger(filename=f'log_{time_stamp}.csv')

    terminate_on_request = TerminateOnRequest()

    # threadsafe not used, as Keras does not allow it
    #@threadsafe_generator
    def custom_generator(iterator):    
        while True:
            batch_x, batch_y = iterator.next()        
            yield (batch_x, batch_y)

    final_model.fit_generator(custom_generator(train_iterator),
                              steps_per_epoch=2000,
                              epochs=200,
                              validation_data=custom_generator(test_iterator),
                              validation_steps=200, 
                              verbose=2,
                              shuffle=True,
                              callbacks=[lr_reducer, checkpoint, early_stopper, tensorboard,csvlogger,terminate_on_request],
                              workers=1,  # only functions with single thread for GPU
                              use_multiprocessing=False)
    
    # test results
    test_datagen = ImageDataGenerator()



    test_iterator = DirectoryIteratorWithBoundingBoxes("../fashion_data_upper/val", 
                                                        test_datagen, 
                                                        bounding_boxes=dict_test, 
                                                        target_size=(200, 200))
    scores = final_model.evaluate_generator(custom_generator(test_iterator), steps=2000)

    print('Multi target loss: ' + str(scores[0]))
    print('Image loss: ' + str(scores[1]))
    print('Bounding boxes loss: ' + str(scores[2]))
    print('Image accuracy: ' + str(scores[3]))
    print('Top-5 image accuracy: ' + str(scores[4]))
    print('Bounding boxes error: ' + str(scores[5]))
