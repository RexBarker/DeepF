#######################################################################
# Train model for individual class detection
# based on the DeepF model, but for individual classes only
#######################################################################

import os
import numpy as np
import pandas as pd
import argparse
from datetime import datetime

# model training
from keras.models import Model, load_model
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, \
                            EarlyStopping, TensorBoard,  \
                            CSVLogger, Callback

from keras import backend as K

# remove old stopping requests
if os.path.exists('./STOP'): os.remove('./STOP')

# command line definition of training & testing
parser = argparse.ArgumentParser()
parser.add_argument('-traindir',type=str,dest='trainDir',required=True,
                    help='Required: training directory containing containing organized classes')

parser.add_argument('-testdir',type=str,dest='testDir',default=None,
                    help='Optional: testing directory containing containing organized classes')

parser.add_argument('-valdir',type=str,dest='valDir',default=None,
                    help='Optional: validation directory to use during training, instead of train_split')

parser.add_argument('-nepochs',type=int,dest='nepochs',default=200,
                    help='Optional: number of training epochs (default=200)')

parser.add_argument('-nsteps',dest='nsteps',default=None,
                    help='Optional: number of steps per epoch (default=n_samples//batch_size)')
                    
parser.add_argument('-vnsteps',dest='vnsteps',default=None,
                    help='Optional: number of steps per epoch for validation (default=valsplit*n_samples//batch_size)')

parser.add_argument('-nbatch',type=int,dest='nbatch',default=32,
                    help='Optional: number of images per batch (default=32)')

parser.add_argument('-nworkers',type=int,dest='nworkers',default=1,
                    help='Optional: number multiprocessing threads (default=1)')

parser.add_argument('-valsplit',type=float,dest='valsplit',default=0.2,
                    help='Optional: fraction of images to use for validation (default=0.2)')

parser.add_argument('-lr', type=float,dest='lr',default=0.0001,
                    help='Optional: learning rate for SGD (default=0.0001)')

parser.add_argument('-opt', type=str,dest='opti',default='SGD',
                    help='Optional: optimizer to use for training: SGD or Adam (default=SGD)')
                    
#parser.add_argument('-zca', dest='zca',action='store_true',
#                    help='Optional: use ZCA whitening (default=False)')

parser.add_argument('-ss', dest='ss',action='store_true',
                    help='Optional: standardize samples (default=False)')

parser.add_argument('-dropout', type=float,dest='dropout',default=None,
                    help='Optional: use dropout fraction (default=None)')

parser.add_argument('-unfreezelast', type=int,dest='unfreezelast',default=12,
                    help='Optional: layers from end of res_net model to unfreeze (default=12)\n' + 
                         '    (indicates layers from end, value supplied by user should be > 0)')

parser.add_argument('-modelfile',type=str,dest='modelname',default=None,
                    help=r'Optional: model name.  Default = ./model_{rootname}.h5')

parser.add_argument('-logfile',type=str,dest='logfile',default=None,
                    help=r'Optional: logfile name.  Default = ./log/log_{timestamp}.csv')

parser.add_argument('-noweights',dest='noweights',action='store_true',
                    help='Optional: do not use weights from previous model' +
                          ' (default: weights from previous model of same name will be used)')

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

#----------------------------------------------------------------------
args = parser.parse_args()

# prepare path information
trainDir = args.trainDir
testDir = args.testDir 
valDir = args.valDir

# check existance
assert os.path.exists(trainDir), f"Error:\n{trainDir}\ndid not exist!"
if testDir is not None: 
    assert os.path.exists(testDir), f"Error:\n{testDir}\ndid not exist!"

# setup information
time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
n_batch = args.nbatch
n_epochs = args.nepochs
n_workers = args.nworkers
valsplit = args.valsplit
learnrate = args.lr
unfreezelast = args.unfreezelast

# class count
class_names = sorted([d for d in os.listdir(trainDir) if not d.startswith('.')]) 
n_classes = len(class_names)
n_samples = sum([ len(f) for _,_,f in os.walk(trainDir)])

#test count information 
if testDir is not None:
    n_samples_t = sum([ len(f) for _,_,f in os.walk(testDir)])
    n_steps_t = n_samples_t // n_batch if args.nsteps is None else int(args.nsteps)

# model name 
basename = os.path.abspath(trainDir).split('/')[-1]
if args.modelname is None:
    model_file = f'./model_{basename}.h5'
else:
    model_file = args.modelname

# log file name
if args.logfile is None:
    logfile = f'./logs/log_{basename}_{time_stamp}.csv'
else:
    logfile = args.logfile

n_s = args.nsteps if args.nsteps is not None else 'not yet determined'
print(f"""
 ------------------------
Initial model Summary:
    model_file    ={model_file}
    ignore prev   ={args.noweights}
    learnrrate    ={learnrate}
    optimizer     ={args.opti}
    use_dropuot   ={args.dropout}
    use_ss        ={args.ss} 
    n_batch       ={n_batch}
    n_epochs      ={n_epochs}
    n_steps       ={n_s}  (per epoch)
    n_classes     ={n_classes}
    n_samples     ={n_samples}
    valsplit      ={valsplit}
    classes       =\n{class_names}\n
""")

if testDir is not None:
    print(f"""Test summary:
    n_samples_t   ={n_samples_t}
    n_steps_t     ={n_steps_t}
    ------------------------
    """)


# Default model
# import and reset model
model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
for layer in model_resnet.layers[:-unfreezelast]:
    layer.trainable = False  # layer from 0:n-unfreeze will be trained

# category classification
x = model_resnet.output
x = Dense(512, activation='elu', kernel_regularizer=l2(0.001))(x)
if args.dropout is not None: x = Dropout(args.dropout)(x)

y = Dense(n_classes, activation='softmax', name='img')(x)

# final model
final_model = Model(inputs=model_resnet.input, outputs=y)

# check if we should use previous training weights
if not args.noweights and os.path.exists(model_file):
    print('...found existing model, attempting to re-use weights')
    old_model = load_model(model_file)

    if old_model.get_config() == final_model.get_config():
        try:
            final_model.set_weights(old_model.get_weights())  
            print('...Successfully transfered previous model weights')
        except:
            print(f'...Failed to load previous training weights from {model_file}...skipping ')
    else:
        print('...Could not load previous model weights due to inconsistent architecture')

# optimization
if args.opti == 'Adam':
    opt = Adam(lr=learnrate, beta_1=0.9,beta_2=0.999,epsilon=1E-8)
else:
    opt = SGD(lr=learnrate, momentum=0.9, nesterov=True)

# compile
final_model.compile(optimizer=opt,
                loss={'img': 'categorical_crossentropy'},
                metrics={'img': ['accuracy', 'top_k_categorical_accuracy']}) # default: top-5

#if args.ss or args.zca:
#    zca_nbatchs = 1
#    zca_tr =   (ImageDataGenerator(rescale=1. / 255.,
#                                   rotation_range=30.,
#                                   shear_range=0.2,
#                                   zoom_range=0.2,
#                                   width_shift_range=0.2,
#                                   height_shift_range=0.2,
#                                   horizontal_flip=True).flow_from_directory(trainDir,
#                                                                            target_size=(200, 200),
#                                                                             batch_size=8, 
#                                                                             shuffle=True))
#
#    zca_tr_x = np.vstack(next(zca_tr)[0] for _ in range(zca_nbatchs))

if valDir is not None: valsplit = 0.0

train_datagen = ImageDataGenerator(
                                rescale= 1. / 255.,
                                rotation_range=30.,
                                samplewise_std_normalization=args.ss,
                                #zca_whitening=args.zca,
                                shear_range=0.2,
                                zoom_range=0.2,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                validation_split=valsplit,
                                horizontal_flip=True)

#if args.ss or args.zca:
#    train_datagen.fit(zca_tr_x)


# train and validation iterators
# this may take some time to load depending upon the sample sizes
print('...creating samples iterators')
print('\nTrain sample iterator:')
if valDir is None:
    train_iterator = train_datagen.flow_from_directory(directory=trainDir,
                                                       shuffle=True,
                                                       batch_size=n_batch,
                                                       seed= 1234,
                                                       class_mode='categorical',
                                                       subset='training',
                                                       target_size=(200, 200))

    print('\nValidation sample iterator:')
    val_iterator  =  train_datagen.flow_from_directory(directory=trainDir,
                                                       shuffle=True,
                                                       batch_size=n_batch,
                                                       seed= 1234,
                                                       class_mode='categorical',
                                                       subset='validation',
                                                       target_size=(200, 200))
else: 
    train_iterator = train_datagen.flow_from_directory(directory=trainDir,
                                                       shuffle=True,
                                                       batch_size=n_batch,
                                                       seed= 1234,
                                                       class_mode='categorical',
                                                       target_size=(200, 200))

    print('\nValidation sample iterator:')
    val_iterator  =  train_datagen.flow_from_directory(directory=valDir,
                                                       shuffle=True,
                                                       batch_size=n_batch,
                                                       seed= 1234,
                                                       class_mode='categorical',
                                                       target_size=(200, 200))


# determine step information
n_steps   = train_iterator.samples // n_batch if args.nsteps is None else int(args.nsteps)
n_steps_v = val_iterator.samples // n_batch if args.vnsteps is None else int(args.vnsteps) 

# save results
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               patience=12,
                               factor=0.5,
                               verbose=1)

tensorboard = TensorBoard(log_dir='./logs')
csvlogger   = CSVLogger(filename=logfile)

early_stopper = EarlyStopping(monitor='val_loss',
                            patience=30,
                            verbose=1)

checkpoint = ModelCheckpoint(model_file)

terminate_on_request = TerminateOnRequest()

final_model.fit_generator(train_iterator,
                          steps_per_epoch=n_steps,
                          epochs=n_epochs,
                          validation_data=val_iterator,
                          validation_steps=n_steps_v, 
                          verbose=2,
                          shuffle=True,
                          workers=n_workers,
                          callbacks=[lr_reducer, checkpoint, early_stopper, \
                                     tensorboard, csvlogger, terminate_on_request])

# test results
if testDir is not None:
    test_datagen =  ImageDataGenerator(rescale=1./255.)
    test_iterator = test_datagen.flow_from_directory(directory=testDir,
                                                      shuffle=True,
                                                      batch_size=n_batch,
                                                      seed= 1234,
                                                      class_mode='categorical',
                                                      target_size=(200, 200))


    scores = final_model.evaluate_generator(test_iterator, steps=n_steps_t)

    print('Final training Summary:\n' \
          '------------------------------')
    print('Image loss: ' + str(scores[0]))
    print('Image accuracy: ' + str(scores[1]))
    print('Top-5 image accuracy: ' + str(scores[2]))

print("Completed successfully!")