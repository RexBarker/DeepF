# given an input model, run batch prediction and pickle results
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-model',type=str,dest='modelfile',required=True,
                    help='Required: model file')
parser.add_argument('-testdir',type=str,dest='testdir',required=True,
                    help='Required: test directory')
parser.add_argument('-nsteps',type=int,dest='nsteps',default = 2000,
                    help='Optional: number of steps (def=2000)')
parser.add_argument('-nbatch',type=int,dest='nbatch',default = 32,
                    help='Optional: number of images per batch (def=32)')

args = parser.parse_args()

assert os.path.exists(args.modelfile), "Could not find model file!"
resultFile = os.path.basename(args.modelfile).split('.')[0] + '.pickle'

model = load_model(args.modelfile)

test_datagen =  ImageDataGenerator(rescale=1./255.)
test_iterator = test_datagen.flow_from_directory(directory=args.testdir,
                                                shuffle=False,
                                                batch_size=args.nbatch,
                                                seed= 1234,
                                                class_mode='categorical',
                                                target_size=(200, 200))

# collect batch predictions
y_test_col = None
y_pred_col = None

for _ in range(args.nsteps):
    x_test,y_test = next(test_iterator)
    y_pred = model.predict_on_batch(x_test)
    
    if y_test_col is None:
        y_test_col = y_test
        y_pred_col = y_pred
    else:
        y_test_col = np.append(y_test_col,y_test,axis=0)
        y_pred_col = np.append(y_pred_col,y_pred,axis=0)


# dump to a pickle file, for use with ROC curve producer
with open(resultFile, 'wb') as fp:
    pickle.dump((y_test_col,y_pred_col),fp)