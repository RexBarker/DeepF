# Training script for the classDetectKinkli model
- model based upon the blog :
https://medium.com/@birdortyedi\_23820/deep-learning-lab-episode-4-deep-fashion-2df9e15a63e1

- Model preparation scripts in the directory ../prepareData

- this builds the model as based on the Kinli blog using hardcoded references
  In the current configuration, this runs with Tensorflow-gpu with the restriction to 'workers=1'. I tried more than this, but it would not function (for the 'multithreading=True' option).  Also, trying to use threadsafing didn't work either.  The workers > 1 option works with Tensforflow-cpu with some complaints.

- currently, the training layers is restricted to -12 based on the original Kinli blog
- model files and data are not included with this repo due to the size

- use the iterationTracker.py file to track the iteration progress during training (takes the log.csv file as an input)


