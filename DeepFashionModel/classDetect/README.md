# Summary of classDetect directory
For archiving purposes

- This folder contains the various training files

- train\_class.py : 
-  the main training script with various input options using the 'elu' activation for last layer
-  examples of use are given in the various .sh scripts

- train\_class\_relu.py :
-  same training examples, except using the 'relu' activation function for the last layer

- file paths are relative:
- data\\  : directory of unsorted classes, simply the image files in original directory as unpacked from the archive
- fdata\\ : directory of classes sorted by train/test/val and by category: upper/lower/full  by class: Blouse/Tee/Top/ etc. 


# Analysis scripts
For archiving purposes

- Build\_CropAndResizeImages.ipynb:
-  Script to crop and resize images based upon bounding box input

- PlotValidationError.ipynb
-  plot validation error of given model

- ResampleDataset.ipynb
-  Upsample or downsample dataset based on various class requirements

- Trial\_CropAndResizeImages.ipynb
-  other attempts at crop and resize...might not be useful

# further analysis aids
- iterationTracker.py
-  plot iterations from model using a given log.csv input

- predict\_and\_pickle.py
-  execute the predict\_on\_batch() function and build the y\_true vs y\_pred matrix for analysis with the ROC curve.  Values are pickled.  The process is slow, since it requires CPU processing for the python data storage side. 
