# DeepF (Deep Fashion)

### Background
Fashion analysis based on the Deep Fashion Dataset.  The following nomenclature applies
- 'category':  clothing classification into 'upper', 'lower', and 'full' body clothing
- 'class'   :  within categories, the different classes of clothing items (e.g. 'Tee', 'Blouse', etc.)

### Setup Environment

This project assumes that you have setup your environment already. This project is based on the
following main dependencies (this is at the time of execution.  Newer versions may also work):
- **classDetect**, **classDetectKinli** : 
   python=3.6.7
   tensorflow-gpu=1.11.0
   keras=2.2.4

- **keras-frcnn** :
   python=3.6.8
   tensorflow-gpu=1.8.0
   keras=2.2.0

(note:  this older version of keras/tensorflow was required as there is a bug in the newer version
which causes a fatal error during model training)

- hints:  I used the anaconda python builds with seperate environments for both models,
 on a Ubuntu 18.04 LTS with Xeon 6-core 3.5 GHz, 12 GB RAM, NVIDIA GTX 970 4GB GPU, Cuda 9.0
- Other systems may work, but GPU training is a must (w/o GPU training, the simple model required > 2 weeks of constant computation on a 4-core Ubuntu laptop and still didn't reach convergence :-/)    

### Download DeepFashion Dataset 
- exercise left to the student... here's basically how you do it:
```sh
# http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html

# The directory structure after downloading and extracting dataset:
# data/
# ---Anno
# ------list_attr_cloth.txt
# ------list_attr_img.txt
# ------list_bbox.txt
# ------list_category_cloth.txt
# ------list_category_img.txt
# ------list_landmarks.txt
# ---Eval
# ------list_eval_partition.txt
# ---Img
# ------img
```

### Create Dataset
-  Once the main archive is unpacked into the ./DeepFashionModel/data/ directory, utilise the directions
 in the ./DeepFashionModel/prepareData/ directory.

### Model Training

- **classDetect**      : ./DeepFashionModel/classDetect/
- **classDetectKinli** : ./DeepFashionModel/classDetectKinli/
- **keras-frcnn**      : ./DeepFashionModel/keras-frcnn/

### Model Analysis

After analysis, utilise the analysis scripts as given in the ./DeepFashionAnalysis/ directories



### RESULTS
***classDetectKinli*** :
![alt text](https://raw.githubusercontent.com/RexBarker/DeepF/blob/master/Results/DeepFashionResult.png "DeepFashion datset")

***keras-frcnn***      :
![alt text](https://raw.githubusercontent.com/RexBarker/DeepF/blob/master/Results/StreetStyleResult.png "StreetStyle datset")


### Acknowledgment
- [DeepFashion Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
- [StreetStyle Dataset](http://streetstyle.cs.cornell.edu/index.html)
