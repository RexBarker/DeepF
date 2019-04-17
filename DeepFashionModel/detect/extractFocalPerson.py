# Extract focal person from image which coincides with the defined bbox
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os
import numpy as np
import json
import argparse 
from imageai.Detection import ObjectDetection
from PIL import Image

# adapted from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_IoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# determine new BB for selected picture
def bb_Refine(bbInner,bbBase):
    x1i, y1i, x2i, y2i = bbInner
    x1b, y1b, x2b, y2b = bbBase
    bbInnerNew = [ int(max(x1i - x1b,0)),
                   int(max(y1i - y1b,0)),
                   int(min(x2i - x1b, x2b - x1b)),
                   int(min(y2i - y1b, y2b - y1b))]
    
    return bbInnerNew

# ------------------------------------------------------------------- #

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sourcepath',dest='sourcepath',required=True)
    parser.add_argument('--refinepath',dest='refinepath',required=True)
    parser.add_argument('--startindex',dest='startindex',default=0)
    parser.add_argument('--incrementindex',dest='incrementindex',default=1)
    parser.add_argument('--imagefilelist',dest='imagefilelist',default=None)
    parser.add_argument('--modelpath',dest='modelpath',default='../model/')
    parser.add_argument('--bboxfile',dest='bboxfile',default='../data/Anno/list_bbox.txt')
    parser.add_argument('--testimage',dest='testimage',default=None) 
    parser.add_argument('--showresult',dest='showresult',action='store_true') 

    args = parser.parse_args()
        
    sourcePath = args.sourcepath
    assert os.path.exists(args.sourcepath), f'\nSource path={args.sourcepath}\n...Did not exist\n'

    refinePath = args.refinepath 
    assert os.path.exists(args.refinepath), f'\nRefine path={args.refinepath}\n...Did not exist\n'
    
    bboxFile = args.bboxfile
    assert os.path.exists(args.bboxfile), f'\nAnnotation file={args.bboxfile}\n...Did not exist\n'

    modelPath = os.path.join(args.modelpath,'yolo.h5')
    assert os.path.exists(modelPath),f"\nCould not find 'yolo.h5' at specified model path={modelPath}\n"
    
    testImage = None
    if args.testimage is not None: 
        testImage = args.testimage
        assert os.path.exists(os.path.join(sourcePath,testImage)), f'Test image file={args.testimage}\n...Did not exist\n'
    else:
        testImage='Anorak/Hooded_Cotton_Canvas_Anorak/img_00000131.jpg'
        if not os.path.exists(os.path.join(sourcePath,testImage)): testImage = None

    imageFiles = []
    dirSet = {}
    if args.imagefilelist is not None:
        assert os.path.exists(args.imagefilelist), f'\nImage list file={args.imagefilelist}\n...Did not exist\n'
        starti = int(args.startindex)
        inci = int(args.incrementindex)
        with open(args.imagefilelist,'r') as imgf:
            imageFiles = [ line.rstrip('\n') for line in imgf][ starti: :inci]
            dirSet = { imgf.split('/')[-2] for imgf in imageFiles}  # list of directories to be created
    else:
        imageFiles = [testImage]


    # load dictionary of bbox annotations
    with open(bboxFile, 'r') as bbox_file:
        list_bbox = [line.rstrip('\n') for line in bbox_file][2:]
        list_bbox = [line.split('/') for line in list_bbox]
        bb_dict= {l[1].split('_')[-1] + '/'+ l[1] + '/' + l[2].split()[0]: [int(xi) for xi in l[2].split()[-4:]] for l in list_bbox}
    
    # check for existance of target directories, create them if they don't exist
    for d in dirSet:
        dirA = os.path.join(refinePath,d.split('_')[-1])
        dirB = os.path.join(dirA,d)
        if not os.path.exists(dirA): os.mkdir(dirA)
        if not os.path.exists(dirB): os.mkdir(dirB)

    # dictionary of actual bboxes as determined from the refinement.  
    bb_dict_refine = {} 

    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(modelPath) 
    detector.loadModel()
    custom_objects = detector.CustomObjects(person=True)

    for imgFile in imageFiles:
        try:
            imgmain, imgitems, imgind = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, 
                                                                              input_image= os.path.join(sourcePath, imgFile), 
                                                                              output_type='array',
                                                                              extract_detected_objects=True,
                                                                              minimum_percentage_probability=80)
        except:
            continue
    
        boxItem = bb_dict[imgFile]
        IoUs = np.array([bb_IoU(boxItem, person['box_points']) for person in imgitems])
        if IoUs.any():
            # person found, select item with maximum IoU
            person = imgind[IoUs.argmax()] #if len(imgind) > 1 else plt.imread(os.path.join(orig_path,test_file)) 
            x1, y1, x2, y2 = bb_Refine(boxItem,imgitems[IoUs.argmax()]['box_points'])
        else:
            # no person found, use original image
            person = np.array(Image.open(os.path.join(sourcePath,imgFile)))
            x1, y1, x2, y2 = boxItem  # use original bbox
        
        bb_dict_refine[imgFile] = [x1, y1, x2, y2] 
        im = Image.fromarray(person)
        im.save(os.path.join(refinePath,imgFile))

        if args.showresult:
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1,edgecolor='r',facecolor='none')
            fig,ax = plt.subplots(1,figsize=(12,12))
            ax.imshow(person)
            ax.add_patch(rect)
            plt.show()
    
    with open(os.path.join(refinePath,f'bbox_refine_{args.startindex}.json'),'w') as bbf:
        json.dump(bb_dict_refine,bbf,indent=4)
        
         
