#######################################################################
# PrepareFiles.py
#  - Purpose:  to sort files into various locations for the model 
#              operation
#######################################################################

import shutil
import os
from pathlib import Path
import re
import cv2

# will use them for creating custom directory iterator
import numpy as np

# regular expression for splitting by whitespace
splitter = re.compile("\s+")
base_path = Path.cwd()
source_path = base_path.joinpath('./fashion_data/Img/')
work_path = base_path.joinpath('./data/')

def process_folders():
    # Read the relevant annotation file and preprocess it
    # Assumed that the annotation files are under '<project folder>/data/anno' path
    with open('./fashion_data/Eval/list_eval_partition.txt', 'r') as eval_partition_file:
        list_eval_partition = [line.rstrip('\n') for line in eval_partition_file][2:]
        list_eval_partition = [splitter.split(line) for line in list_eval_partition]
        list_all = [(v[0][4:], v[0].split('/')[1].split('_')[-1], v[1]) for v in list_eval_partition]
    
    # create main folders
    tasks =set([v[2] for v in list_all])
    styles=set([v[1] for v in list_all]) 
    for t in tasks:
        dirA = work_path.joinpath(t)
        if not dirA.exists(): dirA.mkdir()
        for s in styles:
            dirB = dirA.joinpath(s)
            if not dirB.exists(): dirB.mkdir()
    
    # copy files
    for f,s,t in list_all:
        dn,fn = f.split('/')
        source = source_path.joinpath(f)
        destpath = work_path.joinpath(t).joinpath(s).joinpath(dn)
        if not destpath.exists(): destpath.mkdir()
        shutil.copy(source,destpath.joinpath(fn))


if __name__ == "__main__":
    process_folders()
