#!/usr/bin/bash
# original used with overall file list with splits for multi-processing

python extractFocalPerson.py --sourcepath ../fashion_data/train --refinepath ../fdata_refine/train --imagefilelist ../fashion_data/train_filelist.txt --startindex 0 --incrementindex 3 &
sleep 60
python extractFocalPerson.py --sourcepath ../fashion_data/train --refinepath ../fdata_refine/train --imagefilelist ../fashion_data/train_filelist.txt --startindex 1 --incrementindex 3 &
python extractFocalPerson.py --sourcepath ../fashion_data/train --refinepath ../fdata_refine/train --imagefilelist ../fashion_data/train_filelist.txt --startindex 2 --incrementindex 3 &
