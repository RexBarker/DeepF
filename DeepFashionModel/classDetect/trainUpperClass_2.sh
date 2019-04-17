# run various training schedules for upper class model

#python train_class.py -traindir ../fdata_CropSQRS/train/upper/ -testdir ../fdata_CropSQRS/test/upper/ -modelfile model_upper_u16_lr0001_SGD.h5 -logfile log_u16_lr0001_SGD.csv -lr 0.0001 -unfreezelast 16 -nworkers 6

python train_class.py -traindir ../fdata_CropSQRS/train/upper/ -testdir ../fdata_CropSQRS/test/upper/ -modelfile model_upper_u16_lr0001_Adam_ss.h5 -logfile log_u16_lr0001_Adam_ss.csv -lr 0.0001 -opt Adam -ss -unfreezelast 16 -nworkers 6

python train_class.py -traindir ../fdata_CropSQRS/train/upper/ -testdir ../fdata_CropSQRS/test/upper/ -modelfile model_upper_u16_lr0001_Adam_ss_do.h5 -logfile log_u16_lr0001_Adam_ss_do.csv -lr 0.0001 -opt Adam -ss -dropout 0.5 -unfreezelast 16 -nworkers 6

python train_class.py -traindir ../fdata_CropSQRS/train/upper/ -testdir ../fdata_CropSQRS/test/upper/ -modelfile model_upper_u16_lr00001_Adam.h5 -logfile log_u16_lr00001_Adam.csv -lr 0.00001 -opt Adam -unfreezelast 16 -nworkers 6


#python train_class.py -traindir ../fdata_CropSQRS/train/upper/ -testdir ../fdata_CropSQRS/test/upper/ -modelfile model_upper_u16_lr0001_SGDssdo.h5 -logfile log_u16_lr0001_SGDssdo.csv -lr 0.0001 -ss -dropout 0.5 -unfreezelast 16 -nworkers 6
