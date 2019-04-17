# run various training schedules for upper class model

python train_class.py -traindir ../fashion_data_upper/train/ -testdir ../fashion_data_upper/test/ -valdir ../fashion_data_upper/val/ -modelfile model_upperFullDatRelu_u16_lr0001_SGD.h5 -logfile log_upperFullDatRelu_u16_lr0001_SGD.csv -lr 0.0001 -unfreezelast 16 -nworkers 6 -nsteps 2000 -vnsteps 200

