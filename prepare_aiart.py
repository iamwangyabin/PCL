



# prepare datasets for aiart

train_x.shape
(50000, 640)

(Pdb) train_y
array([90, 15, 51, ..., 48, 88, 24])
(Pdb) train_y.max()
99
(Pdb) train_y==1
array([False, False, False, ..., False, False, False])
(Pdb) sum(train_y==1)
500



考虑到


