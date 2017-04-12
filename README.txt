Market Networks: Understand Ride Sharing Market from A Deep Learning Perspective
CS6200 Machine Learning - Final Project
Shan Jiang (sjiang@ccs.neu.edu)
Dan Guo (guo.dan@husky.neu.edu)

Setup:
1) Install: Keras==1.2, Theano, h5py
2) Update keras.json: {"image_dim_ordering": "th", "epsilon": 1e-07, "floatx": "float32", "backend": "theano"}
Note: Only keras 1.2 with above setting is supported, other version of keras or tensorflow will report error.

Run:
$ python run_uber.py

Results:
ARMA: 6.089453
KNNR: 5.078249
LinReg: 4.561939
MktNet: 2.105124