————————————————————————————————————————————————————————————————————————
Market Networks: Understand Ride Sharing from Deep Learning Perspectives
————————————————————————————————————————————————————————————————————————

————————————————————————————————————————————————————————————————————————
Introduction:
Final Project for CS6200 Machine Learning, Spring 2017
Shan Jiang (sjiang@ccs.neu.edu)
Dan Guo (guo.dan@husky.neu.edu)
————————————————————————————————————————————————————————————————————————

————————————————————————————————————————————————————————————————————————
Setup:
Environments: Linux/Mac/Windows + Python 2.7
Requirements: Keras==1.2, Theano, h5py
Update keras.json: {"image_dim_ordering": "th", "epsilon": 1e-07, "floatx": "float32", "backend": "theano"}
Note: Only keras 1.2 with above setting is supported, other version of keras or tensorflow will report error.
————————————————————————————————————————————————————————————————————————

————————————————————————————————————————————————————————————————————————
Run:
$ python run_uber.py
————————————————————————————————————————————————————————————————————————

————————————————————————————————————————————————————————————————————————
Results:
ARMA: 6.089453
KNNR: 5.078249
LinReg: 4.561939
MktNet: 2.286068
————————————————————————————————————————————————————————————————————————