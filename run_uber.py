# -*- coding: utf-8 -*-
from __future__ import print_function
import cPickle as pickle
import numpy as np
import os, sys, math
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from mktnets.models.MKTNet import mktnet
import mktnets.models.metrics as metrics
from mktnets.datasets import uber
np.random.seed(0)  # for reproducibility

data_path = sys.path[0] + "/mktnets/datasets/"
nb_epoch = 500 # number of epoch at training stage
nb_epoch_cont = 100 # number of epoch at training (cont) stage
batch_size = 20  # batch size
T = 24  # number of time intervals in one day

lr = 0.0001  # learning rate
len_closeness = 4  # length of closeness dependent sequence
len_period = 4  # length of peroid dependent sequence
len_trend = 4  # length of trend dependent sequence
nb_residual_unit = 10  # number of residual units

nb_flow = 2  # demand and supply
days_test = 5 # number of test days (total 40 days)
len_test = T * days_test
map_height, map_width = 20, 29  # grid size
nb_area = 20 * 29
m_factor = math.sqrt(1. * map_height * map_width / nb_area)
print('factor: ', m_factor)
path_result = sys.path[0] + "/results"
path_model = sys.path[0] + "/model"

if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)

def build_model(external_dim):
    c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height,
              map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None

    model = mktnet(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                   external_dim=external_dim, nb_residual_unit=nb_residual_unit)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()
    return model


def main():
    # load data
    print("loading data...")
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = \
        uber.load_data(T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period,
                       len_trend=len_trend, len_test=len_test, meta_data=True, path = data_path)

    print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])

    print('=' * 10)
    print("compiling model...")
    print(
        "**at the first time, it takes a few minites to compile if you use [Theano] as the backend**")
    model = build_model(external_dim)
    hyperparams_name = 'c{}.p{}.t{}.resunit{}.lr{}'.format(
        len_closeness, len_period, len_trend, nb_residual_unit, lr)
    fname_param = os.path.join(sys.path[0], 'model', '{}.best.h5'.format(hyperparams_name))

    early_stopping = EarlyStopping(monitor='val_rmse', patience=5, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    print('=' * 10)
    print("training model...")
    history = model.fit(X_train, Y_train,
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    model.save_weights(os.path.join(sys.path[0],
        'model', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(sys.path[0],
        path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))

    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')

    model.load_weights(fname_param)
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                           0] // 48, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))

    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))

    print('=' * 10)
    print("training model (cont)...")
    fname_param = os.path.join(sys.path[0],
        'model', '{}.cont.best.h5'.format(hyperparams_name))
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')
    history = model.fit(X_train, Y_train, nb_epoch=nb_epoch_cont, verbose=1, batch_size=batch_size, callbacks=[
                        model_checkpoint], validation_data=(X_test, Y_test))
    pickle.dump((history.history), open(os.path.join(sys.path[0], 
        path_result, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
    model.save_weights(os.path.join(sys.path[0], 
        'model', '{}_cont.h5'.format(hyperparams_name)), overwrite=True)

    print('=' * 10)
    print('evaluating using the final model')
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                           0] // 48, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))

    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))

if __name__ == '__main__':
    main()
