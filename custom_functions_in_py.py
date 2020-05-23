import numpy as np
import scipy.stats
from sklearn import preprocessing
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sliced import SlicedInverseRegression
from sklearn.random_projection import *

import warnings
from sklearn.exceptions import DataConversionWarning
###############
#%%
###############


def raw_ds_gen(n=500, p=500, model_features=3, alpha=1.9,
               tr_te_split=0.5, tr_val_split=0.0):
    # p=100; k=7; L=3; n=1000; T=n; alpha=1.7
    # tr_te_split=0.5; tr_val_split = 0.15

    # T = n

    ## this is the old "func": 
    # func = np.vectorize(lambda f1,f2,f3,e: f1+3*f2-2*f3+1+e)
    ## this is the new "func":
    func = np.vectorize(lambda f1, f2, f3, e: f1 * (f2 + f3 + 1) + e)
    aalpha = np.random.uniform(low=.2, high=.8, size=model_features)
    rho = np.random.uniform(low=.2, high=.8, size=[p])
    A = scipy.stats.levy_stable.rvs(alpha/2, 1, loc=0,
                                  scale=np.cos(np.pi*alpha/4)**(2/alpha),
                                  size=[n])
    G = np.random.normal(0, 1, size=[n])
    epsilon = A ** (1 / 2) * G

    A=scipy.stats.levy_stable.rvs(alpha/2,1,loc=0,
                                  scale=np.cos(np.pi*alpha/4)**(2/alpha),
                                  size=[p,n])
    G=np.random.normal(0,1,size=[p,n])
    nu=A**(1/2)*G

    A=scipy.stats.levy_stable.rvs(alpha / 2, 1, loc=0,
                                  scale=np.cos(np.pi*alpha/4)**(2/alpha),
                                  size=[model_features, n])
    G=np.random.normal(0, 1, size=[model_features, n])
    e=A**(1/2)*G

    F=np.zeros((model_features, n))
    for j in range(model_features):
      for t in range(1,n):
        F[j,t]=F[j,t-1]*aalpha[j]+e[j,t]
    #    F[j,t]=e[j,t]

    U=np.zeros((p,n))
    for i in range(p):
      for t in range(1,n):
        U[i,t]=U[i,t-1]*rho[i]+nu[i,t]

    B=np.random.normal(0, 1, size=[p, model_features])
    y=func(F[0,],F[1,],F[2,],epsilon)
    X=np.matmul(B,F)+U
    
    
    n_train = np.int(np.floor(tr_te_split*len(y)))
    X_train, X_test = np.split(X,[n_train],axis=1)
    y_train, y_test = np.split(y,[n_train])

    n_validation = np.int(np.floor(tr_val_split*len(y_train)))
    X_val, X_train = np.split(X_train,[n_validation],axis=1)
    y_val, y_train = np.split(y_train,[n_validation])

    return X_train, X_val, X_test, y_train, y_val, y_test

###############
#%%
###############
# min_max_multiplier = 9


def ripe_ds_gen(n=500, p=500, model_features=3, alpha=1.9,
                tr_te_split=0.5, tr_val_split=0.0,
                min_max_multiplier=9):

    X_train, X_val, X_test, y_train, y_val, y_test = raw_ds_gen(n=n,
                                                                p=p,
                                                                model_features=model_features,
                                                                alpha=alpha,
                                                                tr_te_split=tr_te_split,
                                                                tr_val_split=tr_val_split)

    X_train = X_train.transpose(); X_test = X_test.transpose()
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_scaled = 1 + min_max_scaler.fit_transform(X_train) * min_max_multiplier
    X_test_scaled = 1 + min_max_scaler.fit_transform(X_test) * min_max_multiplier
    X_train_ln = np.log(X_train_scaled); X_train_ln_sqrt = np.sqrt(X_train_ln)
    X_test_ln = np.log(X_test_scaled); X_test_ln_sqrt = np.sqrt(X_test_ln)

    X_train_tot = np.hstack((X_train, X_train_ln, X_train_ln_sqrt))
    X_test_tot = np.hstack((X_test, X_test_ln, X_test_ln_sqrt))


    y_train_scaled = 1 + min_max_scaler.fit_transform(y_train.reshape(-1, 1)) * min_max_multiplier
    y_test_scaled = 1 + min_max_scaler.fit_transform(y_test.reshape(-1, 1)) * min_max_multiplier
    y_train_ln = np.log(y_train_scaled); y_train_ln_sqrt = np.sqrt(y_train_ln)
    y_test_ln = np.log(y_test_scaled); y_test_ln_sqrt = np.sqrt(y_test_ln)


    output_dict = {}
    output_dict['X_train_raw'] = X_train
    output_dict['X_test_raw'] = X_test
    output_dict['X_train_ripe'] = X_train_tot
    output_dict['X_test_ripe'] = X_test_tot
    output_dict['y_train'] = y_train
    output_dict['y_test'] = y_test
    output_dict['y_train_ln'] = y_train_ln
    output_dict['y_test_ln'] = y_test_ln
    output_dict['y_train_ln_sqrt'] = y_train_ln_sqrt
    output_dict['y_test_ln_sqrt'] = y_test_ln_sqrt

    return output_dict

###############
#%%
###############

def after_sir_ds_gen(alpha=1.9, pca_degree=2, pca_features=10,  # Why 40?? maybe 7-10 is better
                     sir_features=9, y_trans_type=''):
    # y_trans_type --> '' , 'ln' , 'ln-sqrt'
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    ds_dict = ripe_ds_gen(alpha=alpha)
    X_train = ds_dict['X_train_ripe']
    X_test = ds_dict['X_test_ripe']

    scaler_z = StandardScaler()
    pca_obj = KernelPCA(kernel='poly', degree=pca_degree, random_state=133)

    X_train_pca = pca_obj.fit_transform(scaler_z.fit_transform(X_train))[:, :pca_features]
    X_train_pca = scaler_z.fit_transform(X_train_pca)
    X_test_pca = pca_obj.transform(scaler_z.fit_transform(X_test))[:, :pca_features]
    X_test_pca = scaler_z.fit_transform(X_test_pca)

    if y_trans_type == '':
        y_train = ds_dict['y_train']
        y_test = ds_dict['y_test']

    elif y_trans_type == 'ln':
        y_train = ds_dict['y_train_ln']
        y_test = ds_dict['y_test_ln']

    elif y_trans_type == 'ln-sqrt':
        y_train = ds_dict['y_train_ln_sqrt']
        y_test = ds_dict['y_test_ln_sqrt']
    else:
        raise ValueError

    y_train = scaler_z.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler_z.fit_transform(y_test.reshape(-1, 1))

    sir_obj = SlicedInverseRegression(n_directions=sir_features)

    X_train_sir = sir_obj.fit_transform(X_train_pca, y_train)
    X_test_sir = sir_obj.transform(X_test_pca)

    return X_train_sir, X_test_sir, y_train, y_test

###############
#%%
###############
# Unit Test:
# X_train_sir, X_test_sir, y_train, y_test = after_sir_ds_gen(pca_degree=2,
#                                                             pca_features=40,
#                                                             sir_features=9,
#                                                             y_trans_type='')

###############
#%%
###############

def after_rp_sir_ds_gen(alpha=1.9,
                        RP_type='Sparse', RP_features=7, sir_features=3):
    '''
    RP_type = 'Sparse' or 'Gaussian'
    '''

    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    ds_dict = ripe_ds_gen(alpha=alpha)
    X_train = ds_dict['X_train_ripe']
    X_test = ds_dict['X_test_ripe']

    scaler_z = StandardScaler()
    if RP_type == 'Sparse':
        rp_obj = SparseRandomProjection(n_components=RP_features)
    elif RP_type == 'Gaussian':
        rp_obj = GaussianRandomProjection(n_components=RP_features)
    else:
        error_msg = 'Type of RandomProjection is not identified.'
        raise ValueError(error_msg)
    X_train_rp = rp_obj.fit_transform(scaler_z.fit_transform(X_train))
    X_train_rp = scaler_z.fit_transform(X_train_rp)
    X_test_rp = rp_obj.transform(scaler_z.fit_transform(X_test))
    X_test_rp = scaler_z.fit_transform(X_test_rp)

    y_train = ds_dict['y_train']
    y_test = ds_dict['y_test']

    y_train = scaler_z.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler_z.fit_transform(y_test.reshape(-1, 1))

    sir_obj = SlicedInverseRegression(n_directions=sir_features)

    X_train_sir = sir_obj.fit_transform(X_train_rp, y_train)
    X_test_sir = sir_obj.transform(X_test_rp)

    return X_train_sir, X_test_sir, y_train, y_test

###############
#%%
###############


def ripe_ds_gen_from_file(X_input, y_input,
                          min_max_multiplier=9,
                          tr_te_split=0.5):

    n_train = np.int(np.floor(tr_te_split*len(y_input)))
    X_train, X_test = np.split(X_input, [n_train])
    y_train, y_test = np.split(y_input, [n_train])


    # X_train = X_train.transpose(); X_test = X_test.transpose()
    # print('X_train.shape, X_test.shape, y_train.shape, y_test.shape:',
    # X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_scaled = 1 + min_max_scaler.fit_transform(X_train) * min_max_multiplier
    X_test_scaled = 1 + min_max_scaler.fit_transform(X_test) * min_max_multiplier
    X_train_ln = np.log(X_train_scaled); X_train_ln_sqrt = np.sqrt(X_train_ln)
    X_test_ln = np.log(X_test_scaled); X_test_ln_sqrt = np.sqrt(X_test_ln)

    X_train_tot = np.hstack((X_train, X_train_ln, X_train_ln_sqrt))
    X_test_tot = np.hstack((X_test, X_test_ln, X_test_ln_sqrt))


    y_train_scaled = 1 + min_max_scaler.fit_transform(y_train.reshape(-1, 1)) * min_max_multiplier
    y_test_scaled = 1 + min_max_scaler.fit_transform(y_test.reshape(-1, 1)) * min_max_multiplier
    y_train_ln = np.log(y_train_scaled); y_train_ln_sqrt = np.sqrt(y_train_ln)
    y_test_ln = np.log(y_test_scaled); y_test_ln_sqrt = np.sqrt(y_test_ln)


    output_dict = {}
    output_dict['X_train_raw'] = X_train
    output_dict['X_test_raw'] = X_test
    output_dict['X_train_ripe'] = X_train_tot
    output_dict['X_test_ripe'] = X_test_tot
    output_dict['y_train'] = y_train
    output_dict['y_test'] = y_test
    output_dict['y_train_ln'] = y_train_ln
    output_dict['y_test_ln'] = y_test_ln
    output_dict['y_train_ln_sqrt'] = y_train_ln_sqrt
    output_dict['y_test_ln_sqrt'] = y_test_ln_sqrt

    return output_dict

###############
#%%
###############


def after_sir_ds_gen_from_file(X_input, y_input,
                               pca_degree=2,
                               pca_features=10,  # Why 40?? maybe 7-10 is better
                               sir_features=9,
                               y_trans_type=''):
    # y_trans_type --> '' , 'ln' , 'ln-sqrt'
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    ds_dict = ripe_ds_gen_from_file(X_input=X_input, y_input=y_input)
    X_train = ds_dict['X_train_ripe']
    X_test = ds_dict['X_test_ripe']

    # print(type(X_train), X_train.shape)
    # print(
    # np.any(np.isnan(X_train)),
    # np.all(np.isfinite(X_train)),
    # np.any(np.isnan(X_test)),
    # np.all(np.isfinite(X_test))
    # )


    scaler_z = StandardScaler()
    pca_obj = KernelPCA(kernel='poly', degree=pca_degree, random_state=133)

    X_train_pca = pca_obj.fit_transform(scaler_z.fit_transform(X_train))[:, :pca_features]
    X_train_pca = scaler_z.fit_transform(X_train_pca)
    X_test_pca = pca_obj.transform(scaler_z.fit_transform(X_test))[:, :pca_features]
    X_test_pca = scaler_z.fit_transform(X_test_pca)

    if y_trans_type == '':
        y_train = ds_dict['y_train']
        y_test = ds_dict['y_test']

    elif y_trans_type == 'ln':
        y_train = ds_dict['y_train_ln']
        y_test = ds_dict['y_test_ln']

    elif y_trans_type == 'ln-sqrt':
        y_train = ds_dict['y_train_ln_sqrt']
        y_test = ds_dict['y_test_ln_sqrt']
    else:
        raise ValueError

    y_train = scaler_z.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler_z.fit_transform(y_test.reshape(-1, 1))

    sir_obj = SlicedInverseRegression(n_directions=sir_features)

    X_train_sir = sir_obj.fit_transform(X_train_pca, y_train)
    X_test_sir = sir_obj.transform(X_test_pca)

    return X_train_sir, X_test_sir, y_train, y_test

###############
#%%
###############


def after_pca_ds_gen(alpha=1.9, pca_degree=2, pca_features=10):  # Why 40?? maybe 7-10 is better

    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    ds_dict = ripe_ds_gen(alpha=alpha)
    X_train = ds_dict['X_train_ripe']
    X_test = ds_dict['X_test_ripe']

    scaler_z = StandardScaler()
    pca_obj = KernelPCA(kernel='poly', degree=pca_degree, random_state=133)

    X_train_pca = pca_obj.fit_transform(scaler_z.fit_transform(X_train))[:, :pca_features]
    X_train_sir = scaler_z.fit_transform(X_train_pca)
    X_test_pca = pca_obj.transform(scaler_z.fit_transform(X_test))[:, :pca_features]
    X_test_sir = scaler_z.fit_transform(X_test_pca)

    y_train = ds_dict['y_train']; y_test = ds_dict['y_test']

    y_train = scaler_z.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler_z.fit_transform(y_test.reshape(-1, 1))

    return X_train_sir, X_test_sir, y_train, y_test

###############
#%%
###############


def after_pca_ds_gen_from_file(X_input, y_input,
                               pca_degree=2,
                               pca_features=10):  # Why 40?? maybe 7-10 is better

    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    ds_dict = ripe_ds_gen_from_file(X_input=X_input, y_input=y_input)
    X_train = ds_dict['X_train_ripe']; X_test = ds_dict['X_test_ripe']
    scaler_z = StandardScaler()
    pca_obj = KernelPCA(kernel='poly', degree=pca_degree, random_state=133)

    X_train_pca = pca_obj.fit_transform(scaler_z.fit_transform(X_train))[:, :pca_features]
    X_train_sir = scaler_z.fit_transform(X_train_pca)
    X_test_pca = pca_obj.transform(scaler_z.fit_transform(X_test))[:, :pca_features]
    X_test_sir = scaler_z.fit_transform(X_test_pca)

    y_train = ds_dict['y_train']; y_test = ds_dict['y_test']

    y_train = scaler_z.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler_z.fit_transform(y_test.reshape(-1, 1))

    return X_train_sir, X_test_sir, y_train, y_test
