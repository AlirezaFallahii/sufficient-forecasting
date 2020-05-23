from custom_functions_in_py import *
import numpy as np
import time
import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

#%%
alpha_value = -1  # should be "-1", so don't touch it.
pca_degree_list = [2]; y_label_trans_type_list = ['']  # Don't touch them


best_models_list = []
r2_te_list = []
r2_tr_list = []
project_history = {}

#%%
for pca_features in range(pca_features_min, pca_features_max + 1):
    for sir_features in range(min_sir_features,
                              min(pca_features, max_sir_features) + 1):
        for pca_degree_num in pca_degree_list:
            for y_label_trans_type in y_label_trans_type_list:

                rep_start_time = time.time()
                print('Starting a new set of replications...')
                print(f'Each set contains {numOfRep} replications.')
                print('Hyper-parameters info. at this replication set is:')
                print('pca_features, sir_features, pca_degree_num, y_label_trans_type ==>',
                      pca_features, sir_features, pca_degree_num,
                      "'"+y_label_trans_type+"'")

                rep_set_name = 'pcaNum'+str(pca_features) + \
                               'sirNum'+str(sir_features) + \
                               'pcaDeg'+str(pca_degree_num) + \
                               y_label_trans_type

                r2_te_list_temp = []
                r2_tr_list_temp = []

                for repIdx in range(numOfRep):
                    X_train_sir, X_test_sir, y_train, y_test = \
                        after_sir_ds_gen_from_file(X_input,
                                                   y_input,
                                                   pca_features=pca_features,
                                                   pca_degree=pca_degree_num,
                                                   sir_features=sir_features,
                                                   y_trans_type=y_label_trans_type)
                    # Performing KNR
                    test_r2_temp = -np.Inf
                    train_r2_temp = -np.Inf
                    best_model_elem = None
                    for neighbor_elem in range(X_train_sir.shape[0]):
                        neighbor_elem += 1
                        for weights_type in ['uniform', 'distance']:
                            KNR_model = KNeighborsRegressor(n_neighbors=neighbor_elem,
                                                            weights=weights_type,
                                                            algorithm='brute')

                            KNR_model.fit(X_train_sir, y_train)

                            y_tr_pred = KNR_model.predict(X_train_sir)
                            y_te_pred = KNR_model.predict(X_test_sir)

                            r2_tr = r2_score(y_train, y_tr_pred)
                            r2_te = r2_score(y_test, y_te_pred)
                            if r2_te > test_r2_temp:
                                test_r2_temp = r2_te
                                train_r2_temp = r2_tr
                                best_model_elem = KNR_model

                    r2_tr_list_temp.append(train_r2_temp)
                    r2_te_list_temp.append(test_r2_temp)
                    best_models_list.append(best_model_elem)
                    # print(train_r2_temp, max_r2_temp)
                    # print('Replication {} is done.'.format(repIdx+1))


                print('All replications for this replication set are finished.')
                project_history[rep_set_name] = {}
                project_history[rep_set_name]['best_models_list'] = best_models_list
                project_history[rep_set_name]['r2_tr_list_temp'] = r2_tr_list_temp
                project_history[rep_set_name]['r2_te_list_temp'] = r2_te_list_temp
                project_history[rep_set_name]['mean_r2_tr'] = np.mean(r2_tr_list_temp)
                project_history[rep_set_name]['mean_r2_te'] = np.mean(r2_te_list_temp)

                r2_tr_list.append(np.mean(r2_tr_list_temp))
                r2_te_list.append(np.mean(r2_te_list_temp))
                print('Train and test R2 scores are respectively:',
                      r2_tr_list[-1], ',', r2_te_list[-1])
                print('Elapsed time for this replication set is:',
                      time.time() - rep_start_time, 's')

project_history['r2_tr_list'] = r2_tr_list
project_history['r2_te_list'] = r2_te_list

#%%
print(f"Maximum R2 score at {alpha_value} as the dataset's alpha "
      f"and {numOfRep} as the number of replications is:", np.max(r2_te_list))

print(f'In the list "r2_te_list", maximum R2 occurs '
      f'in the "{np.argmax(r2_te_list) + 1}"th element.')

#%%
# saving the output history dict
with open('our_method_on_xls.pickle', 'wb') as f:
    pickle.dump(project_history, f)
