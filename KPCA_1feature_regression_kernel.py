from custom_functions_in_py import *
import numpy as np
import time
import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
plt.interactive(False)

#%%
best_models_list = []
r2_te_list = []
r2_tr_list = []
project_history = {}

#%%
for alpha_value in alpha_values_list:
    for pca_features in range(pca_features_min, pca_features_max + 1):

        rep_start_time = time.time()
        print('Starting a new set of replications...')
        print(f'Each set contains {numOfRep} replications.')
        print('Hyper-parameters info. at this replication set is:')
        print('pca_features  ==>', pca_features)

        rep_set_name = 'pcaNum'+str(pca_features)

        r2_te_list_temp = []
        r2_tr_list_temp = []

        for repIdx in range(numOfRep):
            if dataset_type == dataset_type_list[0]:
                X_train_sir, X_test_sir, y_train, y_test = \
                    after_pca_ds_gen(alpha=alpha_value,
                                     pca_degree=2,
                                     pca_features=pca_features)

            if dataset_type == dataset_type_list[1]:
                X_train_sir, X_test_sir, y_train, y_test = \
                    after_pca_ds_gen_from_file(X_input,
                                               y_input,
                                               pca_degree=2,
                                               pca_features=pca_features)

            # Performing KNR
            test_r2_temp = -np.Inf
            train_r2_temp = -np.Inf
            best_model_elem = None
            for neighbor_elem in [X_train_sir.shape[0]-1]:
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
plt.figure()
plt.plot(range(1, len(r2_tr_list) + 1), r2_tr_list)
plt.xlabel("Hyper-parameters order: pca_features, "
           "sir_features, pca_degree_num, y_label_trans_type")
plt.ylabel('mean R2 accuracy criterion')
plt.title('{} Replications -- Best Possible Train R2 Scores'.format(numOfRep))
plt.grid(True)
plt.show()

plt.figure()
plt.plot(range(1, len(r2_te_list) + 1), r2_te_list)
plt.xlabel("Hyper-parameters order: pca_features, "
           "sir_features, pca_degree_num, y_label_trans_type")
plt.ylabel('mean R2 accuracy criterion')
plt.title('{} Replications -- Best Possible Test R2 Scores'.format(numOfRep))
plt.grid(True)
plt.show()

#%%
if dataset_type == dataset_type_list[0]:
    print('\nfrom Stable Distribution\n')
print(f"Maximum R2 score at {alpha_value} as the dataset's alpha "
      f"and {numOfRep} as the number of replications is:", np.max(r2_te_list))

print(f'In the list "r2_te_list", maximum R2 occurs '
      f'in the "{np.argmax(r2_te_list) + 1}"th element.')

#%%
# saving the output history dict
project_hist_name = 'KPCA_1f_reg__' + dataset_type
with open(project_hist_name + '.pickle', 'wb') as f:
    pickle.dump(project_history, f)
