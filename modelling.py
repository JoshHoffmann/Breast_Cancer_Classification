import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from itertools import product

import plotting



def RunLogit(target,features,target_out,features_out):

    '''FS_selected_features = FSActiveSet(target,features)
    plotting.PlotCorrelationMatrix(FS_selected_features, title='Pruned Feature Correlations')
    CrossValidateLogit(target,features)'''

    '''C_space = np.linspace(start=0.1,stop=0.01,num=15)
    l1_space = np.linspace(start=0,stop=0.5,num=10)'''

    C_space = [0.09357142857142858]
    l1_space = [0.16666666666666666]

    c,l1_ratio = OptimiseLogit(target,features,C_space,l1_space)

    print(f'C = {c}')
    print(f'l1_ratio = {l1_ratio}')

    scaler = StandardScaler()
    features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    features = FSActiveSet(target,features)
    logit = LogisticRegression(penalty='elasticnet', C=c,l1_ratio=l1_ratio,solver='saga')
    fitted_logit = logit.fit(features,target)

    int = fitted_logit.intercept_
    betas = fitted_logit.coef_
    f_names = fitted_logit.feature_names_in_

    print(f'intercept  = {int}')
    print('beta = \n', betas)
    print('feature names =  \n', f_names)

    in_sample_classification = fitted_logit.predict(features)
    in_sample_loss = np.mean(in_sample_classification != target)
    in_sample_acc = 1 - in_sample_loss
    print(f'in sample loss = {in_sample_loss}')
    print(f'in sample accuracy = {in_sample_acc}')

    ### Now do the out of sample stuff

    out_scaler = StandardScaler()
    features_out = features_out[f_names]
    features_out = pd.DataFrame(out_scaler.fit_transform(features_out), columns=features_out.columns)
    predictions = fitted_logit.predict(features_out)

    out_sample_loss = np.mean(predictions != target_out)
    out_sample_acc = 1 - out_sample_loss
    print(f'in sample loss = {out_sample_loss}')
    print(f'in sample accuracy = {out_sample_acc}')

    ### ROC stuff
    y_scores = fitted_logit.predict_proba(features_out)[:, 1]
    fpr, tpr, _ = roc_curve(target_out, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve on Unseen Test Data")
    plt.legend(loc="lower right")
    plt.grid(True)


    return None

def OptimiseLogit(target,features,C_space,l1_space):
    best_C = None
    best_l1 = None
    best_mloss = np.inf
    for c, l1 in product(C_space,l1_space):
        print(f'c = {c}, l1_ratio = {l1}')
        m_loss = CrossValidateLogit(target,features,C=c,l1_ratio=l1)
        print('mloss ', m_loss)
        if m_loss < best_mloss:
            best_C, best_l1 = c,l1
            best_mloss = m_loss
    return best_C, best_l1



def CrossValidateLogit(target, features,C=0.5,l1_ratio=0.5):

    kf = KFold(n_splits=5)
    loss_list = []

    for i_train, i_test in kf.split(target):
        features_train, features_test = features.iloc[i_train], features.iloc[i_test]
        target_train, target_test = target.iloc[i_train], target.iloc[i_test]
        scalar_train = StandardScaler()
        scalar_test = StandardScaler()

        features_train = pd.DataFrame(scalar_train.fit_transform(features_train), columns=features_train.columns)

        features_train = FSActiveSet(target_train,features_train) # Get the active set from initial pruning
        features_test = features_test[features_train.columns] # filter test features by the active set

        features_test = pd.DataFrame(scalar_test.fit_transform(features_test), columns=features_test.columns)

        logit_reg = LogisticRegression(penalty='elasticnet', C=C, l1_ratio=l1_ratio,solver='saga')
        fitted_logit = logit_reg.fit(features_train,target_train)
        predicted_classes = fitted_logit.predict(features_test)

        loss = np.mean(predicted_classes != target_test)
        loss_list.append(loss)
    mean_loss = np.mean(loss_list)
    return mean_loss

def FSActiveSet(target,features):

    active_set_found = False
    count = 0
    while not active_set_found and count<=10:
        active_set, active_beta, modified_active_set = ForwardStagewise(target,features)
        features = features[active_set]
        active_set_found = not modified_active_set
        count+=1
    return features


def ForwardStagewise(target, features, step_size = 0.01, max_steps=1500):
    modified_active_set = False
    feature_names = features.columns

    # Convert input types to numpy array
    if type(target) == pd.Series:
        target = target.to_numpy()
    if type(features) == pd.DataFrame:
        features = features.to_numpy()
    # Check input types have been converted to numpy array
    def check_dtypes():
        if type(target) != np.ndarray:
            print('target has not been converted to numpy array')
        if type(features) != np.ndarray:
            print('features has not been converted to numpy array')

    def selectsubset(target,features):

        steps = 0

        target = target.reshape(-1, 1)
        beta = np.zeros(shape=(features.shape[1],1))
        y_bar = np.mean(target)
        beta_0 = np.log(y_bar/(1-y_bar))
        p_0 = 1/(1+np.exp(-beta_0))
        X = features

        p = p_0*np.ones((target.shape[0],1))
        r = target - p

        none_correlated = False
        while not none_correlated:

            corr = np.corrcoef(X,r, rowvar=False)[:-1,-1]
            corr_abs = np.abs(corr)

            if np.all(corr_abs<0.001) or steps>=max_steps:
                
                none_correlated = True

            else:
                correlated_feature = np.argmax(corr_abs)
                x = X[:,correlated_feature:correlated_feature+1]
                # Now calculate the linear regression coefficient of the max correlated feature on the residual
                delta = step_size*np.sign(corr[correlated_feature])
                beta[correlated_feature] = beta[correlated_feature] + delta
                r = r - delta*x
                steps+=1

        return beta

    check_dtypes()
    beta = selectsubset(target,features)

    if np.all(beta!=0):

        return feature_names, beta, modified_active_set
    else:
        active_variable_indices = np.where(beta != 0.0)[0]
        active_set = feature_names[active_variable_indices]
        active_beta = beta[active_variable_indices]

        modified_active_set = True

        return active_set, active_beta, modified_active_set

