from cgi import test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler, scale
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from scipy import stats
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn import decomposition, datasets
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
import pickle
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import streamlit.components.v1 as components
import streamlit as st

# ## Acquiring Data
df=pd.DataFrame(pd.read_excel("diabetes-dataset.xlsx"))

def prepare_data(df):
    print(df.describe)
    corr_mat = df.corr()
    p = df.hist(figsize = (10,10))
    mat_plot(corr_mat)

def mat_plot(corr_mat):
    f,ax = plt.subplots(figsize = (8,6))
    sns.heatmap(corr_mat,cmap = "GnBu",annot = True, fmt = '.1f',ax = ax,annot_kws={"fontsize":75})
    plt.show()

def preprocess(df):
    diab_df_cpy = df.copy(deep = True)
    diab_df_cpy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].replace(0,np.NaN)
    diab_df_cpy['Glucose'].fillna(diab_df_cpy['Glucose'].mean(), inplace = True)
    diab_df_cpy['BloodPressure'].fillna(diab_df_cpy['BloodPressure'].mean(), inplace = True)
    diab_df_cpy['SkinThickness'].fillna(diab_df_cpy['SkinThickness'].median(), inplace = True)
    diab_df_cpy['Insulin'].fillna(diab_df_cpy['Insulin'].median(), inplace = True)
    diab_df_cpy['BMI'].fillna(diab_df_cpy['BMI'].median(), inplace = True)
    return diab_df_cpy


def scale_data(diab_df_cpy):
    sc_x = StandardScaler()
    X =  pd.DataFrame(sc_x.fit_transform(diab_df_cpy.drop(["Outcome"],axis = 1),), columns=['Pregnancies', 
'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    return X


def visualizer(diab_df_cpy):
    fig=px.histogram(diab_df_cpy,x='Age',marginal='violin')
    fig.update_layout(bargap=0.2)
    fig.show()
    sns.countplot(data=diab_df_cpy,x='Outcome',palette='coolwarm')

    fig=px.histogram(diab_df_cpy,x=diab_df_cpy[diab_df_cpy.Outcome==0].Age,marginal='box',title='Age distribution with outcome 0',color_discrete_sequence=['green'])
    fig.update_layout(bargap=0.1)
    fig.show()

    fig=px.histogram(diab_df_cpy,x=diab_df_cpy[diab_df_cpy.Outcome==1].Age,marginal='box',title='Age distribution with outcome 1',color_discrete_sequence=['darkred'])
    fig.update_layout(bargap=0.1)
    fig.show()


    fig = px.box(diab_df_cpy, y="Pregnancies", x="Outcome")
    fig.show()


    plt.subplots(figsize=(15,10))
    sns.boxplot(x='Age', y='BMI', data=diab_df_cpy)
    plt.show()

    data_plot = sns.lmplot('Insulin','Age',data = diab_df_cpy, hue = 'Outcome',fit_reg = 'False')

# ## Split the X and Y variables

def split_train_test(diab_df_cpy):

    y = diab_df_cpy.iloc[:,-1:]
    x = diab_df_cpy.iloc[:,:-1]

    X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size = 0.2,random_state = 8)

    column_lis = list(diab_df_cpy.columns[:-1])

    ### SMOTE ANALYSIS FOR IMBALANCED DATASET
    print("Percentage of Positive Values in training data before Smote :",Y_train.value_counts(normalize=True)[1]/(Y_train.value_counts(normalize=True)[0]+Y_train.value_counts(normalize=True)[1])*100,"%")
    print("Percentage of Negative Values in training data before Smote :",Y_train.value_counts(normalize=True)[0]/(Y_train.value_counts(normalize=True)[0]+Y_train.value_counts(normalize=True)[1])*100,"%")

    smote = SMOTE()
    X_train,Y_train = smote.fit_resample(X_train,Y_train)
    return X_train,Y_train,X_test,Y_test
    #print("Shape of X after SMOTE: ",X_train.shape)

diab_df_cpy = preprocess(df)
diab_df = scale_data(diab_df_cpy)
X_train,Y_train,X_test,Y_test = split_train_test(diab_df_cpy)


def dtree_classifier():
    column_lis = X_train.columns
    dtree = DecisionTreeClassifier(max_depth = 15,random_state = 0, 
                                min_samples_split = 2)
    dtree = dtree.fit(X_train,Y_train)
    Y_pred = dtree.predict(X_test)
    #Change using streamlit lib
    accu_score_dtree = metrics.accuracy_score(Y_test,Y_pred)*100
    vis_dtree(dtree,Y_pred)
    pickle_dt = open("dtree_classifier.pkl",mode = "wb")
    pickle.dump(dtree,pickle_dt)
    pickle_dt.close()
    return 1

def vis_dtree(model,Y_pred):
    column_lis = X_train.columns
    conf_mat = confusion_matrix(Y_test,Y_pred)
    plt.figure(figsize = (25,20))
    heat_dtree = sns.heatmap(conf_mat,annot = True,annot_kws={"fontsize":75})
    plt.savefig("pages/images/heat_dtree.png")
    # st.pyplot(heat_dtree)
    plt.figure(figsize = (15,10))
    pd.Series(model.feature_importances_,index = column_lis).plot(kind = 'barh')
    plot_dtree(model,X_train)
    return


def plot_dtree(model_name,train_data):
    fig = plt.figure(figsize=(250,200))
    _ = tree.plot_tree(
        model_name,
        feature_names = train_data.columns,
        class_names = ['NEGATIVE','POSTIVE'],
        filled = True
    )
    fig.savefig("decision_tree.png")
    return

Y_train_arr = Y_train['Outcome'].ravel()

# ## 2. Random Forest Classifier
def rf_classifier(perm):
    Y_train_arr = np.array(Y_train['Outcome'])
    rfc = RandomForestClassifier(n_estimators = 500)
    rfc.fit(X_train,Y_train_arr) 
    # Overfitted
    y_train_rfc = rfc.predict(X_train)
    # print("TRAINING Accuracy Score = {}%".format(metrics.accuracy_score(Y_train,y_train_rfc)*100))
    print("\n-----RANDOM FOREST ALGORITHM-----\n")
    rfc_cv = hyper_param_model(rfc)
    print("RFC Score = {}%".format(rfc.score(X_test,Y_test)*100))
    y_pred_rfc = rfc_cv.predict(X_test)
    test_accu_score = metrics.accuracy_score(Y_test,y_pred_rfc)*100
    print("Validation Accuracy of Random Forest Classifier = {}".format(test_accu_score))
    y_train_rfc = rfc_cv.predict(X_train)
    print("TRAINING Accuracy Score = {}".format(metrics.accuracy_score(Y_train,y_train_rfc)*100))
    vis_rf(rfc_cv,y_pred_rfc,perm)
    return 1

def vis_rf(model,y_pred_rfc,perm):
    #Classification Report
    report_rfc = classification_report(Y_test,y_pred_rfc)
    print(report_rfc)
    rfc_cros_valid = hyper_param_model(model)
    y_pred_rfc_cv = rfc_cros_valid.predict(X_test)
    
    print("Accuracy Score:- {}%".format(metrics.accuracy_score(Y_test,y_pred_rfc_cv)*100))

    print("Training Accuracy Score = {}%".format(metrics.accuracy_score(Y_train,rfc_cros_valid.predict(X_train))*100))
    mat_rfc_cros_valid = confusion_matrix(Y_test,y_pred_rfc_cv)
    if perm == 0:
        heat_mp_rfc = plt.figure(figsize = (25,20))
        sns.heatmap(mat_rfc_cros_valid,annot = True,annot_kws={"fontsize":75})
        plt.savefig("pages/images/heat_rfc.png")
        # st.pyplot(heat_mp_rfc)
        roc_plot(model)
    elif perm == 1:
        f_imp_plot = plt.figure(figsize = (65,50))
        pd.Series(rfc_cros_valid.feature_importances_,index = X_train.columns).plot(kind = 'barh',fontsize = 80)
        #st.pyplot(f_imp_plot)
        plt.savefig("pages/images/f_imp.png")
    return

def roc_plot(model):
    y_pred_prob = model.predict_proba(X_test)[:,1]
    fpr,tpr, thresholds = roc_curve(Y_test,y_pred_prob)
    plt.figure(figsize=(7,10))
    plt.plot([0,1] , [0,1],'k-')
    plt.plot(fpr,tpr,label = 'Knn')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC Curve')
    print("Area under ROC/AUC Curve = {}%".format(roc_auc_score(Y_test,y_pred_prob)*100))

def hyper_param_model(model):
    
    param_grid_aug = {
        'n_estimators':[500],
        'max_features':['sqrt','log2'], #range of values ex - 5 to 15
        'max_depth':[27],#[i for i in range (25,30)]
        'max_leaf_nodes': [173], #set to range of values 5 to 15 [i for i in range (150,181)]
        'criterion':['gini'],
        'min_samples_leaf':[1], #NOT create range from 20 to 50 in steps of 5
    }

    Y_train_arr = Y_train['Outcome'].ravel()
    CV_rfc_aug = GridSearchCV(estimator = model,param_grid = param_grid_aug,cv = 5,n_jobs = -1)
    CV_rfc_aug.fit(X_train,Y_train_arr)
    rfc_cros_valid = CV_rfc_aug. best_estimator_
    pickle_rf = open("rf_classifier.pkl",mode = "wb")
    pickle.dump(rfc_cros_valid,pickle_rf)
    pickle_rf.close()
    return rfc_cros_valid

# # BEST PARAMETER
# 
# # Pass - 1
# 
# ### MAX DEPTH = 27 
# ### Min_Samples_Leaf = 1 
# 
# ### Accuracy = 99% with 2 params
# 
# # Pass- 2
# 
# ### MAX LEAF MODES = 69
# ### ACC = 91.5%
# 
# # Pass-3
# 
# ### MAX LEAF NODES = 80 when range from 60 to 81
# ### ACC = 94%
# 
# # Pass-4
# 
# ### MAX LEAF NODES = 100 when range from 80 to 101
# ### ACC = 95.5%
# 
# 
# # Pass-5
# ### MAX_LEAF NODES = 116 when range from 100-121
# ### ACC = 97.25%
# 
# 
# # Pass-6
# ### MAX_LEAF_NODES = 126 when range from 110-131
# ### ACC = 98.0%
# 
# # PASS - 7
# 
# ### MAX_LEAF_NODES = 144 when range from 120 to 151
# ### ACC = 99%
# 
# # PASS - 8
# 
# ### MAX_LEAF_NODES = 154 when range from 140 to 161
# ### ACC = 99% [SAME AS PREV]
# 
# 
# ## PASS - 9 [MAXIMUM EFFICIENCY]
# 
# ### MAX_LEAF_NODES = 169 when range from 150 to 181
# ### ACC = 99.25%
# 
# 
# ## PASS - 11 [MAX - 2]
# 
# ### MAX_LEAF_NODES = 163 when range from 160 to 171
# ### ACC = 99.25%
# 
# # PASS - 10 [Reached Peak and Crossed Maxima]
# 
# ### MAX_LEAF_NODES = 180 when range from 160 to 191
# ### ACC = 98.25% [LOW]
# 
# 
# 

# # Cross Validation

# ## 3. Logistic Regression

def log_regression():
    print("\n-----LOGISTIC REGRESSION-----\n")
    log_reg=linear_model.LogisticRegression(max_iter=50000)
    Y_train_arr = Y_train['Outcome'].ravel()
    log_reg.fit(X_train,Y_train_arr)
    log_reg_cv = hyper_logreg(log_reg)
    predicted=log_reg_cv.predict(X_test)
    test_accu_score = metrics.accuracy_score(Y_test,predicted)*100
    vis_logreg(log_reg,predicted)
    return 1

def vis_logreg(model,predicted):
    Y_train_arr = Y_train['Outcome'].ravel()

    cf_matrix = confusion_matrix(Y_test,predicted)
    heat_logreg= plt.figure(figsize = (25,20))
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
                fmt='.2%', cmap='Blues',annot_kws={"fontsize":75})
    #st.pyplot(heat_logreg)
    plt.savefig('pages/images/log_reg.png')

    #Plot results comparison graph
    coeff = list(model.coef_[0])
    labels = list(X_train.columns)
    features = pd.DataFrame()
    features['Features'] = labels
    features['importance'] = coeff
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features['positive'] = features['importance'] > 0
    features.set_index('Features', inplace=True)
    features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
    plt.xlabel('Importance')
    log_reg_cv = hyper_logreg(model)
    roc_plot(log_reg_cv)
    pickle_lr = open("logreg.pkl",mode = "wb")
    pickle.dump(log_reg_cv,pickle_lr)
    pickle_lr.close()
    return

def hyper_logreg(model):
    c_space = np.logspace(-1,2,15)
    param_grid = {'solver':['newton-cg', 'lbfgs', 'liblinear'],'C':c_space}
    log_reg_cv = GridSearchCV(model,param_grid,cv = 5)
    log_reg_cv.fit(X_train,Y_train_arr)
    print("Validation Accuracy of improved model = {}".format(log_reg_cv.best_score_*100))
    return log_reg_cv


# ## 4. SVM Model

def sv_classifier():
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    diab_df_cpy = preprocess(df)
    diab_df_cpy_up = diab_df_cpy[~((diab_df_cpy < (Q1 - 1.5 * IQR)) |(diab_df_cpy > (Q3 + 1.5 * IQR))).any(axis=1)]
    y = diab_df_cpy_up.iloc[:,-1:]
    x = diab_df_cpy_up.iloc[:,:-1]
    x_train_svc,x_test_svc,y_train_svc,y_test_svc = train_test_split(x,y,test_size = 0.2,random_state = 8)
    # x_train_svc,x_test_svc,y_train_svc,y_test_svc = train_test_split(diab_df_cpy_up,diab_df_cpy_up.iloc[:,-1:],test_size = 0.2,random_state=8)
    # print(x_train_svc)
    diab_df_cpy = scale_data(diab_df_cpy_up)
    y_train_arr_svc = y_train_svc['Outcome'].ravel()
    classifier = SVC(kernel = 'rbf',probability = True)
    classifier.fit(x_train_svc,y_train_arr_svc)
    # Hyperparameter Tuning
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf']}
    grid = GridSearchCV(classifier,param_grid,refit = True, verbose = 2)
    grid.fit(x_train_svc,y_train_arr_svc)
    grid_cv = grid.best_estimator_
    grid_prediction = grid_cv.predict(x_test_svc)
    mat_svm = confusion_matrix(y_test_svc['Outcome'],grid_prediction)
    plt.figure(figsize = (7,5))
    sns.heatmap(mat_svm,annot = True)
    print(classification_report(y_test_svc['Outcome'],grid_prediction))
    test_accu_score_svc = accuracy_score(y_test_svc['Outcome'], grid_prediction)*100
    print("Validation Accuracy Score [SVC] = {}%".format(test_accu_score_svc))
    # roc_plot(grid_cv)
    pickle_svm = open("svc_classifier.pkl",mode = "wb")
    pickle.dump(grid_cv,pickle_svm)
    pickle_svm.close()
    return test_accu_score_svc



if __name__ == "__main__":
    rf_classifier(0)
    log_regression()
    sv_classifier()
    dtree_classifier()