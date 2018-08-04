import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics
from scipy import stats

indir = "D:\\Docs_required\\EE559\\HW3\\Dataset_AReM"
folders = ['bending1','bending2','cycling','lying','sitting','standing','walking']
final = []
for folder_name in folders:
    os.chdir(indir+ "\\" + folder_name)
    fileList = glob.glob('*.csv')
    dfList = []
    colnames = ['avg_rss12','var_rss12','avg_rss13','var_rss13','avg_rss23','var_rss23']
    for fname in fileList:
        df = pd.read_csv(fname, skiprows = 4)
        a = df.values
        dfList.append(a[:, 1:])
    final.append(dfList)

finalnp = np.array(final)
bending1 = finalnp[0]
bending2 = finalnp[1]
cycling = finalnp[2]
lying = finalnp[3]
sitting = finalnp[4]
standing = finalnp[5]
walking = finalnp[6]

test = []
train = []
############################### PREPARING TRAINING AND TEST SETS ###########################################
train = bending1[2:]+ bending2[2:]+ cycling[3:]+ lying[3:]+ sitting[3:]+ standing[3:] + walking[3:]
trainnp = np.array(train)
tr_labels = [0]*(trainnp.shape[0])
for f in range(9):
        tr_labels[f] = 1
tr_labels = np.array(tr_labels)   ######## ORIGINAL TRANING LABELS, CONSIDERING BENDING ACTIVITY AS CLASS 1 AND OTHERS AS CLASS 0


test = bending1[:2] + bending2[:2] + cycling[:3] + lying[:3] + sitting[:3] + standing[:3] + walking[:3]
testnp = np.array(test)
print(testnp.shape)
test_labels = [0]*(testnp.shape[0])
for f in range(4):
    test_labels[f] = 1
test_labels = np.array(test_labels)  ######## ORIGINAL TEST LABELS, CONSIDERING BENDING ACTIVITY AS CLASS 1 AND OTHERS AS CLASS 0


######################### OBTAINING THE TIME DOMAIN FEATURES FOR ALL INSTANCES
feature = []
total_feature = []
for i in range(7):
    for j in range(len(finalnp[i])):
        k = pd.DataFrame(finalnp[i][j])
        c = k.describe()
        temp2 = []
        for o in range(6):
            for z in range(1,8):
                temp2.append(c.values[z][o])
                
        temp = []
        for x in range(6):
            temp.append(c.values[1][x])
            temp.append(c.values[2][x])
            temp.append(c.values[6][x])
        feature = np.append(feature,temp)
        total_feature = np.append(total_feature,temp2)


feature_arr = feature.reshape(-1,18)
feature_table = pd.DataFrame(feature_arr)
print('Dataset with features mean, std.dev and 3rd quartile for all time-series')
print(feature_table)


total_feature_arr = total_feature.reshape(-1,42)
total_feature_table = pd.DataFrame(total_feature_arr)
print('Complete feature set with 7 statistical values for each time-series')
print(total_feature_table)


test_l = [0,1,7,8,13,14,15,28,29,30,43,44,45,58,59,60,73,74,75]
whole_l = [i for i in range(88)]
train_l = list(set(test_l)^ set(whole_l))

train_feaure_set = total_feature_table.loc[train_l,:]
train_feaure_set = train_feaure_set.values

test_feature_set = total_feature_table.loc[test_l,:]
test_feature_set = test_feature_set.values

req_df = feature_table.loc[:,[0,1,2,3,4,5,15,16,17]]
req_df.columns = ['mean1', 'std1', '3rd quartile1','mean2', 'std2', '3rd quartile2','mean6', 'std6', '3rd quartile6']
label = []
for i in range(88):
    if i<=12:
        label.append('Bending')
    else:
        label.append('Other')

req_df['label'] = label        
print('Features Mean, std.dev and 3rd quartile for time-series 1,2 and 6')
print(req_df.loc[train_l,:])

########################### SCATTER-PLOT MATRIX ##########################
sns.set(style="ticks")
sns.pairplot(req_df.loc[train_l,:], hue="label", markers=["o", "s"], palette="husl")
plt.show()


####################################### CROSS VALIDATION LOOP TO CHOOSE BEST L #####################################################
CV_err_list = []
for L in range(1,5):
    print(L)
    latest_train = []
    training_labels = [0]*(L*trainnp.shape[0])
    feature_set = []
    for j in range(trainnp.shape[0]):                                   ##### FOR EACH INSTANCE BREAK DOWN THE SERIES BY L TIMES
        for k in range(L):
            trainnp[j][k*(480//L): (k+1)*(480//L),:]
            latest_train += list(trainnp[j][k*(480//L): (k+1)*(480//L),:])
    latest_train = np.array(latest_train)
    latest_train = latest_train.reshape(L*trainnp.shape[0],-1,6)
    for f in range(9*L):
        training_labels[f] = 1
    training_labels = np.array(training_labels)

    for d in range(latest_train.shape[0]):                                 ########## OBTAINING TIME-DOMAIN FEATURES
        k = pd.DataFrame(latest_train[d])
        c = k.describe()
        #print(c)
        temp2 = []
        for o in range(6):
            for z in range(1,8):
                temp2.append(c.values[z][o])
        feature_set = np.append(feature_set,temp2)
    feature_arr = feature_set.reshape(-1,42)      
    print(feature_arr.shape)

    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False) #### STRATIFIED KFOLD - USED BECAUSE OF CLASS IMBALANCE

    tot = 0
    avg_CV_err = 0
    for train_index, val_index in skf.split(feature_arr, training_labels):
        e = 0
        X_train, X_val = feature_arr[train_index], feature_arr[val_index]
        y_train, y_val = training_labels[train_index], training_labels[val_index]

        model = LogisticRegression(C = 10000000000)   ###### VERY HIGH C VALUE TO NULLIFY L2 REGULARIZATION
        rfe = RFE(model,10)                           ###### USING RFE TO GET 10 BEST FEATURES OUT OF 42
        rfe = rfe.fit(X_train, y_train)
        print('RFE Details')
        print(rfe.support_)
        best_features = np.where(rfe.support_ == 1)[0]
        print(best_features)
        X_train = pd.DataFrame(X_train)
        X_train = X_train.loc[:,best_features]
        X_train = X_train.values

        X_val = pd.DataFrame(X_val)
        X_val = X_val.loc[:,best_features]
        X_val = X_val.values

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        e = 1.0 - accuracy_score(y_val, y_pred)
        tot += e
    avg_CV_err = tot/5
    CV_err_list.append(avg_CV_err)
    
######################################## FINDING THE BEST L AND COMPUTING TRANING ERROR, AUC AND CNF MATRIX ########################
print(CV_err_list)        
least_error = min(CV_err_list)
best_l = (CV_err_list.index(least_error)) + 1

model = LogisticRegression(C = 10000000000)
rfe = RFE(model,10)
rfe = rfe.fit(train_feaure_set, tr_labels)
tr_features = np.where(rfe.support_ == 1)[0]
print(tr_features)
X_tr = pd.DataFrame(train_feaure_set)
X_tr = X_tr.loc[:, tr_features]
X_tr = X_tr.values

X_test = pd.DataFrame(test_feature_set)
X_test = X_test.loc[:, tr_features]
X_test = X_test.values

model.fit(X_tr, tr_labels)
print(model.get_params())
print(model.intercept_)
print(model.coef_)
tr_y = model.predict(X_tr)

params = np.append(model.intercept_,model.coef_)
newX = np.append(np.ones((len(X_tr),1)), X_tr, axis=1)
MSE = (sum((tr_labels-tr_y)**2))/(len(newX)-len(newX[0]))

var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params/ sd_b

p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

sd_b = np.round(sd_b,3)
params = np.round(params,4)

myDF3 = pd.DataFrame()
myDF3["Coefficients"],myDF3["P-value"] = [params,p_values]
print(myDF3)


print('Training accuracy')
print(accuracy_score(tr_labels, tr_y))
cnf_matrix = confusion_matrix(tr_y, tr_labels)
print('Training Confusion matrix')
print(cnf_matrix)
fpr, tpr, thresholds = metrics.roc_curve(tr_labels, tr_y)
print('Training AUC')
print(metrics.auc(fpr, tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % metrics.auc(fpr, tpr))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



test_y = model.predict(X_test)
print('Test accuracy')
print(accuracy_score(test_labels, test_y))
cnf_matrix = confusion_matrix(test_y, test_labels)
print('Test Confusion matrix')
print(cnf_matrix)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, test_y)
print('Test AUC')
print(metrics.auc(fpr, tpr))

######################################## SMOTE OVERSAMPLING TO COMPENSATE CLASS IMBALANCE ###################################
from imblearn.over_sampling import SMOTE, ADASYN
X_resampled, y_resampled = SMOTE().fit_sample(X_tr, tr_labels)
model.fit(X_resampled, y_resampled)
resampled_y_pred = model.predict(X_resampled)

print('Training accuracy after resampling')
print(accuracy_score(y_resampled, resampled_y_pred))
res_cnf_matrix = confusion_matrix(y_resampled, resampled_y_pred)
print('Training Confusion matrix after resampling')
print(res_cnf_matrix)
fpr_r, tpr_r, thresholds_r = metrics.roc_curve(y_resampled, resampled_y_pred)
print('Training AUC after resampling')
print(metrics.auc(fpr_r, tpr_r))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr_r, tpr_r, 'b', label = 'AUC = %0.2f' % metrics.auc(fpr_r, tpr_r))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




