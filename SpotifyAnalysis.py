#-----------------------------------------------------------------------------------------------------------------------
# Project by Daniel Blei			Student ID: 16151704
#-----------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pylab
import scipy
import scipy.stats as stats
from sklearn import metrics
from sklearn.model_selection import GridSearchCV,RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

#-----------------------------------------------------------------------------------------------------------------------
# Loading the data
#-----------------------------------------------------------------------------------------------------------------------

TopDir= "/home/danielblei/PycharmProjects/SpotifyProject/train_set_top_AF.csv"
BotDir = "/home/danielblei/PycharmProjects/SpotifyProject/train_set_bot_AF.csv"

TopTest="/home/danielblei/PycharmProjects/SpotifyProject/test_set_top_AF.csv"
BotTest="/home/danielblei/PycharmProjects/SpotifyProject/test_set_bot_AF.csv"

Top = pd.read_csv(TopDir)
Bot = pd.read_csv(BotDir)
Top['Hit'] = 1
Bot['Hit'] = 0

Topt = pd.read_csv(TopTest)
Bott = pd.read_csv(BotTest)
Topt['Hit'] = 1
Bott['Hit'] = 0

#Combining the files

df_Bot = Bot
df_Top = Top
df = df_Top.append(df_Bot)

df_Bot_test = Bott
df_Top_test = Topt
df_test = df_Top_test.append(df_Bot_test)

#Rename unnamed collumns

df.rename(columns={'Unnamed: 0': 'ID'},inplace=True)
df_test.rename(columns={'Unnamed: 0': 'ID'},inplace=True)

df=df.drop(['analysis_url','uri','track_href','type','id','ID'], axis=1)
df_test=df_test.drop(['analysis_url','uri','track_href','type','id','ID'], axis=1)

#-----------------------------------------------------------------------------------------------------------------------
# Data Analysis
#-----------------------------------------------------------------------------------------------------------------------

clean_data = df_Top.append(df_Bot)
clean_data.rename(columns={'Unnamed: 0': 'ID'},inplace=True)
clean_data = clean_data.drop(['analysis_url','uri','track_href','type','id','ID'], axis=1)

# Creating the graphics 
def correlation():
    cor = clean_data.corr()
    sns.heatmap(cor,cmap="YlGnBu",annot=True,
                xticklabels=cor.columns,
                yticklabels=cor.columns)

def histogram():
    clean_data.hist(bins=10, layout=(2, 7))

def boxplot():
    clean_data.plot(kind='box', subplots=True, layout=(2, 7), sharex=True, sharey=False)


def scatter(data):
    stats.probplot(data, dist="norm", plot=pylab)
    pylab.show()


# Analysing the data normality

def shapiro():
    print ""
    print "H0: Data is normal. "
    print "H1: Data is not normal."
    print "Alpha: 0.05 \n"
    for column in clean_data:
        shapiro_results = scipy.stats.shapiro(clean_data[column])
        print "Shapiro Wilk Test of Normality: ",column, "variable"
        print "DF:", len(clean_data[column]), ", W:", shapiro_results[0], " ,P-Value:",shapiro_results[1], "\n"
        if shapiro_results[1] > 0.05:
            print "Warning:"
            print "Do not Reject the null hypothesis at 5% level of significance. \n"
        print "--------------------------------------------------------------------"

#Performing the Statistical tests

Top.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
Top_analysis = Top.drop(['analysis_url', 'uri', 'track_href', 'type', 'id', 'ID','Hit'], axis=1)
Bot.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
Bot_analysis = Bot.drop(['analysis_url', 'uri', 'track_href', 'type', 'id', 'ID','Hit'], axis=1)

def mann_whitney():
    print ""
    print "H0: There is no difference between the Medians, "
    print "H1: There is a difference between the Medians, "
    print "Alpha 0.05\n"
    for column in Top_analysis:
        mann = scipy.stats.mannwhitneyu(Top_analysis[column],Bot_analysis[column],alternative='two-sided')
        print "Mann-Whitney u test: ", column, "variable"
        print "N1:",len(Top_analysis[column]),"N2:",len(Bot_analysis[column]),", U:", mann[0], " ,P-Value:", mann[1]
        if mann[1] > 0.05:
            print "Warning:"
            print "Do not Reject the null hypothesis at 5% level of significance."
        print "--------------------------------------------------------------------"


#-----------------------------------------------------------------------------------------------------------------------
# Data Manipulation
#-----------------------------------------------------------------------------------------------------------------------

#Converting attributes 

df['duration_ms'] = (df.duration_ms/(1000*60))
df_test['duration_ms'] = (df_test.duration_ms/(1000*60))

df['duration_ms'] = df.duration_ms.round(2)
df_test['duration_ms'] = df_test.duration_ms.round(2)

df['tempo'] = df.tempo.round()
df_test['tempo'] = df_test.tempo.round()

df['loudness'] = df.loudness.round(2)
df_test['loudness'] = df_test.loudness.round(2)


#Feature Engineering

#Pandas warming off:
pd.options.mode.chained_assignment = None

df['acoustic'] = pd.qcut(df['acousticness'], 12)
print(df[['acoustic', 'Hit']].groupby(['acoustic'], as_index=False).mean().sort_values(by='acoustic', ascending=True))

df['dance'] = pd.qcut(df['danceability'], 10)
print(df[['dance', 'Hit']].groupby(['dance'], as_index=False).mean().sort_values(by='dance', ascending=True))

df['volume'] = pd.qcut(df['loudness'], 10)
print(df[['volume', 'Hit']].groupby(['volume'], as_index=False).mean().sort_values(by='volume', ascending=True))

df['vocal'] = pd.qcut(df['speechiness'], 5)
print(df[['vocal', 'Hit']].groupby(['vocal'], as_index=False).mean().sort_values(by='vocal', ascending=True))

df['tempo1'] = pd.qcut(df['tempo'], 30)
print(df[['tempo1', 'Hit']].groupby(['tempo1'], as_index=False).mean().sort_values(by='tempo1', ascending=True))

df['flow'] = pd.qcut(df['valence'], 10)
print(df[['flow','Hit']].groupby(['flow'], as_index=False).mean().sort_values(by='flow', ascending=True))

df['acoustic_split'] = 0
df.acoustic_split[df['acousticness'] < 0.27] = 1
df.acoustic_split[(df['acousticness'] >= 0.27) & (df['acousticness'] < 0.53)] = 2
df.acoustic_split[df['acousticness'] >= 0.53] = 3

df_test['acoustic_split'] = 0
df_test.acoustic_split[df_test['acousticness'] < 0.27] = 1
df_test.acoustic_split[(df_test['acousticness'] >= 0.274) & (df_test['acousticness'] < 0.53)] = 2
df_test.acoustic_split[df_test['acousticness'] >= 0.53] = 3

df['dance_split'] = 0
df.dance_split[df['danceability'] < 0.45] = 1
df.dance_split[(df['danceability'] >= 0.45) & (df['danceability'] < 0.53)] = 2
df.dance_split[(df['danceability'] >= 0.53) & (df['danceability'] < 0.63)] = 3
df.dance_split[(df['danceability'] >= 0.63) & (df['danceability'] < 0.83)] = 4
df.dance_split[df['danceability'] >= 0.83] = 5

df_test['dance_split'] = 0
df_test.dance_split[df_test['danceability'] < 0.45] = 1
df_test.dance_split[(df_test['danceability'] >= 0.45) & (df_test['danceability'] < 0.53)] = 2
df_test.dance_split[(df_test['danceability'] >= 0.53) & (df_test['danceability'] < 0.63)] = 3
df_test.dance_split[(df_test['danceability'] >= 0.63) & (df_test['danceability'] < 0.83)] = 4
df_test.dance_split[df_test['danceability'] >= 0.83] = 5

df['volume_split'] = 0
df.volume_split[df['loudness'] < -10.5] = 1
df.volume_split[(df['loudness'] >= -10.5) & (df['loudness'] < -7.5)] = 2
df.volume_split[(df['loudness'] >= -7.5) & (df['loudness'] < -6.8)] = 3
df.volume_split[(df['loudness'] >= -6.8) & (df['loudness'] < -6.1)] = 4
df.volume_split[(df['loudness'] >= -7.5) & (df['loudness'] < -6.8)] = 5
df.volume_split[(df['loudness'] >= -6.8) & (df['loudness'] < -6.1)] = 6
df.volume_split[(df['loudness'] >= -6.1) & (df['loudness'] < -5.0)] = 7
df.volume_split[df['loudness'] >= -5.0] = 8

df_test['volume_split'] = 0
df_test.volume_split[df_test['loudness'] < -10.5] = 1
df_test.volume_split[(df_test['loudness'] >= -10.5) & (df_test['loudness'] < -7.5)] = 2
df_test.volume_split[(df_test['loudness'] >= -7.5) & (df_test['loudness'] < -6.8)] = 3
df_test.volume_split[(df_test['loudness'] >= -6.8) & (df_test['loudness'] < -6.1)] = 4
df_test.volume_split[(df_test['loudness'] >= -7.5) & (df_test['loudness'] < -6.8)] = 5
df_test.volume_split[(df_test['loudness'] >= -6.8) & (df_test['loudness'] < -6.1)] = 6
df_test.volume_split[(df_test['loudness'] >= -6.1) & (df_test['loudness'] < -5.0)] = 7
df_test.volume_split[df_test['loudness'] >= -5.0] = 8

df['voice'] = 0
df.voice[df['speechiness'] < 0.04] = 0
df.voice[df['speechiness'] >= 0.04] = 1

df_test['voice'] = 0
df_test.voice[df_test['speechiness'] < 0.04] = 0
df_test.voice[df_test['speechiness'] >= 0.04] = 1

df['strings'] = 0
df.strings[df['instrumentalness'] <= 0.03] = 1
df.strings[df['instrumentalness'] > 0.03] = 2

df_test['strings'] = 0
df_test.strings[df_test['instrumentalness'] <= 0.03] = 1
df_test.strings[df_test['instrumentalness'] > 0.03] = 2

df['BPM'] = 0
df.BPM[(df['tempo'] >= 50) &(df['tempo'] < 91)] = 1
df.BPM[(df['tempo'] >= 91) &(df['tempo'] < 98)] = 2
df.BPM[(df['tempo'] >= 98) &(df['tempo'] <= 102)] = 3
df.BPM[(df['tempo'] >= 102) &(df['tempo'] < 105)] = 4
df.BPM[(df['tempo'] >= 105) &(df['tempo'] < 120)] = 5
df.BPM[(df['tempo'] >= 120) &(df['tempo'] < 126)] = 6
df.BPM[(df['tempo'] >= 126) &(df['tempo'] < 130)] = 5
df.BPM[(df['tempo'] >= 130)] = 6

df_test['BPM'] = 0
df_test.BPM[(df_test['tempo'] >= 50) &(df_test['tempo'] < 91)] = 1
df_test.BPM[(df_test['tempo'] >= 91) &(df_test['tempo'] < 98)] = 2
df_test.BPM[(df_test['tempo'] >= 98) &(df_test['tempo'] <= 102)] = 3
df_test.BPM[(df_test['tempo'] >= 102) &(df_test['tempo'] < 105)] = 4
df_test.BPM[(df_test['tempo'] >= 105) &(df_test['tempo'] < 120)] = 5
df_test.BPM[(df_test['tempo'] >= 120) &(df_test['tempo'] < 126)] = 6
df_test.BPM[(df_test['tempo'] >= 126) &(df_test['tempo'] < 130)] = 5
df_test.BPM[(df_test['tempo'] >= 130)] = 6

df['spirit'] = 0
df.spirit[(df['valence'] < 0.19)] = 1
df.spirit[(df['valence'] >= 0.19) & (df['valence'] < 0.83)] = 2
df.spirit[(df['valence'] > 0.83)] = 3

df_test['spirit'] = 0
df_test.spirit[(df_test['valence'] < 0.19)] = 1
df_test.spirit[(df_test['valence'] >= 0.19) & (df_test['valence'] < 0.83)] = 2
df_test.spirit[(df_test['valence'] > 0.83)] = 3

df = df.drop(['liveness','time_signature'], axis=1)
df_test = df_test.drop(['liveness','time_signature'],axis=1)

#------------------------------------------------------------------------------------------------------------------------
# Predictive Modelling
#------------------------------------------------------------------------------------------------------------------------

X_train = np.array((df.drop("Hit", axis=1)))
Y_train = np.array(df["Hit"])

kfold = RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=11)

#-----------------------------------------------------------------------------------------------------------------------
#SVM
print("")
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'],
                      'gamma': [0.1],
                      'C': [0.1,0.01],
                      'decision_function_shape': ['ovr']}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring='accuracy', n_jobs= 4, verbose = 1)

gsSVMC.fit(X_train,Y_train)

SVM_best = gsSVMC.best_estimator_

print 'SVM the best Accuracy score:', gsSVMC.best_score_
#-----------------------------------------------------------------------------------------------------------------------
#KNN
print("")
KNN = KNeighborsClassifier()

KNN_param_grid = {'n_neighbors': list(range(1, 25,3)),
                      'algorithm': ['auto'],
                      'p': [1],
                      'leaf_size' : [1,2,5],
                      'weights': ['distance']
                         }

preKNN = GridSearchCV(KNN,param_grid = KNN_param_grid, cv=kfold, scoring='accuracy', n_jobs= 4, verbose = 1)

preKNN.fit(X_train,Y_train)

KNN_best = preKNN.best_estimator_

print 'KNN the best Accuracy score:', preKNN.best_score_
#-----------------------------------------------------------------------------------------------------------------------
#Naive Bayes
print("")
bayes = GaussianNB()

bayes_param_grid = {}

prebayes = GridSearchCV(bayes,param_grid = bayes_param_grid, cv=kfold, scoring='accuracy', n_jobs= 4, verbose = 1)

prebayes.fit(X_train,Y_train)

Bayes_best = prebayes.best_estimator_

print 'Naive Bayes the best Accuracy score:', prebayes.best_score_
#-----------------------------------------------------------------------------------------------------------------------
#Neural Network
print("")
MLPC = MLPClassifier(hidden_layer_sizes=(125))

MLPC_param_grid = {'solver': ['lbfgs']}

preMLPC = GridSearchCV(MLPC,param_grid =MLPC_param_grid, cv=kfold,scoring='accuracy', n_jobs= 4, verbose = 1)

preMLPC.fit(X_train,Y_train)

MLPC_best = preMLPC.best_estimator_

print 'Neural Network the best Accuracy score:', preMLPC.best_score_
#-----------------------------------------------------------------------------------------------------------------------
#Decision Tree
print("")
tree = DecisionTreeClassifier()

tree_param_grid = {'max_features': [10,13],
                       'min_samples_split': [3,5],
                       'min_samples_leaf': [55,60]}

pretree = GridSearchCV(tree,param_grid =tree_param_grid, cv=kfold,scoring='accuracy', n_jobs= 4, verbose = 1)

pretree.fit(X_train,Y_train)

tree_best = pretree.best_estimator_

print 'Decision tree the best Accuracy score:', pretree.best_score_

#Visual representation of the decision tree
def tree_decision():
    dot_data = StringIO()

    export_graphviz(tree_best, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png('decision_tree.png'))
#-----------------------------------------------------------------------------------------------------------------------
#Gradient Boosting
print("")
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["exponential"],
                  'n_estimators' : [100,85,105],
                  'learning_rate': [0.1],
                  'max_depth': [9],
                  'min_samples_leaf': [4,8,12],
                  'max_features': [0.5],
                  'subsample' : [0.8]
                  }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring='accuracy', n_jobs= 4, verbose = 1)

gsGBC.fit(X_train,Y_train)

GB_best = gsGBC.best_estimator_

print 'Gradient Boosting the best Accuracy score:', gsGBC.best_score_
#-----------------------------------------------------------------------------------------------------------------------
#RandomForest:
print("")

RFC = RandomForestClassifier()

rf_param_grid = {"max_depth": [None],
                  "max_features": [10,13],
                  "min_samples_split": [4,6,8],
                  "min_samples_leaf": [3],
                  "n_estimators" :[580],
                  "criterion": ["gini"],
                     'bootstrap': [True],
                     'oob_score': [True],
                    'warm_start': [True],
                     "n_jobs": [4]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring='accuracy', n_jobs= 4, verbose = 1)

gsRFC.fit(X_train,Y_train)

RF_best = gsRFC.best_estimator_

print 'RandomForest the best Accuracy score:',gsRFC.best_score_
#-----------------------------------------------------------------------------------------------------------------------
#AdaBoostClassifier

ada = AdaBoostClassifier()

ada_param_grid= {'n_estimators': [2500],
                     'learning_rate': [0.15]}

preada = GridSearchCV(ada,param_grid =ada_param_grid, cv=kfold,scoring='accuracy', n_jobs= 4, verbose = 1)

preada.fit(X_train,Y_train)

ada_best = preada.best_estimator_

print 'AdaBoost the best Accuracy score:', preada.best_score_

# -----------------------------------------------------------------------------------------------------------------------
#Voting Classifier - The Final Model
#------------------------------------------------------------------------------------------------------------------------
votingC = VotingClassifier(estimators=[('rfc', RF_best), ('gbc',GB_best),('ada',ada_best),('svm', SVM_best),
                                           ('KNN',KNN_best),('bys',Bayes_best),('neu',MLPC_best),('tree',tree_best)]
                                            , voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, Y_train)

df_test_eval = df_test.drop(['Hit'], axis=1)

preds_test = votingC.predict(np.array(df_test_eval))
#------------------------------------------------------------------------------------------------------------------------
# Evaluation
#------------------------------------------------------------------------------------------------------------------------
print "-------------------------------------------------"
print("Voting Classifier Accuracy Score (Testing Set):")
print(metrics.accuracy_score(df_test.Hit, preds_test))
print "-------------------------------------------------"
print("Voting Classifier Recall Score (Testing Set):")
print(metrics.recall_score(df_test.Hit, preds_test))
print "-------------------------------------------------"
print("Voting Classifier Precision Score (Testing Set):")
print(metrics.precision_score(df_test.Hit, preds_test))
print "-------------------------------------------------"
print("Voting Classifier F1-Score (Testing Set):")
print(metrics.fbeta_score(df_test.Hit, preds_test, beta=1))
print "-------------------------------------------------"

#Plot Confusion Matrix
def Confusion(figsize=(6, 4), fontsize=15):
    cm = metrics.confusion_matrix(df_test.Hit, preds_test)
    class_names = ('False', 'True')
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=90, ha='right', fontsize=fontsize)
    plt.title('Confusion Matrix', size=18)
    plt.ylabel('True label',size=12)
    plt.xlabel('Predicted label',size =12)
    return fig

#ROC Curve
def ROC():
    fpr, tpr, threshold = metrics.roc_curve(df_test.Hit, preds_test)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic (ROC)',size=18)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate',size=15)
    plt.xlabel('False Positive Rate',size=15)
    plt.grid()
    plt.show()
