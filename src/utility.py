# %% [code] {"jupyter":{"outputs_hidden":false}}
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import copy
import time

from sklearn.feature_selection import SelectKBest, chi2, f_regression, f_classif
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score

from mlxtend.classifier import StackingCVClassifier

def return_singular_cols(df: pd.DataFrame):
    """Identify columns with only 1 value, these are unlikely to be helpful"""
    col_singular = [col for col in df.columns if df[col].nunique() == 1]
    print('Singular columns: {}'.format(col_singular))

    return col_singular

def get_feature_importances(df: pd.DataFrame, target: str, score_func: str, n_splits):
    """Identify important features"""
    features = list(df.columns)
    features.remove(target)

    X = df[features]
    y = df[target]

    if score_func == 'classification':
        bestfeatures = SelectKBest(score_func=f_classif, k=n_splits)
    else:
        bestfeatures = SelectKBest(score_func=f_regression, k=n_splits)
    fit = bestfeatures.fit(X, y)
    
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    
    # Concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score'] 
    featureScores.sort_values(by='Score', axis=0, ascending=True, inplace=True)
    
    return featureScores

def grid_search(model: object, params: dict, n_splits: int, scoring: str, X: pd.DataFrame, y: pd.Series):
    """Hyper-parameter tuning of model using param grid"""
    CV = RandomizedSearchCV(model
                            , param_distributions=params
                            , scoring = scoring
                            , n_jobs=-1
                            , cv=n_splits)
    CV.fit(X, y)
    output_model = CV.best_estimator_
    
    return output_model

def get_stacked_model(models, stack_model, X, y, n_splits):
    """Stack models with higher model"""
    stack = StackingCVClassifier(classifiers=models,
                             meta_classifier=stack_model,
                             cv=n_splits,
                             stratify=True,
                             shuffle=True,
                             use_probas=True,
                             use_features_in_secondary=True,
                             verbose=0,
                             random_state=5,
                             n_jobs=2)

    stack = stack.fit(X, y)
    stack_out = copy.deepcopy(stack)
    
    return stack_out

def get_model_score(model, X_train, y_train, n_splits, scoring):
    print('Model: {model_name}'.format(model_name=model.__class__.__name__))
    start = time.time()
    score = cross_val_score(model, X_train, y_train, cv=n_splits, scoring=scoring).mean()
    end = time.time()
    elapsed_time = end - start
    print('Start time: ' + str(time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime(start))))
    print('Start time: ' + str(time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime(end))))
    
    if elapsed_time > 60:
        mins = elapsed_time // 60
        secs = elapsed_time - (mins * 60)
    else:
        mins = 0
        secs = elapsed_time
    elapsed = str(mins) + '  minutes, ' + str(int(secs)) + ' seconds'
    print('Elapsed time: ' + str(elapsed))
    print(score)
    print()

    return score