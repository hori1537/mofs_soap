

# programmed by YUKI Horie, Glass Research Center 'yuki.horie@'
# do not use JAPANESE!

from __future__ import print_function

import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import sklearn.ensemble
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import _tree
from sklearn.externals.six import StringIO
from sklearn import linear_model
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge
#ver0.20
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split


import numpy as np
import pandas as pd
import math
import time
import random
import os
import sys
import itertools
import copy

import category_encoders

import matplotlib.pyplot as plt 
import seaborn as sns

import GPy
import GPyOpt

try:
    import dtreeviz.trees
    import dtreeviz.shadow
    import dtreeviz
except:
    print('dtreeviz was not found')
    pass
    
    
import pydotplus

import xgboost as xgb
#from xgboost import plot_tree # sklearn has also plot_tree, so do not import plot_Tree


# #chkprint 
# refer from https://qiita.com/AnchorBlues/items/f7725ba87ce349cb0382
from inspect import currentframe
def chkprint(*args):
    names = {id(v):k for k,v in currentframe().f_back.f_locals.items()}
    print(', '.join(names.get(id(arg),'???')+' = '+repr(arg) for arg in args))

# get variable name
def get_variablename(*args):
    names = {id(v):k for k,v in currentframe().f_back.f_locals.items()}
    return '_' + ', '.join(names.get(id(arg),'???') + '_' + repr(arg) for arg in args)

# fix the random.seed, it can get the same results every time to run this program
np.random.seed(1)
random.seed(1)


# check the desktop path and move to the desktop path
#desktop_path =  os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Desktop"

#desktop_path =  os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "/Desktop"

desktop_path =  '/home'
os.chdir(desktop_path)


#http://cocodrips.hateblo.jp/entry/2015/07/19/120028
# Graphviz path
#http://spacegomi.hatenablog.com/entry/2018/01/26/170721
#sys.path.append('C:\\Users\\1310202\\AppData\\Local\\Continuum\\anaconda3\\envs\\py36\\Library\\bin\\graphviz')
#graphviz_path = 'C:\\Users\\1310202\\Desktop\\graphviz-2.38\\release\\bin\\dot.exe'

# name of theme name
theme_name = 'mofs'

# outputs of dataset
#address_ = 'C:/Users/1310202/Desktop/20180921/horie/data_science/for8ken/'
#address_ = 'E:/8ken/'
address_ = '/home/garaken/mofs'

# CSV name of data
CSV_NAME = "cif_soap.csv"

# csv category information
category_list = []

# csv information
info_num    = 0    # information columns in csv file
input_num   = 0    # input data  columns in csv file
output_num  = 30000     # output data columns in csv file


# evaluate of all candidate or not
is_gridsearch = True

# perform deeplearning or not
is_dl = False

# baysian optimization
is_bo = False


# to make the folder of Machine Learning
if os.path.exists(address_) == False:
    print('no exist ', address_)
    pass


# make the folder for saving the results
def chk_mkdir(paths):
    for path_name in paths:
        if os.path.exists(path_name) == False:
            os.mkdir(path_name)
    return
    

for is_inverse_predict in [False]: # 0,1
    # False  0: forward predict *normal
    # True   1: inverse predict
    
    inv_    = is_inverse_predict
    
    direction_name_list = ['normal', 'inverse']
    direction_name      = direction_name_list[inv_]
    
    paths = [ 'ML',
              'ML\\' + theme_name, 
              'ML\\' + theme_name + '\\' + direction_name,
              'ML\\' + theme_name + '\\' + direction_name + '\\sklearn', 
              'ML\\' + theme_name + '\\' + direction_name + '\\sklearn\\tree', 
              'ML\\' + theme_name + '\\' + direction_name + '\\sklearn\\importance',
              'ML\\' + theme_name + '\\' + direction_name + '\\sklearn\\parameter',
              'ML\\' + theme_name + '\\' + direction_name + '\\sklearn\\predict',
              'ML\\' + theme_name + '\\' + direction_name + '\\sklearn\\traintest',
              'ML\\' + theme_name + '\\' + direction_name + '\\deeplearning',
              'ML\\' + theme_name + '\\' + direction_name + '\\deeplearning\\h5',
              'ML\\' + theme_name + '\\' + direction_name + '\\deeplearning\\traintest',
              'ML\\' + theme_name + '\\' + direction_name + '\\deeplearning\\predict']

    chk_mkdir(paths)
    

# to move the directory to the theme name
os.chdir('ML')
os.chdir(theme_name)


#raw_data_df = pd.read_csv(open(str(address_) + str(CSV_NAME) ,encoding="utf-8_sig"))
raw_data_df = pd.read_csv(open(str(address_) + str(CSV_NAME)))

input_num_plus = 0
output_num_plus = 0


for column_name in category_list:
    column_nth = raw_data_df.columns.get_loc(column_name)
    num_plus = raw_data_df[column_name].nunique()
    #print(num_plus)
    
    if column_nth <= info_num + input_num:
        input_num_plus += num_plus-1
    elif column_nth < info_num + input_num:
        output_num_plus += num_plus-1
    
    pass
    

input_num   += input_num_plus 
output_num  += output_num_plus
list_num    = [input_num, output_num]


def get_ordinal_mapping(obj):

    listx = list()
    for x in obj.category_mapping:
        listx.extend([tuple([x['col']])+ i for i in x['mapping']])
    df_ord_map = pd.DataFrame(listx)
    return df_ord_map

ce_onehot   = category_encoders.OneHotEncoder(cols = category_list, handle_unknown = 'impute')
ce_binary   = category_encoders.BinaryEncoder(cols = category_list, handle_unknown = 'impute')

ce_onehot.fit_transform(raw_data_df)
raw_data_df = ce_binary.fit_transform(raw_data_df)

get_ordinal_mapping(ce_onehot)

print(get_ordinal_mapping(ce_onehot))


print(input_num)
print(output_num)


print(raw_data_df)

# csv information data columns
info_col    = info_num 
input_col   = info_num + input_num 
output_col  = info_num + input_num + output_num


info_df         = raw_data_df.iloc[:, 0         : info_col]
input_df        = raw_data_df.iloc[:, info_col  : input_col]
output_df       = raw_data_df.iloc[:, input_col : output_col]
in_output_df    = raw_data_df.iloc[:, info_col  : output_col]
list_df         = [input_df, output_df]




info_feature_names              = info_df.columns
input_feature_names             = input_df.columns
output_feature_names            = output_df.columns
list_feature_names              = [input_feature_names, output_feature_names]

predict_input_feature_names     = list(map(lambda x:x + '-predict' , input_feature_names))
predict_output_feature_names    = list(map(lambda x:x + '-predict' , output_feature_names))
list_predict_feature_names      = [predict_input_feature_names, predict_output_feature_names]

from sklearn.preprocessing import StandardScaler
in_output_sc_model  = StandardScaler()
input_sc_model      = StandardScaler()
output_sc_model     = StandardScaler()
list_sc_model       = [input_sc_model, output_sc_model]

in_output_std_df    = pd.DataFrame(in_output_sc_model.fit_transform(in_output_df))
input_std_df        = pd.DataFrame(input_sc_model.fit_transform(input_df))
output_std_df       = pd.DataFrame(output_sc_model.fit_transform(output_df))

input_des   = input_df.describe() 
output_des  = output_df.describe() 

input_max   = input_des.loc['max']
input_min   = input_des.loc['min']
output_max  = output_des.loc['max']
output_min  = output_des.loc['min']

list_max    = [input_max, output_max] 
list_min    = [input_min, output_min] 

input_std_des   = input_std_df.describe() 
output_std_des  = output_std_df.describe() 

input_std_max   = input_std_des.loc['max']
input_std_min   = input_std_des.loc['min']
output_std_max  = output_std_des.loc['max']
output_std_min  = output_std_des.loc['min']

list_std_max    = [input_std_max, output_std_max] 
list_std_min    = [input_std_min, output_std_min] 

# split train data and test data from the in_output_std_df
np.random.seed(10)
random.seed(10)
train_std_df, test_std_df   = train_test_split(in_output_std_df, test_size=0.2)
np.random.seed(10)
random.seed(10)
train_df, test_df           = train_test_split(in_output_df, test_size=0.2)

# transform from pandas dataframe to numpy array
train_np = np.array(train_df)
test_np  = np.array(test_df)
train_std_np = np.array(train_std_df)
test_std_np  = np.array(test_std_df)

# split columns to info, input, output
[train_input, train_output]         = np.hsplit(train_np, [input_num])
list_train                          = [train_input, train_output]
[test_input,  test_output]          = np.hsplit(test_np,  [input_num])
list_test                           = [test_input, test_output]

[train_input_std, train_output_std] = np.hsplit(train_std_np, [input_num])
list_train_std                      = [train_input_std  , train_output_std]

[test_input_std,  test_output_std]  = np.hsplit(test_std_np , [input_num])
list_test_std                       = [test_input_std   , test_output_std]

train_input_df                      = pd.DataFrame(train_input  , columns = input_feature_names)
test_input_df                       = pd.DataFrame(test_input   , columns = input_feature_names)
train_output_df                     = pd.DataFrame(train_output , columns = output_feature_names)
test_output_df                      = pd.DataFrame(test_output  , columns = output_feature_names)

list_train_df                       = [train_input_df, train_output_df]
list_test_df                        = [test_input_df, test_output_df]

train_input_std_df                  = pd.DataFrame(train_input_std  , columns = input_feature_names)
test_input_std_df                   = pd.DataFrame(test_input_std   , columns = input_feature_names)
train_output_std_df                 = pd.DataFrame(train_output_std , columns = output_feature_names)
test_output_std_df                  = pd.DataFrame(test_output_std  , columns = output_feature_names)

list_train_std_df                   = [train_input_std_df , train_output_std_df ]
list_test_std_df                    = [test_input_std_df  , test_output_std_df ]



#######  extract descision tree   ################################################

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    #print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        
        with open(str(save_address)+ 'tree.txt', 'a') as f:
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                #print("{}if {} <= {}:".format(indent, name, threshold))
                #print("{}if {} <= {}:".format(indent, name, threshold), file=f)
                recurse(tree_.children_left[node], depth + 1)
                #print("{}else:  # if {} > {}".format(indent, name, threshold), file=f)
                #print("{}else:  # if {} > {}".format(indent, name, threshold))
                recurse(tree_.children_right[node], depth + 1)
            else:
                #print("{}return {}".format(indent, tree_.value[node]))
                #print("{}return {}".format(indent, tree_.value[node]), file=f)
                pass

    recurse(0, 1)

def gridsearch_predict(model_std, model_name):        
    start_time = time.time()

    gridsearch_predict_std_df   = pd.DataFrame(model_std.predict(gridsearch_input_std), columns = list_predict_feature_names[out_n])
    gridsearch_predict_df       = pd.DataFrame(list_sc_model[out_n].inverse_transform(gridsearch_predict_std_df), columns = list_predict_feature_names[out_n])

    end_time = time.time()
    total_time = end_time - start_time
    #print(model_name, ' ' , total_time)

    #gridsearch_predict_df['Tmax-']      = gridsearch_predict_df.max(axis = 1)
    #gridsearch_predict_df['Tdelta-']    = gridsearch_predict_df.min(axis = 1)
    #gridsearch_predict_df['Tmin-']      = gridsearch_predict_df['Tmax-'] - gridsearch_predict_df['Tdelta-']
    
    gridsearch_predict_df = pd.concat([gridsearch_input_df, gridsearch_predict_df], axis = 1)
    gridsearch_predict_df.to_csv(direction_name + '/sklearn/predict/' + str(model_name) + '_predict.csv')

    return
        
#########   search by hyperopt ###############
#import hyperopt

#########   save the tree to pdf ###############

#### dtree viz #####
'''
train_output_  = train_output.flatten()
try:
    viz = dtreeviz.trees.dtreeviz(model_raw,
                                  train_input,
                                  train_output_, 
                                  target_name   = list_predict_feature_names[out_n], 
                                  feature_names = list_predict_feature_names[in_n])

    viz.save(direction_name + '/sklearn/tree/' + 'decisiontree' + str(max_depth) + '.svg')
    #viz.view()
    #sys.exit()

except:
    #print('dtreeviz error')
    # PATH
    # output_num more than 2
    
'''

#########   regression by the scikitlearn model ###############


#print(allmodel_results_df)
        
def regression(model, model_name):
    print(model_name)
    start_time = time.time()
    
    model_raw           = copy.deepcopy(model)
    model_std           = copy.deepcopy(model)

    model_raw.fit(list_train[in_n],     list_train[out_n])
    model_std.fit(list_train_std[in_n], list_train_std[out_n])

    save_regression(model_raw, model_std, model_name)

    return


def save_regression(model_raw, model_std, model_name):

    def save_tree_topdf(model, model_name):

        dot_data = StringIO()
        try:
            sklearn.tree.export_graphviz(model, out_file=dot_data, feature_names = list_feature_names[in_n])
        except:
            xgb.to_graphviz(model,  out_file=dot_data, feature_names = list_feature_names[in_n])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        
        # refer from https://qiita.com/wm5775/items/1062cc1e96726b153e28
        # the Graphviz2.38 dot.exe
        graph.progs = {'dot':graphviz_path}
        
        graph.write_pdf(direction_name + '/sklearn/tree/' + model_name + '.pdf')
        pass
        
        return
        
    def save_importance_features(model, model_name):
    
        importances = pd.Series(model.feature_importances_)
        importances = np.array(importances)

        label       = list_feature_names[in_n]
        chkprint(model_name)
        chkprint(label)
        chkprint(importances)
        if inv_ == True:
            #sys.exit()
            pass

        plt.bar(label, importances)

        plt.xticks(rotation=90)
        plt.xticks(fontsize=8)
        plt.rcParams["font.size"] = 12

        plt.title("importances-" + model_name)
        #plt.show()
        plt.savefig(direction_name + '/sklearn/importance/' + str(model_name)   + ' ' + list_feature_names[out_n][i] + '.png', dpi = 240)
                
        return


    global allmodel_results_df
    
    train_output_predict_std    = model_std.predict(list_train_std[in_n])
    test_output_predict_std     = model_std.predict(list_test_std[in_n])
    
    train_output_predict    = list_sc_model[out_n].inverse_transform(train_output_predict_std)
    test_output_predict     = list_sc_model[out_n].inverse_transform(test_output_predict_std)

    if hasattr(model_std, 'score') == True:
        train_model_score   = model_std.score(list_train_std[in_n] , list_train_std[out_n])
        train_model_score   = np.round(train_model_score,3)
        test_model_score    = model_std.score(list_test_std[in_n],   list_test_std[out_n])
        test_model_score    = np.round(test_model_score,3)
    
    if hasattr(model_std, 'evaluate') == True:
        train_model_score   = model_std.evaluate(list_train_std[in_n] , list_train_std[out_n])
        train_model_score   = np.round(train_model_score,3)
        test_model_score    = model_std.evaluate(test_input_std,   list_test_std[out_n])
        test_model_score    = np.round(test_model_score,3)

    train_output_predict_df = pd.DataFrame(train_output_predict, columns = list_predict_feature_names[out_n])
    train_result_df         = pd.concat([list_train_df[in_n], train_output_predict_df, list_train_df[out_n]], axis=1)

    test_output_predict_df  = pd.DataFrame(test_output_predict, columns = list_predict_feature_names[out_n])
    test_result_df          = pd.concat([list_test_df[in_n], test_output_predict_df, list_test_df[out_n]], axis=1)
    
    train_model_mse     = sklearn.metrics.mean_squared_error(list_train_std[out_n], train_output_predict_std)
    train_model_rmse    = np.sqrt(train_model_mse)
    test_model_mse      = sklearn.metrics.mean_squared_error(list_test_std[out_n], test_output_predict_std)
    test_model_rmse     = np.sqrt(test_model_mse)
    
    results_df = pd.DataFrame([model_name, train_model_mse, train_model_rmse, test_model_mse, test_model_rmse, train_model_score, test_model_score]).T
    results_df.columns = columns_results
    allmodel_results_df = pd.concat([allmodel_results_df, results_df])
            
    #chkprint(model_name)
    
    train_result_df.to_csv(direction_name + '/sklearn/traintest/' + str(model_name) + '_train.csv')
    test_result_df.to_csv( direction_name + '/sklearn/traintest/' + str(model_name) + '_test.csv')
    
    if hasattr(model_std, 'get_params') == True:
        model_params        = model_std.get_params()
        params_df           = pd.DataFrame([model_std.get_params])
    
    if hasattr(model_std, 'intercept_') == True &  hasattr(model_std, 'coef_') == True:
        model_intercept_df  = pd.DataFrame(model_std.intercept_)
        model_coef_df       = pd.DataFrame(model_std.coef_)
        model_parameter_df  = pd.concat([model_intercept_df, model_coef_df])
        model_parameter_df.to_csv(direction_name + '/sklearn/parameter/' + str(model_name) + '_parameter.csv')
        
    if hasattr(model_raw, 'tree_') == True:
        save_tree_topdf(model_raw, model_name)
        pass
        
    if hasattr(model_raw, 'feature_importances_') == True:

        importances = pd.Series(model_raw.feature_importances_)
        importances = np.array(importances)
        #print(importances)
        #importances = importances.sort_values()
        
        label       = list_feature_names[in_n]
        
        # initialization of plt
        plt.clf()
        
        plt.bar(label, importances)
        
        plt.xticks(rotation=90)
        plt.xticks(fontsize=8)
        plt.rcParams["font.size"] = 12

        plt.title("importance in the tree " + str(theme_name))
        #plt.show()
        plt.savefig(direction_name + '/sklearn/importance/' + str(model_name)  + '.png', dpi = 240)

    if hasattr(model_raw, 'estimators_') == True:

        #if 'DecisionTreeRegressor' in str(type(model_raw.estimators_[0])):
        if 'MultiOutput DecisionTreeRegressor' in model_name:
        
            MultiOutput_DTR_estimators = len(model_raw.estimators_)
            
            for i in range(MultiOutput_DTR_estimators):
                
                model_name_i = model_name + '_'  +list_feature_names[out_n][i]
                
                # save_importance_features
                save_importance_features(model_raw.estimators_[i], model_name_i)
                
                # save_tree to pdf
                save_tree_topdf(model_raw.estimators_[i], model_name_i)

        #if 'RandomForestRegressor' in str(type(model_raw)):
        if 'RandomForest' in model_name:
            for i in range(1):
                if hasattr(model_raw.estimators_[i], 'tree_') == True:
                    model_name_i = model_name + '_tree-'+ str(i)
                    
                    # save_importance_features
                    save_importance_features(model_raw.estimators_[i], model_name_i)
                    
                    # save_tree to pdf
                    save_tree_topdf(model_raw.estimators_[i], model_name_i)
                   
        
        #if 'xgboost.sklearn.XGBRegressor' in str(type(model_raw.estimators_[0])):
        '''
        if 'XGB' in model_name:
            for i in range(3):
                
                model_name_ = model_name + '_tree-'+ str(i)
                graph = xgb.to_graphviz(model_raw.estimators_[i], num_trees = i)
                
                graph.render(direction_name + '/sklearn/tree/' + model_name_ +str(i)+ '.png')

            if hasattr(model_raw.estimators_[i], 'get_booster'):
                # refer from https://github.com/dmlc/xgboost/issues/1238
                #print('(model_raw.estimators_[i].get_booster().feature_names')
                #print((model_raw.estimators_[i].get_booster().feature_names))
                
                ##print(type((model_raw.estimators_[i].get_booster)))
                #for x in dir(model_raw.estimators_[i].get_booster):
                #    #print(x)
                #sys.exit()
                pass
            else:
                #print('get booster not found')
                pass
        '''


    # call gridsearch_predict 
    if is_gridsearch == True:
        gridsearch_predict(model_std, model_name)


    # Baysian Optimization
    # refer https://qiita.com/shinmura0/items/2b54ab0117727ce007fd
    # refer https://qiita.com/marshi/items/51b82a7b990d51bd98cd
    
    if is_bo == False:
    
        def function_for_baysian(x):
            return model_std.predict(x)
            
        if inv_ == 0 and list_num[out_n] ==1 :
            bounds = []
            print(list_num[in_n])
            for i in range(list_num[in_n]):
                bounds.append({'name': list_feature_names[in_n][i] , 'type': 'continuous', 'domain': (list_std_min[in_n][i],list_std_max[in_n][i])})
                     
            chkprint(bounds) 
            myBopt = GPyOpt.methods.BayesianOptimization(f=function_for_baysian, domain=bounds)

            myBopt.run_optimization(max_iter=30)
            
            print(list_sc_model[in_n].inverse_transform(np.array([myBopt.x_opt])))
            print(list_sc_model[out_n].inverse_transform(np.array([myBopt.fx_opt])))
        
        return
    

if output_num != 1:
    list_inverse_predict = [False, True]
elif output_num == 1:
    list_inverse_predict = [False, True]


for is_inverse_predict in list_inverse_predict: # 0,1
    chkprint(is_inverse_predict)

    # False  0: forward predict *normal
    # True   1: inverse predict
    
    inv_    = is_inverse_predict
    in_n    = is_inverse_predict        # 0 to 1
    out_n   = not(is_inverse_predict)   # 1 to 0
    
    #print(in_n)
    #print(out_n)
    
    direction_name_list = ['normal', 'inverse']
    direction_name      = direction_name_list[inv_]
    
    columns_results = ['model_name', 'train_model_mse', 'train_model_rmse', 'test_model_mse', 'test_model_rmse', 'train_model_score', 'test_model_score']
    allmodel_results_df = pd.DataFrame(columns = columns_results)
    
    #########   predict of all candidate by the scikitlearn model ###############

    # select the feature value by the random forest regressor
    max_depth = 7
    model       = sklearn.ensemble.RandomForestRegressor(max_depth = max_depth)
    model_name  = ''
    model_name  += 'RandomForestRegressor_'
    model_name  += 'max_depth_'+str(max_depth)
    
    print('list_train[in_n]')
    
    print(list_train[in_n])
    model.fit(list_train[in_n], list_train[out_n])
    
    importances         = np.array(model.feature_importances_)
    chkprint(importances)
    importances_sort    = importances.argsort()[::-1]
    split_base          = np.array([15,13,9,4,4,3,3,3]) # max:758160
    split_base          = np.array([10,7,3,3,3,3,3,3])  # max:51030

    # set the split num from importances rank of random forest regressor
    split_num   = np.full(len(importances_sort),1)
    for i in range(min(len(importances),8)):
        rank_ = importances_sort[i]
        split_num[rank_] = split_base[i]

    def combination(max, min, split_num):
        candidate = []
        for i in range(list_num[in_n]):
            candidate.append(np.linspace(start = max[i], stop = min[i], num = split_num[i]))

        candidate = np.array(candidate)        
        return candidate

    if is_gridsearch == True:

        all_gridsearch_number = split_num.prod()
        candidate = combination(list_max[in_n], list_min[in_n], split_num)

        # refer from https://teratail.com/questions/152110
        # unpack   *candidate
        gridsearch_input        = list(itertools.product(*candidate))
        #print(gridsearch_input)
        gridsearch_input_std    = list_sc_model[in_n].transform(gridsearch_input)

        gridsearch_input_df     = pd.DataFrame(gridsearch_input, columns = list_feature_names[in_n])
        gridsearch_input_std_df = pd.DataFrame(gridsearch_input_std, columns = list_feature_names[in_n])
    


    ##################### Linear Regression #####################

    model = linear_model.LinearRegression()
    model_name = 'linear_regression_'

    regression(model, model_name)

    ##################### Regression of Stochastic Gradient Descent ##################### 
    max_iter = 1000

    model = MultiOutputRegressor(linear_model.SGDRegressor(max_iter = max_iter))
    model_name = 'MultiOutput Stochastic Gradient Descent_'
    model_name += 'max_iter_'+str(max_iter)

    regression(model, model_name)



    ##################### Regression of SVR #####################
    kernel_ = 'rbf'
    C_= 1

    model = MultiOutputRegressor(svm.SVR(kernel = kernel_, C = C_))
    model_name = 'MultiOutput SupportVectorRegressor_'
    model_name += 'kernel_'+str(kernel_)
    model_name += 'C_'+str(C_)

    regression(model, model_name)
    
    # refer https://www.slideshare.net/ShinyaShimizu/ss-11623505


    ##################### Regression of Ridge #####################
    alpha_ = 1.0
    model = linear_model.Ridge(alpha = alpha_)
    model_name = 'Ridge_'
    model_name += 'alpha_'+str(alpha_)

    regression(model, model_name)


    ##################### Regression of KernelRidge #####################
    alpha_ = 1.0
    model = KernelRidge(alpha=alpha_, kernel='rbf')
    model_name = 'KernelRidge_'
    model_name += 'alpha_'+str(alpha_)

    regression(model, model_name)


    ##################### Regression of Lasso #####################
    alpha_ = 1.0
    model = linear_model.Lasso(alpha = alpha_)
    model_name = 'Lasso_'
    model_name += 'alpha_'+str(alpha_)

    regression(model, model_name)


    ##################### Regression of Elastic Net #####################
    alpha_ =1.0
    l1_ratio_ = 0.5
    model = linear_model.ElasticNet(alpha=alpha_, l1_ratio = l1_ratio_)
    model_name = 'ElasticNet_'
    model_name += 'alpha_'+str(alpha_)
    model_name += 'l1_ratio_'+str(l1_ratio_)

    regression(model, model_name)


    ##################### Regression of MultiTaskLassoCV #####################
    max_iter_ = 1000 
    model = linear_model.MultiTaskLassoCV()
    model_name = 'MultiTaskLasso_'
    model_name += 'max_iter_'+str(max_iter)

    regression(model, model_name)

    ##################### Regression of Multi Task Elastic Net CV #####################
    model = linear_model.MultiTaskElasticNetCV()

    model_name = 'MTElasticNet_'
    regression(model, model_name)


    ##################### Regression of OrthogonalMatchingPursuit #####################
    #model = linear_model.OrthogonalMatchingPursuit()
    #model_name = 'OrthogonalMatchingPursuit_'

    #regression(model, model_name)

    ##################### Regression of BayesianRidge #####################
    model = MultiOutputRegressor(linear_model.BayesianRidge())
    model_name = 'MultiOutput BayesianRidge_'

    regression(model, model_name)

    ##################### Regression of PassiveAggressiveRegressor #####################
    #model = MultiOutputRegressor(linear_model.PassiveAggressiveRegressor())
    #model_name = 'MultiOutput PassiveAggressiveRegressor_'

    #regression(model, model_name)

    ##################### Regression of PolynomialFeatures #####################
    '''
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    # http://techtipshoge.blogspot.com/2015/06/scikit-learn.html
    # http://enakai00.hatenablog.com/entry/2017/10/13/145337

    for degree in [2]:
        model = Pipeline([
            ('poly', PolynomialFeatures(degree = 2),
            'linear', MultiOutputRegressor(linear_model.LinearRegression()))
            ])
        
        model_name = 'PolynomialFeatures_'
        model_name += 'degree_' + str(degree)
        
        regression(model, model_name)
    '''

    ##################### Regression of GaussianProcessRegressor #####################
    '''
    from sklearn.gaussian_process import GaussianProcessRegressor

    model = MultiOutputRegressor(GaussianProcessRegressor())
    model_name = 'MultiOutput GaussianProcessRegressor_'
        
    regression(model, model_name)
    '''

    ##################### Regression of GaussianNB #####################

    '''
    from sklearn.naive_bayes import GaussianNB

    model = MultiOutputRegressor(GaussianNB())
    model_name = 'MultiOutput GaussianNB_'

    regression(model, model_name)
    '''

    ##################### Regression of GaussianNB #####################

    '''
    from sklearn.naive_bayes import  ComplementNB

    model = ComplementNB()
    model_name = 'ComplementNB_'
        
    regression(model, model_name)
    '''

    ##################### Regression of MultinomialNB #####################

    '''
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model_name = 'MultinomialNB_'
        
    regression(model, model_name)
    '''

    ##################### Regression of DecisionTreeRegressor #####################
    for max_depth in [7]:
        model = sklearn.tree.DecisionTreeRegressor(max_depth = max_depth)
        model_name = 'DecisionTreeRegressor_'    
        model_name += 'max_depth_'+str(max_depth)

        regression(model, model_name)

    ##################### Regression of Multioutput DecisionTreeRegressor #####################
    for max_depth in [7]:

        model = MultiOutputRegressor(sklearn.tree.DecisionTreeRegressor(max_depth = max_depth))
        model_name = 'MultiOutput DecisionTreeRegressor_'    
        model_name += 'max_depth_'+str(max_depth)

        regression(model, model_name)



    #################### Regression of RandomForestRegressor #####################
    for max_depth in [7]:
        model = sklearn.ensemble.RandomForestRegressor(max_depth = max_depth)
        model_name = ''
        model_name += 'RandomForestRegressor_'
        #model_name += get_variablename(max_depth)
        model_name += 'max_depth_'+str(max_depth)
        
        regression(model, model_name)

    ##################### Regression of XGBoost #####################
    # refer from https://github.com/FelixNeutatz/ED2/blob/23170b05c7c800e2d2e2cf80d62703ee540d2bcb/src/model/ml/CellPredict.py

    estimator__min_child_weight_ = [5] #1,3 
    estimator__subsample_        = [0.9] #0.7, 0.8, 
    estimator__learning_rate_    = [0.1,0.01] #0.1
    estimator__max_depth_        = [7]
    estimator__n_estimators_      = [100]

    for estimator__min_child_weight, estimator__subsample, estimator__learning_rate, estimator__max_depth, estimator__n_estimators \
         in itertools.product(estimator__min_child_weight_, estimator__subsample_, estimator__learning_rate_, estimator__max_depth_,estimator__n_estimators_ ):

        xgb_params = {'estimator__min_child_weight': estimator__min_child_weight,
                      'estimator__subsample': estimator__subsample,
                      'estimator__learning_rate': estimator__learning_rate,
                      'estimator__max_depth': estimator__max_depth,
                      'estimator__n_estimators': estimator__n_estimators,
                      'colsample_bytree': 0.8,
                      'silent': 1,
                      'seed': 0,
                      'objective': 'reg:linear'}

        model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))
        
        model_name = 'MultiOutput-XGBoost'
        model_name += 'min_child_weight_'+str(estimator__min_child_weight)
        model_name += 'subsample_'+str(estimator__subsample)
        model_name += 'learning_rate_'+str(estimator__learning_rate)
        model_name += 'max_depth_'+str(estimator__max_depth)
        model_name += 'n_estimators_'+str(estimator__n_estimators)

        regression(model, model_name)



    ################# to csv ##############################
    allmodel_results_df.to_csv(direction_name + '/comparison of methods.csv')

    #######################################################




    '''
    ################# importances feature by XGBOOST ######
    import matplotlib.pyplot as plt

    importances = pd.Series(reg1_multigbtree.feature_importances_)
    importances = importances.sort_values()
    importances.plot(kind = "barh")
    plt.title("imporance in the xgboost Model")
    plt.show()
    #######################################################
    '''




    '''
    ##################### LIME Explainer #####################
    import lime
    import lime.lime_tabular

    #explainer1 = lime.lime_tabular.LimeTabularExplainer(train_output, feature_names=input_feature_names, kernel_width=3)
    0
    explainer1 = lime.lime_tabular.LimeTabularExplainer(train_input, feature_names= input_feature_names, class_names=output_feature_names, verbose=True, mode='regression')

    np.random.seed(1)
    i = 3
    #exp = explainer.explain_instance(test[2], predict_fn, num_features=10)
    exp = explainer.explain_instance(test[i], reg1_SVR.predict, num_features=5)

    sys.exit()
    # exp.show_in_notebook(show_all=False)
    exp.save_to_file(file_path= str(address_) + 'numeric_category_feat_01', show_table=True, show_all=True)

    i = 3
    exp = explainer.explain_instance(test[i], predict_fn, num_features=10)
    # exp.show_in_notebook(show_all=False)
    exp.save_to_file(file_path=str(address_) + 'numeric_category_feat_02', show_table=True, show_all=True)
    ##########################################################
    '''




    '''
    # import pickle
    # pickle.dump(reg, open("model.pkl", "wb"))
    # reg = pickle.load(open("model.pkl", "rb"))

    pred1_train = reg1_gbtree.predict(train_input)
    pred1_test = reg1_gbtree.predict(test_input)
    #print(mean_squared_error(train_output, pred1_train))
    #print(mean_squared_error(test_output, pred1_test))

    import matplotlib.pyplot as plt

    importances = pd.Series(reg1_gbtree.feature_importances_)
    importances = importances.sort_values()
    importances.plot(kind = "barh")
    plt.title("imporance in the xgboost Model")
    plt.show()
    '''



##################### Deep Learning #####################
if is_dl == False:
    sys.exit()


import keras
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
#from keras.layers.normalization import BatchNomalization
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
import keras.models
from keras.wrappers.scikit_learn import KerasRegressor

from keras.models import load_model
from keras.utils import plot_model

import h5py


def get_model(num_layers, layer_size,bn_where,ac_last,keep_prob, patience):
    
    model =Sequential()
    model.add(InputLayer(input_shape=(input_num,)))
    #model.add(Dense(layer_size))

    for i in range(num_layers):
        if num_layers != 1:

            model.add(Dense(layer_size))

            if bn_where==0 or bn_where ==3:
                model.add(BatchNormalization(mode=0))

            model.add(Activation('relu'))

            if bn_where==1 or bn_where ==3:
                model.add(BatchNormalization(mode=0))

            model.add(Dropout(keep_prob))

            if bn_where==2 or bn_where ==3:
                model.add(BatchNormalization(mode=0))

    if ac_last ==1:
        model.add(Activation('relu'))

    model.add(Dense(output_num))


    model.compile(loss='mse',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

    return model

def get_model2(num_layers, layer_size,bn_where,ac_last,keep_prob, patience):
    model = Sequential()
    model.add(Dense(input_num, input_dim = input_num, activation = 'relu'))

    for i in range(num_layers):
        model.add(Dense(layer_size, activation = 'relu'))
    
    model.add(BatchNormalization(mode=0))
    model.add(Dense(output_num))

    model.compile(loss = 'mean_squared_error', optimizer = 'adam')

    return model
# refer from https://github.com/completelyAbsorbed/ML/blob/0ca17d25bae327fe9be8e3639426dc86f3555a5a/Practice/housing/housing_regression_NN.py


num_layers  = [4,5]
layer_size  = [64, 32, 16]
bn_where    = [3, 0]
ac_last     = [0, 1]
keep_prob   = [0]
patience    = [3000]

'''
for dp_params in itertools.product(num_layers, layer_size, bn_where, ac_last, keep_prob, patience):
    num_layers, layer_size, bn_where, ac_last, keep_prob, patience = dp_params
    
    batch_sixe  = 30
    nb_epochs   = 10000
    cb = keras.callbacks.EarlyStopping(monitor = 'loss'   , min_delta = 0,
                                 patience = patience, mode = 'auto')
                                 
    model = KerasRegressor(build_fn = get_model(*dp_params), nb_epoch=5000, batch_size=5, verbose=0, callbacks=[cb])
    
    model_name =   'deeplearning'
    model_name +=  '_numlayer-'       + str(num_layers)
    model_name +=  '_layersize-'      + str(layer_size)
    model_name +=  '_bn- '            + str(bn_where)
    model_name +=  '_ac-'             + str(ac_last)
    model_name +=  '_k_p-'            + str(keep_prob)
    model_name +=  '_patience-'       + str(patience)
    
    regression(model, model_name, train_input, train_output)
'''


allmodel_results_df.to_csv('comparison of methods.csv')

epochs = 100000
batch_size = 32

for patience_ in [100,3000]:

    es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_, verbose=0, mode='auto')

    for num_layers in [4,3,2]:
        if num_layers !=1:
            for layer_size in[1024,512,256,128,64,32,16]:
                for bn_where in [3,0,1,2]:
                    for ac_last in [0,1]:
                        for keep_prob in [0,0.1,0.2]:

                            model =get_model(num_layers,layer_size,bn_where,ac_last,keep_prob, patience)
                            #model = KerasRegressor(build_fn = model, epochs=5000, batch_size=5, verbose=0, callbacks=[es_cb])

                            if layer_size >= 1024:
                                batch_size = 30
                            elif num_layers >= 4:
                                batch_size = 30
                            elif bn_where ==3:
                                batch_size=30

                            else:
                                batch_size = 30

                            model_name = "deeplearning"
                            model_name +=  '_numlayer-'       + str(num_layers)
                            model_name +=  '_layersize-'      + str(layer_size)
                            model_name +=  '_bn- '            + str(bn_where)
                            model_name +=  '_ac-'             + str(ac_last)
                            model_name +=  '_k_p-'            + str(keep_prob)
                            model_name +=  '_pat-'       + str(patience_)

                            regression(model, model_name, list_train_std[in_n], test_input_std)



                            
                            model.fit(train_input, train_output,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      verbose=1,
                                      validation_data=(test_input, test_output),
                                      callbacks=[es_cb])
                            
                            save_regression(model, model_name, list_train_std[in_n], test_input_std)
                            
                            '''
                            score_test = model.evaluate(test_input, test_output, verbose=1)
                            score_train = model.evaluate(train_input, train_output, verbose=1)
                            test_predict = model.predict(test_input, batch_size=32, verbose=1)
                            train_predict = model.predict(train_input, batch_size=32, verbose=1)

                            df_test_pre = pd.DataFrame(test_predict)
                            df_train_pre = pd.DataFrame(train_predict)

                            df_test_param = pd.DataFrame(test_output)
                            df_train_param = pd.DataFrame(train_output)

                            df_dens_test = pd.concat([df_test_param, df_test_pre], axis=1)
                            df_dens_train = pd.concat([df_train_param, df_train_pre], axis=1)

                            savename = ""
                            savename +=  '_score_train-'    + str("%.3f" % round(score_train[0],3))
                            savename +=  '_score_test-'     + str("%.3f" % round((score_test[0]),3))
                            savename +=  '_numlayer-'       + str(num_layers)
                            savename +=  '_layersize-'      + str(layer_size)
                            savename +=  '_bn- '            + str(bn_where)
                            savename +=  '_ac-'             + str(ac_last)
                            savename +=  '_k_p-'            + str(keep_prob)
                            savename +=  '_patience-'       + str(patience_)

                            df_dens_test.to_csv('deeplearning/traintest/' + savename + '_test.csv')
                            df_dens_train.to_csv('deeplearning/traintest/' + savename + '_train.csv')
                            '''

                            model.save('deeplearning/h5/' + model_name + '.h5')

                            
                            model.summary()

                            
                            '''
                            ### evaluation of deeplearning ###
                            def eval_bydeeplearning(input):
                                output_predict = model.predict(input, batch_size = 1, verbose= 1)
                                output_predict = np.array(output_predict)

                                return output_predict

                            if is_gridsearch == True:

                                gridsearch_output = eval_bydeeplearning(gridsearch_input)
                                
                
                                #print('start the evaluation by deeplearning')
                                #print('candidate is ', candidate_number)
                                
                                start_time = time.time()
                                                           
                                iter_deeplearning_predict_df = pd.DataFrame(gridsearch_output, columns = predict_output_feature_names)                            
                                iter_deeplearning_predict_df['Tmax'] = iter_deeplearning_predict_df.max(axis=1)
                                iter_deeplearning_predict_df['Tmin'] = iter_deeplearning_predict_df.min(axis=1)
                                iter_deeplearning_predict_df['Tdelta'] = iter_deeplearning_predict_df['Tmax'] -  iter_deeplearning_predict_df['Tmin']

                                iter_deeplearning_df = pd.concat([gridsearch_input_std_df, iter_deeplearning_predict_df], axis=1)

                                end_time = time.time()

                                total_time = end_time - start_time
                                #print('total_time 1', total_time)

                                predict_df_s = iter_deeplearning_df.sort_values('Tdelta')

                                predict_df_s.to_csv('deeplearning/predict/'
                                                    + savename
                                                    + '_predict.csv')

                                # evaluate by the for - loop     Not use now
                                
                                
                                i=0                            
                                predict_df = pd.DataFrame()
                                output_delta_temp = 100000
                            
                                start_time = time.time()
                                #print('start the for loop')
                                for xpot_, ypot_, tend_, tside_, tmesh_, hter_ in itertools.product(xpot_candidate, ypot_candidate, tend_candidate, tside_candidate, tmesh_candidate, hter_candidate):
                                
                                    input_ori = [xpot_, ypot_, tend_, tside_, tmesh_, hter_]
                                    
                                    xpot_ = xpot_ / xpot_coef
                                    ypot_ = ypot_ / ypot_coef
                                    tend_ = tend_ / tend_coef
                                    tside_ = tside_ / tside_coef
                                    tmesh_ = tmesh_ / tmesh_coef
                                    hter_ = hter_ / hter_coef
                                    
                                    input = [xpot_, ypot_, tend_, tside_, tmesh_, hter_]
                                    
                                    ##print(input)
                                    input = np.reshape(input, [1,6])
                                    
                                    output = eval_bydeeplearning(input)
                                    output = output * 1000
                                    output_max = float(max(output[0]))
                                    output_min = float(min(output[0]))
                                    ##print(output_max)
                                    ##print(output_min)
                                    output_delta = float(output_max - output_min)

                                    tmp_series = pd.Series([i,input_ori[0],input_ori[1],input_ori[2],input_ori[3],input_ori[4],input_ori[5],output[0][0],output[0][1],output[0][2],output[0][3],output_max,output_min,output_delta])

                                    if output_delta < output_delta_temp * 1.05:
                                        output_delta_temp = min(output_delta,output_delta_temp)
                                        
                                        predict_df = predict_df.append(tmp_series,ignore_index = True)
                                    
                                    i +=1
                                    #if i > 100 :
                                    #    break


                                end_time = time.time()
                                total_time = end_time - start_time
                                #print('loop time is ', total_time)
                                
                            '''

            
        else:
            layer_size=62
            bn_where=1
            keep_prob=0.2
            for ac_last in [1]:
                model = get_model(num_layers, layer_size, bn_where, ac_last, keep_prob)

                model.fit(train_input, train_output,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(test_input, test_output),
                          callbacks=[es_cb])

                score_test = model.evaluate(test_input, test_output, verbose=0)
                score_train = model.evaluate(train_input, train_output, verbose=0)
                test_predict = model.predict(test_input, batch_size=32, verbose=1)
                train_predict = model.predict(train_input, batch_size=32, verbose=1)

                df_test_pre = pd.DataFrame(test_predict)
                df_train_pre = pd.DataFrame(train_predict)

                df_test_param = pd.DataFrame(test_output)
                df_train_param = pd.DataFrame(train_output)

                df_dens_test = pd.concat([df_test_param, df_test_pre], axis=1)
                df_dens_train = pd.concat([df_train_param, df_train_pre], axis=1)

                savename = ""
                savename +=  '_score_train-' + str("%.3f" % round(math.log10(score_train[0]), 3))
                savename +=  '_score_test-' + str("%.3f" % round(math.log10(score_test[0]), 3))
                savename +=  '_numlayer-' + str(num_layers)
                savename +=  '_layersize-' + str(layer_size)
                savename +=  '_bn- ' + str(bn_where)
                savename +=  '_ac-' + str(ac_last)
                savename +=  '_k_p-' + str(keep_prob)
                savename +=  '_patience-' + str(patience_)


                df_dens_test.to_csv('deeplearning/traintest/' + savename + '_test.csv')

                df_dens_train.to_csv('deeplearning/traintest/' + savename + '_train.csv')


                model.save('deeplearning/h5/'
                                       + savename
                                       + '.h5')
                # plot_model(model, to_file='C:\Deeplearning/model.png')
                #plot_model(model, to_file='model.png')

                #print('Test loss:', score_test[0])
                #print('Test accuracy:', score_test[1])

                ##print('predict', test_predict)

                model.summary()                

