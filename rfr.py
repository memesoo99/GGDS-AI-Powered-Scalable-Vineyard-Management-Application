from sklearn.datasets import load_wine
import math
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os

import pickle
import matplotlib.pyplot as plot
from feature_extraction import Contours
from sklearn.model_selection import GridSearchCV

def inference(regressor_path='regressor_model.pkl', csv_path='/workspace/hue_test.csv', image_path='/workspace/regression_data/images3', mask_path = '/workspace/regression_data/masks'):
    """
    model_path : str
        RFR model path
    csv_path : str
        feature csv를 저장할 경로 ex) '/workspace/features.csv'
    """
    with open(regressor_path,"rb") as f:
        regressor = pickle.load(f)
    # for i in os.listdir(image_path):
    #     mask_name = i.split('.')[0]+'_masks.pkl'
    #     print(f"{image_path}/{i}")
    #     features = Contours(f"{mask_path}/{mask_name}", f"{image_path}/{i}",csv_path)
    #     features.run()

    df = pd.read_csv(csv_path)
    X = df.drop(["image"], axis=1).values

    pred = regressor.predict(X)
    pred = pred.astype(int)
    pred_df = pd.DataFrame(pred)
    n = len(df.columns)
    df.insert(n,'predict',pred_df)
    print(df.head())
    print(type(pred))
    result_df = pd.DataFrame({'Image':df["image"].to_numpy().reshape(-1), 'Predicted Values':pred.reshape(-1)})
    print(result_df)
    df.to_csv(csv_path)

def check_accuracy():
    pass

#train
def train(regressor_path = "regressor_model.pkl",csv_path = '/workspace/features3.csv'):
    # for i in os.listdir('/workspace/regression_data/images3'):
    #     print(i)
    #     # image_name = i.replace('.pkl','.jpg')
    #     pkl_name = i.split('.')[0]
    #     pkl_name = pkl_name + '_masks.pkl'
    #     # image_name = image_name.replace("_masks","")
    #     features = Contours(f"/workspace/regression_data/masks/{pkl_name}", f"/workspace/regression_data/images3/{i}",csv_path)
    #     features.run()

    pre_df1 = pd.read_csv("/workspace/features1.csv")
    pre_df2 = pd.read_csv("/workspace/features2.csv") # features csv
    pre_df3 = pd.read_csv("/workspace/features3.csv")
    pre_df = pd.concat([pre_df1,pre_df2,pre_df3],axis=0)
    # pre_df = pd.read_csv(csv_path)
    # gt랑 합치기. gt_df가 최종
 
    gt = pd.read_csv("/workspace/Counts.csv", index_col = False) # GT csv
    gt = gt.to_dict('split')['data']
    gt_df = pd.DataFrame(columns=["image","number of instances","sunburn_ratio","diameter","circularity","density","aspect ratio","grade","average_hue","gt"])
    gt_dict = {}

    for idx, row in enumerate(gt):
        row[0] = row[0][1:-1] # image_name 추출
        gt[idx] = row
        gt_dict[row[0]] = row[1] # image_name : gt

    for i in pre_df.itertuples():
        i = list(i)[1:]
        image_name = i[0].split('/')[-1]
        if image_name not in gt_dict.keys():
            continue
        i.append(gt_dict[image_name])

        #아래 부분 aspect ratio 정상 출력한 다음에 제거해야됨
        # if math.isnan(i[-2]):
       
        #     i[-2] = i[-4]
        # else: 
        #     i[-4] = i[-2]

        # i = i[:-3]+i[-2:]

        gt_df.loc[len(gt_df)] = i

    gt_df.to_csv('train_features.csv',index = False)
    X = gt_df.drop(["gt", "image"], axis=1).values
    Y = gt_df["gt"].values

    X_train, X_test, y_train, y_test = train_test_split(X,Y , test_size= 0.1)

    # Grid Search
    params = {'n_estimators':[10,50,60], 'max_depth':[6,8,10,12,14,40], 'min_samples_leaf':[8,12,18,20], 'min_samples_split':[8,16,20,24]}
    regressor = RandomForestRegressor(random_state = 0, n_jobs = -1)
    grid_cv = GridSearchCV(regressor, param_grid = params, cv=5, n_jobs=-1)
    grid_cv.fit(X_train, y_train)

    best_param = grid_cv.best_params_

    print('최적 하이퍼 파라미터:', best_param)
    # print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))
    estimator = grid_cv.best_estimator_

    regressor = RandomForestRegressor(random_state = 0, n_jobs = -1, criterion ='mse',max_depth = best_param['max_depth'],min_samples_leaf = best_param['min_samples_leaf'],n_estimators = best_param['n_estimators'],min_samples_split = best_param['min_samples_split'])
    regressor.fit(X_train,y_train)

    from sklearn.metrics import mean_squared_error
    some_predicted = regressor.predict(X_test)
    mse = np.sqrt(mean_squared_error(some_predicted, y_test))

    print('평균제곱근오차', mse)

    with open(regressor_path,"wb") as f:
        pickle.dump(regressor, f)

    y_pred = regressor.predict(X_test)
    result_df = pd.DataFrame({'Real Values':y_test.reshape(-1), 'Predicted Values':y_pred.reshape(-1)})
    print(result_df)

    # print(f'테스트 데이터 세트 정확도: {0:.4f}'.format(accuracy_score(y_test,y_pred)))

inference()

# train()