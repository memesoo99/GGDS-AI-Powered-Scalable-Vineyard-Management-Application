from sklearn.datasets import load_wine
import math
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
import argparse
import pickle
import matplotlib.pyplot as plot
from feature_extraction import Contours
from sklearn.model_selection import GridSearchCV

def inference(regressor_path, csv_path, image_path, mask_path):
    """
    model_path : str
        RFR model path
    csv_path : str
        feature csv를 저장할 경로 ex) '/workspace/features.csv'
    """
    with open(regressor_path,"rb") as f:
        regressor = pickle.load(f)

    for i in os.listdir(image_path):
        mask_name = i.split('.')[0]+'_masks.pkl'
        print(f"{image_path}/{i}")
        features = Contours(f"{mask_path}/{mask_name}", f"{image_path}/{i}",csv_path)
        features.run()

    df = pd.read_csv(csv_path)
    X = df.drop(["image"], axis=1).values

    pred = regressor.predict(X)
    pred = pred.astype(int)
    pred_df = pd.DataFrame(pred)
    n = len(df.columns)
    df.insert(n,'predict',pred_df)
    result_df = pd.DataFrame({'Image':df["image"].to_numpy().reshape(-1), 'Predicted Values':pred.reshape(-1)})
    # print(result_df)
    df.to_csv(csv_path)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select inference/save mode')
    parser.add_argument('--train', action='store_true', default=False,
                        help='an integer for the accumulator')
    parser.add_argument('--inference',action='store_true', default=False,
                        help='sum the integers (default: find the max)')
    parser.add_argument('--regressor-path', type=str, default='regressor_model.pkl',
                        help='Trained regressor model path')
    parser.add_argument('--csv-path', type=str, default='/workspace/hue_test.csv',
                        help='csv file path where you want to save results')
    parser.add_argument('--image-path', type=str, default='/workspace/regression_data/images3',
                        help='inference할 cropped된 포도 이미지 folder')
    parser.add_argument('--mask-path', type=str, default='/workspace/regression_data/masks',
                        help='inference할 cropped된 포도 이미지 folder')
    
    args = parser.parse_args()
    if inference:
        inference(regressor_path = args.regressor_path, csv_path=args.csv_path, image_path= args.image_path, mask_path=args.mask_path)
    if train:
        train()
    # train()