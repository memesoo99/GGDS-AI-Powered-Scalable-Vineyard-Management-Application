# Grape-thinning
3-step grape prunning algorithm

## Process
![grape](https://user-images.githubusercontent.com/68745418/174239241-3e8850a0-2b0c-4797-a783-07576b09be3b.jpg)  


## 1. grape_detection (Yolo & DeepSORT)

### Inference
```
python track.py --yolo_weights /grape-thinning/grape_detection/yolov5/run22_best_yolov5.pt --source /grape-thinning/grape_detection/grape_demo.mp4 --save-txt --save-crop
```

## 2. grape_boxinst (Boxinst)

### Inference

```
python demo/demo.py \
  --input  \
  --output ./viz/grape_from_real \
  --mask-path ../regression_data/masks \
  --opts MODEL.WEIGHTS ./training_dir/grape_pretrained.pth
```

input : Inference files (directory O, list of paths O, single path as string O) (***This path will be used in the next step***)
output : path to save visualized inference results
mask-path : path to save mask.pkl (***This path will be used in the next step***)

## 3. grape_feature_regression (Random Forest Regressor)

Extract features from masks and predict total number of grapes.
Output final `result.csv` with all the features

```
python rfr.py \
  --inference \
  --regressor_path ./regressor_model.pkl \
  --csv-path ./sample_result.csv \
  --image-path ../regression_data/images1 \
  --mask-path : ../regression_data/masks
```

- inference : inference mode
- train : train mode
- regressor_path : pretrained regressor path
- csv-path : result csv path
- image-path : image path (directory O, list of paths O, single path as string O) (***Same path as the previous step's input***)  
- mask-path : Inferenced Mask.pkl path (***Same path as the previous step's mask-path***)  
    

### Result CSV
- /grape-thinning/grape_feature_regression/sample_result.csv  

[Output features]
image,number of instances,sunburn_ratio,diameter,circularity,density,aspect ratio,grade(Grape Grade),average_hue,predict,Thinning(need for thinning)

## Pretrained
- grape detection : /grape-thinning/grape_detection/run22_best_yolov5.pt
- boxinst : /grape-thinning/grape_boxinst/training_dir/grape_pretrained.pth
- random forest regressor : /grape-thinning/grape_feature_regression/regressor_model.pkl
