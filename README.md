# grape_rfr
random forest regressor to calculate total number of berries in a 2D RGB image

## Usage

```python
python rfr.py --inference --regressor-path '학습시킨 RFR모델경로' --csv-path '결과출력할 csv path', --image-path 'inference 할 이미지가 들어있는 폴더' --mask-path 'boxinst에서 demo.py돌려서 얻은 mask.pkl들 저장되어 있는 폴더 경로'
```