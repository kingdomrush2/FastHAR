# GenHAR

## Dataset
We use UCI, MotionSense, Shoaib, and HHAR for evaluation. You can replace them with any dataset you prefer. 

## How to run
### (1) training phase
```shell
python main_GenHAR.py --mode train -dataset uci
```
For this command, we use uci dataset to train a model.

### (1) cross dataset test phase
```shell
python main_GenHAR.py --mode cross_dataset -dataset uci -target_dataset hhar
```
For this command, we use the model trained by uci dataset to test on hhar dataset.
