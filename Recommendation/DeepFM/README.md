# DeepFM using tensorflow-keras API

This is a fork from **ChenglongChen**'s implementation of DeepFM[here](!https://github.com/ChenglongChen/tensorflow-DeepFM). The original implementation is in `DeepFM.py` and the corresponding tensorflow-keras implementation is in `DeepFM_tf2.py` file.

# Usage
## Input Format
This implementation requires the input data in the following format:
- [ ] **Xi**: *[[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]*
    - *indi_j* is the feature index of feature field *j* of sample *i* in the dataset
- [ ] **Xv**: *[[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]*
    - *vali_j* is the feature value of feature field *j* of sample *i* in the dataset
    - *vali_j* can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
- [ ] **y**: target of each sample in the dataset (1/0 for classification, numeric number for regression)

Please see `example/DataReader.py` an example how to prepare the data in required format for DeepFM.


# Example
Folder `example` includes an example usage of DeepFM/FM/DNN models for [Porto Seguro's Safe Driver Prediction competition on Kaggle](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction).

Please download the data from the competition website and put them into the `example/data` folder.

To train DeepFM model for this dataset, run

```
$ cd example
$ python main.py    # run with the original implementation
$ python main_tf2.py   # run with keras implementation
```
Please see `example/DataReader.py` how to parse the raw dataset into the required format for DeepFM.


# License
MIT