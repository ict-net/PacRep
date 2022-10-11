# PacRep
Source code for KDD'22 paper: "[Packet Representation Learning for Traffic Classification](https://dl.acm.org/doi/abs/10.1145/3534678.3539085)".
# Requirements
- python: 3.8
- pytorch: 1.8.1
- numpy: 1.19.5
- scikit-learn: 0.24.2
- tensorboard: 2.6.0
- protobuf <= 3.20.0

# Usage
### Train the model
1. Download preprocessed data from [here](https://drive.google.com/drive/folders/1zpLiBsJQJJ02r0q6JBIruewxyJahbYY-?usp=sharing), and unzip it to ```./data/```. You can also use your own data with the same format, and change the data path by ```--data_dir```
2. Run the code
```
python3 run_train.py
```
3. Results can be found in ```./log/sample.log```
### Reproduce our results
1. Download the [trained model](https://drive.google.com/drive/folders/14Aycy7QJvKABQGVKmNoovBtfYy8IGke8?usp=sharing). Save the ```.bin``` in ```./saved_model/```, and save the ```.pth``` in ```./saved_model/sample/```
2. Run the code
```
python3 run_train.py --breakpoint 650
```
3. Results can be found in ```./log/sample.log```
# Others
Please cite our paper if you use this code in your own work:
```
@inproceedings{MengWMLLZ22,
  title     = {Packet Representation Learning for Traffic Classification},
  author    = {Xuying Meng and Yequan Wang and Runxin Ma and Haitong Luo and Xiang Li and Yujun Zhang},
  booktitle = {The 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages     = {3546--3554},
  year      = {2022}
}
```
