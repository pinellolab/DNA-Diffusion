# Deopen
Deopen is a hybrid deep learning based framework to automatically learn the regulatory code of DNA sequences and predict chromatin accessibility.

# Requirements
- h5py
- hickle
- Scikit-learn=0.18.2
- Theano=0.8.0
- Lasagne=0.2.dev1
- nolearn=0.6.0

NOTE: You need to download the most recent version of Theano and Lasagne 0.2.dev1 manually by running: 

`pip install --upgrade https://github.com/Theano/Theano/archive/master.zip` 

`pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip`.  

# Installation
Download Deopen by
```shell
git clone https://github.com/kimmo1019/Deopen
```
Installation has been tested in a Linux/MacOS platform with Python2.7.

# Instructions

 

Preprocessing data for model training
```shell
python Gen_data.py <options> -pos <positive_bed_file> -neg <negative_bed_file> -out <outputfile>
```
```
Arguments:
  positive_bed_file: positive samples (bed format)
  e.g. chr1	9995	10995	
       chr3	564753	565753
       chr7	565935	566935
       
  negative_bed_file: negative samples (bed format)
  e.g. chr1	121471114	121472114	
       chr2	26268350	26269350
       chr5	100783702	100784702
  
  outputfile: preprocessed data for model training (hkl format)
 
Options:
  -l <int> length of sequence (default: 1000)
```
Run Deopen classification model
```shell
THEANO_FLAGS='device=gpu,floatX=float32' python Deopen_classification.py -in <inputfile> -out <outputfile>
```
```
 Arguments:  
  inputfile: preprocessed data for model training (hkl format)  
  outputfile: prediction outcome to be saved (hkl format)
```
 Run Deopen regression model
```shell
THEANO_FLAGS='device=gpu,floatX=float32' python Deopen_regression.py -in <inputfile> -reads <readsfile> -out <outputfile>
```
```
 Arguments:  
  inputfile: preprocessed file containing different features (hkl format)  
  readsfile: reads count for each sample (hkl format)  
  outputfile: trained model to be saved (hkl format)
```
# Citation
Liu, Qiao, et al. "Chromatin accessibility prediction via a hybrid deep convolutional neural network." Bioinformatics 34.5 (2017): 732-738.

```
@article{liu2017chromatin,
  title={Chromatin accessibility prediction via a hybrid deep convolutional neural network},
  author={Liu, Qiao and Xia, Fei and Yin, Qijin and Jiang, Rui},
  journal={Bioinformatics},
  volume={34},
  number={5},
  pages={732--738},
  year={2017},
  publisher={Oxford University Press}
}
```

# License
This project is licensed under the MIT License - see the LICENSE.md file for details
