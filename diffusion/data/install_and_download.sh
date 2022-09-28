#!/bin/bash

genomepy install hg38 --annotation

curl -L -o train_all_classifier_WM20220916.csv.gz "https://www.dropbox.com/s/db6up7c0d4jwdp4/train_all_classifier_WM20220916.csv.gz?dl=2"
gunzip train_all_classifier_WM20220916.csv.gz