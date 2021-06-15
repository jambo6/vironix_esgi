# Vironx - Vermont ESGI 2021

This project contains code for the Vironix Vermont ESGI study group.

To install, setup a python virtualenv and run
```angular2html
pip install -e .
```

This code relies on data taken from `https://github.com/yaleemmlc/admissionprediction`. Steps to get this are as follows

1. Download the `5v_cleandf.RData` file from `https://github.com/yaleemmlc/admissionprediction/tree/master/Results`.
2. Convert the data (using R) to a csv. 
3. Call the csv file 'admissionprediction.csv' and copy it into `/data/raw`.

Once this is done, you can run `python get_data/admission_prediction.py` which will perform some processing of the data and setup some splits that can be worked from. 

An example of some simple models comparing the performance of models trained with generated and real data to predict influenza can be found at `experiments/model_example.py`.