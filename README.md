# DMFLDA
DMFLDA is a deep learning framework for predicting lncRNA–disease associations.

# Requirements

tensorflow==1.3.0

numpy==1.11.2

scikit-learn==0.18

scipy==0.18.1

# Usage

  In this GitHub project, we give a demo to show how DMFLDA works. In data_processing folder, we give three datasets we used in our study.
  
  1. interMatrix.mat is the raw lncRNA-disease interaction matrix with matlab format. Its shape is 577 lncRNAs x 272 diseases.
  
  2. matrix.npy is the lncRNA-disease interaction matrix with numpy format.
  
  3. data.pkl is used to store the sampled positive and negative samples.
  
  You can use these python files which provided by us in data_processing folder to obtain them.
  
  In our demo, we provide a leave-one-out cross validation to evaluate our model. You can use cross_validation.py to see experimental results and predict lncRNA related diseases. If you want to tune some hyper-parameters, you can change some values of hyper-parameters in hyperparams.py. 

  The other details can see the paper and the codes.
  

# Citation

# License
This project is licensed under the MIT License - see the LICENSE.txt file for details
