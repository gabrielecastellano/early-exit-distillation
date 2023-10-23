# Early Classification with Network Distillation

This repository implements a pipeline for introducing a bottleneck to an existing Deep Neural Network so that it can be easily split 
 for distributed deployment (inspired by the work in [Head Network Distillation](https://github.com/yoshitomo-matsubara/head-network-distillation)).
 Jointly, the pipeline trains a local smaller classifier that can perform early prediction on the bottleneck output. Multiple local 
 classifiers are supported: Logistic, KNN, K-means, and a novel Gaussian Mixture Layer developed in PyTorch. The pipeline can be further 
 extended with additional classifiers adding the corresponding module in [src/early_classifier](src/early_classifier) 

 
