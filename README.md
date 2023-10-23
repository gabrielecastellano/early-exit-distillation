# Early Classification with Network Distillation

This repository implements a pipeline for introducing a bottleneck to an existing Deep Neural Network so that it can be easily split  for 
 distributed deployment (inspired by the work in [Head Network Distillation](https://github.com/yoshitomo-matsubara/head-network-distillation)).
 Jointly, the pipeline trains a local smaller classifier that can perform early prediction on the bottleneck output. Bottleneck and early 
 classifier are trained jointly by means of a regularization process that leads to producing high-quality embeddings as the bottleneck output,
 hence helping the early classification task. Multiple early classifiers are supported: Logistic, KNN, K-means, and a novel 
 [Gaussian Mixture Layer](https://github.com/gabrielecastellano/pytorch-gaussian-mixture-layer) developed in PyTorch. The pipeline can be 
 further extended with additional classifiers implementing the desired modules under [src/early_classifier](src/early_classifier).

 
