* removed first 2 samples from liu because they had too many outlier values
* save detected be/em stars in test sets with ground truth not be/em
* autoencoder for stars
  * Add as feature
* unsupervised star modeling (GMM)
  * use test set to check that it works
* Test single feature classifiers for EM/BE
* Run training test size exp
* Fix and generalize class distribution experiments
* Add Experiment name parser 
* Add progress indicators to sklearn models
* Check that missing values in EM and BE indicate confirmed non EM and non BE stars
* 