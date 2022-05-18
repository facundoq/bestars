
* evaluate on liu
* Violin plots of features, also per class
* bar plots for spectral info
* box plot of features
* add "feature" that removes w features
* change train set percentage precision to 0.2f when printing to avoid 0.30000000000000001-like strings

* removed first 2 samples from liu because they had too many outlier values
* save detected be/em stars in test sets with ground truth not be/em
* Add values in text to bar plots 
* autoencoder for stars
  * Add as feature
* unsupervised star modeling (GMM)
  * use test set to check that it works
* Test single feature classifiers for EM/BE
* Fix and generalize class distribution experiments
* Add progress indicators to sklearn models
* Check that missing values in EM and BE indicate confirmed non EM and non BE stars


* Separar B de Be vs Separar todo de Be
* Repetir experimentos anteriores
* Separar Be tempranas de Be tardías

* Surveys s+ y j+ son nuevos, aplicar modelo con estos datos.
* Probar quitando el filtro U que es el más costoso

* Son todas estrellas calientes
* Automatizar la búsqueda de espectros de candidatas en LAMOST. Usar los que tienen LAMOST para testear y el resto para entrenar. También se puede buscar en el SLOAN los espectros.

* Para las tempranas, están el 50% aprox del tiempo en emisión -> verificar distribución en los espectros que sea también del 50%

* Ojo con los filtros del wise (w1 w2) si w1 es superior a 8 hacer un tratamiento especial? O tener cuidado porque pueden ser poco confiable