Technologies for DL
*********************

## Data Base Management System
* mariadb/mysql
* Mongodb
* Clickhouse

## Text Wrangling
- Spacy & sus modelos
- lib propia: preprocess (depend on scipy, nltk & sklearn)
- Stanford library y su api de python (inside NLTK)

## Data Analysis

- Amazon SageMaker
- GCloud ML, Bigquery ML
- Sklearn
- spark.ml
- Scipy
- Pandas

## Deep Learning

* [Tensorflow](/media/DATA/PyData/myBooksData/04_ML/2017_Hands_on_ML/2017_Hands_On_ML_with_Sklearn_and_Tf.pdf)
* Keras High Level API
* __Model Visualization__: Tensorboard

## API

* [Chalice](https://aws.github.io/chalice/index)
* [Flask]()
* [tensorflow-serving](https://www.kdnuggets.com/2020/07/building-rest-api-tensorflow-serving-part-1.html)

API Consortium for n-dimensional arrays and dataframes
https://pycoders.com/link/4702/bnjp1v6tix

### Other Techs

| Optimization      | Scaling Services       | UI      |
| :---------------- | :--------------------: | ------: |
| cython            | AWS                    | VUE     |
| PyCuda            | Apache Spark           |         |
| Fortran           | Hadoop                 |         |
| Apache Spark      | GCloud                 |         |
| __ . __ . __ . __ | __ . __ . __ . __ . __ | __ . __ |

## Optimizations

* https://github.com/ray-project/tune-sklearn

    - Consistency with the scikit-learn API: You usually only need to change a couple lines of code to use Tune-sklearn (example).
    - Accessibility to modern hyperparameter tuning techniques: It is easy to change your code to utilize techniques like bayesian optimization, early stopping, and distributed execution
    - Framework support: There is not only support for scikit-learn models, but other scikit-learn wrappers such as Skorch (PyTorch), KerasClassifiers (Keras), and XGBoostClassifiers (XGBoost).
    - Scalability: The library leverages Ray Tune, a library for distributed hyperparameter tuning, to efficiently and transparently parallelize cross validation on multiple cores and even multiple machines.

* Parallelize or distribute your training with joblib and Ray

    - https://joblib.readthedocs.io/en/latest/
    - https://docs.ray.io/en/master/index.html
