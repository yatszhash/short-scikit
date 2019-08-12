
# short-scikit
[![CircleCI](https://circleci.com/gh/yatszhash/short-scikit.svg?style=svg&circle-token=df34123a17a59d41904fdceaeb4a876c5b076594)](https://circleci.com/gh/yatszhash/short-scikit)
[![codecov](https://codecov.io/gh/yatszhash/short-scikit/branch/master/graph/badge.svg)](https://codecov.io/gh/yatszhash/short-scikit)

This project is aimed to be provide **"short-circuit" to pre-processing and feature extraction with scikit-learn**
by reducing to write boiler codes.  

## Caution!

This project is under construction and the interfaces and the modules are subject to the dynamic changes.

## features

The checked features have been implemented.

- [x] save and load for intermediate data
- [x] generator support
- [x] dataframe support
- [ ] dask support
- [x] batch adaptation of transformations
- [ ] branchable pipeline
- [ ] delayed evaluation
- [x] prepared transformers and estimators (inherit sklearn's base classes) for frequent used pre-processing and feature
      extractions

##  How to develop

Please see [here](development_guid.md)