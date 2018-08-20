# pygrid

Minimalistic multi-gpu / cpu grid search for pytorch.

requires:

* python >= 3.6
* pytorch

features:

* csv: params, status, results are contained in a csv file
* multi-processing: distributes isolated pytorch processes among gpus / cpus
* timestamps: isolates jobs in timestamped output directories
* logging: isolates logging facility per job
* single-file: self-contained within a single file for reproducibility

setup:
1. fill in TODOs with your training code and list of parameters
2. set list of gpu / cpu device ids
3. run
