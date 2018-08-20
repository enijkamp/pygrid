# pygrid

Minimalistic multi-gpu / cpu grid search for pytorch.

requires:

* python >= 3.6
* pytorch

features:

* csv: params, status, results are contained in a simple csv file
* multi-processing: distributes isolated pytorch processes among gpus / cpus
* timestamps: isolates experiments in timestamped output directories
* logging: isolated logging facility per run
* single-file: self-contained within a single file for simple reproducibility

setup:
1. fill in TODOs with your training code
2. set list of gpu / cpu device ids
3. run
