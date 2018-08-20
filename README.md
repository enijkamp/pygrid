# pygrid

Minimalistic multi-gpu / cpu grid search for pytorch.

requires:

* pytorch
* python >= 3.6

features:

* csv: params, status, results are contained in a simple csv file
* multi-processing: distributes isolated pytorch processes among gpus / cpus
* timestamps: isolates experiments in timestamped output directories
* logging: isolated logging facility per run
* single-file: self-contained within a single file for simple reproducibility
