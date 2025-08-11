This is the code repository of my master's thesis project _Machine Learning for Regulation of Media
Platforms: A Case Study of Polarization Detection_.

This repository contains all code that has been used throughout the project. Legacy code can be found in the 'old_code' directory.

A snapshot of the model is included as a git submodule, linking to repository [1]. In order to include the snapshot, after cloning this repository you must run _git submodule init_ and _git submodule update_. This is enough to be able to run the basic demo functionalities of the model, which can be seen by running as main _polarization_model_1st_stage.py_ and _polarization_model_2nd_stage.py_ respectively. These two files contain the architectural code of the model, including the training process of the first stage (second stage is non-trainable), as well as evaluation routines.

If you wish to train, evaluate or otherwise make use of the dataset, you need to:
- Go to the original SemEval 2020 task 11 propaganda code repository [2] and retrieve the 'train', 'dev', 'test' directories. 
- Go to this project's data repository [3] and download its two directories.
- Put all of these folders in a 'data' directory at the root of the project.
- Disable the _exclude_data_ flag in the _load_objects_ function call in _polarization_model_1st_stage.py_ and _polarization_model_2nd_stage.py_ respectively.
- Continue with the desired operations.

We document the principal dependencies of this project:
- Transformers 4.45.1
- Torch 2.4.1+cu118
- Numpy 1.26.4
- Datasets 3.1.0
- Pandas 2.2.3
- Matplotlib 3.9.2
- Tqdm 4.66.5
- Spacy 3.8.2

We include all dependencies in 'requirements.txt'.

[1] https://huggingface.co/victor-nonea/roberta_for_polarizing_language-model_checkpoint

[2] https://propaganda.math.unipd.it/fine-grained-propaganda-emnlp.html navigate to the 'Code' button at the buttom of the page, from the downloaded archive navigate to 'data/protechn_corpus_eval'

[3] https://www.kaggle.com/datasets/victornonea/polarizing-language-dataset/data
