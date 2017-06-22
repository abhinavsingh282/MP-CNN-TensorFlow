# MP-CNN-TF
A TensorfFlow implementation of [Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks](http://aclweb.org/anthology/D/D15/D15-1181.pdf)


## Requirements

- Python 2.7
- Tensorflow > 1.0
- Numpy


Optional Arguments:

--dev_sample_percentage        :    Percentage of the training data to use for validation<br>
--training_datafilename        :    training data file containing sentence pairs and similarity score in one line<br>
--embedding_dim                :    Dimensionality of character embedding (default: 128)<br>
--filter_sizes                 :    Comma-separated filter sizes (default: '1,2,3')<br>
--num_filters                  :    Number of filters per filter size (default: 128)<br>
--batch_size                   :    Batch Size <br>
--num_epochs                   :    Number of training epochs <br>
--evaluate_every               :    Evaluate model on dev set after this many steps <br>
--checkpoint_every             :    Save model after this many steps <br>
--num_checkpoints              :    Number of checkpoints to store <br>
--allow_soft_placement         :    Allow device soft device placement<br>
--log_device_placement         :    Log placement of ops on devices<br>
--debug                        :    Run with tf debugger<br>





## Train:

```
python train.py
```
## Evaluating

```bash
python eval.py --checkpoint_dir="/path/to/checkpoints/"  --eval=True
```
Optional Arguments for eval.py:

test_datafilename            :    test data file containing sentence pairs and similarity score in one line



## Pending features:
1) Initialise word embeddings with word2vec 


# License
The MIT License

Copyright (c) 2017 Abhinav Singh
