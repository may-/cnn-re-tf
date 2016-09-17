# Attention-based Convolutional Neural Network for Relation Extraction

**Note:** This project is mostly based on https://github.com/yuhaozhang/sentence-convnet

---


## Requirements

- [Python 2.7](https://www.python.org/)
- [Tensorflow](https://www.tensorflow.org/) (tested with version 0.10.0rc0)
- [Numpy](http://www.numpy.org/)

To download wikipedia articles (`distant_supervision.py`)

- [Beautifulsoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Pandas](http://pandas.pydata.org/)

To visualize the result (`eval.py`)

- [Matplotlib](http://matplotlib.org/)



## Data
- `data` directory includes preprocessed data:
    ```
    cnn-re-tf
    ├── ...
    ├── word2vec
    └── data
        ├── er              # single-label single-instance dataset
        │   ├── source.txt  #   source sentences
        │   └── target.txt  #   target labels
        └── mlmi            # multi-label multi-instance dataset
            ├── source.att  #   attention
            ├── source.txt  #   source sentences
            └── target.txt  #   target labels
    ```    
    To reproduce: 
    ```
    python ./distant_supervision.py
    ```
    
- `word2vec` directory is empty. Please download the Google News pretrained vector data from 
[this Google Drive link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit), 
and unzip it to the directory. It will be a `.bin` file.



## Usage
### Preprocess

```sh
python ./util.py
```
It creates `vocab.txt`, `ids.txt` and `emb.npy` files.

### Training

- Single-label single-instance learning on the provided dataset:
    ```sh
    python ./train.py --sent_len=3 --vocab_size=11208 --num_classes=2 \
    --data_dir=./data/er --attention=False --multi_label=False --use_pretrain=False
    ```

- Multi-label multi-instance learning on the provided dataset:
    ```sh
    python ./train.py --sent_len=255 --vocab_size=36112 --num_classes=23 \
    --data_dir=./data/mlmi --attention=True --multi_label=True --use_pretrain=True
    ```
    
- Context-wise learning on the provided dataset:
    ```sh
    python ./train_context.py --sent_len=102 --vocab_size=36112 --num_classes=23 \
    --data_dir=./data/mlmi --attention=True --multi_label=True --use_pretrain=True
    ```

**Caution:** A wrong value for input-data-dependent options (`sent_len`, `vocab_size` and `num_class`) 
may cause an error. If you want to train the model on another dataset, please check these values.

### Evaluation

```sh
python ./eval.py --train_dir=./train/1473898241
```


### Run TensorBoard

```sh
tensorboard --logdir=./train/1473898241
```


## References

* https://github.com/yuhaozhang/sentence-convnet
* https://github.com/dennybritz/cnn-text-classification-tf
* http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
* http://tkengo.github.io/blog/2016/03/14/text-classification-by-cnn/

