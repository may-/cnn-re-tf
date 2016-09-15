# Convolutional Neural Network for Relation Extraction

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
        ├── clean.att   # attention
        ├── clean.label # label (class names)
        └── clean.txt   # raw sentences
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
It creates `vocab.txt` and `ids.txt` files in `data` directory.

### Training

For multi-label multi-instance learning on provided dataset:
```sh
python ./train.py --train_dir=./train --data_dir=./data \
--sent_len=371 --vocab_size=36393 --num_classes=23 \
--attention=True --multi_label=True --use_pretrain=True
```
If you want to train the model on other dataset, please modify the path variables in `main()` func 
and specify the option values(`sent_len`, `vocab_size`, `num_class`) properly.

### Evaluation

```sh
python ./eval.py --train_dir=./train/1473898241
```
It creates a png image file of Precision-Recall curve in the `train_dir`.

### Run TensorBoard

```sh
tensorboard --logdir=./train/1473898241
```


## References

Implementation:

* https://github.com/yuhaozhang/sentence-convnet
* https://github.com/dennybritz/cnn-text-classification-tf
* http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
* http://tkengo.github.io/blog/2016/03/14/text-classification-by-cnn/

Theory:

* 
* 
* 

