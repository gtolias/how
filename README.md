# HOW local descriptors

This is the official Python/PyTorch implementation of the HOW local descriptors from our [ECCV 2020 paper](https://arxiv.org/abs/2007.13172):

```
@InProceedings{TJ20,
  author      = "Giorgos Tolias and Tomas Jenicek and Ond\v{r}ej Chum}",
  title       = "Learning and aggregating deep local descriptors for instance-level recognition",
  booktitle   = "European Conference on Computer Vision",
  year        = "2020"
}
```


## Running the Code

1. Install the cirtorch package (see [cirtorch github](https://github.com/filipradenovic/cnnimageretrieval-pytorch/) for details)

```
# cirtorch
wget "https://github.com/filipradenovic/cnnimageretrieval-pytorch/archive/v1.2.zip"
unzip v1.2.zip
rm v1.2.zip
export PYTHONPATH=${PYTHONPATH}:$(realpath cnnimageretrieval-pytorch-1.2)
```

2. Install the asmk package with dependencies (see [asmk github](https://github.com/jenicek/asmk#running-the-code) for details)

```
# asmk
git clone https://github.com/jenicek/asmk.git
pip3 install pyaml numpy faiss-gpu
cd asmk
python3 setup.py build_ext --inplace
rm -r build
cd ..
export PYTHONPATH=${PYTHONPATH}:$(realpath asmk)
```

3. Install pip3 requirements

```
pip3 install -r requirements.txt
```

4. Run `examples/demo_how.py` with two arguments &ndash; mode (`train` or `eval`) and any `.yaml` parameter file from `examples/params/*/*.yml`


### Evaluating ECCV 2020 HOW models

Reproducing results from **Table 2.** with the publicly available models

- R18<sub>how</sub> (n = 1000): &nbsp; `examples/demo_how.py eval examples/params/eccv20/eval_how_r18_1000.yml -e official_how_r18_1000` &ensp; _ROxf (M): 75.1, RPar (M): 79.4_
- -R50<sub>how</sub> (n = 1000): &nbsp; `examples/demo_how.py eval examples/params/eccv20/eval_how_r50-_1000.yml -e official_how_r50-_1000` &ensp; _ROxf (M): 78.3, RPar (M): 80.1_
- -R50<sub>how</sub> (n = 2000): &nbsp; `examples/demo_how.py eval examples/params/eccv20/eval_how_r50-_2000.yml -e official_how_r50-_2000` &ensp; _ROxf (M): 79.4, RPar (M): 81.6_


### Training HOW models

- R18<sub>how</sub>:
    - train: `examples/demo_how.py train examples/params/eccv20/train_how_r18.yml -e train_how_r18`
    - eval (n = 1000): `examples/demo_how.py eval examples/params/eccv20/train_how_r18_1000.yml -ml train_how_r18`
- -R50<sub>how</sub>:
    - train: `examples/demo_how.py train examples/params/eccv20/eval_how_r50-.yml -e train_how_r50-`
    - eval (n = 1000): `examples/demo_how.py eval examples/params/eccv20/eval_how_r50-_1000.yml -ml train_how_r50-`
    - eval (n = 2000): `examples/demo_how.py eval examples/params/eccv20/eval_how_r50-_2000.yml -ml train_how_r50-`

Dataset shuffling during the training is done according to the cirtorch package; randomness in the results is caused by cudnn and by kmeans for codebook creation during evaluation.
