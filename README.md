# CDRE
This is the source code of CDRE model.

## Requirements
* Python (tested on 3.7.2)
* CUDA (tested on 11.3)
* [PyTorch](http://pytorch.org/) (tested on 1.11.0)
* [Transformers](https://github.com/huggingface/transformers) (tested on 4.18.0)
* numpy (tested on 1.21.6)
* spacy (tested on 2.3.7)
* [opt-einsum](https://github.com/dgasmith/opt_einsum) (tested on 3.3.0)
* tqdm
* sklearn
* scipy (tested on 1.5.2)

## Pretrained models
Download pretrained language model from [huggingface](https://huggingface.co/bert-base-uncased) and put it into the `./bert-base-uncased` directory. 

## Data

Put the `DocRED` and  `ReDocRED`  directories into the `./data` directory. 

## Run
### DocRED
Train the CDRE model on DocRED dataset under different memory sizes with the following command:

```bash
>> python main.py --task DocRED --memory_size 10  # memory size = 10
>> python main.py --task DocRED --memory_size 5  # memory size = 5
>> python main.py --task DocRED --memory_size 20  # memory size = 20
```

### ReDocRED
Train the CDRE model on ReDocRED dataset under different memory sizes  with the following command:
```bash
>> python main.py --task ReDocRED --memory_size 10  # memory size = 10
>> python main.py --task ReDocRED --memory_size 5  # memory size = 5
>> python main.py --task ReDocRED --memory_size 20  # memory size = 20
```

