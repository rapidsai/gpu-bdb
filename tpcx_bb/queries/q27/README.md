# Query 27

## Query  Details:
In this query, we find "competitor" company names in the product reviews for a given product.

The final output is review id, product id, "competitor’s" company name and the related sentence from the online review. 

We have two implementations for this query: 

## 1. [HuggingFace Implementation](tpcx_bb_query_hf_27.py) 


This implementation uses [HuggingFace's](https://huggingface.co/) [token-classification](https://github.com/huggingface/transformers/tree/master/examples/token-classification) to do NER. We suggest choosing between the following models for optimal speed and accuracy. 


1. [distilbert-base-cased](https://huggingface.co/distilbert-base-cased) (2.5x Spacy Implementation, `89.6 F1` on conll-2003)
2. [base-base-ner](https://huggingface.co/dslim/bert-base-NER) (1.7x Spacy Implementation, `91.95 F1` on conll-2003)

### Setup:
#### 1. Distilbert-base-cased

To use `distil-bert-cased` model:

a.  Download it from  https://huggingface.co/distilbert-base-cased 

b. Fine tune the model to on [conll-2003](https://www.clips.uantwerpen.be/conll2003/ner/) by following HuggingFace's example at [link](https://huggingface.co/transformers/v2.4.0/examples.html#named-entity-recognition)

c. Place it on your shared directory `data_dir` +`../../q27_hf_model`

**Commands Used:**
```CUDA_VISIBLE_DEVICES=0 python run_ner.py config-distilbert.json```

**config-distilbert.json**
```json
{
    "data_dir": "./data",
    "labels": "./data/labels.txt",
    "model_name_or_path": "distilbert-base-cased",
    "output_dir": "distilbert-base-cased",
    "max_seq_length": 512,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 16,
    "save_steps": 878,
    "seed": 1,
    "do_train": true,
    "do_eval": true,
    "do_predict": true,
    "--fp16": true
}
```

#### 2. Bert-base-ner

a. Download it from https://huggingface.co/dslim/bert-base-NER
b. Place it on your shared directory `data_dir` +`../../q27_hf_model`


## 2. [spaCy Implementation](tpcx_bb_query_27.py)

This implementation relies on spaCy's [entityrecognizer](https://spacy.io/api/entityrecognizer ) model. 

Download the spaCy model via :
```
python -m spacy download en_core_web_sm
```
