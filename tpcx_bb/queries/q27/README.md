# Query 27

### Query  Details:
In this query, we find "competitor" company names in the product reviews for a given product.

The final output is  review id, product id, "competitor’s" company name and the related sentence from the online review . 


We have two implimentations for this query: 

#### 1. [HuggingFace Implimentation](tpcx_bb_query_hf_27.py) 


This implimentation uses [HuggingFace's](https://huggingface.co/) [distilbert-base-cased](https://huggingface.co/distilbert-base-cased) model fine tuned for ner token-classification on [conll-2003](https://www.clips.uantwerpen.be/conll2003/ner/). 

To fine tune the model please follow HuggingFace's example.  https://huggingface.co/transformers/v2.4.0/examples.html#named-entity-recognition .

**Commands Used:**
```CUDA_VISIBLE_DEVICES=0 python run_ner.py config-distilbert.json```

**config-distilbert.json**
```json
{
    "data_dir": "./data",
    "labels": "./data/labels.txt",
    "model_name_or_path": "distilbert-base-cased",
    "output_dir": "distilbert-base-cased",
    "max_seq_length": 128,
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
**Accuracy Stats**
```
eval_loss = 0.15505129464577716
eval_precision = 0.8869037294015611
eval_recall = 0.9059177888022679
eval_f1 = 0.8963099307564203
```


Advantages of this implimentation are:

- This uses the full context of reviews for ner prediction
- Avoids host->device->host round trips and is 2.6x times faster 

**Implimentation Details:**

This uses [subword_tokenize](https://docs.rapids.ai/api/cudf/nightly/api.html?highlight=subword_tokenize#cudf.core.column.string.StringMethods.subword_tokenize) on a cudf_series which gives you tokenized tensors

These tensors with appropiate padding are fed into the model for inference 

**RunTime:** 
This runs in `10.5 s` on sf-10k on 136 GPU's.


#### 2. [SPACY Implimentation](tpcx_bb_query_27.py)



This implimentation relies on SPACY's [entityrecognizer](https://spacy.io/api/entityrecognizer ) model. 

Spacy's `entityrecognizer` model requires inputs to be on host so requires a copy to host->device which slows this implimentation down.  


**RunTime:** 
This runs in `27.5 s` on sf-10k on 136 GPU's. 
