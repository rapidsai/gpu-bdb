import cupy as cp
import numpy as np
import torch
import logging
import cudf
import time
from torch.utils.dlpack import from_dlpack


def get_stride(seq_len):
    max_len = seq_len-2
    stride = int(max_len*0.5)
    return stride

## ----Tokenization Utils----
def append_seq_len(df,sequence_len_ls,vocab_hash_file):
    """
        Appends column sequence length to the dataframe
    """
    df['input_id'] = cp.arange(0,len(df),dtype=np.int32)
    ### here stride is set to ensure non repeated rows as we want to gather sequence_length
    ### -2 because of padding of special chars
    d = tokenize_text_series(df['pr_review_content'],
                             max(sequence_len_ls),
                             stride=max(sequence_len_ls)-2,
                             vocab_hash_file = vocab_hash_file)
    
    metadata_df = get_token_metadata_df(d['metadata'],sequence_len_ls)
    metadata_df = df[['pr_review_sk','pr_item_sk','input_id']].merge(metadata_df)
    metadata_df = metadata_df.groupby('pr_review_sk').sequence_len.max()
    metadata_df = metadata_df.reset_index(drop=False)
    df = df.merge(metadata_df)
    
    output_columns = ['pr_review_sk','pr_review_content','pr_item_sk','sequence_len']
    return df[output_columns]

def get_token_metadata_df(metadata,sequence_len_ls):
    """
        Adds the sequence_length from the sequence_len_ls to be used 
        for inference
        
        Args:
            metadata: nx3 cupy array(input_id,start_id,stop_id)
            
            sequence_len_ls: list of int sequence_lengths we can have
                (eg:[128,256,512])
        Returns:
            a Cudf Dataframe ([input_id,start_id,stop_id])
    """
    sequence_len_ls = sorted(sequence_len_ls)
    metadata_df = cudf.DataFrame()
    metadata_df['input_id'] = metadata[:,0]
    metadata_df['start_id'] = metadata[:,1]
    metadata_df['stop_id'] = metadata[:,2]
    
    metadata_df['sequence_len']=max(sequence_len_ls)
    for output_size in sorted(sequence_len_ls,reverse=True):
        output_flag = metadata_df['stop_id']<output_size
        metadata_df['sequence_len'][output_flag] = output_size
    
    return metadata_df

def get_df_partitioned_by_seq(df,sequence_len_ls):
    """
        We get dataframe partitioned by sequences
    """
    sq_part_d = {}
    for s_len  in sequence_len_ls:
        subset_df = df[df['sequence_len']==s_len].reset_index(drop=True)
        sq_part_d[s_len]=subset_df
    
    return sq_part_d

def tokenize_text_series(text_ser,seq_len,stride,vocab_hash_file):
    """
        This function tokenizes a text series using the bert subword_tokenizer and vocab-hash
    """
    if len(text_ser)==0:
        return {'token_tensor':None,'masks':None,'metadata':None}
    
    max_num_chars = text_ser.str.len().sum()+1
    max_rows_tensor = len(text_ser)*2
    max_length = seq_len-2

    tokens,attention_masks,metadata = text_ser.str.subword_tokenize(vocab_hash_file,
                                            do_lower=False,
                                            max_num_strings = max_rows_tensor,
                                            max_rows_tensor = max_rows_tensor,
                                            max_num_chars = max_num_chars,              
                                            stride = stride,
                                            max_length =max_length,
                                            do_truncate = False,
                                             )
  
    tokens = tokens.reshape(-1,max_length)
    output_rows = tokens.shape[0]
    padded_tokens = cp.zeros(shape = (output_rows,seq_len),dtype=np.uint32)
    padded_tokens[:,1:-1] = tokens
    # Pad first token with [CLS] token to mark start of sequence
    padded_tokens[:,0]=101
    # Pad last token with [SEP] token to mark end of sequence
    padded_tokens[:,-1]=102
   
    del tokens
    attention_masks = attention_masks.reshape(-1,max_length)
    padded_attention_mask = cp.zeros(shape = (output_rows,seq_len),dtype=np.uint32)
    padded_attention_mask[:,1:-1] = attention_masks
    padded_attention_mask[:,0]=1
    padded_attention_mask[:,-1]=1
    del attention_masks
    
    return {'token_ar':padded_tokens,'attention_ar':padded_attention_mask,'metadata':metadata.reshape(-1,3)}


## ----Inference Utils----
def run_inference_on_tensor(model,token_ar,attention_ar,batchsize):
    import time    
    if token_ar is None:
        return []
    
    prediction_ls = []

    batch_st = 0
    total_batches = token_ar.shape[0]//batchsize+1
    with torch.no_grad():
        token_tensor = from_dlpack(token_ar.astype(np.int32).toDlpack()).long()
        attention_tensor = from_dlpack(attention_ar.astype(np.int32).toDlpack()).long()
        for batch_index in range(0,total_batches):
            batch_st = batch_index*batchsize
            batch_end = min(batch_st + batchsize,token_tensor.shape[0])
            
            if batch_end==batch_st:
                break

            current_batch_tensor = token_tensor[batch_st:batch_end]
            current_batch_attention = attention_tensor[batch_st:batch_end]
            
            outputs = model(current_batch_tensor,current_batch_attention)
            prediction_ls.append(outputs[0])

                
    return torch.cat(prediction_ls).argmax(dim=2)


def run_inference_on_df(df,model,vocab_hash_file, batchsize=64, sequence_len_th_ls=[512,256,128,64,32,16,8]):
    """
         The function has following steps:
           a. Distribute df based on sequence_length (we do this because inference time is prop to sequence length)
           b. For each part we run tokenization
           c. For each part we run inference using the passed model

        #TODO: add params/return 

        Returns a token_d,prediction_d  with key=seq_len
    """
    ## Get max sequence_length for a particular sequence
    df = append_seq_len(df,sequence_len_th_ls,vocab_hash_file)

    ### Tokenize dataframe
    token_d = {}
    max_seq_len = max(sequence_len_th_ls)
    ### Partition each df by sequence length
    for sequence_len,sub_df in get_df_partitioned_by_seq(df,sequence_len_th_ls).items():
        if sequence_len==max_seq_len:
            stride = get_stride(max_seq_len)
        else:
            # -2 for padding
            stride = sequence_len-2
        token_d[sequence_len] = tokenize_text_series(sub_df['pr_review_content'],
                                                    sequence_len,
                                                    stride,
                                                    vocab_hash_file=vocab_hash_file)
        token_d[sequence_len]['df']=sub_df

    del df

    ## Run Inference
    prediction_d = {}
    for seqlen,batch_d in token_d.items():
        prediction_d[seqlen]= run_inference_on_tensor(model,batch_d['token_ar'],batch_d['attention_ar'],batchsize)
    return token_d,prediction_d


