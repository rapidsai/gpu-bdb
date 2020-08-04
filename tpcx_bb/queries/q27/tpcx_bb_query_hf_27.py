#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import time
import rmm
import cupy as cp
import numpy as np
import distributed
from numba import cuda

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    left_semi_join,
    run_query,
)

from xbb_tools.text import create_sentences_from_reviews, create_words_from_sentences
from xbb_tools.readers import build_reader
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, wait

#Pytorch/HF imports
from torch.utils.dlpack import from_dlpack
from transformers import AutoModelForTokenClassification
import torch



# -------- Q27 -----------
q27_pr_item_sk = 10002
EOL_CHAR = "."


def get_stride(seq_len):
    max_len = seq_len-2
    stride = int(max_len*0.5)
    return stride



def read_tables(config):
    ### splitting by row groups for better parallelism
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=True,
    )
    product_reviews_cols = ["pr_item_sk", "pr_review_content", "pr_review_sk"]
    product_reviews_df = table_reader.read(
        "product_reviews", relevant_cols=product_reviews_cols
    )
    return product_reviews_df

## ----Tokenization Utils----

def append_seq_len(df,sequence_len_ls):
    """
        Appends column sequence length to the dataframe
    """
    df['input_id'] = cp.arange(0,len(df),dtype=np.int32)
    ### here stride is set to ensure non repeated rows as we want to gather sequence_length
    ### -2 because of padding of special chars
    d = tokenize_text_series(df['pr_review_content'],max(sequence_len_ls),
                             stride=max(sequence_len_ls)-2)
    
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
    for sqlen  in sequence_len_ls:
        subset_df = df[df['sequence_len']==sqlen].reset_index(drop=True)
        sq_part_d[sqlen]=subset_df
    
    return sq_part_d

def tokenize_text_series(text_ser,seq_len,stride=None,vocab_file='/raid/vjawa/torch_ner_q27/tpcx-bb-2/tpcx_bb/queries/q27/vocab-hash.txt'):
    """
        This function tokenizes a text series using the bert subword_tokenizer and vo
    """
    if len(text_ser)==0:
        return {'token_tensor':None,'masks':None,'metadata':None}
    
    max_num_chars = text_ser.str.len().sum()+1
    max_rows_tensor = len(text_ser)*2
    max_num_strings = max_rows_tensor
    max_length = seq_len-2
    if stride==None:
        stride = get_stride(seq_len)

    tokens,attention_masks,metadata = text_ser.str.subword_tokenize(vocab_file,
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

    token_tensor = from_dlpack(padded_tokens.astype(np.int32).toDlpack()).long()
    attention_masks_tensor = from_dlpack(padded_attention_mask.astype(np.int32).toDlpack()).long()
    
    return {'token_tensor':token_tensor,'masks':attention_masks_tensor,'metadata':metadata.reshape(-1,3)}


## ----Inference Utils----
def run_inference_on_tensor(model,token_tensor,attention_tensor,batchsize):
    import time
    prediction_ls = []
    
    if token_tensor is None:
        return prediction_ls

    batch_st = 0
    total_batches = token_tensor.shape[0]//batchsize+1
    with torch.no_grad():   
        for batch_index in range(0,total_batches):
            batch_st = batch_index*batchsize
            batch_end = min(batch_st + batchsize,token_tensor.shape[0])
            batch_st_time = time.time()
            
            if batch_end==batch_st:
                break

            # for batch_end in range(batchsize
            batch_f_st = time.time()
            current_batch_tensor = token_tensor[batch_st:batch_end]
            current_batch_attention = attention_tensor[batch_st:batch_end]
            batch_f_et = time.time()
            
            #print('batch fetch took = {0:.2f} s'.format(batch_f_et-batch_f_st))

            outputs = model(current_batch_tensor,current_batch_attention)
            prediction_ls.append(outputs[0])

            if batch_index%20==0:
                 print(f"batch_index = {batch_index}/{total_batches} took = {time.time()-batch_st_time}")
                
    return torch.cat(prediction_ls).argmax(dim=2)


def run_inference_on_df(df,model,batchsize=64, sequence_len_th_ls=[512,256,128,64,32,16,8]):
    """
        Runs inference on a single dataframe

        Returns a token_d,prediction_d  with key=seq_len
    """
    ## Get Sequence length
    df = append_seq_len(df,sequence_len_th_ls)

    ### Tokenize dataframe
    token_d = {}
    for sequence_len,sub_df in get_df_partitioned_by_seq(df,sequence_len_th_ls).items():
        token_d[sequence_len] = tokenize_text_series(sub_df['pr_review_content'],sequence_len)
        token_d[sequence_len]['df']=sub_df

    del df

    ## Run Inference
    total_st = time.time()
    prediction_d = {}
    for seqlen,batch_d in token_d.items():
        print(f"Started batch of sequence = {seqlen}")
        st = time.time()
        prediction_d[seqlen]= run_inference_on_tensor(model,batch_d['token_tensor'],batch_d['masks'],batchsize)
        et = time.time()
        print(f"Inference for sequence of len {seqlen} took {et-st}")
    total_et = time.time()
    print("Total time taken = {}".format(total_et-total_st))

    return token_d,prediction_d

#### Gathering Sentence Utils:

def get_sentence_boundaries(metadata_df,token_ar,stride,fs_index_ls): 
    """
        Given token array and meta-data we create sentence boundaries 
        We consider a sentence boundary as one which is at eol-chars (1012,29625) or start/end of a review
    """
    seq_len = token_ar.shape[1]
    
    fullstop_flag = None
    for fs_token_idx in fs_index_ls:
        if fullstop_flag is None:
            fullstop_flag = token_ar==fs_token_idx
        else:
            fullstop_flag = (fullstop_flag) | (token_ar==fs_token_idx)
        
    fullstop_row,fullstop_col =  torch.nonzero(fullstop_flag,as_tuple=True)    
    
    min_row_df = metadata_df.groupby('input_text_index').seq_row.min().reset_index(drop=False)
    min_row_df.rename(columns = {'seq_row':'min_row'},inplace=True)
    max_row_df = metadata_df.groupby('input_text_index').seq_row.max().reset_index(drop=False)
    max_row_df.rename(columns = {'seq_row':'max_row'},inplace=True)

    metadata_df = metadata_df.merge(min_row_df).merge(max_row_df)


    ### Can filter to only sequences that have the org if below becomes a bottleneck 

    fullstop_df = cudf.DataFrame()
    fullstop_df['seq_row'] = cudf.Series(fullstop_row)
    fullstop_df['fs_seq_col'] = cudf.Series(fullstop_col)
    fullstop_df = fullstop_df.merge(metadata_df)
    
    fullstop_df.rename(columns={'seq_row':'fs_seq_row'},inplace=True)
    
    first_row_df = cudf.DataFrame()
    first_row_df['input_text_index']=min_row_df['input_text_index']
    first_row_df['fs_seq_row']=min_row_df['min_row']
    first_row_df['fs_seq_col']=1
    first_row_df['min_row']=min_row_df['min_row']
    first_row_df = first_row_df.merge(max_row_df[['input_text_index','max_row']])
    
    
    last_row_df = cudf.DataFrame()
    last_row_df['input_text_index']=max_row_df['input_text_index']
    last_row_df['fs_seq_row']=max_row_df['max_row']
    last_row_df['fs_seq_col']=seq_len-1
    last_row_df['max_row']=max_row_df['max_row']
    last_row_df = last_row_df.merge(min_row_df[['input_text_index','min_row']])
    
    
    fullstop_df = cudf.concat([fullstop_df,first_row_df,last_row_df])
    
    ## -2-> for padding
    valid_region = (seq_len-2) - stride + 1
    
    ### only keep sentences in the valid_region
    valid_flag = fullstop_df['fs_seq_col']<valid_region
    valid_flag = valid_flag | (fullstop_df['fs_seq_row']==fullstop_df['max_row'])
    fullstop_df = fullstop_df[valid_flag]
    
    fullstop_df['flat_loc_fs']=fullstop_df['fs_seq_row']*seq_len+fullstop_df['fs_seq_col']
    
    
    return fullstop_df[['input_text_index','fs_seq_row','fs_seq_col','flat_loc_fs']]


def get_org_sentences(sentence_boundary_df,org_df):
    """
        We return sentences that contain org,
        Given the sentence_boundary_df and org_df
    """
    merged_df = sentence_boundary_df.merge(org_df,on='input_text_index')
    merged_df['left_loc']=merged_df['flat_loc_org']-merged_df['flat_loc_fs']
    merged_df['right_loc']=merged_df['flat_loc_fs']-merged_df['flat_loc_org']
    
    ### Better way to get the closeset row/col maybe 
    valid_right_loc = merged_df[merged_df['right_loc']>0].sort_values(by=['flat_loc_org','right_loc']).groupby('flat_loc_org').nth(0)
    valid_left_loc = merged_df[merged_df['left_loc']>0].sort_values(by=['flat_loc_org','left_loc']).groupby('flat_loc_org').nth(0)
    
    cols_2_keep = ['input_text_index','fs_seq_row','fs_seq_col','org_seq_row','org_seq_col']
    valid_left_loc = valid_left_loc[cols_2_keep]
    valid_left_loc.rename(columns = {'fs_seq_row':'l_fs_seq_row', 
                           'fs_seq_col':'l_fs_seq_col'},inplace=True)
    
    
    valid_right_loc.rename(columns = {'fs_seq_row':'r_fs_seq_row', 
                           'fs_seq_col':'r_fs_seq_col'},inplace=True)
    
    valid_right_loc = valid_right_loc[['r_fs_seq_row','r_fs_seq_col']]
    
    valid_left_loc['r_fs_seq_row']=valid_right_loc['r_fs_seq_row']
    valid_left_loc['r_fs_seq_col']=valid_right_loc['r_fs_seq_col']
    
    
    return valid_left_loc


@cuda.jit
def get_output_sen_word_kernel(start_r_ar,start_c_ar,
                       end_r_ar,end_c_ar,
                       t_row_ar,t_col_ar,
                       valid_region,
                       mat,
                       output_mat,
                       label_ar,
                    ):
    
    """
        Returns the output sentences given start,end
    """
    
    #token_ls = []
    rnum = cuda.grid(1)
    
    if rnum < (start_r_ar.size):  # boundary guard
        start_r,start_c = start_r_ar[rnum], start_c_ar[rnum]
        end_r,end_c =  end_r_ar[rnum],  end_c_ar[rnum]
        t_row,t_col = t_row_ar[rnum], t_col_ar[rnum]

        i = 0
        for curr_r in range(start_r,end_r+1):
            if curr_r==start_r:
                col_loop_s = start_c
            else:
                col_loop_s = 1

            if curr_r == end_r:
                col_loop_e = end_c
            else:
                col_loop_e = valid_region

            for curr_c in range(col_loop_s,col_loop_e): 
                token = mat[curr_r][curr_c]
                if token!=0:
                    output_mat[rnum][i]=token
                    i+=1
                    if (curr_r == t_row) and (curr_c == t_col):
                        label_ar[rnum] = i

        
    return 


### CPU part of workflow
def convert_to_sentence(row,target_index,id2vocab):
    """
        Given a row of token_ids , we convert to a sentence
        We also combine subtokens back to get back the input sentence
    """
    row = row[row!=0]
    output_ls = []
    tr_index = -1
    row = id2vocab[row]
    for t_num,token in enumerate(row):
        if t_num==target_index:
            tr_index = len(output_ls)-1
        
        if token.startswith("##"):
            output_ls[-1]+=token[2:]
            
        else:
            output_ls.append(token)
        
        
    if output_ls[0]=='.':
        output_sen = ' '.join(output_ls[1:])
    else:
        output_sen = ' '.join(output_ls)

    return output_sen,output_ls[tr_index]


def get_review_sentence(tokenized_d,predicted_label_t,vocab2id,id2vocab,org_labels = [2,5]):
    """
        Given a tokenized_d and predicted_label_t
        we gather the  setenceces that contain labels for either 5 or 3 
    """
    seq_len = tokenized_d['token_tensor'].shape[1]
    stride = get_stride(seq_len)
    
    metadata_df = cudf.DataFrame()
    metadata_df['input_text_index'] = tokenized_d['metadata'][:,0]
    metadata_df['seq_row'] = cp.arange(len(metadata_df))
    
    pr_label_f = None
    for label in org_labels:
        if pr_label_f is None:
            pr_label_f = predicted_label_t==label
        else:
            pr_label_f = pr_label_f | (predicted_label_t==label)
    
    org_r,org_c  = torch.nonzero(pr_label_f, as_tuple=True)
    
    org_df = cudf.DataFrame()
    org_df['seq_row'] = cudf.Series(org_r)
    org_df['org_seq_col'] = cudf.Series(org_c)
    org_df = org_df.merge(metadata_df)
    org_df = org_df.rename(columns={"seq_row":'org_seq_row'})
    
    org_df['flat_loc_org']=org_df['org_seq_row']*seq_len+org_df['org_seq_col']
    

    ### Because we have repeations in our boundaries we 
    ### We create a valid region boundary to prevent copying
    valid_region = (seq_len-2) - stride + 1
    ### This gives us all the valid sentence boundaries
    sentence_boundary_df = get_sentence_boundaries(metadata_df,
                                                 tokenized_d['token_tensor'],
                                                 stride=stride,
                                                 fs_index_ls = [vocab2id['.'],vocab2id['##.']]
                                                )
    
    
    ### This df contains the sentences that intersect with org
    org_senten_df = get_org_sentences(sentence_boundary_df,org_df)
    org_senten_df = org_senten_df.reset_index(drop=False)
    
    
    input_mat = cp.array(tokenized_d['token_tensor'])
    
    l_r_ar = org_senten_df['l_fs_seq_row']
    l_c_ar = org_senten_df['l_fs_seq_col']
    
    r_r_ar = org_senten_df['r_fs_seq_row']
    r_c_ar = org_senten_df['r_fs_seq_col']
    
    o_r_ar = org_senten_df['org_seq_row']
    o_c_ar = org_senten_df['org_seq_col']
    
    output_mat = cp.zeros(shape=(len(o_c_ar),1024*2),dtype=np.int32)
    label_ar = cp.zeros(shape=(len(o_c_ar),1),dtype=np.int32)
        
    get_output_sen_word_kernel.forall(len(l_r_ar))(l_r_ar,l_c_ar,
                      r_r_ar,r_c_ar,
                      o_r_ar,o_c_ar,
                      valid_region,
                      input_mat,
                      output_mat,
                      label_ar,
                      )
    

    output_mat = cp.asnumpy(output_mat)
    label_ar = cp.asnumpy(label_ar).flatten()
    
    ### CPU logic to gather sentences begins here
    sen_ls = []
    target_ls = []
    st = time.time()
    for row,tnum in zip(output_mat,label_ar):
        s,t = convert_to_sentence(row,tnum,id2vocab)
        sen_ls.append(s)
        target_ls.append(t)
    et = time.time()
    print(f"Creating sentence took = {et-st} for sq = {seq_len}")


    df = cudf.DataFrame()
    df['sentence'] = sen_ls
    df['company'] = target_ls
    df['input_text_index']= org_senten_df['input_text_index']
    return df

def create_vocab_table(vocabpath):
    """
        Create Vocabulary tables 
    """
    id2vocab = []
    vocab2id = {}
    with open(vocabpath) as f:
        for index, line in enumerate(f):
            token = line.split()[0]
            id2vocab.append(token)
            vocab2id[token] = index
    return np.array(id2vocab),vocab2id


def run_single_part_workflow(df,model_path,vocab2id,id2vocab):
    """
    This function runs the workflow end2end on a single GPU
    """

    model =  AutoModelForTokenClassification.from_pretrained(model_path)
    model.cuda()
    model.eval()

    token_d,prediction_d= run_inference_on_df(df,model)
    output_d = {}
    for seq,pred_label in prediction_d.items():
        if len(pred_label)!=0:
            sen_df = get_review_sentence(token_d[seq],prediction_d[seq],vocab2id,id2vocab)
            review_df = token_d[seq]['df'][['pr_review_sk','pr_item_sk']]
            review_df = review_df.reset_index(drop=False)
            review_df.rename(columns={'index':'input_text_index'},inplace=True)
            output_d[seq] = sen_df.merge(review_df)[['sentence','company','pr_review_sk','pr_item_sk']]

    
    output_df = cudf.concat([o_df for o_df in  output_d.values()])
    return output_df.drop_duplicates()


def main(client, config):
    product_reviews_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )
    product_reviews_df = product_reviews_df[
        product_reviews_df.pr_item_sk == q27_pr_item_sk
    ]
    product_reviews_df = product_reviews_df.repartition(npartitions=16).persist()
    #wait(product_reviews_df)

    path = './distilbert-base-en-cased'
    id2vocab,vocab2id = create_vocab_table(f'{path}/vocab.txt')
    meta_d = {
        'sentence':'',
        'company':'',
        'pr_review_sk': np.ones(1, dtype=np.int64),
        'pr_item_sk': np.ones(1, dtype=np.int64),
    }
    meta_df = cudf.DataFrame(meta_d)
        ### Serializng a pytorch model seems to be taking forever
    ### Reading from disk write now
    model_path = '/raid/vjawa/torch_ner_q27/tpcx-bb-2/tpcx_bb/queries/q27/distilbert-base-en-cased'
    st = time.time()
    output_df = product_reviews_df.map_partitions(run_single_part_workflow,model_path,vocab2id,id2vocab,meta=meta_df)
    output_df = output_df.persist()
    wait(output_df)
    et = time.time()
    print("workflow took = {}s".format(et-st))
    return output_df


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
