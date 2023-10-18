import sys
from features import fs_encode
import lightgbm as lgb
import numpy as np
import torch
from transformers import T5Tokenizer, T5Model,T5EncoderModel
import re
from Bio import SeqIO
import argparse

model_esm, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
def esm_infer(seq):
    batch_converter = alphabet.get_batch_converter()
    model_esm.eval()  # disables dropout for deterministic results
    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        ("tmp", seq),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    batch_tokens = batch_tokens
    # batch_tokens = batch_tokens

    with torch.no_grad():
        results = model_esm(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    token_representations = token_representations.detach().cpu().numpy()
    # print(token_representations.shape)
    token_representations = token_representations[0][1:-1,:]
    # (7, 1280)
    return token_representations.sum(axis=0)

'''
pip install transformers
pip install SentencePiece
'''

tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50',do_lower_case=False)
model_t5 = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
def T5_infer(seq):
    sequences_Example = [" ".join(list(seq))]
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
    ids = tokenizer.batch_encode_plus(sequences_Example,
                                      add_special_tokens=True,
                                      padding=False)

    input_ids = torch.tensor(ids['input_ids'])
    attention_mask = torch.tensor(ids['attention_mask'])

    input_ids = input_ids
    attention_mask = attention_mask

    with torch.no_grad():
        # embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=None)
        embedding = model_t5(input_ids=input_ids,
                          attention_mask=attention_mask,
                          # decoder_input_ids=input_ids,
                          )

    # For feature extraction we recommend to use the encoder embedding
    encoder_embedding = embedding.last_hidden_state[0,:-1].detach().cpu().numpy()
    return encoder_embedding.sum(axis=0)

model_fs = lgb.Booster(model_file="./models/model.fs")
model_tr = lgb.Booster(model_file="./models/model.transformer")

def get_data(path):
    id_seq_dict = {}
    rx = SeqIO.parse(path,format="fasta")
    for x in list(rx):
        id = str(x.id)
        seq = str(x.seq)
        id_seq_dict[id] = seq
    return id_seq_dict

def encode(id_seq):
    id_encode_dict= {}
    for id,seq in id_seq.items():
        id = id
        seq = seq
        fsvec = fs_encode(seq)
        res_esm = esm_infer(seq)
        res_t5 = T5_infer(seq)
        res2 = np.concatenate([res_t5,res_esm])
        id_encode_dict[id] = [fsvec,res2]
    return id_encode_dict

def predict(id_encode_dict):
    ress = {}
    for id,vec in id_encode_dict.items():
        vec_fs = np.array(vec[0])[np.newaxis,:]
        vec_ts = np.array(vec[1])[np.newaxis,:]
        p1 = model_fs.predict(vec_fs).flatten()[0]
        p2 = model_tr.predict(vec_ts).flatten()[0]
        t = 0.4
        if p1>0.4 and p2>0.4:
            ress[id] = max([p1,p2])
        elif p1<0.4 and p2<0.4:
            ress[id]=min([p1,p2])
        else:
            ress[id]=np.mean([p1,p2])
    return ress

def write_res(ress_dict):
    f = open("predict_results.csv","w")
    for id,ps in ress_dict.items():
        tmp = id+","+str(ps)+"\n"
        f.write(tmp)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--path",help="add a path")
    args = parser.parse_args()
    path = args.path
    id_seq_dict = get_data(path)
    id_encode_dict = encode(id_seq_dict)
    ress_dict = predict(id_encode_dict)
    write_res(ress_dict)