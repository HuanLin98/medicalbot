import os
import time
import datetime
from datasets.utils import logging

import pandas as pd
import numpy as np
import logging
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from nltk.translate.bleu_score import sentence_bleu
torch.manual_seed(42)

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
# from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_metric
import string, re
import pickle

def save_obj(obj, filepath ):
    with open(filepath + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(filepath ):
    with open(filepath + '.pkl', 'rb') as f:
        return pickle.load(f)

logging.basicConfig(filename='C:/Users/wjtay/Documents/GitHub/medicalbot/test_data/logs/bleurt_eval.txt', level=logging.INFO)

test_set = pd.read_csv('test_set.csv')
logging.info(f'Expected: 604 , testset {test_set.shape}')
save_path =  'C:/Users/wjtay/Documents/GitHub/medicalbot/test_data/evaluations/'
logging.info('save path: {save_path}')

def process_string(answer):
    p_answer = answer.translate(str.maketrans('','', string.punctuation))
    curr_ans = p_answer.split(' ')
    return curr_ans

gpt_paths = [
    'C:/Users/wjtay/Documents/GitHub/medicalbot/gpt_models/model_save_2_lasseOnly_20',
    'C:/Users/wjtay/Documents/GitHub/medicalbot/gpt_models/model_save_3_lasseOnly_30',
    'C:/Users/wjtay/Documents/GitHub/medicalbot/gpt_models/model_save_4_lasseTranslated_5',
    'C:/Users/wjtay/Documents/GitHub/medicalbot/gpt_models/model_save_BERT_Lasse_FAISS_30',
    'C:/Users/wjtay/Documents/GitHub/medicalbot/gpt_models/model_save_GPT-med_MEDBERT_Lasse_15',
    'C:/Users/wjtay/Documents/GitHub/medicalbot/gpt_models/model_save_MBERT_Lasse_FAISS_30',
    'C:/Users/wjtay/Documents/GitHub/medicalbot/gpt_models/model_save_MEDBERT_Lasse_FAISS_30',
]

def get_prediction(n, query):
    prompt = f"<|startoftext|><|question|>{query}<|answer|>"

    generated = torch.tensor(gpt_tokenizer.encode(prompt)).unsqueeze(0)
    # generated = generated.to(device)

    # logging.info(generated)

    sample_outputs = gpt_model.generate(
                                    generated, 
                                    #bos_token_id=random.randint(1,30000),
                                    do_sample=True,   
                                    top_k=50, 
                                    max_length = 300,
                                    top_p=0.95, 
                                    num_return_sequences=n
                                    )
    decoded_outputs = []

    for i, sample_output in enumerate(sample_outputs):
        # logging.info("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
        output = gpt_tokenizer.decode(sample_output, skip_special_tokens=True)
        output = output.split('<|answer|>')[1]
        decoded_outputs.append(output)

            
    # logging.info(decoded_outputs)
    return decoded_outputs

results = pd.DataFrame(columns=['model', 'bluert'])

for gpt_path in gpt_paths:
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium

    # gpt_path = 'C:/Users/wjtay/Documents/GitHub/medicalbot/gpt_models/model_save_3_lasseOnly_30'
    logging.info(f'Starting --> {gpt_path} ')
    gpt_model =  GPT2LMHeadModel.from_pretrained(gpt_path)
    gpt_model.resize_token_embeddings(len(gpt_tokenizer))

    gpt_model.eval()

    metric = load_metric("bleurt")

    for idx in range(test_set.shape[0]):
        row = test_set.iloc[idx]
        references = [row['answer']]
        predictions = get_prediction(1, row['question'])
        # print(f"predictions {predictions} ")
        # print(f"references {references} ")
        metric.add(prediction=predictions, reference=references)
        # metric.add(prediction=["hello world"], reference=["hello worlds"])
        # print(metric.compute(prediction=["hello world"], reference=["hello worlds"]))
        if(idx % 5 == 0):
            logging.info(f"question {row['question']}")
            logging.info(f"predictions {predictions} ")
            logging.info(f"references {references} ")

    score = metric.compute()
    print(sum(score['scores']) / len(score['scores']))
    datapoint = {"model": gpt_path.split('/')[-1]}
    datapoint.update({"bluert" : sum(score['scores']) / len(score['scores'])})
    results = results.append(datapoint, ignore_index=True)

    save_name = save_path + gpt_path.split('/')[-1]
    save_obj(score,save_name)
    logging.info(f'Result-{gpt_path}')
    logging.info(f'Saving score to -{save_name}')
    logging.info(score)

results.to_csv(save_path + 'results.csv')