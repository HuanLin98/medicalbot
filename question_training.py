from transformers import BertTokenizer, BertForMaskedLM
import torch
import pandas as pd
import torch.nn as nn
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(torch.cuda.is_available())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))