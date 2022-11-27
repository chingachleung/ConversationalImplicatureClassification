import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
import torch
from torch import cuda
import argparse
import logging

from ImplicatureData import ImplicatureData
from T5Class import T5Class
from train_model import *

logging.basicConfig(level=logging.ERROR)
args = argparse.ArgumentParser(description='implicature classification using T5')
args.add_argument('-a', '--train_file', type=str, help='train file', required=True)
args.add_argument('-v', '--val_file', type=str, help='val file', required=True)
args = args.parse_args()

#global variables

MAX_LEN = 100
BATCH_SIZE = 25
EPOCHS = 4
LEARNING_RATE = 1e-05

if __name__=="__main__":

    # Setting up the device for GPU usage
    device = 'cuda' if cuda.is_available() else 'cpu'
    model = T5Class()
    tokenizer = T5Tokenizer.from_pretrained("t5-small",truncation=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    #for gpu further training
    #model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.to(device)

    train_file = args.train_file
    val_file = args.val_file


    train_params = {'batch_size': BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    #load the data
    train_data = pd.read_csv(train_file, encoding='utf-8-sig')
    val_data = pd.read_csv(val_file, encoding='utf-8-sig')

    train_context = train_data["context"]
    train_utterance = train_data["utterance"]
    train_categories = train_data["implicature"]

    val_context = val_data["context"]
    val_utterance = val_data["utterance"]
    val_categories = val_data["implicature"]



    #data preparation
    onehot_encoder = OneHotEncoder(handle_unknown="ignore")  # set to zeros if new categories in test set occur
    training_onehot_targets = onehot_encoder.fit_transform(train_categories.values.reshape(-1, 1)).toarray()
    #context, utterance, targets, tokenizer, max_len
    training_set = ImplicatureData(train_context, train_utterance, training_onehot_targets, tokenizer, MAX_LEN)
    training_loader = DataLoader(training_set, **train_params)

    val_onehot_targets = onehot_encoder.transform(val_categories.values.reshape(-1, 1)).toarray()
    val_set = ImplicatureData(val_context, val_utterance, val_onehot_targets, tokenizer, MAX_LEN)
    val_loader = DataLoader(val_set)

    #start training
    model, loss, optimizer = train(model,optimizer, EPOCHS, training_loader, val_loader,onehot_encoder)
    #save the model
    torch.save({'epoch': EPOCHS,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'encoder': onehot_encoder},'implicature_model1.pt')

    print('All files saved')