
import torch
from torch import cuda
import argparse
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from ImplicatureData import ImplicatureData
from BertClass import BertClass
from transformers import AutoTokenizer


from validation import valid
device = 'cuda' if cuda.is_available() else 'cpu'

args = argparse.ArgumentParser(description='validating the Roberta model')
args.add_argument('-a', '--testing_file', type=str, help='testing_file', required=True)
args.add_argument('-m', '--model_file', type=str, help='saved model', required=True)
args = args.parse_args()

LEARNING_RATE = 1e-05
MAX_LEN = 100 # matching the training parameters


def make_confusion_matrix(labels, predictions):
    fig, ax2 = plt.subplots(figsize=(14, 12))

    label_names = sorted(set(labels))
    cm = confusion_matrix(labels, predictions, labels=label_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=label_names)
    disp.plot(ax=ax2)
    plt.show()
    plt.savefig('testing_cm.png')


if __name__=="__main__":

    model_file = args.model_file
    testing_file = args.testing_file

    model = BertClass()
    model.to(device)

    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.5).to(device))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", truncation=True)

    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])


    test_data = pd.read_csv(testing_file, encoding='utf-8-sig')

    test_context = test_data["context"]
    test_utterance = test_data["utterance"]
    test_categories = test_data["implicature"]

    testing_set = ImplicatureData(test_context, test_utterance, test_categories, tokenizer, MAX_LEN)
    testing_loader = DataLoader(testing_set)

    epoch_loss, predictions, labels = valid(model, testing_loader,loss_function)
    print('predictions:')
    for p in predictions:
       print(p)
    print('labels:')
    for l in labels:
       print(l)
    scores = classification_report(labels, predictions)
    print(scores)