

"""
creating a T5  class
"""
import torch
from transformers import T5Tokenizer, T5Model

class T5Class(torch.nn.Module):
    def __init__(self):
        super(T5Class, self).__init__()
        self.l1 = T5Model.from_pretrained("t5-small")
        #chinese BERT hidden dimensions
        self.pre_classifier = torch.nn.Linear(21128, 21128)
        self.dropout = torch.nn.Dropout(0.2)
        # two classes: Yes, No
        self.classifier = torch.nn.Linear(21128, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0] # returning sequence outputs, if it's -1, it's all the hidden states
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output