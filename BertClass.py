

"""
creating a Bert  class
"""
import torch

from transformers import BertModel

class BertClass(torch.nn.Module):
    def __init__(self):
        super(BertClass, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-uncased')
        #chinese BERT hidden dimensions
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.2)
        # two classes: Yes, No
        self.classifier = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask,token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        #output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask,decoder_input_ids=decoder_input_ids)
        hidden_state = output_1[0] # returning sequence outputs, if it's -1, it's all the hidden states
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output