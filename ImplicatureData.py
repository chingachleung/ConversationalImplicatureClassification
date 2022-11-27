import torch
from torch.utils.data import Dataset

class ImplicatureData(Dataset):
    #changed from data frame to texts
    def __init__(self, context, utterance, targets, tokenizer, max_len):
        self.tokenizer = tokenizer
        #self.data = dataframe
        self.context = context
        self.utterance = utterance
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.context + self.utterance)

    def __getitem__(self, index):
        context = str(self.context[index])
        utterance = str(self.utterance[index])
        #text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            context,
            utterance,
            add_special_tokens=True,
            max_length=self.max_len,
            #pad to the max length
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float),
        }