# Conversational Implicature Classification
We train 3 models: synthetic model, naturally occurring model and transfer model using different types of data by fine-tuning BERT. Please refer to section [Data][#data]


# Model Architecture

`BertClass.py` creates a Bert class that takes the final hidden layer from [BERT](https://huggingface.co/bert-base-uncased) with one additional hidden layer.  

# Model Training 
To start training a model, use the command `python create_bert_model --train_file <training data> --val_file <validation data>`. A model `model.pt` will be created upon 
completion of training.

# Model Testing

Once you have created your model, run `python test_model.py --testing_file <test_data> --model_file <model_file>` to get a model predictions as well as classicfication report.
# Data 

All the datasets contain pairs of polar-questions and answers that are labeled as 'context', 'uttereance' and 'implicature'.
1. Circa dataset ([Louis et al., 2020](https://aclanthology.org/2020.emnlp-main.601/)): Contains 32,044 pairs, used for training and validation of the synthetic model. 
2. Friends QIA dataset ([Damgaard et al., 2021](https://www.semanticscholar.org/paper/%E2%80%9CI%E2%80%99ll-be-there-for-you%E2%80%9D%3A-The-One-with-Understanding-Damgaard-Toborek/50a98dff869ab316adc88a6d2a6b1978e9a721c6)): Contains 4,580 pairs,used for training and validation of the naturally occurring model. 
3. Conversational Implicature dataset ([George & Mamida, 2020](https://arxiv.org/abs/1911.10704)): Contains 733 pairs labeled as 'context', 'uttereance' and 'implicature' used for testing all the models

The Circa and and Friends datasets are used for training the Transfer model, which will be validated on the Friends dataset only. 
