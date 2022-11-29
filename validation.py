import numpy as np
from tqdm import tqdm
import torch
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'



def valid(model, validation_loader):
    model.eval()
    loss_function = torch.nn.BCEWithLogitsLoss()
    tr_loss = 0
    nb_tr_steps = 0
    predictions = np.array([])
    labels = np.array([])

    with torch.no_grad():
        for _, data in tqdm(enumerate(validation_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device)
            labels = np.append(labels, targets.cpu())
            outputs = model(ids, mask, token_type_ids)
            preds = torch.round(torch.sigmoid(outputs))
            predictions = np.append(predictions, preds.cpu())
            loss = loss_function(outputs.view(-1), targets)
            tr_loss += loss.item()
            nb_tr_steps += 1

    epoch_loss = tr_loss / nb_tr_steps
    return epoch_loss, predictions, labels