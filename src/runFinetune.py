import os
import argparse
from copy import deepcopy
from dotenv import load_dotenv

DATA_DIR = '../data'
# Load environment variables from .env file
parser  = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="Finetune_5step")
env_name = './env/' + parser.parse_args().env + '.env'
load_dotenv(env_name)

# Access environment variables
seed = int(os.getenv("SEED"))
epochs = list(map(int, os.getenv("EPOCHS").split()))
batch_size = int(os.getenv("BATCH_SIZE"))
bert_learning_rate = float(os.getenv("BERT_LEARNING_RATE"))
learning_rate = float(os.getenv("LEARNING_RATE"))
gpu = os.getenv("GPU")
n_labeled = int(os.getenv("N_LABELED"))
tasks = os.getenv("TASKS").split()

# Set CUDA_VISIBLE_DEVICES environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW

from model.basemodel import BaseModel
from read_data import merge_labeling, prepare_dataloaders

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = device
n_gpu = torch.cuda.device_count()

dataset_classes = {
    'amazon'  : 5,
    'yelp'    : 5,
    'yahoo'   : 6,
    'ag'      : 4,
    'dbpedia' : 5,
    '20NG' : 6,
}


def train_step(model, optimizer, cls_CR, x, y):
    model.train()
    logits = model(x)
    loss = cls_CR(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def validation(model, t, validation_loaders):
    '''
    Compute the validation accuracy on the first (t + 1) tasks,
    return the average accuracy over (t + 1) tasks and detailed accuracy
    on each task.
    '''
    model.eval()
    acc_list = []
    with torch.no_grad():
        avg_acc = 0.0
        for i in range(t + 1):
            valid_loader = validation_loaders[i]
            total = 0
            correct = 0
            for x, mask, y in valid_loader:
                x, y = x.to(device), y.to(device)
                batch_size = x.size(0)
                logits = model(x)
                _, pred_cls = logits.max(1)
                correct += pred_cls.eq(y.view_as(pred_cls)).sum().item()
                total += batch_size
            print("acc on task {} : {}".format(i, correct * 100.0 / total))
            avg_acc += correct * 100.0 / total
            acc_list.append(correct * 100.0 / total)

    return avg_acc / (t + 1), acc_list


def runFineTune():
    np.random.seed(0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    task_num = len(tasks)
    task_classes = [dataset_classes[task] for task in tasks]
    total_classes, offsets = merge_labeling(tasks, task_classes)
    train_loaders, validation_loaders, test_loaders = \
        prepare_dataloaders(DATA_DIR, tasks, offsets, n_labeled,
                            2000, batch_size, 128, 128)

    # Reset random seed by the torch seed
    np.random.seed(torch.randint(1000, [1]).item())

    model = BaseModel(total_classes).to(device)
    cls_CR = torch.nn.CrossEntropyLoss()

    for task_id in range(task_num):
        data_loader = train_loaders[task_id]
        length = len(data_loader)

        optimizer = AdamW(
            [
                {"params": model.bert.parameters(), "lr": bert_learning_rate},
                {"params": model.classifier.parameters(), "lr": learning_rate},
            ]
        )

        best_acc = 0
        best_model = deepcopy(model.state_dict())

        acc_track = []

        for epoch in range(epochs[task_id]):
            iteration = 1
            for x, mask, y in tqdm(data_loader, total=length, ncols=100):
                x, y = x.to(device), y.to(device)
                train_step(model, optimizer, cls_CR, x, y)

                if iteration % 250 == 0:
                    print("----------------Validation-----------------")
                    avg_acc, acc_list = validation(model, task_id, validation_loaders)
                    acc_track.append(acc_list)

                    if avg_acc > best_acc:
                        print("------------------Best Model Till Now------------------------")
                        best_acc = avg_acc
                        best_model = deepcopy(model.state_dict())

                iteration += 1

        if len(acc_track) > 0:
            print("ACC Track: {}".format(acc_track))

        model.load_state_dict(deepcopy(best_model))
        print("------------------Best Result------------------")
        avg_acc, _ = validation(model, task_id, test_loaders)
        print("Best avg acc: {}".format(avg_acc))


if __name__ == '__main__':
    runFineTune()