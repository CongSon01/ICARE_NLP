import os
from dotenv import load_dotenv
from copy import deepcopy
import numpy as np
import torch
import argparse
from tqdm import tqdm
from transformers import AdamW

from model.basemodel import BaseModel
from read_merge_data import merge_labeling, prepare_dataloaders


DATA_DIR = '../data'

# Load environment variables from .env file
parser  = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="Replay_5step")
env_name = './env/' + parser.parse_args().env + '.env'
load_dotenv(env_name)

# Access environment variables
seed = int(os.getenv("SEED"))
epochs = list(map(int, os.getenv("EPOCHS").split()))
batch_size = int(os.getenv("BATCH_SIZE"))
bert_learning_rate = float(os.getenv("BERT_LEARNING_RATE"))
learning_rate = float(os.getenv("LEARNING_RATE"))
replay_freq = int(os.getenv("REPLAY_FREQ"))
gpu = os.getenv("GPU")
n_labeled = int(os.getenv("N_LABELED"))
store_ratio = float(os.getenv("STORE_RATIO"))
tasks = os.getenv("TASKS").split()

# Set CUDA_VISIBLE_DEVICES environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

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


class Memory(object):
    def __init__(self):
        self.examples = []
        self.labels = []
        self.tasks = []

    def append(self, example, label, task):
        self.examples.append(example)
        self.labels.append(label)
        self.tasks.append(task)

    def get_random_batch(self, batch_size, task_id=None):
        if task_id is None:
            permutations = np.random.permutation(len(self.labels))
            index = permutations[:batch_size]
            mini_examples = [self.examples[i] for i in index]
            mini_labels = [self.labels[i] for i in index]
            mini_tasks = [self.tasks[i] for i in index]
        else:
            index = [i for i in range(len(self.labels)) if self.tasks[i] == task_id]
            np.random.shuffle(index)
            index = index[:batch_size]
            mini_examples = [self.examples[i] for i in index]
            mini_labels = [self.labels[i] for i in index]
            mini_tasks = [self.tasks[i] for i in index]
        return torch.tensor(mini_examples), torch.tensor(mini_labels), torch.tensor(mini_tasks)

    def __len__(self):
        return len(self.labels)


def train_step(model, optimizer, cls_CR, x, y):
    model.train()
    logits = model(x)
    loss = cls_CR(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def validation(model, t, validation_loaders):
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


def random_class_aware_representation(buffer, data_loader, task_id):
    x_list = []
    y_list = []
    for x, mask, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        x_list.append(x)
        y_list.append(y)
    x_list = torch.cat(x_list, dim=0).data.cpu().numpy()
    y_list = torch.cat(y_list, dim=0).data.cpu().numpy()
    permutations = np.random.permutation(len(x_list))
    index = permutations[:int(store_ratio * len(x_list))]
    for j in index:
        buffer.append(x_list[j], y_list[j], task_id)

    print("Buffer size:{}".format(len(buffer)))
    b_lbl = np.unique(buffer.labels)
    for i in b_lbl:
        print("Label {} in Buffer: {}".format(i, buffer.labels.count(i)))


def runReplay():
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

    buffer = Memory()
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
                if iteration % replay_freq == 0 and task_id > 0:
                    # replay once every replay_freq batches, starting from the 2nd task
                    total_x, total_y = x, y
                    for j in range(task_id):
                        old_x, old_y, old_t = buffer.get_random_batch(batch_size, j)
                        total_x = torch.cat([old_x, total_x], dim=0)
                        total_y = torch.cat([old_y, total_y], dim=0)
                    permutation = np.random.permutation(total_x.shape[0])
                    total_x = total_x[permutation, :]
                    total_y = total_y[permutation]
                    for j in range(task_id + 1):
                        x = total_x[j * batch_size: (j + 1) * batch_size, :]
                        y = total_y[j * batch_size: (j + 1) * batch_size]
                        x, y = x.to(device), y.to(device)
                        train_step(model, optimizer, cls_CR, x, y)
                else:
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

            if task_id == 0:
                print("----------------Validation-----------------")
                avg_acc, _ = validation(model, task_id, validation_loaders)

                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_model = deepcopy(model.state_dict())
                    print("------------------Best Model Till Now------------------------")

        if len(acc_track) > 0:
            print("ACC Track: {}".format(acc_track))

        model.load_state_dict(deepcopy(best_model))
        print("------------------Best Result------------------")
        avg_acc, _ = validation(model, task_id, test_loaders)
        print("Best avg acc: {}".format(avg_acc))

        random_class_aware_representation(buffer, data_loader, task_id)


if __name__ == '__main__':
    runReplay()