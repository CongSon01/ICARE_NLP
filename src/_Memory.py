import numpy as np
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = gpu

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = device

class Memory(object):
    def __init__(self):
        self.examples = []
        self.masks = []
        self.labels = []
        self.tasks = []
        self.features = []
    
    def append(self, example, mask, label, task):
        self.examples.append(example)
        self.masks.append(mask)
        self.labels.append(label)
        self.tasks.append(task)

    def store_features(self, model):
        """

        Args:
            model: The model trained just after previous task

        Returns: None

        store previous features before trained on new class
        """
        self.features = []
        length = len(self.labels)
        model.eval()
        with torch.no_grad():
            for i in range(length):
                x = torch.tensor(self.examples[i]).view(1, -1).to(device)
                mask = torch.tensor(self.masks[i]).view(1, -1).to(device)
                old_fea, pruned_fea, _, _, _ = model(x, mask)
                fea = torch.cat([old_fea, pruned_fea], dim=1).view(-1).data.cpu().numpy()
                self.features.append(fea)
        print(len(self.features))
        print(len(self.labels))

    def get_random_batch(self, batch_size, task_id=None):
        if task_id is None:
            # Hoán vị
            permutations = np.random.permutation(len(self.labels)) 
            index = permutations[:batch_size]
            mini_examples = [self.examples[i] for i in index]
            mini_masks = [self.masks[i] for i in index]
            mini_labels = [self.labels[i] for i in index]
            mini_tasks = [self.tasks[i] for i in index]
            mini_features = [self.features[i] for i in index]
        else:
            index = [i for i in range(len(self.labels)) if self.tasks[i] == task_id]
            np.random.shuffle(index)
            index = index[:batch_size]
            mini_examples = [self.examples[i] for i in index]
            mini_masks = [self.masks[i] for i in index]
            mini_labels = [self.labels[i] for i in index]
            mini_tasks = [self.tasks[i] for i in index]
            mini_features = [self.features[i] for i in index]
        return torch.tensor(mini_examples), torch.tensor(mini_masks), torch.tensor(mini_labels), \
               torch.tensor(mini_tasks), torch.tensor(mini_features)
    
    def get_minibatch(self, batch_size):
        length = len(self.labels)
        permutations = np.random.permutation(length)
        for s in range(0, length, batch_size):
            if s + batch_size >= length:
                break
            index = permutations[s:s + batch_size]
            mini_examples = [self.examples[i] for i in index]
            mini_masks = [self.masks[i] for i in index]
            mini_labels = [self.labels[i] for i in index]
            mini_tasks = [self.tasks[i] for i in index]
            mini_features = [self.features[i] for i in index]
            yield torch.tensor(mini_examples), torch.tensor(mini_masks), torch.tensor(mini_labels), \
                  torch.tensor(mini_tasks), torch.tensor(mini_features)

    def __len__(self):
        return len(self.labels)