import torch
import numpy as np
from torchmetrics.functional.text import char_error_rate, word_error_rate
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


class CustomSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(CustomSchedule, self).__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch + 1
        arg1 = (step ** -0.5)
        arg2 = step * (self.warmup_steps ** -1.5)
        return [((self.d_model ** -0.5) * min(arg1, arg2)) for base_lr in self.base_lrs]


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha, gamma, ignore_index=0, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.weight = weight
        self.nll_loss = torch.nn.NLLLoss(reduction='none', ignore_index=self.ignore_index)

    def forward(self, output, target):
        log_p = torch.log_softmax(output, dim=-1)
        ce = self.nll_loss(log_p, target)

        # get true class column from each row
        all_rows = torch.arange(len(output))
        log_pt = log_p[all_rows, target]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        focal_loss = focal_term * ce
        return focal_loss.mean()


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_metric = float('inf')  # Initialize with positive infinity for loss
        self.early_stop = False

    def __call__(self, current_metric):
        if self.best_metric - current_metric > self.delta:
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")
        return self.early_stop


def decode_input(input_decoder, vocab):
    # input_decoder: (batch_size, seq_len)
    index = vocab['<pad>']
    vocabulary = list(vocab.keys())
    sequences = []
    for batch in input_decoder:
        text = ''.join([vocabulary[i] for i in batch if i != index])
        sequences.append(text)
    return sequences


def decode_output(output_model, vocab):
    # output_model: (batch_size, seq_len, vocab_size)
    index = vocab['<pad>']
    vocabulary = list(vocab.keys())
    sequences = []
    for batch in output_model:
        output = torch.argmax(batch, dim=-1)
        text = ''.join([vocabulary[i] for i in output if i != index])
        sequences.append(text)
    return sequences


def compute_cer(input_decoder, output_model, vocab):
    target = decode_input(input_decoder, vocab)
    predict = decode_output(output_model, vocab)
    cer = char_error_rate(predict, target)
    wer = word_error_rate(predict, target)
    return wer.item()


def compute_recall_precision_f1_acc(input_decoder, output_model):
    y, y_temp = [], []
    y_pred, y_pred_temp = [], []
    for batch in output_model:
        output = torch.argmax(batch, dim=-1)
        y_pred_temp.extend([i.cpu() for i in output])
    for batch in input_decoder:
        y_temp.extend([i.cpu() for i in batch])
    for i, o in zip(y_pred_temp, y_temp):
        if o != 0:
            y.append(o.item())
            y_pred.append(i.item())
    acc = accuracy_score(np.asarray(y), np.asarray(y_pred))
    recall = recall_score(np.asarray(y), np.asarray(y_pred), average='micro', zero_division=0)
    precision = precision_score(np.asarray(y), np.asarray(y_pred), average='micro', zero_division=0)
    f1 = f1_score(np.asarray(y), np.asarray(y_pred), average='micro', zero_division=0)
    return recall, precision, f1, acc


def compute_metrics(input_decoder, output_model, vocab):
    input_decoder = input_decoder[:, 1:].contiguous()
    output_model = output_model[:, :-1, :].contiguous()
    recall, precision, f1, acc = compute_recall_precision_f1_acc(input_decoder, output_model)
    cer = compute_cer(input_decoder, output_model, vocab)
    return cer, recall, precision, f1, acc


def compute_total_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
