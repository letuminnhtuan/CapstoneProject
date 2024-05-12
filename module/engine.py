import torch
import torch.nn.functional as F
from module.metrics import compute_cer

def train_fn(wandb, model, vocab, optimizer, criterion, train_loader, device):
    model.train()
    vocab_size_decoder = len(vocab)
    loss_value = 0.0
    cer_value = 0.0
    for batch_idx, (input_encoder, input_decoder) in enumerate(train_loader):
        # Predict
        optimizer.zero_grad()
        input_encoder, input_decoder = input_encoder.to(device), input_decoder.to(device)
        output_model = model(input_encoder, input_decoder)
        # Compute loss
        target = input_decoder[:, 1:].contiguous()
        pred = output_model[:, :-1, :].contiguous()
        loss = criterion(pred.view(-1, vocab_size_decoder), target.view(-1).long())
        loss.backward()
        optimizer.step()
        cer = compute_cer(target, pred, vocab)
        cer_value += cer
        loss_value += loss.item()
        wandb.log({'loss per step': loss.item(), 'cer per step': cer})
    loss_train = loss_value / len(train_loader)
    cer_train = cer_value / len(train_loader)
    return loss_train, cer_train

def val_fn(model, vocab, criterion, val_loader, device):
    model.eval()
    vocab_size_decoder = len(vocab)
    loss_value = 0.0
    cer_value = 0.0
    with torch.no_grad():
        for batch_idx, (input_encoder, input_decoder) in enumerate(val_loader):
            # Predict
            input_encoder, input_decoder = input_encoder.to(device), input_decoder.to(device)
            output_model = model(input_encoder, input_decoder)
            # Compute loss
            target = input_decoder[:, 1:].contiguous()
            pred = output_model[:, :-1, :].contiguous()
            loss = criterion(pred.view(-1, vocab_size_decoder), target.view(-1).long())
            cer = compute_cer(target, pred, vocab)
            cer_value += cer
            loss_value += loss.item()
        loss_val = loss_value / len(val_loader)
        cer_val = cer_value / len(val_loader)
    return loss_val, cer_val