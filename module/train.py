import regex
import torch
import warnings
import wandb
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from module.model_builder.model import Model
from module.data_setup import CustomDataset
from module.metrics import EarlyStopping, CustomSchedule, FocalLoss
from module.metrics import compute_metrics, compute_total_norm
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.login(key='e7ca95062f4bf972322db5f37645ac5fa1e0afb0', relogin=True)
wandb.init(project='ocr_vn', reinit=True)

# ------- Get train_loader, val_loader -------
train_path = '../dataset/augment_data'
train_file = '../dataset/train.txt'
val_file = '../dataset/val.txt'
vocab_file = '../dataset/augment_labels.txt'
batch_size = 128
seq_length = 144
image_size = (384, 384)
train_dataset = CustomDataset(train_path, train_file, vocab_file, seq_length, image_size)
val_dataset = CustomDataset(train_path, val_file, vocab_file, seq_length, image_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
vocab = train_dataset.vocab.string_to_index
# Create instance model
n_dim_model = 512
# --- Encoder Parameters ---
input_chanel_encoder = 3
hidden_dim_encoder = 512
n_head_encoder = 8
n_expansion_encoder = 4
n_layer_encoder = 6
# --- Decoder Parameters ---
n_head_decoder = 8
seq_length_decoder = seq_length
vocab_size_decoder = len(vocab)
n_expansion_decoder = 4
n_layer_decoder = 6
model = Model(n_dim_model, input_chanel_encoder, hidden_dim_encoder, n_head_encoder, n_expansion_encoder, n_layer_encoder,
              n_head_decoder, seq_length_decoder, vocab_size_decoder, n_expansion_decoder, n_layer_decoder).to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters: {}".format(pytorch_total_params))
print("Num of Vocabularies: {}".format(len(vocab)))
print("Vocabularies: {}".format(vocab))
print("-----------------------------------------------------------------------")
# Loss function and Optimizer
epochs = 150
criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
# criterion = FocalLoss(alpha=0.75, gamma=2.0, ignore_index=vocab['<pad>'])
optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-5)
lr_scheduler = CustomSchedule(optimizer, d_model=n_dim_model, warmup_steps=4000)
early_stopping = EarlyStopping(patience=5, delta=1e-3, verbose=True)
# Training
for epoch in range(epochs):
    model.train()
    loss_value = 0.0
    cer_value = 0.0
    recall_value, precision_value, f1_value, acc_value = 0.0, 0.0, 0.0, 0.0
    for batch_idx, (input_encoder, input_decoder) in enumerate(train_loader):
        # Predict
        lr_scheduler.step()
        input_encoder, input_decoder = input_encoder.to(device), input_decoder.to(device)
        output_model = model(input_encoder, input_decoder[:, :-1])
        # Compute loss
        output_dim = output_model.shape[-1]
        output = output_model.contiguous().view(-1, output_dim)
        target = input_decoder[:, 1:].contiguous().view(-1).long()
        loss = criterion(output, target)
        loss_value += loss.item()
        # Compute metrics
        # current_lr = optimizer.param_groups[0]['lr']
        # cer, recall, precision, f1, acc = compute_metrics(input_decoder, output_model, vocab)
        # cer_value += cer
        # recall_value += recall
        # precision_value += precision
        # f1_value += f1
        # acc_value += acc
        loss.backward()
        total_norm = compute_total_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()
        wandb.log({
            'loss per step': loss.item(),
            # 'lr': current_lr,
            # 'cer per step': cer,
            # 'recall per step': recall,
            # 'precision per step': precision,
            # 'f1 per step': f1,
            # 'acc per step': acc,
            # 'grad_norm per step': total_norm
        })
    loss_train = loss_value / len(train_loader)
    # cer_train = cer_value / len(train_loader)
    # recall_value = recall_value / len(train_loader)
    # precision_value = precision_value / len(train_loader)
    # f1_value = f1_value / len(train_loader)
    # acc_value = acc_value / len(train_loader)
    wandb.log({
        'train loss': loss_train,
        # 'train cer': cer_train,
        # 'train recall': recall_value,
        # 'train precision': precision_value,
        # 'train f1': f1_value,
        # 'train acc': acc_value,
    })
    # Evaluate
    loss_value = 0.0
    cer_value = 0.0
    # recall_value, precision_value, f1_value, acc_value = 0.0, 0.0, 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, (input_encoder, input_decoder) in enumerate(val_loader):
            # Predict
            input_encoder, input_decoder = input_encoder.to(device), input_decoder.to(device)
            output_model = model(input_encoder, input_decoder[:, :-1])
            # Compute loss
            output_dim = output_model.shape[-1]
            output = output_model.contiguous().view(-1, output_dim)
            target = input_decoder[:, 1:].contiguous().view(-1).long()
            loss = criterion(output, target).item()
            cer, recall, precision, f1, acc = compute_metrics(input_decoder, output_model, vocab)
            cer_value += cer
            loss_value += loss
            recall_value += recall
            precision_value += precision
            f1_value += f1
            acc_value += acc
        loss_val = loss_value / len(val_loader)
        # cer_val = cer_value / len(val_loader)
        # recall_value = recall_value / len(val_loader)
        # precision_value = precision_value / len(val_loader)
        # f1_value = f1_value / len(val_loader)
        # acc_value = acc_value / len(val_loader)
        wandb.log({
            'val loss': loss_val,
            # 'val cer': cer_val,
            # 'val recall': recall_value,
            # 'val precision': precision_value,
            # 'val f1': f1_value,
            # 'val acc': acc_value,
        })
    print(f"Epoch: {epoch + 1} | train_loss: {loss_train:.2f} | val_loss: {loss_val:.2f}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab
    }, '../checkpoints2/checkpoint_v3.pth.tar')
    if early_stopping(loss_val):
        print("Early Stopping Training Progress!")
        break
torch.save(model.state_dict(), '../model.pth')
wandb.finish()