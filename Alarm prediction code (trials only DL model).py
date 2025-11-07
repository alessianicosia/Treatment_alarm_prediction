# ====================== ALARM PREDICTION CODE ======================
# TEST 1 DL MODEL
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from numpy import array, hstack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- SEED AND DEVICE ----------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---------- ORIGINAL FUNCTIONS ----------
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def create_splitted_df(df, n_steps):
    historical_df = df[:-20]
    features = ['dmparent', 'dmchild', 'tmp', 'pt3', 'pt4', 'pt5', 'bld', 'diffFlow',
                'encoderDelivPump', 'encoderUFPump', 'delivPumpActuation', 'ufPressureActuation']
    stacked_features = [historical_df[col].to_numpy().reshape((len(historical_df), 1)) for col in features]
    alarm_type = df['type'].to_numpy()[20:].reshape((len(df)-20, 1))
    dataset = hstack(stacked_features + [alarm_type])
    X, y = split_sequences(dataset, n_steps)
    return X, y, features

# ---------- UPLOADING AND PRE-PROCESSING ----------
CSV_PATH = r"dataset.csv"
df = pd.read_csv(CSV_PATH)
df = df.rename(columns={'dT(CP)': 'dT'})  # se non esiste, non fa danni
df['type'] = df['type'].replace({'alarm': '1', 'normal': '0', 'override': '0'})

N_STEPS = 3
X_all, y_all, feat_names = create_splitted_df(df, N_STEPS)
y_all = y_all.astype(int).ravel()

print("Raw shapes -> X:", X_all.shape, " y:", y_all.shape)  # (n_samples, n_steps, n_features)

# SPLIT 15% TEST, 15% VAL
X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all, test_size=0.15, random_state=SEED, stratify=y_all)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=SEED, stratify=y_temp)

# STANDARDIZATION BY FEATURE USING TRAINING SET
n_features = X_train.shape[2]
scaler = StandardScaler()
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_val_flat   = X_val.reshape((X_val.shape[0], -1))
X_test_flat  = X_test.reshape((X_test.shape[0], -1))

scaler.fit(X_train_flat)
X_train_std = scaler.transform(X_train_flat).reshape(X_train.shape)
X_val_std   = scaler.transform(X_val_flat).reshape(X_val.shape)
X_test_std  = scaler.transform(X_test_flat).reshape(X_test.shape)

# ---------- TORCH DATASET ----------
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        # Conv1d si aspetta (B, C, T) -> quindi trasponiamo a (features, steps)
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (N, C, T)
        self.y = torch.tensor(y, dtype=torch.long)  # classi 0/1
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = SeqDataset(X_train_std, y_train)
val_ds   = SeqDataset(X_val_std,   y_val)
test_ds  = SeqDataset(X_test_std,  y_test)

BATCH_SIZE = 256
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ---------- MODEL: TCN + TEMPORAL ATTENTION ----------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        self.act1 = nn.PReLU()
        self.do1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        self.act2 = nn.PReLU()
        self.do2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.act_out = nn.PReLU()

        # INIT
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")
            nn.init.zeros_(self.downsample.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.act1(out)
        out = self.do1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.act2(out)
        out = self.do2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.act_out(out + res)

class TemporalAttention(nn.Module):
    def __init__(self, in_ch, attn_dim=64):
        super().__init__()
        self.W = nn.Linear(in_ch, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h):
        h_t = h.transpose(1, 2)
        score = torch.tanh(self.W(h_t))
        e = self.v(score).squeeze(-1)
        alpha = torch.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), h_t).squeeze(1)
        return context, alpha

class TCNWithAttention(nn.Module):
    def __init__(self, in_ch, num_classes=2, channels=(64, 64, 128), kernel_size=3, drop=0.1):
        super().__init__()
        layers = []
        prev = in_ch
        dil = 1
        for ch in channels:
            layers.append(TemporalBlock(prev, ch, kernel_size=kernel_size, dilation=dil, dropout=drop))
            prev = ch
            dil *= 2
        self.tcn = nn.Sequential(*layers)
        self.attn = TemporalAttention(in_ch=prev, attn_dim=64)
        self.bn = nn.BatchNorm1d(prev)
        self.head = nn.Linear(prev, num_classes)

        nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = self.tcn(x)
        h = self.bn(h)
        ctx, alpha = self.attn(h)
        logits = self.head(ctx)
        return logits, alpha

# ---------- CLASS WEIGHTS ----------
def compute_class_weights(y):
    cnt = Counter(y)
    classes = sorted(cnt.keys())
    total = sum(cnt.values())
    weights = {c: total / (len(classes) * cnt[c]) for c in classes}
    return torch.tensor([weights[c] for c in classes], dtype=torch.float32)

# ---------- TRAIN / VAL ----------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_y, all_p = [], []
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits, _ = model(xb)
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        all_y.append(yb.cpu().numpy())
        all_p.append(pred.cpu().numpy())
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, pos_label=1)
    return acc, f1, y_true, y_pred

# ---------- HYPERPARAMETERS ----------
CHANNELS = (64, 64, 128)
KERNEL_SIZE = 3
DROPOUT = 0.15
LR = 1e-3
EPOCHS = 60
PATIENCE = 10
WEIGHT_DECAY = 1e-4

# ---------- MODEL / LOSS / OPTIMIZ ----------
model = TCNWithAttention(in_ch=n_features, num_classes=2,
                         channels=CHANNELS, kernel_size=KERNEL_SIZE, drop=DROPOUT).to(DEVICE)

class_weights = compute_class_weights(y_train).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print(model)

# ---------- EARLY STOPPING ----------
best_f1 = -1.0
pat_count = 0
best_path = "best_tcn_attention.pt"

for epoch in range(1, EPOCHS + 1):
    tr_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    val_acc, val_f1, _, _ = evaluate(model, val_loader)
    print(f"Epoch {epoch:03d} | TrainLoss {tr_loss:.4f} | ValAcc {val_acc:.4f} | ValF1 {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        pat_count = 0
        torch.save(model.state_dict(), best_path)
    else:
        pat_count += 1
        if pat_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best Val F1={best_f1:.4f}")
            break

# ---------- FINAL EVALUATION ----------
model.load_state_dict(torch.load(best_path, map_location=DEVICE))

val_acc, val_f1, yv_true, yv_pred = evaluate(model, val_loader)
test_acc, test_f1, yt_true, yt_pred = evaluate(model, test_loader)

print("\n=== FINAL RESULTS ===")
print(f"Validation  -> Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
print(f"Test        -> Acc: {test_acc:.4f} | F1: {test_f1:.4f}")

print("\nClassification report (Validation):")
print(classification_report(yv_true, yv_pred, digits=4))
print("Class balance (Train):", Counter(y_train))

# ---------- CONFUSION MATRICES ----------
def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

plot_cm(yv_true, yv_pred, "TCN+Attention - Validation Confusion Matrix")
plot_cm(yt_true, yt_pred, "TCN+Attention - Test Confusion Matrix")

# ---------- VIEW ATTENTION WEIGHT DISTRIBUTION ----------
@torch.no_grad()
def visualize_attention(model, loader, n_samples=5):
    model.eval()
    xb, yb = next(iter(loader))
    xb = xb.to(DEVICE)
    logits, alpha = model(xb)  # alpha: (B, T)
    alpha = alpha.detach().cpu().numpy()
    T = alpha.shape[1]
    ns = min(n_samples, alpha.shape[0])
    plt.figure(figsize=(8, 3*ns))
    for i in range(ns):
        plt.subplot(ns, 1, i+1)
        plt.stem(range(T), alpha[i], use_line_collection=True)
        plt.title(f"Sample {i} - Attention over time steps (0..{T-1})")
        plt.xlabel("Time step")
        plt.ylabel("Weight")
    plt.tight_layout()
    plt.show()