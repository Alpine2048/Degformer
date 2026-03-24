import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from tqdm import tqdm

# =====================
# CONFIG
# =====================
MAX_LEN = 28
BATCH_SIZE = 512
EPOCHS = 50
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

torch.backends.cudnn.benchmark = True

# =====================
# VOCAB
# =====================
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY*"
aa_to_idx = {aa: i for i, aa in enumerate(AA_VOCAB)}

# =====================
# DATASET
# =====================
class PeptideDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def encode(self, seq):
        return torch.tensor([aa_to_idx[a] for a in seq], dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = self.encode(self.sequences[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y

# =====================
# MODEL
# =====================
class PeptideTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, MAX_LEN + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        B = x.size(0)
        x = self.embedding(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.head(x[:, 0]).squeeze(-1)

# =====================
# METRICS
# =====================
def evaluate(model, loader, loss_fn):
    model.eval()
    preds, trues = [], []
    total_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE=="cuda")):
                p = model(x)
                loss = loss_fn(p, y)

            total_loss += loss.item()
            preds.append(p.cpu())
            trues.append(y.cpu())

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    pearson = pearsonr(preds, trues)[0]
    r2 = r2_score(trues, preds)
    avg_loss = total_loss / len(loader)

    return avg_loss, pearson, r2

# =====================
# LOAD DATA
# =====================
print("Loading data...")
df = pd.read_csv("data.csv")

sequences = df.iloc[:, 0].values
control = df.iloc[:, 1].values.reshape(-1, 1)

# Scale
control_scaler = StandardScaler()
control_scaled = control_scaler.fit_transform(control).flatten()

# Split
train_seq, val_seq, train_y, val_y = train_test_split(
    sequences, control_scaled, test_size=0.1, random_state=42
)

train_dataset = PeptideDataset(train_seq, train_y)
val_dataset = PeptideDataset(val_seq, val_y)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# =====================
# INIT
# =====================
model = PeptideTransformer(vocab_size=len(AA_VOCAB)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
scaler_amp = torch.amp.GradScaler("cuda")

# 🔥 Reduce LR on plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)

best_val_loss = float("inf")
history = []

# =====================
# TRAIN
# =====================
print("Training...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for x, y in tqdm(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE=="cuda")):
            pred = model(x)
            loss = loss_fn(pred, y)

        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    val_loss, val_p, val_r2 = evaluate(model, val_loader, loss_fn)
    tr_loss_eval, tr_p, tr_r2 = evaluate(model, train_loader, loss_fn)

    # 🔥 step scheduler
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]['lr']

    print(f"\nEpoch {epoch+1}")
    print(f"LR: {current_lr:.6e}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Train Pearson: {tr_p:.3f} | Val Pearson: {val_p:.3f}")
    print(f"Train R2: {tr_r2:.3f} | Val R2: {val_r2:.3f}")

    history.append({
        "epoch": epoch+1,
        "lr": current_lr,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_pearson": tr_p,
        "val_pearson": val_p,
        "train_r2": tr_r2,
        "val_r2": val_r2,
    })

    # =====================
    # SAVE EVERY EPOCH
    # =====================
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "control_scaler": control_scaler
    }, checkpoint_path)

    # =====================
    # SAVE BEST MODEL
    # =====================
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_path = os.path.join(CHECKPOINT_DIR, "best.pt")

        torch.save({
            "model_state_dict": model.state_dict(),
            "control_scaler": control_scaler
        }, best_path)

        print(f"✅ Saved new best model (epoch {epoch+1})")

# =====================
# FINAL SAVE
# =====================
torch.save({
    "model_state_dict": model.state_dict(),
    "control_scaler": control_scaler
}, "peptide_model_control_only_final.pt")

pd.DataFrame(history).to_csv("training_metrics_control_only.csv", index=False)

print("Done. Model + metrics saved.")