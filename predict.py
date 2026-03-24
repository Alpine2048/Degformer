import argparse
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

# =====================
# CONFIG
# =====================
MAX_LEN = 28
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY*"
aa_to_idx = {aa: i for i, aa in enumerate(AA_VOCAB)}

# =====================
# MODEL (UPDATED: 1 OUTPUT)
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

        # 🔥 single output
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

        return self.head(x[:, 0]).squeeze(-1)  # [B]

# =====================
# HELPERS
# =====================
def encode(seq):
    return torch.tensor([aa_to_idx[a] for a in seq], dtype=torch.long)

def generate_sat_mut(seq):
    mutants = []
    L = len(seq)
    for i in range(L):
        orig_aa = seq[i]
        for aa in AA_VOCAB:
            if aa == "*" or aa == orig_aa:
                continue
            mutant = list(seq)
            mutant[i] = aa
            mutants.append(("".join(mutant), i, aa))
    return mutants

def generate_scan_mut(seq, residue):
    mutants = [(seq, -1, "WT")]
    for i in range(len(seq)):
        if seq[i] == residue or seq[i] == "*":
            continue
        mutant = list(seq)
        mutant[i] = residue
        mutants.append(("".join(mutant), i, residue))
    return mutants

def generate_protein_windows(seq):
    windows = []
    for i in range(len(seq) - MAX_LEN + 1):
        window = seq[i:i+MAX_LEN]
        windows.append((window, i))
    return windows

# =====================
# ARGPARSE
# =====================
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="CSV: name (col1), sequence (col2)")
parser.add_argument("--mode", default="default",
                    choices=["default", "sat_mut", "scan_mut", "protein"])
parser.add_argument("--residue", default="A")
args = parser.parse_args()

# =====================
# LOAD MODEL
# =====================
checkpoint = torch.load("peptide_model_epoch50.pt", map_location=DEVICE, weights_only=False)

model = PeptideTransformer(vocab_size=len(AA_VOCAB)).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 🔥 only one scaler now
control_scaler = checkpoint["control_scaler"]

# =====================
# LOAD INPUT
# =====================
df = pd.read_csv(args.input)
names = df.iloc[:, 0].values
sequences = df.iloc[:, 1].values

all_sequences = []
all_names = []

# =====================
# MODE HANDLING
# =====================
for idx, seq in enumerate(sequences):
    name = names[idx]

    if args.mode == "default":
        all_sequences.append(seq)
        all_names.append(name)

    elif args.mode == "sat_mut":
        all_sequences.append(seq)
        all_names.append(name + "_WT")
        mutants = generate_sat_mut(seq)
        for m_seq, pos, aa in mutants:
            all_sequences.append(m_seq)
            all_names.append(f"{name}_{pos+1}{aa}")

    elif args.mode == "scan_mut":
        mutants = generate_scan_mut(seq, args.residue)
        for m_seq, pos, aa in mutants:
            if pos == -1:
                all_names.append(name + "_WT")
            else:
                all_names.append(f"{name}_{pos+1}{aa}")
            all_sequences.append(m_seq)

    elif args.mode == "protein":
        windows = generate_protein_windows(seq)
        for window, start in windows:
            all_sequences.append(window)
            all_names.append(f"{name}_{start+1}")

# =====================
# ENCODE
# =====================
X = torch.stack([encode(seq) for seq in all_sequences]).to(DEVICE)

# =====================
# PREDICT
# =====================
with torch.no_grad():
    preds = model(X).cpu().numpy()

# 🔥 inverse scale (single output)
control_pred = control_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

# =====================
# SAVE OUTPUT
# =====================
out = pd.DataFrame({
    "name": all_names,
    "sequence": all_sequences,
    "pred_controlPSI": control_pred
})

out.to_csv("predict_output.csv", index=False)
print("Saved predict_output.csv")