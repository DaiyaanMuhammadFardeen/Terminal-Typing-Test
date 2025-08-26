# run_char_rnn.py
import os, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 1) load text & vocab
text = open("./mark_twain_cleaned.txt", "r", encoding="utf-8").read()
chars = sorted(set(text))
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}
data = [stoi[c] for c in text]
vocab_size = len(chars)

# 2) dataset
class CharDataset(Dataset):
    def __init__(self, data, seq_len=100):
        self.data, self.seq_len = data, seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

# 3) model
class CharRNN(nn.Module):
    def __init__(self, vocab_size, emb=256, hid=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hid, num_layers=3, batch_first=True)
        self.fc = nn.Linear(hid, vocab_size)
    def forward(self, x, h=None):
        x = self.embedding(x)
        out, h = self.lstm(x, h)
        return self.fc(out), h

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = CharDataset(data, seq_len=100)
loader = DataLoader(dataset, batch_size=24, shuffle=True, drop_last=True,
                    pin_memory=True, num_workers=0)  # safer on ROCm

model = CharRNN(vocab_size).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
crit = nn.CrossEntropyLoss()

checkpoint = "mark_twain_story.pt"

# 4) train only if checkpoint not found
if os.path.exists(checkpoint):
    print(f"Loading existing model from {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
else:
    print("No checkpoint found, training model...")
    for epoch in range(10):
        model.train()
        pbar = tqdm(loader)
        avg_loss = 0.0
        num_batches = 0
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = crit(logits.view(-1, vocab_size), y.view(-1))
            # Update running average incrementally
            num_batches += 1
            avg_loss = avg_loss + (loss.item() - avg_loss) / num_batches
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.set_description(f"e{epoch} avg_loss {avg_loss:.4f}")
        torch.save(model.state_dict(), checkpoint)
        print(f"Saved checkpoint: {checkpoint}")
    torch.cuda.empty_cache()  # free VRAM
# 5) simple sample
def sample(model, start="The", length=200, temp=1.0):
    model.eval()
    idxs = [stoi[c] for c in start]
    token = torch.tensor([idxs], dtype=torch.long).to(device)
    h = None
    out = start
    with torch.no_grad():
        logits, h = model(token, h)
        last = token[0, -1].unsqueeze(0).unsqueeze(0)
        for _ in range(length):
            logits, h = model(last, h)
            logits = logits[:, -1, :] / temp
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            out += itos[nxt]
            last = torch.tensor([[nxt]], dtype=torch.long).to(device)
    return out

print(sample(model, length=200, temp=0.8))

