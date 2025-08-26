import os
import pickle
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class TextGenerator:
    def __init__(self, pkl_file="./TextGens/mark_twain_preprocessed.pkl", checkpoint="./TextGens/mark_twain_story.pt"):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        # Load preprocessed data
        if not os.path.exists(pkl_file):
            
            sys.exit(1)
        
        with open(pkl_file, "rb") as f:
            saved_data = pickle.load(f)
        self.chars = saved_data["chars"]
        self.stoi = saved_data["stoi"]
        self.itos = saved_data["itos"]
        self.data = saved_data["data"]
        self.vocab_size = saved_data["vocab_size"]

        # Initialize and load model
        
        self.model = CharRNN(self.vocab_size).to(self.device)
        if not os.path.exists(checkpoint):
            
            sys.exit(1)
        
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.model.eval()

    def generate(self, start="The ", sentences=5, temp=1.0, max_length=1000):
        idxs = [self.stoi[c] for c in start]
        token = torch.tensor([idxs], dtype=torch.long).to(self.device)
        h = None
        out = start
        sentence_count = 0
        char_count = 0
        with torch.no_grad():
            logits, h = self.model(token, h)
            last = token[0, -1].unsqueeze(0).unsqueeze(0)
            while sentence_count < sentences and char_count < max_length:
                logits, h = self.model(last, h)
                logits = logits[:, -1, :] / temp
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1).item()
                next_char = self.itos[nxt]
                out += next_char
                char_count += 1
                if next_char in ['.', '!', '?', ';', ':', '\n']:
                    sentence_count += 1
                last = torch.tensor([[nxt]], dtype=torch.long).to(self.device)
        return out
