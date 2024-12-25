import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import string

########################################################################
# Check GPU availability
########################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

########################################################################
# Text Dataset (character-level)
########################################################################
class TextDataset(Dataset):
    """
    Creates samples of length 'seq_len' from a list of encoded tokens.
    The next token in the sequence is the 'target' for each position
    (language modeling approach).
    """
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        # Each sample is 'seq_len' tokens, and we predict the *next* token
        # So we need at least (seq_len+1) tokens.
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # x: [idx .. idx+seq_len-1], y: [idx+1 .. idx+seq_len]
        x_seq = self.data[idx : idx + self.seq_len]
        y_seq = self.data[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x_seq, dtype=torch.long), torch.tensor(y_seq, dtype=torch.long)

########################################################################
# Hyena Block (deeper variant with dropout + layernorm)
########################################################################
class HyenaBlock(nn.Module):
    """
    A single 'Hyena-style' block:
      - Linear -> Conv1D -> Linear
      - Dropout
      - LayerNorm
    This block operates on shape: (batch_size, seq_len, hidden_dim).
    """
    def __init__(self, hidden_dim, filter_size=3, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.conv1d = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=filter_size,
            padding=filter_size // 2
        )
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, hidden_dim)
        """
        # 1) Linear
        x = self.linear1(x)

        # 2) Conv1d expects (batch_size, channels, seq_len)
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, seq_len)
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # back to (batch_size, seq_len, hidden_dim)

        # 3) Another linear
        x = self.linear2(x)

        # 4) Dropout
        x = self.dropout(x)

        # 5) LayerNorm
        x = self.layernorm(x)
        return x

########################################################################
# Deep Hyena model
########################################################################
class DeepHyena(nn.Module):
    """
    A deeper Hyena-like model:
      - Embedding
      - Stacked HyenaBlocks
      - Final linear to get logits for each token
    """
    def __init__(self, vocab_size, embed_dim=128, num_layers=4, filter_size=3, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Stacked blocks
        self.blocks = nn.ModuleList([
            HyenaBlock(embed_dim, filter_size=filter_size, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Final linear: from hidden_dim -> vocab
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len) of token IDs
        returns: (batch_size, seq_len, vocab_size)
        """
        # 1) Embed tokens: (batch_size, seq_len, embed_dim)
        x = self.embedding(x)

        # 2) Pass through each HyenaBlock
        for layer in self.blocks:
            x = layer(x)

        # 3) Final linear -> (batch_size, seq_len, vocab_size)
        logits = self.fc_out(x)
        return logits

########################################################################
# Training step function
########################################################################
def train_for_n_epochs(model, dataloader, criterion, optimizer, start_epoch, end_epoch):
    """
    Train from epoch 'start_epoch+1' up to 'end_epoch' (inclusive).
    Prints timing and final loss. Returns the last loss from the block.
    """
    model.train()
    last_loss = None
    for epoch in range(start_epoch + 1, end_epoch + 1):
        epoch_start_time = time.time()
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            # model output: (batch_size, seq_len, vocab_size)
            logits = model(x_batch)
            # Flatten predictions & targets for cross-entropy
            # shape: (batch_size * seq_len, vocab_size)
            # shape: (batch_size * seq_len)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y_batch.view(-1)
            )
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

        epoch_time = time.time() - epoch_start_time
        print(f"  [Epoch {epoch}/{end_epoch}] time={epoch_time:.2f}s | last-batch-loss={last_loss:.4f}")

    print(f"Finished epochs {start_epoch+1} to {end_epoch}; final last-batch-loss={last_loss:.4f}")
    return last_loss

########################################################################
# Generation function
########################################################################
@torch.no_grad()
def generate_text(model, seed_text, length, char_to_idx, idx_to_char, seq_len=70):
    """
    Generate 'length' tokens from the model, given a seed text.
    For each step, we run a forward pass on the last 'seq_len' tokens.
    This is a naive approach: we feed 1 token at a time in an autoregressive loop.
    """
    model.eval()

    # Convert seed text to token IDs
    seed_ids = [char_to_idx.get(ch, 0) for ch in seed_text.lower()]
    # We'll keep a buffer of the last 'seq_len' tokens (like an RNN BPTT approach).
    context = seed_ids[-seq_len:]  # if seed is shorter than seq_len, that's okay
    context = ([0] * (seq_len - len(context))) + context  # pad from the left if needed

    generated = list(seed_ids)  # so we can show the entire sequence

    for _ in range(length):
        # We'll create a batch_size=1 input
        inp = torch.tensor([context], dtype=torch.long, device=device)
        # Forward pass -> shape: (1, seq_len, vocab_size)
        logits = model(inp)
        # We only want the last position's distribution: (1, vocab_size)
        last_logits = logits[:, -1, :]  # shape (1, vocab_size)
        probs = nn.functional.softmax(last_logits, dim=-1).squeeze(0).cpu().numpy()
        next_id = np.random.choice(len(probs), p=probs)
        generated.append(next_id)

        # shift context
        context = context[1:] + [next_id]

    # Convert generated IDs back to chars
    out_str = "".join(idx_to_char[id_] for id_ in generated)
    return out_str

########################################################################
# Main script
########################################################################
def main():
    ####################################################################
    # Hard-coded text file path
    ####################################################################
    text_file = (
        r"C:\Users\Scott\Downloads\Hyena-Hierarchyo1Pro-master\Hyena-Hierarchyo1Pro-master\o1Prochatdata.txt"
    )

    ####################################################################
    # Hyperparameters
    ####################################################################
    seq_len = 70       # sequence length for training
    embed_dim = 128    # embedding size
    num_layers = 4     # how many HyenaBlocks
    filter_size = 3
    dropout = 0.1
    lr = 0.001
    batch_size = 64    # reduce if you get memory issues
    initial_end_epoch = 1  # we'll double this
    generated_sample_length = 200  # length of text to generate each time

    ####################################################################
    # Load data & create char->idx
    ####################################################################
    with open(text_file, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    text = text.lower()

    # Unique characters
    chars = sorted(list(set(text)))  # sorting for stable index
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}

    print(f"Vocabulary size: {vocab_size}")

    # Encode entire text -> list of IDs
    encoded_text = [char_to_idx[ch] for ch in text]

    ####################################################################
    # Create dataset & dataloader
    ####################################################################
    dataset = TextDataset(encoded_text, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    ####################################################################
    # Build model
    ####################################################################
    model = DeepHyena(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        filter_size=filter_size,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ####################################################################
    # Indefinite doubling schedule
    ####################################################################
    print("==== Starting Indefinite Doubling Training ====")
    start_epoch = 0
    end_epoch = initial_end_epoch

    try:
        while True:
            block_start_time = time.time()

            # Train from (start_epoch+1) to end_epoch
            _ = train_for_n_epochs(model, dataloader, criterion, optimizer, start_epoch, end_epoch)

            block_end_time = time.time()
            block_duration = block_end_time - block_start_time
            print(f" -> Block {start_epoch+1} to {end_epoch} took {block_duration:.2f}s")

            # Generate text from a sample seed
            seed_text = "The quick brown fox"
            sample_output = generate_text(
                model,
                seed_text=seed_text,
                length=generated_sample_length,
                char_to_idx=char_to_idx,
                idx_to_char=idx_to_char,
                seq_len=seq_len
            )
            print(f" => Generated text after {end_epoch} epoch(s):\n{sample_output}\n")

            # Prepare for next doubling
            start_epoch = end_epoch
            end_epoch *= 2
            print(f"==== Next training block will be from epoch {start_epoch+1} to {end_epoch} ====\n")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl + C). Exiting gracefully...")

if __name__ == "__main__":
    main()
