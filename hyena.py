import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import string

########################################################################
# Device setup
########################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

########################################################################
# Dataset definition
########################################################################
class TextDataset(Dataset):
    def __init__(self, text, seq_len):
        self.text = text
        self.seq_len = seq_len

    def __len__(self):
        return len(self.text) - self.seq_len

    def __getitem__(self, index):
        x = torch.tensor(self.text[index : index + self.seq_len])
        y = torch.tensor(self.text[index + self.seq_len])
        return x, y

########################################################################
# Hyena model (shape-fixed to avoid mismatch)
########################################################################
class Hyena(nn.Module):
    def __init__(self, input_dim, output_dim, filter_size, depth):
        super(Hyena, self).__init__()
        self.depth = depth
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.filter_size = filter_size
        
        # Linear: maps input_dim -> (output_dim+1)*depth
        self.linear1 = nn.Linear(input_dim, (output_dim + 1) * depth)
        
        # Conv1d: operates over 'depth' channels
        self.conv1d = nn.Conv1d(
            in_channels=depth,
            out_channels=depth,
            kernel_size=filter_size,
            padding=filter_size // 2
        )
        
        # Linear: from flattened dimension -> output_dim
        # After conv1d, sequence length is (output_dim + 1),
        # so flatten dimension is depth * (output_dim + 1).
        self.linear2 = nn.Linear(depth * (output_dim + 1), output_dim)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = x.float()
        
        # 1) Linear -> shape (batch_size, (output_dim+1)*depth)
        x = self.linear1(x)
        
        # 2) Reshape -> (batch_size, (output_dim+1), depth)
        #    then transpose -> (batch_size, depth, (output_dim+1))
        x = x.view(x.size(0), (self.output_dim + 1), self.depth).transpose(1, 2)
        
        # 3) Convolution -> shape remains (batch_size, depth, (output_dim+1))
        x = self.conv1d(x)
        
        # 4) Transpose -> (batch_size, (output_dim+1), depth), then flatten
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1)
        
        # 5) Final linear -> (batch_size, output_dim)
        x = self.linear2(x)
        return x

########################################################################
# Train function for a given epoch range
########################################################################
def train_for_n_epochs(model, dataloader, criterion, optimizer, start_epoch, end_epoch):
    """
    Trains 'model' from epoch 'start_epoch+1' up to 'end_epoch' inclusive.
    Prints timing and the last batch's loss each epoch.
    """
    model.train()
    last_loss = None

    for epoch in range(start_epoch + 1, end_epoch + 1):
        epoch_start_time = time.time()
        for seqs, targets in dataloader:
            seqs, targets = seqs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
        
        epoch_time = time.time() - epoch_start_time
        print(f"  [Epoch {epoch}/{end_epoch}] time={epoch_time:.2f}s | last-batch-loss={last_loss:.4f}")

    print(f"Finished epochs {start_epoch+1} to {end_epoch}; final last-batch-loss={last_loss:.4f}")

########################################################################
# Generation function
########################################################################
def generate_text(model, seed_text, length, char_to_idx, idx_to_char, vocab, input_dim):
    """
    Generates 'length' new characters from model, beginning with 'seed_text'.
    """
    model.eval()
    with torch.no_grad():
        # Convert seed text to indices, fallback to random for unknown chars
        seed_indices = torch.LongTensor([
            char_to_idx.get(c, random.randint(0, vocab - 1))
            for c in seed_text.lower()
        ])
        
        # Pad if seed is shorter than input_dim
        if len(seed_indices) < input_dim:
            seed_indices = torch.cat([
                seed_indices,
                torch.zeros(input_dim - len(seed_indices), dtype=torch.long)
            ])
        seed_indices = seed_indices.to(device)
        
        out_chars = []
        for _ in range(length):
            seed_input = seed_indices.float().unsqueeze(0).to(device)
            outputs = model(seed_input)
            probs = nn.functional.softmax(outputs[-1], dim=0).cpu().numpy()
            next_idx = np.random.choice(len(probs), p=probs)
            out_chars.append(idx_to_char[next_idx])
            # Shift seed by 1
            seed_indices[:-1] = seed_indices[1:].clone()
            seed_indices[-1] = next_idx
    
    return ''.join(out_chars)

########################################################################
# Main script
########################################################################
def main():
    ####################################################################
    # Hard-coded path to the text file add yours here
    ####################################################################
    text_file = (
        r"C:\Users\Downloads\Hyena-Hierarchyo1Pro-master\Hyena-Hierarchyo1Pro-master\o1Prochatdata.txt"
    )

    ####################################################################
    # Hyperparameters
    ####################################################################
    input_dim = 70     
    filter_size = 3
    depth = 3
    lr = 0.001
    batch_size = 128
    
    # Doubling schedule for epochs
    epoch_schedule = [1, 2, 4, 8]  # feel free to extend if needed

    ####################################################################
    # Load data & build vocab
    ####################################################################
    with open(text_file, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    text = text.lower()
    
    # Unique characters
    chars = list(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    vocab_size = len(chars)
    
    # Encode entire text
    encoded_text = [char_to_idx[ch] for ch in text]
    
    # Dataset & DataLoader
    dataset = TextDataset(encoded_text, input_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)

    ####################################################################
    # Build model & training objects
    ####################################################################
    model = Hyena(input_dim, vocab_size, filter_size, depth).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ####################################################################
    # TRAIN using the doubling schedule
    ####################################################################
    print("==== Starting Doubling-Schedule Training ====")
    start_epoch = 0
    for end_epoch in epoch_schedule:
        block_start = time.time()
        train_for_n_epochs(model, dataloader, criterion, optimizer, start_epoch, end_epoch)
        block_end = time.time()

        duration = block_end - block_start
        print(f" -> Block {start_epoch+1} to {end_epoch} took {duration:.2f}s")

        # After each block of epochs, generate text
        seed_text = "The quick brown fox"
        generated_length = 70
        generated_output = generate_text(
            model, seed_text, generated_length,
            char_to_idx, idx_to_char, vocab_size, input_dim
        )
        print(f" => Generated text after {end_epoch} epoch(s): {generated_output}\n")

        start_epoch = end_epoch  # move to next doubling

    print("==== All scheduled training complete. ====")

if __name__ == '__main__':
    main()
message.txt
9 KB
