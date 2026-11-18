# Hyena Hierarchy

This code trains a character-level language model using a variant of a convolutional neural network architecture called Hyena. The trained model can be used to generate random text given a seed string.

### Requirements

To run this code, you need to have Python 3 and the following packages installed:

- torch

You can install these packages by running `pip install -r requirements.txt`.

### Usage

To train the model, run `train_hyena_model()` with the following parameters:

- `text_file`: Path to a text file used for training the model.
- `input_dim`: The number of characters to feed into the model at once.
- `output_dim`: The number of output classes (characters) of the model.
- `filter_size`: The size of the convolutional filters used in the model.
- `depth`: The number of convolutional filters in the model.
- `positional_dim`: The dimensionality of the output of the convolutional layer.
- `lr`: The learning rate used during optimization.
- `num_epochs`: The number of epochs to train the model.

The function returns the trained model, a list of characters in the vocabulary, and a dictionary that maps characters to their indices in the vocabulary.

To generate random text using the trained model, run `generate_text()` with the following parameters:

- `model`: The trained model.
- `seed_text`: The seed string used to start text generation.
- `length`: The length of the text to generate.
- `char_to_idx`: The dictionary that maps characters to their indices in the vocabulary.
- `idx_to_char`: The dictionary that maps indices in the vocabulary to characters.
- `vocab`: The number of characters in the vocabulary.

### Example

You can run the example code in `main()` to train the model on a randomly generated text and generate random text given a seed string.


## Hyena Model code overview

### Text Dataset

- $\text{TextDataset}(text, seq\_len)$: Initializes the text dataset with the given `text` and `seq_len`.
    - `text` (string): input text.
    - `seq_len` (int): sequence length.

- `__len__()`: Returns the length of the dataset.

- `__getitem__(index)`: Returns the tensor of the sequence and target at the given `index`.
    - `index` (int): index of the sequence.

### Hyena Model

- $\text{Hyena}(input\_dim, output\_dim, filter\_size, depth, positional\_dim)$: Initializes the Hyena model with the given parameters.
    - `input_dim` (int): input dimension.
    - `output_dim` (int): output dimension.
    - `filter_size` (int): filter size for convolution.
    - `depth` (int): depth of the model.
    - `positional_dim` (int): positional dimension of the model.

- `forward(x)`: Computes the forward pass of the Hyena model with the given input tensor `x`.
    - `x` (tensor): input tensor.

### Training Hyena Model

- `train_hyena_model(text_file, input_dim, filter_size, depth, positional_dim, lr, num_epochs, batch_size=128)`: Trains the Hyena model with the given parameters and returns the trained model, character list, and character-to-index dictionary.
    - `text_file` (string): input text file path.
    - `input_dim` (int): input dimension.
    - `filter_size` (int): filter size for convolution.
    - `depth` (int): depth of the model.
    - `positional_dim` (int): positional dimension of the model.
    - `lr` (float): learning rate.
    - `num_epochs` (int): number of epochs.
    - `batch_size` (int): batch size

### Text Generation

- `generate_text(model, seed_text, length, char_to_idx, idx_to_char, vocab)`: Generates text using the trained Hyena model with the given parameters.
    - `model` (Hyena): trained Hyena model.
    - `seed_text` (string): seed text.
    - `length` (int): length of generated text.
    - `char_to_idx` (dict): character-to-index dictionary.
    - `idx_to_char` (dict): index-to-character dictionary.
    - `vocab` (int): vocabulary size.
    - `input_dim` (int): input dimension

### Main Function

- `main()`: Runs the main function which generates random text, trains the Hyena model, and generates text using the trained model.


# Credits

This code is inspired by the papers 
## [A Convolutional Neural Network for Modelling Sentences"](https://arxiv.org/abs/1404.2188)
 by Nal Kalchbrenner, Edward Grefenstette, and Phil Blunsom.


## [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/pdf/2302.10866.pdf)
by: Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, Christopher Ré

Affiliations:
1. Department of Computer Science, Stanford University, Stanford, CA, USA
2. Mila - Quebec AI Institute and DIRO, Université de Montréal, Montréal, QC, Canada

Below is a **Kali Linux**–friendly version of the same script. The primary difference is simply using a Linux file path (rather than a Windows `C:\...` path). Everything else—model definition, data loading, training loop—is platform-agnostic, so it works the same on Kali (or most Linux distros) as long as you have **Python**, **PyTorch**, and other required packages installed.

---

```python
#!/usr/bin/env python3
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import string
import os

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
    return last_loss

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
            seed_input = seed_indices.float().unsqueeze(0)
            outputs = model(seed_input)
            # Probability over vocab for the next character
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
    # Example file path for Kali or Linux
    # (Adjust this to wherever your dataset is placed)
    ####################################################################
    text_file = r"/root/o1Prochatdata.txt"
    # Alternatively: "/home/kali/o1Prochatdata.txt" or any path you prefer

    ####################################################################
    # Hyperparameters
    ####################################################################
    input_dim = 70     
    filter_size = 3
    depth = 3
    lr = 0.001
    batch_size = 128
    
    ####################################################################
    # Load data & build vocab
    ####################################################################
    if not os.path.isfile(text_file):
        print(f"ERROR: The file '{text_file}' does not exist.")
        print("Please adjust text_file path to an actual text dataset.")
        return

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
    # TRAIN indefinitely with doubling epochs
    ####################################################################
    print("==== Starting Indefinite Doubling Training ====")
    start_epoch = 0
    end_epoch = 1  # We'll double this each loop

    try:
        while True:
            block_start = time.time()
            
            # Train from (start_epoch+1) to end_epoch
            last_loss = train_for_n_epochs(
                model, dataloader, criterion, optimizer,
                start_epoch, end_epoch
            )
            
            block_end = time.time()
            duration = block_end - block_start
            print(f" -> Block {start_epoch+1} to {end_epoch} took {duration:.2f}s")
            
            # Generate text after this block
            seed_text = "The quick brown fox"
            generated_length = 70
            generated_output = generate_text(
                model, seed_text, generated_length,
                char_to_idx, idx_to_char, vocab_size, input_dim
            )
            print(f" => Generated text after {end_epoch} epoch(s):")
            print(generated_output)
            print()
            
            # Prepare for the next doubling
            start_epoch = end_epoch
            end_epoch *= 2  # double the epoch count
            print("==== Next training block will be from epoch "
                  f"{start_epoch+1} to {end_epoch} ====\n")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting gracefully...")

if __name__ == '__main__':
    main()
```

---

# Running on Kali Linux

1. **Install Python & PyTorch**  
   - Make sure Python 3 is installed (Kali usually has it by default).  
   - Install PyTorch (CPU or CUDA version) via pip:
     ```bash
     pip3 install torch torchvision torchaudio
     ```
   - Or install from conda if you use the Anaconda/Miniconda environment.

2. **Place Your Data File**  
   - Copy your text file to `/root/o1Prochatdata.txt` (or whichever path you set in the script).  

3. **Save This Script**  
   - For example, call it `hyena_train_kali.py`.  

4. **Run the Script**  
   ```bash
   python3 hyena_train_kali.py
   ```
   - If everything is correct, it will read `/root/o1Prochatdata.txt`, build a vocabulary, start training, and periodically generate text.  

5. **Notes/Customizations**:  
   - Change `text_file` to whatever path you like (e.g., `/home/kali/my_data.txt`).  
   - Tune **hyperparameters** (learning rate, batch size, etc.) to your system.  
   - If you have an NVIDIA GPU with CUDA drivers installed, the script will automatically use it. Otherwise it will fall back to CPU, which can be slower.  
   - The indefinite doubling schedule (1 → 2 → 4 → 8 epochs, etc.) is just a demonstration. You can adapt it to your needs.

With this setup on **Kali Linux**, you’ll be able to run the Hyena text model training script just as you would on Windows, macOS, or any other Linux distro—only the file path changed to a Linux location.

