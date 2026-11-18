"""Small educational Hyena-inspired character LM; not a paper reproduction."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def causal_fft(x, kernel):
    """Depthwise linear convolution. x=B,C,L; kernel=C,L."""
    length = x.shape[-1]
    n = 2 * length
    return torch.fft.irfft(
        torch.fft.rfft(x, n=n) * torch.fft.rfft(kernel, n=n), n=n
    )[..., :length]


class HyenaOperator(nn.Module):
    def __init__(self, width, context, order=2):
        super().__init__()
        self.width, self.context, self.order = width, context, order
        self.project = nn.Linear(width, width * (order + 1))
        self.short = nn.Conv1d(width * (order + 1), width * (order + 1),
                               3, groups=width * (order + 1), padding=2)
        self.filter = nn.Sequential(nn.Linear(5, width), nn.SiLU(),
                                    nn.Linear(width, order * width))
        self.decay = nn.Parameter(torch.zeros(order, width))
        self.skip = nn.Parameter(torch.zeros(order, width))
        self.output = nn.Linear(width, width)

    def forward(self, x):
        length = x.shape[1]
        # Normalize against fixed maximum context, never current length:
        # otherwise appending future tokens would change earlier filters.
        t = torch.arange(length, device=x.device, dtype=x.dtype) / self.context
        pos = torch.stack([t, torch.sin(2*math.pi*t), torch.cos(2*math.pi*t),
                           torch.sin(4*math.pi*t), torch.cos(4*math.pi*t)], -1)
        kernels = self.filter(pos).reshape(length, self.order, self.width).permute(1,2,0)
        kernels = kernels * torch.exp(-F.softplus(self.decay)[...,None] * t)
        parts = self.short(self.project(x).transpose(1,2))[...,:length].chunk(self.order+1,1)
        value = parts[-1]
        for i, gate in enumerate(reversed(parts[:-1])):
            value = causal_fft(value, kernels[i]) + self.skip[i][None,:,None] * value
            value = value * gate
        return self.output(value.transpose(1,2))


class Block(nn.Module):
    def __init__(self, width, context, order):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.op = HyenaOperator(width, context, order)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width,width))
    def forward(self, x):
        x = x + self.op(self.norm1(x))
        return x + self.ff(self.norm2(x))


class CharacterLM(nn.Module):
    def __init__(self, vocab_size, context=64, width=32, layers=2, order=2):
        super().__init__()
        if min(vocab_size,context,width,layers,order) < 1:
            raise ValueError("all model dimensions must be positive")
        self.config = dict(vocab_size=vocab_size,context=context,width=width,layers=layers,order=order)
        self.embedding = nn.Embedding(vocab_size,width)
        self.blocks = nn.Sequential(*(Block(width,context,order) for _ in range(layers)))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width,vocab_size)
    def forward(self, tokens):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config["context"]:
            raise ValueError("expected B,L tokens with 1 <= L <= context")
        return self.head(self.norm(self.blocks(self.embedding(tokens))))


def split_text(text, context, fraction=.1):
    cut = int(len(text)*(1-fraction))
    train, val = text[:cut], text[cut:]
    if min(len(train),len(val)) <= context:
        raise ValueError("both disjoint text splits must contain more than context characters")
    return train, val


def encode(text, chars):
    mapping = {c:i for i,c in enumerate(chars)}
    unknown = sorted(set(text)-set(chars))
    if unknown:
        raise ValueError("text contains characters absent from checkpoint vocabulary: "+repr(unknown))
    return torch.tensor([mapping[c] for c in text],dtype=torch.long)


def batch(data, context, size, rng, device):
    starts = torch.randint(len(data)-context,(size,),generator=rng)
    x = torch.stack([data[i:i+context] for i in starts])
    y = torch.stack([data[i+1:i+context+1] for i in starts])
    return x.to(device),y.to(device)


def loss_for(model, x, y):
    return F.cross_entropy(model(x).flatten(0,1),y.flatten())


def save_checkpoint(path, model, optimizer, chars, step, rng, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    state = dict(format_version=1,config=model.config,chars=chars,step=step,
                 model=model.state_dict(),optimizer=optimizer.state_dict(),
                 sampling_rng=rng.get_state(),torch_rng=torch.get_rng_state(),
                 metadata=metadata)
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    tmp = path.with_name(path.name+".tmp")
    torch.save(state,tmp)
    os.replace(tmp,path)


def load_checkpoint(path, device="cpu"):
    state = torch.load(path,map_location="cpu",weights_only=True)
    if state["format_version"] != 1:
        raise ValueError("unsupported checkpoint format")
    model = CharacterLM(**state["config"]).to(device)
    model.load_state_dict(state["model"])
    return model,state


@torch.no_grad()
def generate(model, chars, prompt, length=100, temperature=.8):
    if not prompt or length < 0 or temperature < 0:
        raise ValueError("prompt must be nonempty; length/temperature must be nonnegative")
    ids = encode(prompt,chars).tolist()
    device = next(model.parameters()).device
    model.eval()
    for _ in range(length):
        x = torch.tensor([ids[-model.config["context"]:]],device=device)
        logits = model(x)[0,-1]
        token = logits.argmax() if temperature == 0 else torch.multinomial(
            (logits/temperature).softmax(-1),1)[0]
        ids.append(int(token))
    return "".join(chars[i] for i in ids)


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    text = Path(args.text).read_text(encoding="utf8")
    digest = hashlib.sha256(text.encode()).hexdigest()
    rng = torch.Generator().manual_seed(args.seed)
    if args.resume:
        model,state = load_checkpoint(args.resume,args.device)
        if state["metadata"]["data_sha256"] != digest:
            raise ValueError("resume requires identical dataset")
        chars,step = state["chars"],state["step"]
        rng.set_state(state["sampling_rng"])
        torch.set_rng_state(state["torch_rng"])
        if args.device.startswith("cuda") and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    else:
        chars = sorted(set(text))
        model = CharacterLM(len(chars),args.context,args.width,args.layers,args.order).to(args.device)
        step = 0
    context = model.config["context"]
    train_text,val_text = split_text(text,context)
    train_ids,val_ids = encode(train_text,chars),encode(val_text,chars)
    opt = torch.optim.AdamW(model.parameters(),lr=args.lr)
    if args.resume:
        opt.load_state_dict(state["optimizer"])
    # Validation batches are fixed and drawn from a separate RNG, no training RNG consumption.
    eval_rng = torch.Generator().manual_seed(1729)
    eval_batches = [batch(val_ids,context,args.batch_size,eval_rng,args.device) for _ in range(4)]
    metadata = dict(data_sha256=digest,train_characters=len(train_text),
                    validation_characters=len(val_text),seed=args.seed,
                    description="educational Hyena-inspired character language model",
                    batch_size=args.batch_size,learning_rate=opt.param_groups[0]["lr"])
    log = []
    def evaluate():
        model.eval()
        with torch.no_grad():
            result = sum(float(loss_for(model,x,y)) for x,y in eval_batches)/len(eval_batches)
        model.train()
        return result
    log.append(dict(step=step,validation_loss=evaluate()))
    try:
        for _ in range(args.steps):
            model.train()
            x,y = batch(train_ids,context,args.batch_size,rng,args.device)
            opt.zero_grad(set_to_none=True)
            loss = loss_for(model,x,y)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            opt.step()
            step += 1
            if step % args.save_every == 0:
                log.append(dict(step=step,training_loss=float(loss.detach()),validation_loss=evaluate()))
                print(json.dumps(log[-1]),flush=True)
                save_checkpoint(args.output,model,opt,chars,step,rng,metadata)
    finally:
        # Save current state on Ctrl-C or error; interruption mid-update is not an exact replay boundary.
        save_checkpoint(args.output,model,opt,chars,step,rng,metadata)
        log.append(dict(step=step,validation_loss=evaluate()))
        Path(args.output+".metrics.json").write_text(json.dumps(log,indent=2)+"\n")
    print(json.dumps(log[-1]))
    print("parameters:",sum(p.numel() for p in model.parameters()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command",required=True)
    t = sub.add_parser("train")
    t.add_argument("--text",required=True)
    t.add_argument("--output",default="checkpoints/latest.pt")
    t.add_argument("--resume")
    for name,default in [("context",64),("width",32),("layers",2),("order",2),
                         ("steps",200),("batch-size",16),("seed",42),("threads",2),("save-every",50)]:
        t.add_argument("--"+name,type=int,default=default)
    t.add_argument("--lr",type=float,default=.001)
    t.add_argument("--device",default="cpu")
    g = sub.add_parser("generate")
    g.add_argument("--checkpoint",required=True)
    g.add_argument("--prompt",required=True)
    g.add_argument("--length",type=int,default=100)
    g.add_argument("--temperature",type=float,default=.8)
    g.add_argument("--seed",type=int,default=42)
    args = p.parse_args()
    if args.command == "train":
        if min(args.steps,args.batch_size,args.threads,args.save_every) < 1:
            p.error("steps, batch size, threads and save interval must be positive")
        train(args)
    else:
        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        model,state = load_checkpoint(args.checkpoint)
        print(generate(model,state["chars"],args.prompt,args.length,args.temperature))

if __name__ == "__main__":
    main()
