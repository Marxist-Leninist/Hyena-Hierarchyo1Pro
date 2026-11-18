import tempfile
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
from hyena_lm import CharacterLM, causal_fft, split_text, encode, generate, save_checkpoint, load_checkpoint, batch, loss_for

torch.set_num_threads(2)

class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = CharacterLM(4,context=16,width=8,layers=1,order=2)
    def test_fft_matches_direct(self):
        x,k = torch.randn(2,3,11),torch.randn(3,11)
        expected = F.conv1d(F.pad(x,(10,0)),k.flip(-1)[:,None],groups=3)
        torch.testing.assert_close(causal_fft(x,k),expected,atol=2e-6,rtol=1e-5)
    def test_causal_future_and_prefix(self):
        x = torch.randint(4,(2,16))
        changed = x.clone(); changed[:,8:] = (changed[:,8:]+1)%4
        a,b = self.model(x),self.model(changed)
        torch.testing.assert_close(a[:,:8],b[:,:8],atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(a[:,:8],self.model(x[:,:8]),atol=2e-6,rtol=1e-5)
    def test_gradients_reach_filter_and_gates(self):
        self.model(torch.randint(4,(2,16))).square().mean().backward()
        for p in self.model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(self.model.blocks[0].op.filter[0].weight.grad.abs().sum(),0)
    def test_shapes_all_lengths(self):
        for n in [1,2,7,16]:
            self.assertEqual(self.model(torch.zeros(2,n,dtype=torch.long)).shape,(2,n,4))
        with self.assertRaises(ValueError): self.model(torch.zeros(2,17,dtype=torch.long))
    def test_dataset_and_unknown(self):
        with self.assertRaises(ValueError): split_text("abc",16)
        with self.assertRaises(ValueError): encode("z",list("abcd"))
        a,b=split_text("a"*90+"b"*10,4)
        self.assertEqual((a,b),("a"*90,"b"*10))
    def test_generate_short_long_greedy(self):
        for prompt in ["a","abcd"*12]:
            result=generate(self.model,list("abcd"),prompt,5,0)
            self.assertEqual(len(result),len(prompt)+5)
            self.assertTrue(result.startswith(prompt))
        with self.assertRaises(ValueError): generate(self.model,list("abcd"),"",1)
    def test_checkpoint_logits_optimizer_and_rng(self):
        opt=torch.optim.AdamW(self.model.parameters())
        rng=torch.Generator().manual_seed(13)
        data=torch.tensor([0,1,2,3]*40)
        x,y=batch(data,16,2,rng,"cpu")
        loss_for(self.model,x,y).backward(); opt.step(); opt.zero_grad()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"model.pt"
            save_checkpoint(path,self.model,opt,list("abcd"),1,rng,{"test":True})
            other,state=load_checkpoint(path)
            torch.testing.assert_close(self.model(x),other(x))
            rng2=torch.Generator(); rng2.set_state(state["sampling_rng"])
            x1,y1=batch(data,16,2,rng,"cpu"); x2,y2=batch(data,16,2,rng2,"cpu")
            torch.testing.assert_close(x1,x2)
            opt2=torch.optim.AdamW(other.parameters()); opt2.load_state_dict(state["optimizer"])
            for model,optim,xx,yy in [(self.model,opt,x1,y1),(other,opt2,x2,y2)]:
                loss_for(model,xx,yy).backward(); optim.step()
            for p,q in zip(self.model.parameters(),other.parameters()):
                torch.testing.assert_close(p,q)
    def test_tiny_overfit(self):
        opt=torch.optim.AdamW(self.model.parameters(),lr=.02)
        x=torch.tensor([[0,1,2,3]*4]*4); y=(x+1)%4
        before=float(loss_for(self.model,x,y).detach())
        for _ in range(35):
            opt.zero_grad(); loss=loss_for(self.model,x,y); loss.backward(); opt.step()
        self.assertLess(float(loss_for(self.model,x,y).detach()),before*.3)

if __name__=="__main__": unittest.main()
