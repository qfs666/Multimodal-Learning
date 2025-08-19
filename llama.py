'''
Descripttion: 手搓LLaMA
version: 1.0
Date: 2025-08-19 10:15:24
LastEditTime: 2025-08-19 10:24:19
'''
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from modelscope import AutoTokenizer
import math
from typing import Optional
from rope import RotaryEmbedding

# ==========================
# Embedding + MHA + FFN
# ==========================
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model,padding_idx=151643)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor):
        return self.embed(x) * self.scale

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, use_rope: bool = True):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.att_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(self.d_k)

    def forward(self, query, key, value, mask: Optional[torch.Tensor] = None):
        B, L, _ = query.shape
        q = self.q_linear(query).view(B, L, self.num_heads, self.d_k).transpose(1,2)
        k = self.k_linear(key).view(B, L, self.num_heads, self.d_k).transpose(1,2)
        v = self.v_linear(value).view(B, L, self.num_heads, self.d_k).transpose(1,2)

        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)

        att = torch.matmul(q, k.transpose(-2,-1)) / self.scale
        if mask is not None:
            att = att.masked_fill(mask, torch.finfo(att.dtype).min)
        att = torch.softmax(att, dim=-1)
        att = self.att_drop(att)
        out = torch.matmul(att, v)
        out = out.transpose(1,2).contiguous().view(B,L,self.d_model)
        out = self.out_linear(out)
        out = self.out_drop(out)
        return out

class LLaMA_FFN(nn.Module):
    def __init__(self,d_model:int,d_hidden:int,dropout:float):
        super().__init__()
        self.w1 = nn.Linear(d_model,d_hidden)
        self.w2 = nn.Linear(d_model,d_hidden)
        self.w3 = nn.Linear(d_hidden,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(self.dropout(self.w1(x)*F.silu(self.w2(x)))))

class TransformerDecoderLayer(nn.Module):
    def __init__(self,d_model:int,d_hidden:int,num_heads:int,dropout:float=0.0):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model)
        self.mha = MultiHeadAttention(d_model,num_heads,dropout)
        self.ln2 = nn.RMSNorm(d_model)
        self.ffn = LLaMA_FFN(d_model,d_hidden,dropout)

    def forward(self,x,attn_mask=None):
        x = x + self.mha(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask)
        x = x + self.ffn(self.ln2(x))
        return x

# ==========================
# Transformer Decoder
# ==========================
class TransformerDecoder(nn.Module):
    def __init__(self,vocab_size,d_model,d_hidden,num_heads,num_layers,pad_token_id=0):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.num_heads = num_heads
        self.embed = TokenEmbedding(vocab_size,d_model)
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model,d_hidden,num_heads) for _ in range(num_layers)])

    def forward(self, input_ids):
        B,L = input_ids.shape
        x = self.embed(input_ids)

        # pad mask
        pad_mask = (input_ids==self.pad_token_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
        pad_mask = pad_mask.expand(B,self.num_heads,L,L)

        # causal mask
        causal_mask = torch.triu(torch.ones(L,L,dtype=torch.bool,device=input_ids.device),diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B,self.num_heads,L,L)

        attn_mask = pad_mask | causal_mask

        for layer in self.layers:
            x = layer(x,attn_mask)
        return x

class LLaMAModel(nn.Module):
    def __init__(self,vocab_size,d_model=128,d_hidden=512,num_heads=8,num_layers=2,pad_token_id=0):
        super().__init__()
        self.decoder = TransformerDecoder(vocab_size,d_model,d_hidden,num_heads,num_layers,pad_token_id)
        self.output_linear = nn.Linear(d_model,vocab_size)

    def forward(self,input_ids):
        dec_out = self.decoder(input_ids)
        logits = self.output_linear(dec_out)
        return logits

class QADataset(Dataset):
    def __init__(self, data, max_len=64):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        self.data = data
        self.max_len = max_len
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
    def __getitem__(self, idx):
        item = self.data[idx]
        q = item["question"]
        a = item["answer"]

        # 明确不加额外 special tokens，手动控制
        q_ids = self.tokenizer(q, add_special_tokens=False)["input_ids"]
        a_ids = self.tokenizer(a, add_special_tokens=False)["input_ids"]

        # 一个简单的分隔符：空格（也可换成固定模板，比如 "\nAnswer: " 的 token 序列）
        sep_s_ids = self.tokenizer("<Answer>", add_special_tokens=False)["input_ids"]
        sep_e_ids = self.tokenizer("</Answer>", add_special_tokens=False)["input_ids"]
        full_ids = q_ids + sep_s_ids + a_ids + sep_e_ids + [self.eos_id]

        # 截断
        full_ids = full_ids[:self.max_len]

        # labels：问题(+sep)位置 -100，答案+eos 保留；padding 全 -100
        q_len_with_sep = min(len(q_ids), len(full_ids))
        labels = [-100] * q_len_with_sep + full_ids[q_len_with_sep:]

        # padding
        pad_len = self.max_len - len(full_ids)
        input_ids = full_ids + [self.pad_id] * pad_len
        labels    = labels   + [-100] * pad_len

        return {
            "question": q,
            "answer":a,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self):
        return len(self.data)

def validate(model,dataset,device,max_seq_len = 64):

    tok = dataset.tokenizer
    model.eval()
    
    with torch.no_grad():
        for data in dataset:
            input_ids = torch.tensor([tok(data["question"], add_special_tokens=False)["input_ids"]]).to(device)
            que_len = len(input_ids[0])
            outs_ = []
            for i in range(max_seq_len):
                print(input_ids.shape)
                outs = model(input_ids)
                outputs = torch.argmax(outs[:,-1,:],dim=-1)
                print(outputs.item())
                if outputs.item()==tok.eos_token_id:
                    break
                input_ids = torch.cat([input_ids,outputs.unsqueeze(1)],dim=1)
                outs_.append(outputs.item())
            print("que:",data["question"])
            print("pred:",tok.decode(outs_))
            print("gt:",data["answer"])
            print("==================")

# ==========================
# Training Example
# ==========================
if __name__=="__main__":
    data = [
        {"question":"What is the capital of France?","answer":"Paris."},
        {"question":"Who wrote Hamlet?","answer":"William Shakespeare."},
        {"question":"What is 2+2?","answer":"4."},
        {"question":"Name the largest planet in the Solar System.","answer":"Jupiter."},
        {"question":"Who painted the Mona Lisa?","answer":"Leonardo da Vinci."},
    ]

    dataset = QADataset(data)
    dataloader = DataLoader(dataset,batch_size=2,shuffle=True)

    vocab_size = dataset.tokenizer.vocab_size+1
    model = LLaMAModel(vocab_size).cuda() if torch.cuda.is_available() else LLaMAModel(vocab_size)
    device = next(model.parameters()).device

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for epoch in range(3000):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)   # [B, L, V]
            loss = criterion(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1)
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.10f}")
        validate(model,dataset,device)
