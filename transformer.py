'''
Descripttion: 手搓transformer
version: 1.0
Date: 2025-08-14 11:36:07
LastEditTime: 2025-08-19 10:25:01
'''
import math
import torch
from torch import nn
from typing import Optional

class PositionEncoding(nn.Module):
    def __init__(self, d_model: int ,max_sqe_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros((max_sqe_len,d_model),dtype = torch.float32)

        position = torch.arange(0,max_sqe_len,dtype=torch.float32).unsqueeze(1)
        div_item = torch.exp(torch.arange(0,d_model,2,dtype=torch.float32)*-math.log(10000)/d_model)

        pe[:,::2] = torch.sin(position*div_item)
        pe[:,1::2] = torch.cos(position*div_item)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe",pe)

    def forward(self, x: torch.Tensor):
        '''
            [batch,seq_len,dim]
        '''
        batch,seq_len,_ = x.shape
        return x + self.pe[:,:seq_len,:]


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,d_model)
        self.scale = math.sqrt(d_model)

    def forward(self,x: torch.Tensor):
        '''
            [batch,seq_len]
        '''
        return self.embed(x)*self.scale


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)
        self.q_linear = nn.Linear(d_model,d_model)
        self.k_linear = nn.Linear(d_model,d_model)
        self.v_linear = nn.Linear(d_model,d_model)
        self.out_linear = nn.Linear(d_model,d_model)

        self.att_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor]=None):
        batch, que_len, _ = query.shape

        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)
        
        query = query.view(batch,-1,self.num_heads,self.d_k).transpose(1,2)
        key = key.view(batch,-1,self.num_heads,self.d_k).transpose(1,2)
        value = value.view(batch,-1,self.num_heads,self.d_k).transpose(1,2)

        att_ = torch.matmul(query,key.transpose(-2,-1))/self.scale    # [batch,num_heads,que_len,key_len]

        if mask is not None:
            att_ = att_.masked_fill(mask,torch.finfo(att_.dtype).min)

        att_score = torch.softmax(att_,dim=-1)
        att_score = self.att_drop(att_score)
        att_out = torch.matmul(att_score,value)

        att_out = att_out.transpose(1,2).contiguous().view(batch,que_len,self.d_model)
        att_out = self.out_linear(att_out)
        att_out = self.out_drop(att_out)
        return att_out


class FeedForward(nn.Module):
    def __init__(self,d_model: int, d_hidden: int, dropout: float = 0.0, activation: str = "gelu"):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(d_model,d_hidden),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden,d_model),
            nn.Dropout(dropout) 
        )
    
    def forward(self,x: torch.Tensor):
        return self.ffn(x)



def generate_pad_mask(q_tokens: torch.Tensor,k_tokens: torch.Tensor,pad_token_id: int,num_heads: int):
    '''
        q_tokens: [batch,q_len],
        k_tokens: [batch,k_len]
    '''
    batch,q_len = q_tokens.shape
    batch,k_len = k_tokens.shape
    mask = (k_tokens==pad_token_id).unsqueeze(1).unsqueeze(1)
    return mask.expand(batch,num_heads,q_len,k_len).to(q_tokens.device)

def generate_causal_mask(target_tokens: torch.Tensor,num_heads: int):
    '''
        target_tokens: [batch, target_len]
    '''
    batch, target_len = target_tokens.shape
    causal_mask = torch.triu(torch.ones(target_len,target_len,dtype=torch.bool),diagonal=1)
    return causal_mask.expand(batch,num_heads,target_len,target_len).to(target_tokens.device)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, num_heads: int, dropout: float=0.0):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model,num_heads,dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model,d_hidden,dropout)

    def forward(self,x: torch.Tensor, mask: Optional[torch.Tensor]=None):
        norm_x = self.layer_norm1(x)
        x = x + self.mha(norm_x,norm_x,norm_x,mask)
        x = x + self.ffn(self.layer_norm2(x))
        return x
    

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, num_heads: int, dropout: float=0.0):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model,num_heads,dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.cross_mha = MultiHeadAttention(d_model,num_heads,dropout)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model,d_hidden,dropout)

    def forward(self,x: torch.Tensor,enc_out: torch.Tensor, self_attn_mask:Optional[torch.Tensor]=None, cross_attn_mask: Optional[torch.Tensor]=None):

        norm_x = self.layer_norm1(x)
        x = x + self.mha(norm_x,norm_x,norm_x,self_attn_mask)
        x = x + self.cross_mha(self.layer_norm2(x),enc_out,enc_out,cross_attn_mask)
        x = x + self.ffn(self.layer_norm3(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_hidden: int,
        num_heads: int,
        num_layers: int,
        pad_token_id: int = 0,
        max_len: int = 5000,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.pad_token_id = pad_token_id
        self.embed = TokenEmbedding(vocab_size,d_model)
        self.position_encoding = PositionEncoding(d_model,max_len)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model,d_hidden,num_heads,dropout) for i in range(num_layers)])
    
    def forward(self,x: torch.Tensor):
        enc_mask = generate_pad_mask(x,x,self.pad_token_id,self.num_heads)
        x = self.position_encoding(self.embed(x))
        for layer in self.layers:
            x = layer(x,enc_mask)
        
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_hidden: int,
        num_heads: int,
        num_layers: int,
        pad_token_id: int = 0,
        max_len: int = 5000,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.pad_token_id = pad_token_id
        self.embed = TokenEmbedding(vocab_size,d_model)
        self.position_encoding = PositionEncoding(d_model,max_len)
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model,d_hidden,num_heads,dropout) for i in range(num_layers)])
    
    def forward(self,target_tokens: torch.Tensor,src_tokens: torch.Tensor, enc_out: torch.Tensor):
        x = self.position_encoding(self.embed(target_tokens))
        dec_pad_mask = generate_pad_mask(target_tokens,target_tokens,self.pad_token_id,self.num_heads)
        dec_causal_mask = generate_causal_mask(target_tokens,self.num_heads)
        dec_self_attn_mask = torch.logical_or(dec_pad_mask,dec_causal_mask)
        cross_attn_mask = generate_pad_mask(target_tokens,src_tokens,self.pad_token_id,self.num_heads)
        for layer in self.layers:
            x = layer(x,enc_out,dec_self_attn_mask,cross_attn_mask)
        return x
    

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        d_hidden: int = 2048,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        pad_token_id: int = 0,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.encoder = TransformerEncoder(src_vocab_size,d_model,d_hidden,num_heads,num_encoder_layers,pad_token_id,max_len,dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size,d_model,d_hidden,num_heads,num_decoder_layers,pad_token_id,max_len,dropout)

        self.output_linear = nn.Linear(d_model,tgt_vocab_size)

    def forward(self,src_tokens: torch.Tensor, tgt_tokens: torch.Tensor):
        enc_out = self.encoder(src_tokens)
        dec_out = self.decoder(tgt_tokens,src_tokens,enc_out)
        logits = self.output_linear(dec_out)
        return logits




if __name__=="__main__":
    import numpy as np
    import random
    # 固定随机种子
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 如果使用 GPU，还要固定 CUDA 的随机种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多卡
    # 配置参数
    src_vocab_size = 1000   # 源语言词表大小
    tgt_vocab_size = 1000   # 目标语言词表大小
    batch_size = 2          # batch 大小
    src_len = 10            # 源序列长度
    tgt_len = 8             # 目标序列长度
    d_model = 512
    d_hidden = 2048
    num_heads = 8
    num_encoder_layers = 2
    num_decoder_layers = 2
    pad_token_id = 0

    # 初始化模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        d_hidden=d_hidden,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        pad_token_id=pad_token_id,
        max_len=50,
        dropout=0.1
    )

    # 随机生成输入序列（整数 token）
    src_tokens = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt_tokens = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))

    # 前向推理
    logits = model(src_tokens, tgt_tokens)

    
    