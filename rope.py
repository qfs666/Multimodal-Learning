'''
Descripttion: 手搓旋转位置编码
version: 1.0
Date: 2025-08-19 10:15:26
LastEditTime: 2025-08-19 10:24:46
'''
import torch
from torch import nn
import math

class RotaryEmbedding(nn.Module):
    def __init__(self,d_model:int ,max_len: int = 5000, base: int = 10000):
        super().__init__()
        '''
            d_model: d_head维度
            max_len: 最大序列长度
            base: 旋转频率基数
        '''
        theta = 1.0 / (base ** (torch.arange(0,d_model,2).float()/d_model)) # [seq_len,d_model//2]

        position = torch.arange(0,max_len).unsqueeze(1)

        freqs = position*theta

        embed = torch.cat((freqs,freqs),dim=1) # [seq_len,d_model//2] -> [seq_len,d_model]

        cos_cached = torch.cos(embed).unsqueeze(0).unsqueeze(1)
        sin_cached = torch.sin(embed).unsqueeze(0).unsqueeze(1)
        self.register_buffer("cos_cached",cos_cached,persistent=False)
        self.register_buffer("sin_cached",sin_cached,persistent=False)

    def forward(self,x: torch.Tensor):
        '''
            x: [batch,num_heads,seq_len,d_k]
            offset: 推理使用的起始位置索引（KV cache）
        '''
        batch,num_heads,seq_len,d_k = x.shape

        sin = self.sin_cached[:,:,:seq_len,:].to(x.device)
        cos = self.cos_cached[:,:,:seq_len,:].to(x.device)
        
        return self.apply_rope(x,sin,cos)
    
    def rotate_half(self,x: torch.Tensor):
        '''
            x: [batch,num_heads,seq_len,d_k]
        '''
        x1 = x[:,:,:,:x.shape[-1]//2]
        x2 = x[:,:,:,x.shape[-1]//2:]
        return torch.cat((-x2,x1),dim=-1)
    
    def apply_rope(self, x: torch.Tensor, sin: torch.Tensor,cos: torch.Tensor):
        '''
            x: [batch,num_heads,seq_len,d_k]
            sin: [1,1,,seq_len,d_k]
            cos: [1,1,,seq_len,d_k]
        '''
        return x*cos+ self.rotate_half(x)*sin
    