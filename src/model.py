########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
logger = logging.getLogger(__name__)

########################################################################################################
# Block: RWKV Time-mix + RWKV Channel-mix
########################################################################################################

class RWKV_TimeMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.layer_id = layer_id
        self.ctx_size = config.ctx_size
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head

        self.time_w = nn.Parameter(torch.ones(self.n_head, config.ctx_size))
        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, config.ctx_size))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, config.ctx_size, 1))
        self.time_gamma = nn.Parameter(torch.ones(config.ctx_size, 1))
        self.register_buffer("mask", torch.tril(torch.ones(config.ctx_size, config.ctx_size)))
                
        self.time_shift = nn.ZeroPad2d((0,0,1,0))

        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.receptance = nn.Linear(config.n_embd, config.n_embd)
       
        self.output = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()            
        TT = self.ctx_size
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT-1:] # w is now a circulant matrix
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]
        w = w.masked_fill(self.mask[:T, :T] == 0, 0)

        x = torch.cat([self.time_shift(x)[:, :-1, :C//2], x[:, :, C//2:]], dim = -1)
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        k = torch.exp(k)
        sum_k = torch.cumsum(k, dim=1)

        k = k.view(B, T, self.n_head, self.head_size)
        v = v.view(B, T, self.n_head, self.head_size)

        wkv = (torch.einsum('htu,buhc->bthc', w, k * v)).contiguous().view(B, T, C)
        y = torch.sigmoid(r) * wkv / sum_k

        y = self.output(y) * self.time_gamma[:T, :]
        return y

class RWKV_ChannelMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0,0,1,0))
        
        self.key = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.value = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.weight = nn.Linear(3 * config.n_embd, config.n_embd)
        self.receptance = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        
        x = torch.cat([self.time_shift(x)[:, :-1, :C//2], x[:, :, C//2:]], dim = -1)
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        
        wkv = self.weight(F.mish(k) * v) # mish is a bit better than gelu
        y = torch.sigmoid(r) * wkv

        return y

########################################################################################################
# Block: Multi-head Attention + Rotary Encoding + GeGLU FFN
########################################################################################################

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return self.cos_cached, self.sin_cached

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos[...,:q.shape[2],:], sin[...,:q.shape[2],:]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RotaryMHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.ctx_size = config.ctx_size
        self.head_size = config.n_embd // config.n_head

        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.register_buffer("mask", torch.tril(torch.ones(config.ctx_size, config.ctx_size)))
        
        self.rotary_ndims = int(self.head_size * 0.5)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)

        self.output = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()

        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)         # (B, T, C) -> (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)

        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]          
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)                                     # rotary encoding
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)  
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))                 # self-attention: (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        att = att.masked_fill(self.mask[:T,:T] == 0, float('-inf'))                     # causal mask
        att = F.softmax(att, dim = -1)                                                  # softmax        

        x = att @ v                                                                     # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        x = x.transpose(1, 2).contiguous().view(B, T, C)                                # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        x = self.output(x)                                                              # output projection
        return x

class GeGLU(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.value = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.weight = nn.Linear(3 * config.n_embd, config.n_embd)

    def forward(self, x):
        k = self.key(x)
        v = self.value(x)        
        y = self.weight(F.gelu(k) * v)
        return y

########################################################################################################
# Block: MHA+ (with even more tricks)
########################################################################################################        

class RotaryMHA_Plus(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.ctx_size = config.ctx_size
        self.head_size = config.n_embd // config.n_head

        self.time_w = nn.Parameter(torch.ones(self.n_head, config.ctx_size))
        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, config.ctx_size))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, config.ctx_size, 1))
        self.time_gamma = nn.Parameter(torch.ones(config.ctx_size, 1))
        self.register_buffer("mask", torch.tril(torch.ones(config.ctx_size, config.ctx_size)))        

        self.time_shift = nn.ZeroPad2d((0,0,1,0))
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        self.rotary_ndims = int(self.head_size * 0.5)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)

        self.head_mix = nn.Conv2d(self.n_head, self.n_head, kernel_size=1, bias=False) # talking heads

        self.output = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        TT = self.ctx_size
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT-1:] # w is now a circulant matrix
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]

        x = torch.cat([self.time_shift(x)[:, :-1, :C//2], x[:, :, C//2:]], dim = -1)    # time-mixing
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)         # (B, T, C) -> (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)

        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]          
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)                                     # rotary encoding
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)  
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))                 # self-attention: (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        att = att.masked_fill(self.mask[:T,:T] == 0, float('-inf'))                     # causal mask
        att = F.softmax(att, dim = -1)                                                  # softmax        
        att = att * w                                                                   # time-weighting
        att = self.head_mix(att)                                                        # talking heads

        x = att @ v                                                                     # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        x = x.transpose(1, 2).contiguous().view(B, T, C)                                # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        x = self.output(x) * self.time_gamma[:T, :]
        return x

########################################################################################################
# The GPT Model with our blocks
########################################################################################################

class LabelSmoothingCrossEntropy(nn.Module): # can avoid nan loss
    def __init__(self, smoothing=0.0):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))   

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.dd = d ** (-1. / 2)
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        x_normed = x / (norm_x * self.dd + 1e-12)
        return self.weight * x_normed

class FixedNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.dd = d ** (-1. / 2)

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        x_normed = x / (norm_x * self.dd + 1e-12)
        return x_normed

########################################################################################################        

class GPTConfig:
    def __init__(self, vocab_size, ctx_size, **kwargs):
        self.vocab_size = vocab_size
        self.ctx_size = ctx_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        if config.model_type == 'RWKV':
            self.ln1 = FixedNorm(config.n_embd)
            self.ln2 = FixedNorm(config.n_embd)
            self.attn = RWKV_TimeMix(config, layer_id)
            self.mlp = RWKV_ChannelMix(config, layer_id)
        elif config.model_type == 'RotaryMHA':
            self.attn = RotaryMHA(config)
            self.mlp = GeGLU(config)
        elif config.model_type == 'MHA-Plus':
            self.attn = RotaryMHA_Plus(config)
            self.mlp = RWKV_ChannelMix(config)

    def forward(self, x):

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        self.blocks = nn.Sequential(*[Block(config, i) for i in range(config.n_layer)])

        if config.model_type == 'RWKV':
            self.ln_f = FixedNorm(config.n_embd)
        else:
            self.ln_f = nn.LayerNorm(config.n_embd)

        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.ctx_size = config.ctx_size
        self.apply(self._init_weights)

        if self.config.model_type == 'RWKV': # improve orthogonal weight init
            ww = self.state_dict()
            for k in ww:
                if 'tok_emb' in k:
                    if self.config.vocab_size > self.config.n_embd:
                        ww[k] *= math.sqrt(self.config.vocab_size)
                    else:
                        ww[k] *= math.sqrt(self.config.n_embd)
                    ww[k] *= 0.4
                elif 'head.weight' in k:
                    ww[k] *= 0.2
                elif 'blocks.' in k:
                    block_id = int(k.split('.')[1])
                    if 'receptance.weight' in k:
                        ww[k] *= 0.5
                    elif 'attn.key.weight' in k:
                        ww[k] *= 0.2

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_ctx_size(self):
        return self.ctx_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if self.config.model_type == 'RWKV':
                gain = 1.0
                if isinstance(module, nn.Linear):
                    if module.weight.data.shape[0] > module.weight.data.shape[1]:
                        gain = math.sqrt(module.weight.data.shape[0] / module.weight.data.shape[1])
                nn.init.orthogonal_(module.weight, gain=gain)
            else:
                module.weight.data.normal_(mean=0.0, std=0.01)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def configure_optimizers(self, train_config):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (RMSNorm, nn.LayerNorm, nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias') or ('time' in fpn) or ('head' in fpn):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.ctx_size, "Cannot forward, model block size is exhausted."

        x = self.tok_emb(idx)

        x = self.blocks(x)

        x = self.ln_f(x)
        x = self.head(x)

        loss = None
        if targets is not None:
            loss = LabelSmoothingCrossEntropy(smoothing=1e-6)(x.view(-1, x.size(-1)), targets.view(-1))

        return x, loss
