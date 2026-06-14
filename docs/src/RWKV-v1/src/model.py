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
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################

def RWKV_Init(module, config): # fancy initialization of all lin & emb layer in the module
    for m in module.modules():
        if not isinstance(m, (nn.Linear, nn.Embedding)):
            continue
        with torch.no_grad():
            name = '[unknown weight]'
            for name, parameter in module.named_parameters(): # find the name of the weight
                if id(m.weight) == id(parameter):
                    break

            shape = m.weight.data.shape
            gain = 1.0  # positive: gain for orthogonal, negative: std for normal
            scale = 1.0 # extra scale for gain

            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == config.vocab_size and shape[1] == config.n_embd: # final projection?
                    scale = config.rwkv_emb_scale

            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == config.vocab_size and shape[1] == config.n_embd: # token emb?
                    scale = config.rwkv_emb_scale

            if hasattr(m, 'scale_init'):
                scale = m.scale_init

            print(str(shape[0]).ljust(5), str(shape[1]).ljust(5), f'{round(scale,2):g}'.ljust(4), name)

            gain *= scale
            if gain == 0:
                nn.init.zeros_(m.weight) # zero init is great for some RWKV matrices
            elif gain > 0:
                nn.init.orthogonal_(m.weight, gain=gain)
            else:
                nn.init.normal_(m.weight, mean=0, std=-gain)

class RWKV_TimeMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        assert config.n_attn % config.n_head == 0
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_head = config.n_head
        self.head_size = config.n_attn // config.n_head

        with torch.no_grad(): # initial time_w curves for better convergence
            ww = torch.ones(config.n_head, config.ctx_len)
            curve = torch.tensor([-(config.ctx_len - 1 - i) for i in range(config.ctx_len)]) # the distance
            for h in range(config.n_head):
                if h < config.n_head - 1:
                    decay_speed = math.pow(config.ctx_len, -(h+1)/(config.n_head-1))
                else:
                    decay_speed = 0.0
                ww[h] = torch.exp(curve * decay_speed)
                # print('layer', layer_id, 'head', h, 'decay_speed', round(decay_speed, 4), ww[h][:5].numpy(), '...', ww[h][-5:].numpy())
        self.time_w = nn.Parameter(ww)

        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, config.ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, config.ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(config.ctx_len, 1))
                
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))

        self.key = nn.Linear(config.n_embd, config.n_attn)
        self.value = nn.Linear(config.n_embd, config.n_attn)
        self.receptance = nn.Linear(config.n_embd, config.n_attn)

        # if config.rwkv_tiny_attn > 0:
        #     self.tiny_att = RWKV_TinyAttn(config)

        self.output = nn.Linear(config.n_attn, config.n_embd)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        TT = self.ctx_len
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT-1:] # w is now a circulant matrix
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]

        x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)
        # if hasattr(self, 'tiny_att'):
        #     tiny_att = self.tiny_att(x, self.mask)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        k = torch.clamp(k, max=30, min=-60) # clamp extreme values. e^30 = 10^13
        k = torch.exp(k)
        sum_k = torch.cumsum(k, dim=1)

        kv = (k * v).view(B, T, self.n_head, self.head_size)

        wkv = (torch.einsum('htu,buhc->bthc', w, kv)).contiguous().view(B, T, -1)

        rwkv = torch.sigmoid(r) * wkv / sum_k

        rwkv = self.output(rwkv)
        # if hasattr(self, 'tiny_att'):
        #     rwkv += tiny_att

        return rwkv * self.time_gamma[:T, :]

class RWKV_ChannelMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        
        hidden_sz = 5 * config.n_ffn // 2 # can use smaller hidden_sz because of receptance gating
        self.key = nn.Linear(config.n_embd, hidden_sz)
        self.value = nn.Linear(config.n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, config.n_embd)
        self.receptance = nn.Linear(config.n_embd, config.n_embd)

        self.receptance.scale_init = 0
        self.weight.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        
        x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        
        wkv = self.weight(F.mish(k) * v) # i find mish is a bit better than gelu

        rwkv = torch.sigmoid(r) * wkv

        return rwkv

class RWKV_TinyAttn(nn.Module): # extra tiny attention
    def __init__(self, config):
        super().__init__()
        self.d_attn = config.rwkv_tiny_attn
        self.n_head = config.rwkv_tiny_head
        self.head_size = self.d_attn // self.n_head

        self.qkv = nn.Linear(config.n_embd, self.d_attn * 3)
        self.out = nn.Linear(self.d_attn, config.n_embd)

    def forward(self, x, mask):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim = -1)

        if self.n_head > 1:
            q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)      # (B, T, C) -> (B, nh, T, hs)
            k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)      # (B, T, C) -> (B, nh, T, hs)
            v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)      # (B, T, C) -> (B, nh, T, hs)

        qk = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))     # (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        qk = qk.masked_fill(mask == 0, float('-inf'))
        qk = F.softmax(qk, dim = -1)
        qkv = qk @ v                                                           # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)

        if self.n_head > 1:
            qkv = qkv.transpose(1, 2).contiguous().view(B, T, -1)              # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
       
        return self.out(qkv)

########################################################################################################
# MHA_rotary: Multi-head Attention + Rotary Encoding + GeGLU FFN
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
    cos, sin = cos[...,:q.shape[-2],:], sin[...,:q.shape[-2],:]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class MHA_rotary(nn.Module):
    def __init__(self, config, layer_id, time_shift = False):
        super().__init__()
        self.layer_id = layer_id
        assert config.n_attn % config.n_head == 0
        self.n_head = config.n_head
        self.ctx_len = config.ctx_len
        self.head_size = config.n_attn // config.n_head

        if time_shift:
            self.time_shift = nn.ZeroPad2d((0,0,1,-1))

        self.query = nn.Linear(config.n_embd, config.n_attn)
        self.key = nn.Linear(config.n_embd, config.n_attn)
        self.value = nn.Linear(config.n_embd, config.n_attn)

        self.register_buffer("mask", torch.tril(torch.ones(config.ctx_len, config.ctx_len)))
        
        self.rotary_ndims = int(self.head_size * 0.5)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)

        self.output = nn.Linear(config.n_attn, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()

        if hasattr(self, 'time_shift'):
            x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)

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
        x = x.transpose(1, 2).contiguous().view(B, T, -1)                               # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        x = self.output(x)
        return x

class GeGLU(torch.nn.Module):
    def __init__(self, config, layer_id, time_shift = False):
        super().__init__()
        self.layer_id = layer_id

        if time_shift:
            self.time_shift = nn.ZeroPad2d((0,0,1,-1))

        hidden_sz = 3 * config.n_ffn
        self.key = nn.Linear(config.n_embd, hidden_sz)
        self.value = nn.Linear(config.n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        if hasattr(self, 'time_shift'):
            x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)
        
        k = self.key(x)
        v = self.value(x)        
        y = self.weight(F.gelu(k) * v)
        return y

########################################################################################################
# MHA_pro: with more tricks
########################################################################################################

class MHA_pro(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        assert config.n_attn % config.n_head == 0
        self.n_head = config.n_head
        self.ctx_len = config.ctx_len
        self.head_size = config.n_attn // config.n_head

        self.time_w = nn.Parameter(torch.ones(self.n_head, config.ctx_len))
        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, config.ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, config.ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(config.ctx_len, 1))
        self.register_buffer("mask", torch.tril(torch.ones(config.ctx_len, config.ctx_len)))

        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        self.query = nn.Linear(config.n_embd, config.n_attn)
        self.key = nn.Linear(config.n_embd, config.n_attn)
        self.value = nn.Linear(config.n_embd, config.n_attn)
        
        self.rotary_ndims = int(self.head_size * 0.5)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)

        self.head_mix = nn.Conv2d(self.n_head, self.n_head, kernel_size=1, bias=False)  # talking heads

        self.output = nn.Linear(config.n_attn, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        TT = self.ctx_len
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT-1:] # w is now a circulant matrix
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]

        x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)      # time-shift mixing
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
        x = x.transpose(1, 2).contiguous().view(B, T, -1)                               # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        x = self.output(x) * self.time_gamma[:T, :]
        return x

########################################################################################################
# The GPT Model with our blocks
########################################################################################################

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
    def __init__(self, vocab_size, ctx_len, **kwargs):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        for k,v in kwargs.items():
            setattr(self, k, v)

class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        if config.model_type == 'RWKV':
            # self.ln1 = FixedNorm(config.n_embd)
            # self.ln2 = FixedNorm(config.n_embd)
            self.attn = RWKV_TimeMix(config, layer_id)
            self.mlp = RWKV_ChannelMix(config, layer_id)

        elif config.model_type == 'MHA_rotary':
            self.attn = MHA_rotary(config, layer_id)
            self.mlp = GeGLU(config, layer_id)
        
        elif config.model_type == 'MHA_shift':
            self.attn = MHA_rotary(config, layer_id, time_shift=True)
            self.mlp = GeGLU(config, layer_id, time_shift=True)
        
        elif config.model_type == 'MHA_pro':
            self.attn = MHA_pro(config, layer_id)
            self.mlp = RWKV_ChannelMix(config, layer_id)

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

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.time_out = nn.Parameter(torch.ones(1,config.ctx_len,1)) # reduce confidence of early tokens
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.head_q = nn.Linear(config.n_embd, 256)
        self.head_q.scale_init = 0.01
        self.head_k = nn.Linear(config.n_embd, 256)
        self.head_k.scale_init = 0.01
        self.register_buffer("copy_mask", torch.tril(torch.ones(config.ctx_len, config.ctx_len)))

        self.ctx_len = config.ctx_len

        if self.config.model_type == 'RWKV':
            RWKV_Init(self, config)
        else:
            self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_ctx_len(self):
        return self.ctx_len

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
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
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps)
        return optimizer

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

        x = self.tok_emb(idx)

        x = self.blocks(x)

        x = self.ln_f(x)

        q = self.head_q(x)[:,:T,:]
        k = self.head_k(x)[:,:T,:]
        c = (q @ k.transpose(-2, -1)) * (1.0 / 256)
        c = c.masked_fill(self.copy_mask[:T,:T] == 0, 0)
        c = c @ F.one_hot(idx, num_classes = self.config.vocab_size).float()

        x = x * self.time_out[:, :T, :] # reduce confidence of early tokens
        x = self.head(x) + c

        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))

        return x, loss
