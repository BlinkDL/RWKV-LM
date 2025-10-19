# File: 251018_rosa_4bit_GPU_run.py
# Description: GPU-optimized version of the ROSA model.
# The core 'rosa' algorithm has been re-implemented in pure PyTorch for 100% GPU execution,
# eliminating the CPU <-> GPU bottleneck.

import torch
import random
from torch import nn
import torch.nn.functional as F

# Recommended performance settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

####################################################################################################
# --- GPU-Optimized ROSA Algorithm ---
####################################################################################################

def rosa_batch_gpu_exact(x: torch.Tensor, output_uint8: bool = False) -> torch.Tensor:
    """
    Massively parallel GPU version of the ROSA algorithm.
    It reproduces the original logic using tensor operations.

    Args:
        x (torch.Tensor): 2D Tensor (B, T) of type long or uint8.
        output_uint8 (bool): If True, the result is clamped at 0 and converted to uint8,
                             matching the behavior of the original 'rosa_batch_python'.
    """
    B, T = x.shape
    device = x.device

    # Step 1: Equality matrix. eq[b, i, j] is True if x[b, i] == x[b, j].
    # This compares all tokens with all others in parallel.
    eq = (x.unsqueeze(2) == x.unsqueeze(1))  # Shape (B, T, T)

    # Step 2: Calculate the length of common suffixes via dynamic programming.
    # L[b, i, j] = length of the match ending at positions i and j.
    # The recurrence L[i, j] = (L[i-1, j-1] + 1) * eq[i, j] is implemented
    # via a loop over the sequence length, which is very fast on GPU.
    L = torch.zeros((B, T, T), device=device, dtype=torch.int16)
    for i in range(1, T):
        # We shift the diagonal of the previous sub-matrix.
        prev_diag = F.pad(L[:, i - 1, :-1], (1, 0))
        L[:, i, :] = (prev_diag + 1) * eq[:, i, :]

    # Step 3: Masking to consider only matches in the past (j < i).
    mask = torch.tril(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=-1)

    # Step 4: Find the best match.
    # To break ties, we favor the most recent matches (largest j),
    # which mimics the behavior of a suffix automaton.
    j_indices = torch.arange(T, device=device, dtype=torch.float32)
    tie_breaker = j_indices * 1e-5  # Small bias for tie-breaking
    
    L_scored = L.float() + tie_breaker.unsqueeze(0).unsqueeze(1)
    L_scored.masked_fill_(~mask.unsqueeze(0), -1.0)  # Ignore future and present matches

    max_vals, best_j = L_scored.max(dim=2)  # best_j[b, i] is the optimal j for position i

    # Step 5: Retrieve the prediction.
    # The prediction is the token that follows the end of the found match: x[best_j + 1].
    pred_idx = (best_j + 1).clamp(max=T - 1)
    # torch.gather allows for advanced parallel indexing for the whole batch.
    predictions = torch.gather(x, 1, pred_idx)

    # Step 6: Handle cases where no match was found.
    # If the maximum length (without the tie-breaker) was 0, it's a "no match".
    no_match_mask = (L.masked_fill(~mask.unsqueeze(0), 0).max(dim=2).values == 0)
    predictions[no_match_mask] = -1

    # Step 7: Format the output to match the original function's requirements.
    if output_uint8:
        return predictions.clamp_min(0).to(torch.uint8)
    else:
        # Ensure the output dtype matches the input dtype for 'long' types.
        return predictions.to(x.dtype)

####################################################################################################
# --- Model Layers (Using the GPU-optimized version) ---
####################################################################################################

class rosa_emb_layer(nn.Module):
    def __init__(self, V, C):
        super().__init__()
        self.emb = nn.Embedding(V, C)

    def forward(self, idx):
        idx_pred = rosa_batch_gpu_exact(idx, output_uint8=False)
        out = self.emb(idx_pred.clamp_min(0))
        return out.masked_fill(idx_pred.eq(-1).unsqueeze(-1), 0.0)

class rosa_4bit_layer(nn.Module):
    def __init__(self, C: int, eps: float = 1e-5):
        super().__init__()
        assert C % 4 == 0
        self.emb0 = nn.Parameter(torch.full((1, 1, C), -eps))
        self.emb1 = nn.Parameter(torch.full((1, 1, C), eps))
        self.print_lang = False  # Set to True for debugging

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        Cg = C // 4
        b = (x.reshape(B, T, Cg, 4) > 0).to(torch.uint8)
        tok2d = (b[..., 0] | (b[..., 1] << 1) | (b[..., 2] << 2) | (b[..., 3] << 3)).permute(0, 2, 1).reshape(-1, T).contiguous()
        
        if self.print_lang:
            for c in range(min(Cg, B * Cg)):  # Limit the print output
                print('lang ' + ''.join(f'{v:X}' for v in tok2d[c].tolist()))
            print()
        
        idx_q = rosa_batch_gpu_exact(tok2d, output_uint8=True).reshape(B, Cg, T).transpose(1, 2).contiguous()

        e0 = self.emb0.expand(B, T, -1).reshape(B, T, Cg, 4)
        e1 = self.emb1.expand(B, T, -1).reshape(B, T, Cg, 4)
        bits = torch.stack([(idx_q >> i) & 1 for i in range(4)], dim=-1).bool()
        return torch.where(bits, e1, e0).reshape(B, T, C)

####################################################################################################
# --- Main Execution Script ---
####################################################################################################

V, C, B, T = 11, 64, 128, 128

print(f"Device: {device}")
print('Run EmbROSA + ROSA 4bit x 4layer (GPU Optimized)')

def batch(B, T, nn=None):
    s = []
    for _ in range(B):
        if nn is None:
            k = random.randint(1, 3)
            lo = 0 if k == 1 else 10**(k - 1)
            n = random.randint(lo, 10**k - 1)
        else:
            assert B == 1
            n = nn
        a = [10]
        while len(a) < T:
            a += [ord(c) - 48 for c in str(n)] + [10]
            n += 1
        s.append(a[:T])
    return torch.tensor(s, device=device, dtype=torch.long)

class MODEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = nn.Embedding(V, C)
        self.emb_rosa = rosa_emb_layer(V, C)
        self.rosa1 = rosa_4bit_layer(C)
        self.lin = nn.Linear(C, C)
        self.rosa2 = rosa_4bit_layer(C)
        self.lin1 = nn.Linear(C, C)
        self.rosa3 = rosa_4bit_layer(C)
        self.lin2 = nn.Linear(C, C)
        self.rosa4 = rosa_4bit_layer(C)
        self.o = nn.Linear(C, V)

    def forward(self, x):
        x = self.e(x) + self.emb_rosa(x)
        x = x + self.rosa1(x)
        x = x + self.lin(x)
        x = x + self.rosa2(x)
        x = x + self.lin1(x)
        x = x + self.rosa3(x)
        x = x + self.lin2(x)
        x = x + self.rosa4(x)
        x = self.o(x)
        return x

try:
    model = MODEL().to(device)
    model.load_state_dict(torch.load('251018_rosa_4bit_run.pth', map_location=device, mmap=True, weights_only=True))
    print("Model instantiated AND pre-trained weights loaded successfully from '251018_rosa_4bit_run.pth'.")
except FileNotFoundError:
    print("ERROR: Weight file '251018_rosa_4bit_run.pth' not found.")
    print("Please ensure the file is in the same directory as the script.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")

print('#' * 100)

with torch.no_grad():
    model.eval()
    S = '0123456789A'  # 'A' represents token 10

    for SAMPLE in range(5):
        x = batch(1, 128, int(3.5**(SAMPLE + 1)))
        y = x[:, 1:]
        z = model(x[:, :-1]).argmax(-1)
        n = y.numel()
        
        # Use the GPU version for the 'rosa' baseline comparison as well
        r = rosa_batch_gpu_exact(x)[:, :-1]
        
        rr = ''.join([S[val] if val >= 0 else 'X' for val in r[0].tolist()])
        xx = ''.join(S[t] for t in x[0, :-1].tolist())
        yy = ''.join(S[t] for t in y[0].tolist())
        zz = ''.join(S[t] for t in z[0].tolist())
        
        ry = ''.join('.' if r[0, i].item() == y[0, i].item() else '^' for i in range(y.size(1)))
        zy = ''.join('.' if z[0, i].item() == y[0, i].item() else '^' for i in range(y.size(1)))
        
        nry = (r == y).sum().item()
        nzy = (z == y).sum().item()
        
        print(f'in  ', xx)
        print(f'gold', yy)
        print(f'rosa', rr)
        print(f'diff', ry)
        print(f'correct {nry}/{n}  acc {nry / n:.3f}')
        print(f'gold', yy)
        print(f'pred', zz)
        print(f'diff', zy)
        print(f'correct {nzy}/{n}  acc {nzy / n:.3f}')
        print('#' * 100)