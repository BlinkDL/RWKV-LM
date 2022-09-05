########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset


class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        if args.data_type == "binidx":
            self.data = MMapIndexedDataset(args.data_file)
            self.vocab_size = args.vocab_size
            print("current vocab size =", self.vocab_size, "(make sure it's correct)")
            self.data_size = len(self.data._bin_buffer) // 2
            print(f"data has {self.data_size} tokens.")
        elif args.data_type == "numpy":
            self.data = np.load(args.data_file).astype("int")
            self.vocab_size = args.vocab_size
            print("current vocab size =", self.vocab_size, "(make sure it's correct)")
            self.data_size = len(self.data)
            print(f"data has {self.data_size} tokens.")
        else:
            self.data = open(args.data_file, "r", encoding=args.data_type).read()
            print("building token list...", end=" ")
            unique = sorted(list(set(self.data)))
            self.vocab_size = len(unique)
            # print()
            # for u in unique:
            #     print(u, end=' ')
            # print('\n\n')
            xx = 0
            xxObj = {}
            for u in unique:
                xxObj[xx] = u
                xx += 1
            with open(f"{args.proj_dir}/vocab.json", "w", encoding="utf-16le") as vocab_file:
                vocab_file.write(json.dumps(xxObj, ensure_ascii=False))
            self.data_size = len(self.data)
            print("data has %d tokens, %d unique." % (self.data_size, self.vocab_size))
            self.stoi = {ch: i for i, ch in enumerate(unique)}
            self.itos = {i: ch for i, ch in enumerate(unique)}

    def __len__(self):
        return self.args.epoch_steps * int(self.args.devices) * self.args.micro_bsz

    def __getitem__(self, idx):
        #
        # we are cheating: pick a random spot in dataset
        #
        ctx_len = self.args.ctx_len
        req_len = ctx_len + 1
        i = np.random.randint(0, self.data_size - req_len)
        if "MMapIndexedDataset" in str(type(self.data)):
            dix = self.data.get(idx=0, offset=i, length=req_len).astype(int)
        elif "numpy" in str(type(self.data)):
            dix = self.data[i : i + req_len]
        else:
            dix = [self.stoi[s] for s in self.data[i : i + req_len]]

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
