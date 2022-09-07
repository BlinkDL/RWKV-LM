########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset
from .utils import MaybeIsPrime


class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        if args.data_type == "binidx":
            self.data = MMapIndexedDataset(args.data_file)
            self.vocab_size = args.vocab_size
            print("Current vocab size =", self.vocab_size, "(make sure it's correct)")
            self.data_size = len(self.data._bin_buffer) // 2
            print(f"Data has {self.data_size} tokens.")

            if args.my_pile_stage > 0:
                assert self.data_size == 332115325534 and self.vocab_size == 50277
                self.samples_per_epoch = args.epoch_steps * args.real_bsz
                assert self.samples_per_epoch == 40320
                print(f"########## Pile 20b-tokenized stage {args.my_pile_stage} ##########")
                dataset_slot = self.data_size // args.ctx_len
                assert MaybeIsPrime(args.magic_prime)
                assert args.magic_prime % 3 == 2
                assert args.magic_prime / dataset_slot > 0.999999 and args.magic_prime / dataset_slot <= 1
        elif args.data_type == "numpy":
            self.data = np.load(args.data_file).astype("int")
            self.vocab_size = args.vocab_size
            print("Current vocab size =", self.vocab_size, "(make sure it's correct)")
            self.data_size = len(self.data)
            print(f"Data has {self.data_size} tokens.")
        else:
            if args.data_type == "dummy":
                print("Building dummy data...")
                self.data = ""
                for i in range(100000):
                    aa = (i) % 10000
                    bb = (i * i) % 10000
                    cc = aa + bb
                    self.data += f".{aa}+{bb}={cc}."
            else:
                self.data = open(args.data_file, "r", encoding=args.data_type).read()
            print("Building token list...")
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
            print("Data has %d tokens, %d vocab size." % (self.data_size, self.vocab_size))
            self.stoi = {ch: i for i, ch in enumerate(unique)}
            self.itos = {i: ch for i, ch in enumerate(unique)}

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        #
        # we are cheating: pick a random spot in dataset
        #
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")

        ctx_len = args.ctx_len
        req_len = ctx_len + 1

        if args.my_pile_stage > 0:
            ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank
            factor = (math.sqrt(5) - 1) / 2
            factor = int(args.magic_prime * factor)
            i = ((factor * ii * ii * ii) % args.magic_prime) * ctx_len
            i = i + args.my_pile_shift
            # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} ii {ii} pos {round(i / self.data_size, 3)}")
        else:
            i = np.random.randint(0, self.data_size - req_len)

        if args.data_type == "binidx":
            dix = self.data.get(idx=0, offset=i, length=req_len).astype(int)
        elif args.data_type == "numpy":
            dix = self.data[i : i + req_len]
        else:
            dix = [self.stoi[s] for s in self.data[i : i + req_len]]

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
