########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

if __name__ == "__main__":
    print("\n!!! NOTE: THIS IS STILL WIP !!!\n")
    import os, warnings, math, datetime, sys, time
    import numpy as np
    from argparse import ArgumentParser
    import torch
    from torch.utils.data import DataLoader
    import deepspeed
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning import seed_everything
    from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
    from pytorch_lightning.callbacks import TQDMProgressBar
    from pytorch_lightning import Callback

    # print("WARNING: THIS IS ONLY FOR DEBUG")
    # seed_everything(42)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")

    ########################################################################################################

    # example: train a simple L6-D512 RWKV from scratch
    #
    # python train.py --load_model "" --wandb "" --proj_dir "out" \
    # --data_file "../data/enwik8" --data_type "utf-8" --vocab_size 0 \
    # --ctx_len 512 --epoch_steps 5000 --epoch_count 500 --epoch_begin 0 --epoch_save 5 \
    # --micro_bsz 12 --n_layer 6 --n_embd 512 --pre_ffn 0 --head_qk 0 \
    # --lr_init 8e-4 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    # --accelerator gpu --devices 1 --precision bf16 --strategy ddp_find_unused_parameters_false --grad_cp 0

    # example: fine-tune RWKV 1.5B using 8xA100 40G = 1.76it/s = 115k token/s, VRAM 37477M
    #
    # python train.py --load_model "/fsx/BlinkDL/CODE/FP16/out_1b2/all-8040.pth" --wandb "" --proj_dir "out" \
    # --data_file "../data/train.npy" --data_type "numpy" --vocab_size 50277 \
    # --ctx_len 1024 --epoch_steps 1000 --epoch_count 1000 --epoch_begin 0 --epoch_save 5 \
    # --micro_bsz 8 --n_layer 24 --n_embd 2048 --pre_ffn 0 --head_qk 0 \
    # --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
    # --accelerator gpu --devices 8 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0

    # example: fine-tune RWKV 1.5B using 1 GPU fp16 (VRAM 16G) NOTE: fp16 might overflow
    #
    # python train.py --load_model "/fsx/BlinkDL/CODE/FP16/out_1b2/all-8040.pth" --wandb "" --proj_dir "out" \
    # --data_file "../data/train.npy" --data_type "numpy" --vocab_size 50277 \
    # --ctx_len 1024 --epoch_steps 200 --epoch_count 1000 --epoch_begin 0 --epoch_save 1 \
    # --micro_bsz 11 --n_layer 24 --n_embd 2048 --pre_ffn 0 --head_qk 0 \
    # --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
    # --accelerator gpu --devices 1 --precision fp16 --strategy deepspeed_stage_2_offload --grad_cp 1

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--load_model", default="", type=str)
    parser.add_argument("--wandb", default="", type=str)  # wandb project name
    parser.add_argument("--proj_dir", default="out", type=str)

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str)
    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--epoch_steps", default=1000, type=int)  # a mini "epoch" has xxx steps
    parser.add_argument("--epoch_count", default=500, type=int)
    parser.add_argument("--epoch_begin", default=0, type=int)
    parser.add_argument("--epoch_save", default=5, type=int)

    parser.add_argument("--micro_bsz", default=12, type=int)  # micro batch size (batch size per GPU)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--pre_ffn", default=0, type=int)
    parser.add_argument("--head_qk", default=0, type=int)

    parser.add_argument("--lr_init", default=6e-4, type=float)
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)
    parser.add_argument("--adam_eps", default=1e-8, type=float)

    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--my_pile_mode", default=0, type=int)  # my special pile mode

    args = parser.parse_args()
    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = 1.0
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = -1  # continue forever
    args.betas = (args.beta1, args.beta2)

    if args.my_pile_mode > 0:
        args.epoch_steps = 40320 // (int(args.devices) * args.micro_bsz)
        assert args.epoch_steps * int(args.devices) * args.micro_bsz == 40320
        if args.my_pile_mode == 2:
            assert args.lr_final == args.lr_init

    samples_per_epoch = args.epoch_steps * int(args.devices) * args.micro_bsz
    tokens_per_epoch = samples_per_epoch * args.ctx_len
    rank_zero_info(
        f"""
############################################################################
#
# RWKV-4 {args.precision.upper()} on {args.devices} x {args.accelerator.upper()}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
#
# Data = {args.data_file} ({args.data_type}), ProjDir = {args.proj_dir}
#
# Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
#
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
#
# Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, β {args.betas}, eps {args.adam_eps}
#
# Found torch {torch.__version__}, recommend 1.12.1+cu116 or newer
# Found deepspeed {deepspeed.__version__}, recommend 0.7.0 (faster than newer versions)
# Found pytorch_lightning {pl.__version__}, recommend 1.7.4 or newer
#
############################################################################
"""
    )
    rank_zero_info(str(vars(args)) + "\n")

    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)

    assert args.data_type in ["utf-8", "utf-16le", "numpy", "binidx"]
    assert len(args.data_file) > 0

    if args.lr_final == 0 or args.lr_init == 0:
        rank_zero_info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if args.precision == "fp16":
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    os.environ["RWKV_JIT_ON"] = "1"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"

    import torch

    torch.backends.cudnn.benchmark = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    ########################################################################################################

    class train_callback(pl.Callback):
        def __init__(self, args):
            super().__init__()
            self.args = args

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            args = self.args
            g_step = trainer.global_step

            # LR schedule
            w_step = args.warmup_steps
            if g_step < w_step:
                lr = args.lr_init * (g_step / w_step)
            else:
                if args.lr_final == args.lr_init:
                    lr = args.lr_init
                else:
                    progress = (g_step - w_step) / (args.epoch_count * args.epoch_steps - w_step - 1)
                    progress = min(1, max(0, progress))

                    if args.lr_final == 0 or args.lr_init == 0:  # linear decay
                        lr = args.lr_init + (args.lr_final - args.lr_init) * progress
                    else:  # exp decay
                        lr = args.lr_init * math.exp(math.log(args.lr_final / args.lr_init) * pow(progress, 1))

            for param_group in trainer.optimizers[0].param_groups:
                if self.args.my_pile_mode == 0:
                    param_group["lr"] = lr * param_group["my_lr_scale"]
                elif self.args.my_pile_mode == 2:
                    if param_group["my_lr_scale"] > 1:
                        param_group["lr"] = lr * 5
                    else:
                        param_group["lr"] = lr
                # print(param_group["lr"], param_group["my_lr_scale"])

            trainer.my_lr = lr
            # rank_zero_info(f"{g_step} {lr}")

            if g_step == 0:
                if trainer.is_global_zero:  # logging
                    trainer.my_loss_sum = 0
                    trainer.my_loss_count = 0
                    trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
                    trainer.my_log.write(f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n")
                    try:
                        print(f"\n{trainer.strategy.config}\n")
                        trainer.my_log.write(f"{trainer.strategy.config}\n")
                    except:
                        pass
                    trainer.my_log.flush()
                    if len(args.wandb) > 0:
                        print("Login to wandb...")
                        import wandb

                        model_name = str(args.vocab_size) + "-" + str(args.ctx_len) + "-" + str(args.n_layer) + "-" + str(args.n_embd)
                        wandb.init(
                            project=args.wandb,
                            name=model_name + "-" + args.my_timestamp,
                            config=args,
                            save_code=False,
                        )
                        trainer.my_wandb = wandb

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            args = self.args
            if trainer.is_global_zero:  # logging
                t_now = time.time_ns()
                try:
                    t_cost = (t_now - trainer.my_time_ns) / 1e9
                    self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                    self.log("token/s", args.ctx_len * float(args.devices) * args.micro_bsz / t_cost, prog_bar=True, on_step=True)
                except:
                    pass
                trainer.my_time_ns = t_now
                trainer.my_loss = trainer.my_loss_all.float().mean().item()
                trainer.my_loss_sum += trainer.my_loss
                trainer.my_loss_count += 1
                trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
                self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
                self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)

                if len(args.wandb) > 0:
                    trainer.my_wandb.log(
                        {"loss": trainer.my_loss, "lr": trainer.my_lr},
                        step=trainer.global_step,
                    )

        def on_train_epoch_start(self, trainer, pl_module):
            args = self.args
            dataset = trainer.train_dataloader.dataset.datasets
            assert "MyDataset" in str(dataset)
            dataset.global_rank = trainer.global_rank
            dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
            dataset.world_size = trainer.world_size

        def on_train_epoch_end(self, trainer, pl_module):
            args = self.args
            if trainer.is_global_zero:  # logging & save state_dict
                if (args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0) or trainer.current_epoch == args.epoch_count - 1:
                    torch.save(
                        pl_module.state_dict(),
                        f"{args.proj_dir}/rwkv-{args.epoch_begin + trainer.current_epoch}.pth",
                    )
                trainer.my_log.write(f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n")
                trainer.my_log.flush()

                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0

    @rank_zero_only
    def generate_init_weight(model, temp_name):
        try:
            os.remove(temp_name)
        except:
            pass
        mm = model.generate_init_weight()
        print(f"Saving to {temp_name}...")
        torch.save(mm, temp_name)

    ########################################################################################################

    from src.dataset import MyDataset
    from src.model import RWKV

    train_data = MyDataset(args)
    args.vocab_size = train_data.vocab_size

    model = RWKV(args)

    if len(args.load_model) == 0:
        args.load_model = f"{args.proj_dir}/rwkv-init.pth"
        generate_init_weight(model, args.load_model)  # save initial weights to tmp file

    print(f"########## Loading {args.load_model}... ##########")
    load_dict = torch.load(args.load_model, map_location="cpu")
    model.load_state_dict(load_dict)

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[train_callback(args)],
    )

    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)

    trainer.fit(model, data_loader)
