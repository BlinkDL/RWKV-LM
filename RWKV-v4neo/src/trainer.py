import os, math, time, datetime
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.callbacks import TQDMProgressBar

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
            if args.layerwise_lr > 0:
                if self.args.my_pile_mode == 0:
                    param_group["lr"] = lr * param_group["my_lr_scale"]
                elif self.args.my_pile_mode == 2:
                    if param_group["my_lr_scale"] > 1:
                        param_group["lr"] = lr * 5
                    else:
                        param_group["lr"] = lr
                # print(param_group["lr"], param_group["my_lr_scale"])
            else:
                param_group["lr"] = lr

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
