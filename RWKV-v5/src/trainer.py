import os, math, time, datetime, subprocess
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import deepspeed

# xzl: seesm callbacks supplied to Trainer class

def my_save(args, trainer, dd, ff):
    if '14b-run1' in ff:
        fn = ff.split('/')[-1]
        fff = '/dev/shm/' + fn
        torch.save(dd, fff)
        subprocess.Popen(f" aws s3 mv {fff} s3://rwkv-14b-4k/{fn} --quiet", shell=True)
    elif ('world/14b' in ff) or ('world/7b' in ff):
        aa = ff.split('/')[1]
        fn = ff.split('/')[-1]
        fff = f'/dev/shm/{aa}-{fn}'
        torch.save(dd, fff)
        subprocess.Popen(f" aws s3 mv {fff} s3://rwkv-world/{aa}-{fn} --quiet", shell=True)
    else:
        if 'deepspeed_stage_3' in args.strategy:
            trainer.save_checkpoint(ff, weights_only=True)
        else:  # xzl: looks like we hit this line... dd: state_dict
            torch.save(dd, ff)

class train_callback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_after_backward(self, trainer, pl_module):     #xzl add
        args = self.args
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
        
        # xzl -- logging grads ---
        # note deepspeed api
        # If all processes don’t participate these utilities will hang waiting for all processes to send their contribution                
        nlayers=len(pl_module.blocks)
        layers=[nlayers//5, nlayers//2, nlayers-1]
        lll = {}
        logstr = ""
        for ly in layers:
            if 'x052att' in os.environ["RWKV_MY_TESTING"] or 'x052xzl' in os.environ["RWKV_MY_TESTING"]:
                # ---- att ----- # 
                param = pl_module.blocks[ly].att.receptance1.weight
                nm = torch.linalg.matrix_norm(deepspeed.utils.safe_get_full_grad(param)) 
                lll[f"GRAD: layer {ly} att receptance1"] = nm.item()
                logstr += f"layer {ly} receptance.grad {nm.item()}\n"

                param = pl_module.blocks[ly].att.receptance2.weight
                nm = torch.linalg.vector_norm(deepspeed.utils.safe_get_full_grad(param)) 
                lll[f"GRAD: layer {ly} att receptance2"] = nm.item()
                logstr += f"layer {ly} receptance.grad {nm.item()}\n"            

                param = pl_module.blocks[ly].att.key1.weight
                nm = torch.linalg.matrix_norm(deepspeed.utils.safe_get_full_grad(param)) 
                lll[f"GRAD: layer {ly} att key1"] = nm.item()
                logstr += f"layer {ly} key.grad {nm.item()}\n"

                param = pl_module.blocks[ly].att.key2.weight
                nm = torch.linalg.matrix_norm(deepspeed.utils.safe_get_full_grad(param)) 
                lll[f"GRAD: layer {ly} att key2"] = nm.item()
                logstr += f"layer {ly} key.grad {nm.item()}\n"

                param = pl_module.blocks[ly].att.value1.weight
                nm = torch.linalg.matrix_norm(deepspeed.utils.safe_get_full_grad(param)) 
                lll[f"GRAD: layer {ly} att value1"] = nm.item()
                logstr += f"layer {ly} value.grad {nm.item()}\n"

                param = pl_module.blocks[ly].att.value2.weight
                nm = torch.linalg.matrix_norm(deepspeed.utils.safe_get_full_grad(param)) 
                lll[f"GRAD: layer {ly} att value2"] = nm.item()
                logstr += f"layer {ly} value.grad {nm.item()}\n"
                
            if not args.NoDiag: 
                # ---- att.diag ----- (r only, there's more...) # 
                param = pl_module.blocks[ly].att.receptance_diag
                nm = torch.linalg.vector_norm(deepspeed.utils.safe_get_full_grad(param)) 
                lll[f"GRAD: layer {ly} att receptance_diag"] = nm.item()
                logstr += f"layer {ly} receptance.grad {nm.item()}\n"

                # --- ffn diag --- # 
                param = pl_module.blocks[ly].ffn.receptance_diag
                nm = torch.linalg.vector_norm(deepspeed.utils.safe_get_full_grad(param)) 
                lll[f"GRAD: layer {ly} ffn receptance_diag"] = nm.item()
                logstr += f"layer {ly} receptance.grad {nm.item()}\n"


            if 'x052ffn' in os.environ["RWKV_MY_TESTING"] or 'x052xzl' in os.environ["RWKV_MY_TESTING"]:
                # ---- ffn.key ----- # 
                param = pl_module.blocks[ly].ffn.key.weight
                if param.requires_grad:
                    nm = torch.linalg.matrix_norm(deepspeed.utils.safe_get_full_grad(param)) 
                    lll[f"GRAD: layer {ly} ffn key"] = nm.item()

                '''
                param = pl_module.blocks[ly].ffn.key1.weight
                nm = torch.linalg.matrix_norm(deepspeed.utils.safe_get_full_grad(param)) 
                lll[f"GRAD: layer {ly} ffn key1"] = nm.item()
                # logstr += f"layer {ly} key.grad {nm.item()}\n"

                param = pl_module.blocks[ly].ffn.key2.weight
                nm = torch.linalg.matrix_norm(deepspeed.utils.safe_get_full_grad(param)) 
                lll[f"GRAD: layer {ly} ffn key2"] = nm.item()
                # logstr += f"layer {ly} key.grad {nm.item()}\n"
                '''
                
                # ---- ffn.receptance ----- # 
                # breakpoint()
                param = pl_module.blocks[ly].ffn.receptance1.weight
                if param.requires_grad:
                    nm = torch.linalg.matrix_norm(deepspeed.utils.safe_get_full_grad(param)) 
                    lll[f"GRAD: layer {ly} ffn receptance1"] = nm.item()
                    # logstr += f"layer {ly} key.grad {nm.item()}\n"

                param = pl_module.blocks[ly].ffn.receptance2.weight
                if param.requires_grad:
                    nm = torch.linalg.matrix_norm(deepspeed.utils.safe_get_full_grad(param)) 
                    lll[f"GRAD: layer {ly} ffn receptance2"] = nm.item()

                # ---- ffn.value ----- # 
                param = pl_module.blocks[ly].ffn.value.weight
                if param.requires_grad:
                    nm = torch.linalg.matrix_norm(deepspeed.utils.safe_get_full_grad(param)) 
                    lll[f"GRAD: layer {ly} ffn value"] = nm.item()
                '''
                param = pl_module.blocks[ly].ffn.value1.weight
                nm = torch.linalg.matrix_norm(deepspeed.utils.safe_get_full_grad(param)) 
                lll[f"GRAD: layer {ly} ffn value1"] = nm.item()
                # logstr += f"layer {ly} key.grad {nm.item()}\n"

                param = pl_module.blocks[ly].ffn.value2.weight
                nm = torch.linalg.matrix_norm(deepspeed.utils.safe_get_full_grad(param)) 
                lll[f"GRAD: layer {ly} ffn value2"] = nm.item()
                '''
            else: # original 
                param = pl_module.blocks[ly].ffn.receptance.weight
                nm = torch.linalg.matrix_norm(deepspeed.utils.safe_get_full_grad(param)) 
                lll[f"GRAD: layer {ly} ffn receptance"] = nm.item()
                            
        param = pl_module.ln_out.weight
        if param.requires_grad:
            nm = torch.linalg.vector_norm(deepspeed.utils.safe_get_full_grad(param)) 
            lll[f"GRAD: ln_out weight"] = nm.item()

        param = pl_module.head.weight
        if param.requires_grad:
            nm = torch.linalg.matrix_norm(deepspeed.utils.safe_get_full_grad(param)) 
            lll[f"GRAD: head weight"] = nm.item()

        if trainer.is_global_zero:
            # textual... (too much info)
            # if trainer.my_log:
            #     # trainer.my_log.write(f"step: {int(real_step)} grad {nm.item()}\n")
            #     trainer.my_log.write(logstr)
            #     trainer.my_log.flush()
            if len(args.wandb) > 0 and hasattr(trainer, 'my_wandb'):
                trainer.my_wandb.log(lll, step=int(real_step)) 

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        # if args.cuda_cleanup > 0:
        #     torch.cuda.empty_cache()
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        # LR schedule
        w_step = args.warmup_steps
        if args.lr_final == args.lr_init or args.epoch_count == 0:
            lr = args.lr_init
        else:
            decay_step = real_step - args.my_pile_edecay * args.epoch_steps
            decay_total = (args.epoch_count - args.my_pile_edecay) * args.epoch_steps
            progress = (decay_step - w_step + 1) / (decay_total - w_step)
            progress = min(1, max(0, progress))

            if args.lr_final == 0 or args.lr_init == 0:  # linear decay
                lr = args.lr_init + (args.lr_final - args.lr_init) * progress
            else:  # exp decay
                lr = args.lr_init * math.exp(math.log(args.lr_final / args.lr_init) * pow(progress, 1))
            # if trainer.is_global_zero:
            #     print(trainer.global_step, decay_step, decay_total, w_step, progress, lr)

        if args.my_exit_tokens != 0: # cosine decay
            real_tokens = real_step * args.ctx_len * args.real_bsz
            warmup_tokens = w_step * args.ctx_len * args.real_bsz
            progress = (real_tokens - warmup_tokens) / (abs(args.my_exit_tokens) - warmup_tokens)
            progress = max(0, min(1, progress))
            lr_final_factor = args.lr_final / args.lr_init                
            lr_mult = (0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress)
            if args.my_exit_tokens > 0:
                lr = args.lr_init * lr_mult
            else:
                lr = (lr + args.lr_init * lr_mult) / 2
            if progress >= 1:
                if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):
                    my_save(
                        args, trainer,
                        pl_module.state_dict(),
                        f"{args.proj_dir}/rwkv-final.pth",
                    )
                    exit(0)
        if trainer.global_step < w_step:
            lr = lr * (0.2 + 0.8 * trainer.global_step / w_step)

        if args.weight_decay_final > 0:
            wd_now = args.weight_decay * math.exp(math.log(args.weight_decay_final / args.weight_decay) * progress)
        else:
            wd_now = args.weight_decay

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now
            if args.layerwise_lr > 0:
                param_group["lr"] = lr * param_group["my_lr_scale"]
                # print(param_group["lr"], param_group["my_lr_scale"])
            else:
                param_group["lr"] = lr

        trainer.my_lr = lr
        trainer.my_wd = wd_now
        # rank_zero_info(f"{real_step} {lr}")

        if trainer.global_step == 0:
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
                    wandb.init(
                        project=args.wandb,
                        name=args.run_name + " " + args.my_timestamp,
                        # name=args.run_name + " FAC " + f"{FAC}" + args.my_timestamp, # xzl
                        config=args,
                        save_code=False,
                    )
                    trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_step = args.ctx_len * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
        if trainer.is_global_zero:  # logging
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            if pl.__version__[0]=='2':
                trainer.my_loss = outputs["loss"]
            else:
                trainer.my_loss = trainer.my_loss_all.float().mean().item()
            trainer.my_loss_sum += trainer.my_loss          # xzl: trainer will cal loss already?
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)
            # self.log("s", real_step, prog_bar=True, on_step=True)

            if len(args.wandb) > 0:
                lll = {"loss": trainer.my_loss, "lr": trainer.my_lr, "wd": trainer.my_wd, "Gtokens": real_step * token_per_step / 1e9}
                if kt_s > 0:
                    lll["kt/s"] = kt_s  # xzl: k tokens per sec??
                trainer.my_wandb.log(lll, step=int(real_step))
        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy): # save pth
            if args.magic_prime > 0:
                expand_factor = 2 if args.my_qa_mask > 0 else 1
                if int(real_step) == int(args.magic_prime * expand_factor // args.real_bsz) - 1 + int(args.my_random_steps):
                    to_save_dict = pl_module.state_dict()
                    my_save(
                        args, trainer,
                        to_save_dict,
                        f"{args.proj_dir}/rwkv-final.pth",
                    )
                

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        if pl.__version__[0]=='2':
            dataset = trainer.train_dataloader.dataset
        else:
            dataset = trainer.train_dataloader.dataset.datasets
        assert "MyDataset" in str(dataset)
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        dataset.world_size = trainer.world_size
        # print(f'########## world_size {dataset.world_size} global_rank {dataset.global_rank} real_epoch {dataset.real_epoch} ##########')

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        to_save_dict = {}
        save_model_path = ""
        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):  # save pth
            if (args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0) or (trainer.current_epoch == args.epoch_count - 1):
                if args.data_type == 'wds_img':
                    raw_dict = pl_module.state_dict()
                    for k in raw_dict:
                        if k.startswith('encoder.') or k.startswith('decoder.'):
                            to_save_dict[k] = raw_dict[k]
                else:
                    to_save_dict = pl_module.state_dict()
                try:
                    save_model_path = f"{args.proj_dir}/rwkv-{args.epoch_begin + trainer.current_epoch}"
                    my_save(
                        args, trainer,
                        to_save_dict,
                        save_model_path+".pth",
                    )
                except Exception as e:
                    print('Error\n\n', e, '\n\n')

        if trainer.is_global_zero:  # logging
            trainer.my_log.write(f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n")
            trainer.my_log.flush()

            # call lm_eval and log. 
            # NB: save_model_path has no .pth
            if save_model_path != "" and args.lm_eval_n: # we've just saved a model file
                if args.finetune == 1 or 'x052xzl' == os.environ["RWKV_MY_TESTING"]:
                    from src.svd import recover_save
                    eval_model_path = save_model_path + "-recover"
                    recover_save(save_model_path.replace(".pth",""), eval_model_path.replace(".pth",""), 
                                 args.n_layer, args.n_embd)
                else: # pretrain 
                    eval_model_path = save_model_path
                from .run_lm_eval import do_eval
                from .run_lm_eval import clean_cache
                res = do_eval(eval_model_path)
                clean_cache() # otherwise run_lm_eval will cache for future runs

                import json
                trainer.my_log.write(json.dumps(res)+'\n')
                trainer.my_log.flush()

                # works, but no need 
                # if len(args.wandb) > 0 and hasattr(trainer, 'my_wandb'):
                #     args = self.args
                #     real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
                #     trainer.my_wandb.log(res, step=int(real_step)) 

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0
            if (args.epoch_begin + trainer.current_epoch) >= args.my_exit:
                exit(0)


@rank_zero_only
def generate_init_weight(model, init_weight_name):
    mm = model.generate_init_weight()

    if model.args.my_pile_stage == 1:
        if len(model.args.load_model) > 0:
            print(f"Combine weights from {model.args.load_model}...")
            load_dict = torch.load(model.args.load_model, map_location="cpu")
            for k in load_dict:
                try:
                    assert k in mm
                except:
                    print('missing', k)
                    exit(0)
                src = load_dict[k]
                try:
                    mm[k] = src.reshape(mm[k].shape)
                except:
                    tmp = mm[k].squeeze().clone()
                    print(k, src.shape, '-->', mm[k].shape)
                    ss = src.shape[0]
                    dd = tmp.shape[0]
                    for i in range(dd):
                        pos = i / dd * ss
                        if pos >= ss - 1:
                            tmp[i] = src[ss-1]
                        else:
                            p0 = int(math.floor(pos))
                            ii = pos - p0
                            tmp[i] = src[p0] * (1-ii) + src[p0+1] * (ii)
                    mm[k] = tmp.reshape(mm[k].shape)
                    sss = src.squeeze().float().cpu().numpy()
                    print(sss[:10], '...', sss[-10:])
                    mmm = mm[k].squeeze().float().cpu().numpy()
                    print(mmm[:10], '...', mmm[-10:])

    print(f"Save to {init_weight_name}...")
    torch.save(mm, init_weight_name)

    if model.args.my_pile_stage == 1:
        print("Done. Now go for stage 2.")
        exit(0)
