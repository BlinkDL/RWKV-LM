
understand "compressed cls" code

here's chat code using a 0.1B model with compressed cls head
https://github.com/fxlin/RWKV-LM/blob/4dbe46bc3bc5b1a59e68ead30bc1d527af040770/RWKV-v5/src/test-rwkv-chat.py#L34
 
[3:24 PM] Lin, Felix (xl6yq)
here is the code for inference with the "cls" model 
https://github.com/fxlin/RWKV-LM/blob/4dbe46bc3bc5b1a59e68ead30bc1d527af040770/rwkv/model.py#L1705
RWKV-LM/rwkv/model.py at 4dbe46bc3bc5b1a59e68ead30bc1d527af040770 · fxlin/RWKV-LM
RWKV is an RNN with transformer-level LLM performance. It can be directly trained like a GPT (parallelizable). So it&#39;s combining the best of RNN and transformer - great performance, fast in...
 
[3:26 PM] Lin, Felix (xl6yq)
training code: 
https://github.com/fxlin/RWKV-LM/blob/4dbe46bc3bc5b1a59e68ead30bc1d527af040770/RWKV-v5/src/model.py
search for "self.head_l1" "self.head_l2"
RWKV-LM/RWKV-v5/src/model.py at 4dbe46bc3bc5b1a59e68ead30bc1d527af040770 · fxlin/RWKV-LM
RWKV is an RNN with transformer-level LLM performance. It can be directly trained like a GPT (parallelizable). So it&#39;s combining the best of RNN and transformer - great performance, fast in...
 
[3:27 PM] Lin, Felix (xl6yq)
workflow to generate the cls head:
https://github.com/fxlin/RWKV-LM/blob/4dbe46bc3bc5b1a59e68ead30bc1d527af040770/README-xzl-notes
RWKV-LM/README-xzl-notes at 4dbe46bc3bc5b1a59e68ead30bc1d527af040770 · fxlin/RWKV-LM

RWKV is an RNN with transformer-level LLM performance. It can be directly trained like a GPT (parallelizable). So it&#39;s combining the best of RNN and transformer - great performance, fast in...

---------------------------------------------------------------

ffn.receptance1,2 ... if init as 0, then no graidents. (wjhy???                                                       
shall init as scale=1 (norm

---------------------------------------------------------------

clustserd cls head    8/28/2024

TODO: 
- 0.1B: try it on: a newer x59 chkpt, e.g. one that can achieve higher accuracy
- 0.4B: try it out on: x59
- play with [minK, maxK, probs] and see how sensitive they are 


WORKFLOW: 

* create a workdir, e.g. out/01b-cls-mine/    <---- use my version as the template

* pick a pretrained model chkpt (official or ours, can be x52,x58,x59). copy 
  the chkpt to the workdir, e.g. 
    out/01b-cls-mine/from-hpc/rwkv-823.pth

* run svd.py to cluster tokens based on the model's emb.weight, e.g. 

  python3 svd.py --decompose 2 --orig_model out/01b-cls-mine/from-hpc/rwkv-823

  the result is a binary file:  clusters of token labels in the vocab (size=64K), e.g. 
    ---> ./out/01b-cls-mine/from-hpc/rwkv-823-cls.npy

* start training on xsel01/02: run-train.sh 
  (NB: ./submit-train.sh not tested, which may be overkill b/c the training should be 
  fast enough) 

  key training code: src/model.py  
    training_step() around line 1717
      implements several approaches in computing labels & losses, cf comments
        approach 4 with KV divergence loss seems the best 
        apporach 3 with cross entropy (CE) also works, but slightly worse

    three approaches were tried for cls head: 
      forward_cls0:   both head_l1 head_2. not working well 
        (results in: run1-train-both-l1-and-l2/)
      forward_cls1:   only train head_l1 as a single linear layer. works 
        (results in: run2-CE-loss/ run3-KL-loss/)
      forward_cls2:   head_l1 as MLP. trained model generates nonsense. TBD
        (results in: run4-KL-loss-MLP/ run5-KL-loss-MLP-KaimingInit/)
    
* typical training experienkce (xsel02, A6000): 
    CE loss - several hours, loss goes down to ~0.1
    KV loss - several hours, loss goes down to several hundred 
    
* try eval:
  test-rwkv-chat.py
  #Only head.l1 tuned. KL loss (good
  /data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/run3-KL-loss/rwkv-43

    Elon Musk has made a lot of noise about using the same number as the rest of us, and in the process, he has given us a little extra life to enjoy our freedom.
    And so, it is time for Elon Musk to start bringing back the world’s greatest company.
    The company was founded in 1992, by Bill Gates and Jerry Yang, and they have built up a worldwide fortune that has grown since then.
    But this year, Musk has made it clear that he wants to stay true to his principles, not get distracted by another titan.
    The company will build on its initial funding in order to invest in a new generation of high tech companies, but that won’t happen without
    Elon Musk’s help.
    And as much as you may like Elon’s newfound wealth and trust, I urge you to invest in the company right now, instead of keeping your eyes open for a potential one.
    Here are a few ways you can get involved with Elon Musk:
    1)
    stats: runs: 200       cls/run 34.91       tokens/run 0.13

  ^^ last line means: each run (forward pass) only loads ~13% of the total cls 
    head weights   

  run_lm_eval.py 
  #Only head.l1 tuned, KL loss
  # acc: .331 (openai). minK=3, maxK=100, minProb=.95 <--- NEED TO CAREFULLY VERIFY
  MODEL_NAME='/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/run3-KL-loss/rwkv-43'
    
---------------------------------------------------------------
(below old) 

  which specifies load_partial, load_token_cls, and HEAD_K (# of clusters)==200
  it's very slow. b/c my unopt code for training individual cls clusters...
  after one night of training (9 epochs on 4090), loss drops from 1000 to 3.17 
  which matches the orignal cls head

 233/2520 [06:49<1:07:02,  1.76s/it, loss=3.170, lr=6.16e-5, REAL it/s=0.569, Kt/s=4.660]

results: 
out/L12-D768-F4-x052xzlNoReLu-cls
---->  rwkv-66.pth
  
---------------------------------------------------------------
notes on tokenzier and vocab

cf RWKV_v5_demo-svd.py

vocab.txt
each line: idx, token, len_of_token

although # of distinct tokens <65536, vocab_size is set to 65536
for ease (?)

---------------------------------------------------------------

#deploy to lambda VM
# from local machine
rsync.sh

# on VM: 
pip install pytorch-lightning==1.9.5 deepspeed wandb ninja --upgrade 

# install pybind pybind? 
#   local python installation
# pip install pybind11
#  global python?? via apt get
sudo apt -y install python3-pybind11

getdata.sh

# or create a pyenv

---------------------------------------------------------------

#expected by deepspeed installation 
export CUDA_HOME=/sw/ubuntu-22.04/cuda/12.4.0/
pip3 install deepspeed wandb ninja

# to train .... 
source env-xzl.sh
./prep.sh
./run.sh 

command output 

---------------------------------------------------------------

768               blocks.11.ln1.weight
768               blocks.11.ln1.bias
768               blocks.11.ln2.weight
768               blocks.11.ln2.bias
1     1     768   blocks.11.att.time_mix_k
1     1     768   blocks.11.att.time_mix_v
1     1     768   blocks.11.att.time_mix_r
1     1     768   blocks.11.att.time_mix_g
12    64          blocks.11.att.time_decay
12    64          blocks.11.att.time_faaaa
192   768         blocks.11.att.receptance1.weight [scale 1.414]
768   192         blocks.11.att.receptance2.weight [scale 1.414]
192   768         blocks.11.att.key1.weight [scale 0.32]
768   192         blocks.11.att.key2.weight [scale 0.32]
192   768         blocks.11.att.value1.weight [scale 1.414]
768   192         blocks.11.att.value2.weight [scale 1.414]
192   768         blocks.11.att.output1.weight [scale 0]
768   192         blocks.11.att.output2.weight [scale 0]
192   768         blocks.11.att.gate1.weight [scale 0.32]
768   192         blocks.11.att.gate2.weight [scale 0.32]
768               blocks.11.att.ln_x.weight
768               blocks.11.att.ln_x.bias
1     1     768   blocks.11.ffn.time_mix_k
1     1     768   blocks.11.ffn.time_mix_r
2688  768         blocks.11.ffn.key.weight [scale 1.0]
768   768         blocks.11.ffn.receptance.weight [scale 0]
768   2688        blocks.11.ffn.value.weight [scale 0]
768               ln_out.weight
768               ln_out.bias
65536 768         head.weight [scale 4.618802153517006]


---------------------------------------------------------------
own model 
./demo-training-prepare.sh
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.16.0-unknown is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.1build1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 0.1.43ubuntu1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.16.0-unknown is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.1build1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 0.1.43ubuntu1 is an invalid version and will not be supported in a future release
  warnings.warn(
INFO:torch.distributed.nn.jit.instantiator:Created a temporary directory at /tmp/tmpkefkligu
INFO:torch.distributed.nn.jit.instantiator:Writing /tmp/tmpkefkligu/_remote_module_non_scriptable.py
INFO:pytorch_lightning.utilities.rank_zero:########## work in progress ##########
[2024-05-01 20:18:14,091] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Warning: The default cache directory for DeepSpeed Triton autotune, /u/xl6yq/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recom
mended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.                                                                                                                                                                  [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
 [WARNING]  NVIDIA Inference is only supported on Ampere and newer architectures
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.0
 [WARNING]  using untested triton version (2.0.0), only 1.0.0 is known to be compatible
INFO:numexpr.utils:Note: NumExpr detected 40 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
INFO:numexpr.utils:NumExpr defaulting to 8 threads.
INFO:pytorch_lightning.utilities.rank_zero:
############################################################################
#
# RWKV-5 BF16 on 1x1 CPU, bsz 1x1x1=1, deepspeed_stage_2 with grad_cp
#
# Data = data/minipile (binidx), ProjDir = out/L12-D768-x052
#
# Epoch = 0 to 71 (will continue afterwards), save every 1 epoch
#
# Each "epoch" = 40320 steps, 40320 samples, 20643840 tokens
#
# Model = 12 n_layer, 768 n_embd, 512 ctx_len
#
# Adam = lr 1e-05 to 1e-05, warmup 10 steps, beta (0.9, 0.99), eps 1e-08
#
# Found torch 2.0.1+cu117, recommend latest torch
# Found deepspeed 0.14.2, recommend latest deepspeed
# Found pytorch_lightning 1.9.5, recommend 1.9.5
#
############################################################################

INFO:pytorch_lightning.utilities.rank_zero:{'load_model': '', 'wandb': '', 'proj_dir': 'out/L12-D768-x052', 'random_seed': -1, 'train_type': '', 'data_file': 'data/minipile', 'data_type': 'binidx', 'vocab_size': 65536, 'ctx_len': 512, 
'epoch_steps': 40320, 'epoch_count': 72, 'epoch_begin': 0, 'epoch_save': 1, 'micro_bsz': 1, 'n_layer': 12, 'n_embd': 768, 'dim_att': 768, 'dim_ffn': 2688, 'pre_ffn': 0, 'head_qk': 0, 'tiny_att_dim': 0, 'tiny_att_layer': -999, 'lr_init': 1e-05, 'lr_final': 1e-05, 'warmup_steps': 10, 'beta1': 0.9, 'beta2': 0.99, 'adam_eps': 1e-08, 'grad_cp': 1, 'dropout': 0, 'weight_decay': 0.0, 'weight_decay_final': -1, 'my_pile_version': 1, 'my_pile_stage': 1, 'my_pile_shift': 0, 'my_pile_edecay': 0, 'layerwise_lr': 1, 'ds_bucket_mb': 200, 'my_sample_len': 0, 'my_ffn_shift': 1, 'my_att_shift': 1, 'head_size_a': 64, 'head_size_divisor': 8, 'my_pos_emb': 0, 'load_partial': 0, 'magic_prime': 2926181, 'my_qa_mask': 0, 'my_random_steps': 0, 'my_testing': 'x052', 'my_exit': 99999999, 'my_exit_tokens': 1498226207, 'logger': False, 'enable_checkpointing': False, 'default_root_dir': None, 'gradient_clip_val': 1.0, 'gradient_clip_algorithm': None, 'num_nodes': 1, 'num_processes': None, 'devices': '1', 'gpus': None, 'auto_select_gpus': None, 'tpu_cores': None, 'ipus': None, 'enable_progress_bar': True, 'overfit_batches': 0.0, 'track_grad_norm': -1, 'check_val_every_n_epoch': 100000000000000000000, 'fast_dev_run': False, 'accumulate_grad_batches': None, 'max_epochs': -1, 'min_epochs': None, 'max_steps': -1, 'min_steps': None, 'max_time': None, 'limit_train_batches': None, 'limit_val_batches': None, 'limit_test_batches': None, 'limit_predict_batches': None, 'val_check_interval': None, 'log_every_n_steps': 100000000000000000000, 'accelerator': 'cpu', 'strategy': 'deepspeed_stage_2', 'sync_batchnorm': False, 'precision': 'bf16', 'enable_model_summary': True, 'num_sanity_val_steps': 0, 'resume_from_checkpoint': None, 'profiler': None, 'benchmark': None, 'reload_dataloaders_every_n_epochs': 0, 'auto_lr_find': False, 'replace_sampler_ddp': False, 'detect_anomaly': False, 'auto_scale_batch_size': False, 'plugins': None, 'amp_backend': None, 'amp_level': None, 'move_metrics_to_cpu': False, 'multiple_trainloader_mode': 'max_size_cycle', 'inference_mode': True, 'my_timestamp': '2024-05-01-20-18-22', 'betas': (0.9, 0.99), 'real_bsz': 1, 'run_name': '65536 ctx512 L12 D768'}                                                                                                                                                                                   
INFO:pytorch_lightning.utilities.rank_zero:Current vocab size = 65536 (make sure it's correct)
INFO:pytorch_lightning.utilities.rank_zero:Data has 1498226207 tokens.
INFO:pytorch_lightning.utilities.rank_zero:########## Pile 20b-tokenized stage 1 ##########
RWKV_MY_TESTING x052
Using /u/xl6yq/.cache/torch_extensions/py310_cu117 as PyTorch extensions root...
Detected CUDA files, patching ldflags
Emitting ninja build file /u/xl6yq/.cache/torch_extensions/py310_cu117/wkv5/build.ninja...
Building extension module wkv5...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module wkv5...

############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################

65536 768         emb.weight [scale -0.0001]
768               blocks.0.ln1.weight
768               blocks.0.ln1.bias
768               blocks.0.ln2.weight
768               blocks.0.ln2.bias
768               blocks.0.ln0.weight
768               blocks.0.ln0.bias
1     1     768   blocks.0.att.time_mix_k
1     1     768   blocks.0.att.time_mix_v
1     1     768   blocks.0.att.time_mix_r
1     1     768   blocks.0.att.time_mix_g
12    64          blocks.0.att.time_decay
12    64          blocks.0.att.time_faaaa
384   768         blocks.0.att.receptance1.weight [scale 1.0]
768   384         blocks.0.att.receptance2.weight [scale 1.0]
384   768         blocks.0.att.key1.weight [scale 0.1]
768   384         blocks.0.att.key2.weight [scale 0.1]
384   768         blocks.0.att.value1.weight [scale 1.0]
768   384         blocks.0.att.value2.weight [scale 1.0]
384   768         blocks.0.att.output1.weight [scale 0]
768   384         blocks.0.att.output2.weight [scale 0]
384   768         blocks.0.att.gate1.weight [scale 1.0]
768   384         blocks.0.att.gate2.weight [scale 1.0]
768               blocks.0.att.ln_x.weight
768               blocks.0.att.ln_x.bias
1     1     768   blocks.0.ffn.time_mix_k
1     1     768   blocks.0.ffn.time_mix_r
2688  768         blocks.0.ffn.key.weight [scale 1.0]
768   768         blocks.0.ffn.receptance.weight [scale 0]
768   2688        blocks.0.ffn.value.weight [scale 0]
768               blocks.1.ln1.weight
768               blocks.1.ln1.bias
768               blocks.1.ln2.weight
768               blocks.1.ln2.bias
1     1     768   blocks.1.att.time_mix_k
1     1     768   blocks.1.att.time_mix_v
1     1     768   blocks.1.att.time_mix_r
1     1     768   blocks.1.att.time_mix_g
12    64          blocks.1.att.time_decay
12    64          blocks.1.att.time_faaaa
384   768         blocks.1.att.receptance1.weight [scale 1.0]
768   384         blocks.1.att.receptance2.weight [scale 1.0]
384   768         blocks.1.att.key1.weight [scale 0.1]
768   384         blocks.1.att.key2.weight [scale 0.1]
384   768         blocks.1.att.value1.weight [scale 1.0]
768   384         blocks.1.att.value2.weight [scale 1.0]
384   768         blocks.1.att.output1.weight [scale 0]
768   384         blocks.1.att.output2.weight [scale 0]
384   768         blocks.1.att.gate1.weight [scale 1.0]
768   384         blocks.1.att.gate2.weight [scale 1.0]
768               blocks.1.att.ln_x.weight
768               blocks.1.att.ln_x.bias
1     1     768   blocks.1.ffn.time_mix_k
1     1     768   blocks.1.ffn.time_mix_r
2688  768         blocks.1.ffn.key.weight [scale 1.0]
768   768         blocks.1.ffn.receptance.weight [scale 0]
768   2688        blocks.1.ffn.value.weight [scale 0]
768               blocks.2.ln1.weight
768               blocks.2.ln1.bias
768               blocks.2.ln2.weight
768               blocks.2.ln2.bias
1     1     768   blocks.2.att.time_mix_k
1     1     768   blocks.2.att.time_mix_v
1     1     768   blocks.2.att.time_mix_r
1     1     768   blocks.2.att.time_mix_g
12    64          blocks.2.att.time_decay
12    64          blocks.2.att.time_faaaa
384   768         blocks.2.att.receptance1.weight [scale 1.0]
768   384         blocks.2.att.receptance2.weight [scale 1.0]
384   768         blocks.2.att.key1.weight [scale 0.1]
768   384         blocks.2.att.key2.weight [scale 0.1]
384   768         blocks.2.att.value1.weight [scale 1.0]
768   384         blocks.2.att.value2.weight [scale 1.0]
384   768         blocks.2.att.output1.weight [scale 0]
768   384         blocks.2.att.output2.weight [scale 0]
384   768         blocks.2.att.gate1.weight [scale 1.0]
768   384         blocks.2.att.gate2.weight [scale 1.0]
768               blocks.2.att.ln_x.weight
768               blocks.2.att.ln_x.bias
1     1     768   blocks.2.ffn.time_mix_k
1     1     768   blocks.2.ffn.time_mix_r
2688  768         blocks.2.ffn.key.weight [scale 1.0]
768   768         blocks.2.ffn.receptance.weight [scale 0]
768   2688        blocks.2.ffn.value.weight [scale 0]
768               blocks.3.ln1.weight
768               blocks.3.ln1.bias
768               blocks.3.ln2.weight
768               blocks.3.ln2.bias
1     1     768   blocks.3.att.time_mix_k
1     1     768   blocks.3.att.time_mix_v
1     1     768   blocks.3.att.time_mix_r
1     1     768   blocks.3.att.time_mix_g
12    64          blocks.3.att.time_decay
12    64          blocks.3.att.time_faaaa
384   768         blocks.3.att.receptance1.weight [scale 1.0]
768   384         blocks.3.att.receptance2.weight [scale 1.0]
384   768         blocks.3.att.key1.weight [scale 0.1]
768   384         blocks.3.att.key2.weight [scale 0.1]
384   768         blocks.3.att.value1.weight [scale 1.0]
768   384         blocks.3.att.value2.weight [scale 1.0]
384   768         blocks.3.att.output1.weight [scale 0]
768   384         blocks.3.att.output2.weight [scale 0]
384   768         blocks.3.att.gate1.weight [scale 1.0]
768   384         blocks.3.att.gate2.weight [scale 1.0]
768               blocks.3.att.ln_x.weight
768               blocks.3.att.ln_x.bias
1     1     768   blocks.3.ffn.time_mix_k
1     1     768   blocks.3.ffn.time_mix_r
2688  768         blocks.3.ffn.key.weight [scale 1.0]
768   768         blocks.3.ffn.receptance.weight [scale 0]
768   2688        blocks.3.ffn.value.weight [scale 0]
768               blocks.4.ln1.weight
768               blocks.4.ln1.bias
768               blocks.4.ln2.weight
768               blocks.4.ln2.bias
1     1     768   blocks.4.att.time_mix_k
1     1     768   blocks.4.att.time_mix_v
1     1     768   blocks.4.att.time_mix_r
1     1     768   blocks.4.att.time_mix_g
12    64          blocks.4.att.time_decay
12    64          blocks.4.att.time_faaaa
384   768         blocks.4.att.receptance1.weight [scale 1.0]
768   384         blocks.4.att.receptance2.weight [scale 1.0]
384   768         blocks.4.att.key1.weight [scale 0.1]
768   384         blocks.4.att.key2.weight [scale 0.1]
384   768         blocks.4.att.value1.weight [scale 1.0]
768   384         blocks.4.att.value2.weight [scale 1.0]
384   768         blocks.4.att.output1.weight [scale 0]
768   384         blocks.4.att.output2.weight [scale 0]
384   768         blocks.4.att.gate1.weight [scale 1.0]
768   384         blocks.4.att.gate2.weight [scale 1.0]
768               blocks.4.att.ln_x.weight
768               blocks.4.att.ln_x.bias
1     1     768   blocks.4.ffn.time_mix_k
1     1     768   blocks.4.ffn.time_mix_r
2688  768         blocks.4.ffn.key.weight [scale 1.0]
768   768         blocks.4.ffn.receptance.weight [scale 0]
768   2688        blocks.4.ffn.value.weight [scale 0]
768               blocks.5.ln1.weight
768               blocks.5.ln1.bias
768               blocks.5.ln2.weight
768               blocks.5.ln2.bias
1     1     768   blocks.5.att.time_mix_k
1     1     768   blocks.5.att.time_mix_v
1     1     768   blocks.5.att.time_mix_r
1     1     768   blocks.5.att.time_mix_g
12    64          blocks.5.att.time_decay
12    64          blocks.5.att.time_faaaa
384   768         blocks.5.att.receptance1.weight [scale 1.0]
768   384         blocks.5.att.receptance2.weight [scale 1.0]
384   768         blocks.5.att.key1.weight [scale 0.1]
768   384         blocks.5.att.key2.weight [scale 0.1]
384   768         blocks.5.att.value1.weight [scale 1.0]
768   384         blocks.5.att.value2.weight [scale 1.0]
384   768         blocks.5.att.output1.weight [scale 0]
768   384         blocks.5.att.output2.weight [scale 0]
384   768         blocks.5.att.gate1.weight [scale 1.0]
768   384         blocks.5.att.gate2.weight [scale 1.0]
768               blocks.5.att.ln_x.weight
768               blocks.5.att.ln_x.bias
1     1     768   blocks.5.ffn.time_mix_k
1     1     768   blocks.5.ffn.time_mix_r
2688  768         blocks.5.ffn.key.weight [scale 1.0]
768   768         blocks.5.ffn.receptance.weight [scale 0]
768   2688        blocks.5.ffn.value.weight [scale 0]
768               blocks.6.ln1.weight
768               blocks.6.ln1.bias
768               blocks.6.ln2.weight
768               blocks.6.ln2.bias
1     1     768   blocks.6.att.time_mix_k
1     1     768   blocks.6.att.time_mix_v
1     1     768   blocks.6.att.time_mix_r
1     1     768   blocks.6.att.time_mix_g
12    64          blocks.6.att.time_decay
12    64          blocks.6.att.time_faaaa
384   768         blocks.6.att.receptance1.weight [scale 1.0]
768   384         blocks.6.att.receptance2.weight [scale 1.0]
384   768         blocks.6.att.key1.weight [scale 0.1]
768   384         blocks.6.att.key2.weight [scale 0.1]
384   768         blocks.6.att.value1.weight [scale 1.0]
768   384         blocks.6.att.value2.weight [scale 1.0]
384   768         blocks.6.att.output1.weight [scale 0]
768   384         blocks.6.att.output2.weight [scale 0]
384   768         blocks.6.att.gate1.weight [scale 1.0]
768   384         blocks.6.att.gate2.weight [scale 1.0]
768               blocks.6.att.ln_x.weight
768               blocks.6.att.ln_x.bias
1     1     768   blocks.6.ffn.time_mix_k
1     1     768   blocks.6.ffn.time_mix_r
2688  768         blocks.6.ffn.key.weight [scale 1.0]
768   768         blocks.6.ffn.receptance.weight [scale 0]
768   2688        blocks.6.ffn.value.weight [scale 0]
768               blocks.7.ln1.weight
768               blocks.7.ln1.bias
768               blocks.7.ln2.weight
768               blocks.7.ln2.bias
1     1     768   blocks.7.att.time_mix_k
1     1     768   blocks.7.att.time_mix_v
1     1     768   blocks.7.att.time_mix_r
1     1     768   blocks.7.att.time_mix_g
12    64          blocks.7.att.time_decay
12    64          blocks.7.att.time_faaaa
384   768         blocks.7.att.receptance1.weight [scale 1.0]
768   384         blocks.7.att.receptance2.weight [scale 1.0]
384   768         blocks.7.att.key1.weight [scale 0.1]
768   384         blocks.7.att.key2.weight [scale 0.1]
384   768         blocks.7.att.value1.weight [scale 1.0]
768   384         blocks.7.att.value2.weight [scale 1.0]
384   768         blocks.7.att.output1.weight [scale 0]
768   384         blocks.7.att.output2.weight [scale 0]
384   768         blocks.7.att.gate1.weight [scale 1.0]
768   384         blocks.7.att.gate2.weight [scale 1.0]
768               blocks.7.att.ln_x.weight
768               blocks.7.att.ln_x.bias
1     1     768   blocks.7.ffn.time_mix_k
1     1     768   blocks.7.ffn.time_mix_r
2688  768         blocks.7.ffn.key.weight [scale 1.0]
768   768         blocks.7.ffn.receptance.weight [scale 0]
768   2688        blocks.7.ffn.value.weight [scale 0]
768               blocks.8.ln1.weight
768               blocks.8.ln1.bias
768               blocks.8.ln2.weight
768               blocks.8.ln2.bias
1     1     768   blocks.8.att.time_mix_k
1     1     768   blocks.8.att.time_mix_v
1     1     768   blocks.8.att.time_mix_r
1     1     768   blocks.8.att.time_mix_g
12    64          blocks.8.att.time_decay
12    64          blocks.8.att.time_faaaa
384   768         blocks.8.att.receptance1.weight [scale 1.0]
768   384         blocks.8.att.receptance2.weight [scale 1.0]
384   768         blocks.8.att.key1.weight [scale 0.1]
768   384         blocks.8.att.key2.weight [scale 0.1]
384   768         blocks.8.att.value1.weight [scale 1.0]
768   384         blocks.8.att.value2.weight [scale 1.0]
384   768         blocks.8.att.output1.weight [scale 0]
768   384         blocks.8.att.output2.weight [scale 0]
384   768         blocks.8.att.gate1.weight [scale 1.0]
768   384         blocks.8.att.gate2.weight [scale 1.0]
768               blocks.8.att.ln_x.weight
768               blocks.8.att.ln_x.bias
1     1     768   blocks.8.ffn.time_mix_k
1     1     768   blocks.8.ffn.time_mix_r
2688  768         blocks.8.ffn.key.weight [scale 1.0]
768   768         blocks.8.ffn.receptance.weight [scale 0]
768   2688        blocks.8.ffn.value.weight [scale 0]
768               blocks.9.ln1.weight
768               blocks.9.ln1.bias
768               blocks.9.ln2.weight
768               blocks.9.ln2.bias
1     1     768   blocks.9.att.time_mix_k
1     1     768   blocks.9.att.time_mix_v
1     1     768   blocks.9.att.time_mix_r
1     1     768   blocks.9.att.time_mix_g
12    64          blocks.9.att.time_decay
12    64          blocks.9.att.time_faaaa
384   768         blocks.9.att.receptance1.weight [scale 1.0]
768   384         blocks.9.att.receptance2.weight [scale 1.0]
384   768         blocks.9.att.key1.weight [scale 0.1]
768   384         blocks.9.att.key2.weight [scale 0.1]
384   768         blocks.9.att.value1.weight [scale 1.0]
768   384         blocks.9.att.value2.weight [scale 1.0]
384   768         blocks.9.att.output1.weight [scale 0]
768   384         blocks.9.att.output2.weight [scale 0]
384   768         blocks.9.att.gate1.weight [scale 1.0]
768   384         blocks.9.att.gate2.weight [scale 1.0]
768               blocks.9.att.ln_x.weight
768               blocks.9.att.ln_x.bias
1     1     768   blocks.9.ffn.time_mix_k
1     1     768   blocks.9.ffn.time_mix_r
2688  768         blocks.9.ffn.key.weight [scale 1.0]
768   768         blocks.9.ffn.receptance.weight [scale 0]
768   2688        blocks.9.ffn.value.weight [scale 0]
768               blocks.10.ln1.weight
768               blocks.10.ln1.bias
768               blocks.10.ln2.weight
768               blocks.10.ln2.bias
1     1     768   blocks.10.att.time_mix_k
1     1     768   blocks.10.att.time_mix_v
1     1     768   blocks.10.att.time_mix_r
1     1     768   blocks.10.att.time_mix_g
12    64          blocks.10.att.time_decay
12    64          blocks.10.att.time_faaaa
384   768         blocks.10.att.receptance1.weight [scale 1.0]
768   384         blocks.10.att.receptance2.weight [scale 1.0]
384   768         blocks.10.att.key1.weight [scale 0.1]
768   384         blocks.10.att.key2.weight [scale 0.1]
384   768         blocks.10.att.value1.weight [scale 1.0]
768   384         blocks.10.att.value2.weight [scale 1.0]
384   768         blocks.10.att.output1.weight [scale 0]
768   384         blocks.10.att.output2.weight [scale 0]
384   768         blocks.10.att.gate1.weight [scale 1.0]
768   384         blocks.10.att.gate2.weight [scale 1.0]
768               blocks.10.att.ln_x.weight
768               blocks.10.att.ln_x.bias
1     1     768   blocks.10.ffn.time_mix_k
1     1     768   blocks.10.ffn.time_mix_r
2688  768         blocks.10.ffn.key.weight [scale 1.0]
768   768         blocks.10.ffn.receptance.weight [scale 0]
768   2688        blocks.10.ffn.value.weight [scale 0]
768               blocks.11.ln1.weight
768               blocks.11.ln1.bias
768               blocks.11.ln2.weight
768               blocks.11.ln2.bias
1     1     768   blocks.11.att.time_mix_k
1     1     768   blocks.11.att.time_mix_v
1     1     768   blocks.11.att.time_mix_r
1     1     768   blocks.11.att.time_mix_g
12    64          blocks.11.att.time_decay
12    64          blocks.11.att.time_faaaa
384   768         blocks.11.att.receptance1.weight [scale 1.0]
768   384         blocks.11.att.receptance2.weight [scale 1.0]
384   768         blocks.11.att.key1.weight [scale 0.1]
768   384         blocks.11.att.key2.weight [scale 0.1]
384   768         blocks.11.att.value1.weight [scale 1.0]
768   384         blocks.11.att.value2.weight [scale 1.0]
384   768         blocks.11.att.output1.weight [scale 0]
768   384         blocks.11.att.output2.weight [scale 0]
384   768         blocks.11.att.gate1.weight [scale 1.0]
768   384         blocks.11.att.gate2.weight [scale 1.0]
768               blocks.11.att.ln_x.weight
768               blocks.11.att.ln_x.bias
1     1     768   blocks.11.ffn.time_mix_k
1     1     768   blocks.11.ffn.time_mix_r
2688  768         blocks.11.ffn.key.weight [scale 1.0]
768   768         blocks.11.ffn.receptance.weight [scale 0]
768   2688        blocks.11.ffn.value.weight [scale 0]
768               ln_out.weight
768               ln_out.bias
65536 768         head.weight [scale 4.618802153517006]
model params 192807936
Save to out/L12-D768-x052/rwkv-init.pth...
Done. Now go for stage 2.











~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
./demo-training-prepare.sh

/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.16.0-unknown is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.1build1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 0.1.43ubuntu1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.16.0-unknown is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.1build1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 0.1.43ubuntu1 is an invalid version and will not be supported in a future release
  warnings.warn(
INFO:torch.distributed.nn.jit.instantiator:Created a temporary directory at /tmp/tmpijr47csv
INFO:torch.distributed.nn.jit.instantiator:Writing /tmp/tmpijr47csv/_remote_module_non_scriptable.py
INFO:pytorch_lightning.utilities.rank_zero:########## work in progress ##########
[2024-04-29 15:00:52,347] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Warning: The default cache directory for DeepSpeed Triton autotune, /u/xl6yq/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when 
DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.                                                                                                          [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
 [WARNING]  NVIDIA Inference is only supported on Ampere and newer architectures
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.0
 [WARNING]  using untested triton version (2.0.0), only 1.0.0 is known to be compatible
INFO:numexpr.utils:Note: NumExpr detected 40 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
INFO:numexpr.utils:NumExpr defaulting to 8 threads.
INFO:pytorch_lightning.utilities.rank_zero:
############################################################################
#
# RWKV-5 BF16 on 1x1 CPU, bsz 1x1x1=1, deepspeed_stage_2 with grad_cp
#
# Data = data/minipile (binidx), ProjDir = out/L12-D768-x052
#
# Epoch = 0 to 71 (will continue afterwards), save every 1 epoch
#
# Each "epoch" = 40320 steps, 40320 samples, 20643840 tokens
#
# Model = 12 n_layer, 768 n_embd, 512 ctx_len
#
# Adam = lr 1e-05 to 1e-05, warmup 10 steps, beta (0.9, 0.99), eps 1e-08
#
# Found torch 2.0.1+cu117, recommend latest torch
# Found deepspeed 0.14.2, recommend latest deepspeed
# Found pytorch_lightning 1.9.5, recommend 1.9.5
#
############################################################################

INFO:pytorch_lightning.utilities.rank_zero:{'load_model': '', 'wandb': '', 'proj_dir': 'out/L12-D768-x052', 'random_seed': -1, 'train_type': '', 'data_file': 'data/minipile', 'data_type': 'binidx', 'vocab_si
ze': 65536, 'ctx_len': 512, 'epoch_steps': 40320, 'epoch_count': 72, 'epoch_begin': 0, 'epoch_save': 1, 'micro_bsz': 1, 'n_layer': 12, 'n_embd': 768, 'dim_att': 768, 'dim_ffn': 2688, 'pre_ffn': 0, 'head_qk': 0, 'tiny_att_dim': 0, 'tiny_att_layer': -999, 'lr_init': 1e-05, 'lr_final': 1e-05, 'warmup_steps': 10, 'beta1': 0.9, 'beta2': 0.99, 'adam_eps': 1e-08, 'grad_cp': 1, 'dropout': 0, 'weight_decay': 0.0, 'weight_decay_final': -1, 'my_pile_version': 1, 'my_pile_stage': 1, 'my_pile_shift': 0, 'my_pile_edecay': 0, 'layerwise_lr': 1, 'ds_bucket_mb': 200, 'my_sample_len': 0, 'my_ffn_shift': 1, 'my_att_shift': 1, 'head_size_a': 64, 'head_size_divisor': 8, 'my_pos_emb': 0, 'load_partial': 0, 'magic_prime': 2926181, 'my_qa_mask': 0, 'my_random_steps': 0, 'my_testing': 'x052', 'my_exit': 99999999, 'my_exit_tokens': 1498226207, 'logger': False, 'enable_checkpointing': False, 'default_root_dir': None, 'gradient_clip_val': 1.0, 'gradient_clip_algorithm': None, 'num_nodes': 1, 'num_processes': None, 'devices': '1', 'gpus': None, 'auto_select_gpus': None, 'tpu_cores': None, 'ipus': None, 'enable_progress_bar': True, 'overfit_batches': 0.0, 'track_grad_norm': -1, 'check_val_every_n_epoch': 100000000000000000000, 'fast_dev_run': False, 'accumulate_grad_batches': None, 'max_epochs': -1, 'min_epochs': None, 'max_steps': -1, 'min_steps': None, 'max_time': None, 'limit_train_batches': None, 'limit_val_batches': None, 'limit_test_batches': None, 'limit_predict_batches': None, 'val_check_interval': None, 'log_every_n_steps': 100000000000000000000, 'accelerator': 'cpu', 'strategy': 'deepspeed_stage_2', 'sync_batchnorm': False, 'precision': 'bf16', 'enable_model_summary': True, 'num_sanity_val_steps': 0, 'resume_from_checkpoint': None, 'profiler': None, 'benchmark': None, 'reload_dataloaders_every_n_epochs': 0, 'auto_lr_find': False, 'replace_sampler_ddp': False, 'detect_anomaly': False, 'auto_scale_batch_size': False, 'plugins': None, 'amp_backend': None, 'amp_level': None, 'move_metrics_to_cpu': False, 'multiple_trainloader_mode': 'max_size_cycle', 'inference_mode': True, 'my_timestamp': '2024-04-29-15-01-02', 'betas': (0.9, 0.99), 'real_bsz': 1, 'run_name': '65536 ctx512 L12 D768'}                                                                              
INFO:pytorch_lightning.utilities.rank_zero:Current vocab size = 65536 (make sure it's correct)
INFO:pytorch_lightning.utilities.rank_zero:Data has 1498226207 tokens.
INFO:pytorch_lightning.utilities.rank_zero:########## Pile 20b-tokenized stage 1 ##########
RWKV_MY_TESTING x052
Using /u/xl6yq/.cache/torch_extensions/py310_cu117 as PyTorch extensions root...
Creating extension directory /u/xl6yq/.cache/torch_extensions/py310_cu117/wkv5...
Detected CUDA files, patching ldflags
Emitting ninja build file /u/xl6yq/.cache/torch_extensions/py310_cu117/wkv5/build.ninja...
Building extension module wkv5...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)


[1/3] c++ -MMD -MF wkv5_op.o.d -DTORCH_EXTENSION_NAME=wkv5 -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /u/
xl6yq/.local/lib/python3.10/site-packages/torch/include -isystem /u/xl6yq/.local/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /u/xl6yq/.local/lib/python3.10/site-packages/torch/include/TH -isystem /u/xl6yq/.local/lib/python3.10/site-packages/torch/include/THC -isystem /sw/ubuntu-22.04/cuda/12.4.0/include -isystem /usr/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -c /u/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/cuda/wkv5_op.cpp -o wkv5_op.o                                                                                                                                        [2/3] /sw/ubuntu-22.04/cuda/12.4.0/bin/nvcc  -DTORCH_EXTENSION_NAME=wkv5 -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\
" -isystem /u/xl6yq/.local/lib/python3.10/site-packages/torch/include -isystem /u/xl6yq/.local/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /u/xl6yq/.local/lib/python3.10/site-packages/torch/include/TH -isystem /u/xl6yq/.local/lib/python3.10/site-packages/torch/include/THC -isystem /sw/ubuntu-22.04/cuda/12.4.0/include -isystem /usr/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 --compiler-options '-fPIC' -res-usage --use_fast_math -O3 -Xptxas -O3 --extra-device-vectorization -D_N_=64 -std=c++17 -c /u/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/cuda/wkv5_cuda.cu -o wkv5_cuda.cuda.o                                                                                                                                                                                                       ptxas info    : 3 bytes gmem, 24 bytes cmem[4]
ptxas info    : Compiling entry function '_Z15kernel_backwardIN3c108BFloat16EEviiiiPKT_S4_S4_PKfS6_S4_S4_PS2_S7_S7_S7_S7_' for 'sm_75'
ptxas info    : Function properties for _Z15kernel_backwardIN3c108BFloat16EEviiiiPKT_S4_S4_PKfS6_S4_S4_PS2_S7_S7_S7_S7_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 168 registers, 1536 bytes smem, 464 bytes cmem[0]
ptxas info    : Compiling entry function '_Z14kernel_forwardIN3c108BFloat16EEviiiiPKT_S4_S4_PKfS4_PS2_' for 'sm_75'
ptxas info    : Function properties for _Z14kernel_forwardIN3c108BFloat16EEviiiiPKT_S4_S4_PKfS4_PS2_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 102 registers, 1024 bytes smem, 416 bytes cmem[0]
[3/3] c++ wkv5_op.o wkv5_cuda.cuda.o -shared -L/u/xl6yq/.local/lib/python3.10/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -L/sw/ubuntu-22.04/cuda/12.4.0/lib64 -lc
udart -o wkv5.so                                                                                                                                                                                               Loading extension module wkv5...

############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################

65536 768         emb.weight [scale -0.0001]
768               blocks.0.ln1.weight
768               blocks.0.ln1.bias
768               blocks.0.ln2.weight
768               blocks.0.ln2.bias
768               blocks.0.ln0.weight
768               blocks.0.ln0.bias
1     1     768   blocks.0.att.time_mix_k
1     1     768   blocks.0.att.time_mix_v
1     1     768   blocks.0.att.time_mix_r
1     1     768   blocks.0.att.time_mix_g
12    64          blocks.0.att.time_decay
12    64          blocks.0.att.time_faaaa
768   768         blocks.0.att.receptance.weight [scale 1.0]
768   768         blocks.0.att.key.weight [scale 0.1]
768   768         blocks.0.att.value.weight [scale 1.0]
768   768         blocks.0.att.output.weight [scale 0]
768   768         blocks.0.att.gate.weight [scale 0.1]
768               blocks.0.att.ln_x.weight
768               blocks.0.att.ln_x.bias
1     1     768   blocks.0.ffn.time_mix_k
1     1     768   blocks.0.ffn.time_mix_r
2688  768         blocks.0.ffn.key.weight [scale 1.0]
768   768         blocks.0.ffn.receptance.weight [scale 0]
768   2688        blocks.0.ffn.value.weight [scale 0]
768               blocks.1.ln1.weight
768               blocks.1.ln1.bias
768               blocks.1.ln2.weight
768               blocks.1.ln2.bias
1     1     768   blocks.1.att.time_mix_k
1     1     768   blocks.1.att.time_mix_v
1     1     768   blocks.1.att.time_mix_r
1     1     768   blocks.1.att.time_mix_g
12    64          blocks.1.att.time_decay
12    64          blocks.1.att.time_faaaa
768   768         blocks.1.att.receptance.weight [scale 1.0]
768   768         blocks.1.att.key.weight [scale 0.1]
768   768         blocks.1.att.value.weight [scale 1.0]
768   768         blocks.1.att.output.weight [scale 0]
768   768         blocks.1.att.gate.weight [scale 0.1]
768               blocks.1.att.ln_x.weight
768               blocks.1.att.ln_x.bias
1     1     768   blocks.1.ffn.time_mix_k
1     1     768   blocks.1.ffn.time_mix_r
2688  768         blocks.1.ffn.key.weight [scale 1.0]
768   768         blocks.1.ffn.receptance.weight [scale 0]
768   2688        blocks.1.ffn.value.weight [scale 0]
768               blocks.2.ln1.weight
768               blocks.2.ln1.bias
768               blocks.2.ln2.weight
768               blocks.2.ln2.bias
1     1     768   blocks.2.att.time_mix_k
1     1     768   blocks.2.att.time_mix_v
1     1     768   blocks.2.att.time_mix_r
1     1     768   blocks.2.att.time_mix_g
12    64          blocks.2.att.time_decay
12    64          blocks.2.att.time_faaaa
768   768         blocks.2.att.receptance.weight [scale 1.0]
768   768         blocks.2.att.key.weight [scale 0.1]
768   768         blocks.2.att.value.weight [scale 1.0]
768   768         blocks.2.att.output.weight [scale 0]
768   768         blocks.2.att.gate.weight [scale 0.1]
768               blocks.2.att.ln_x.weight
768               blocks.2.att.ln_x.bias
1     1     768   blocks.2.ffn.time_mix_k
1     1     768   blocks.2.ffn.time_mix_r
2688  768         blocks.2.ffn.key.weight [scale 1.0]
768   768         blocks.2.ffn.receptance.weight [scale 0]
768   2688        blocks.2.ffn.value.weight [scale 0]
768               blocks.3.ln1.weight
768               blocks.3.ln1.bias
768               blocks.3.ln2.weight
768               blocks.3.ln2.bias
1     1     768   blocks.3.att.time_mix_k
1     1     768   blocks.3.att.time_mix_v
1     1     768   blocks.3.att.time_mix_r
1     1     768   blocks.3.att.time_mix_g
12    64          blocks.3.att.time_decay
12    64          blocks.3.att.time_faaaa
768   768         blocks.3.att.receptance.weight [scale 1.0]
768   768         blocks.3.att.key.weight [scale 0.1]
768   768         blocks.3.att.value.weight [scale 1.0]
768   768         blocks.3.att.output.weight [scale 0]
768   768         blocks.3.att.gate.weight [scale 0.1]
768               blocks.3.att.ln_x.weight
768               blocks.3.att.ln_x.bias
1     1     768   blocks.3.ffn.time_mix_k
1     1     768   blocks.3.ffn.time_mix_r
2688  768         blocks.3.ffn.key.weight [scale 1.0]
768   768         blocks.3.ffn.receptance.weight [scale 0]
768   2688        blocks.3.ffn.value.weight [scale 0]
768               blocks.4.ln1.weight
768               blocks.4.ln1.bias
768               blocks.4.ln2.weight
768               blocks.4.ln2.bias
1     1     768   blocks.4.att.time_mix_k
1     1     768   blocks.4.att.time_mix_v
1     1     768   blocks.4.att.time_mix_r
1     1     768   blocks.4.att.time_mix_g
12    64          blocks.4.att.time_decay
12    64          blocks.4.att.time_faaaa
768   768         blocks.4.att.receptance.weight [scale 1.0]
768   768         blocks.4.att.key.weight [scale 0.1]
768   768         blocks.4.att.value.weight [scale 1.0]
768   768         blocks.4.att.output.weight [scale 0]
768   768         blocks.4.att.gate.weight [scale 0.1]
768               blocks.4.att.ln_x.weight
768               blocks.4.att.ln_x.bias
1     1     768   blocks.4.ffn.time_mix_k
1     1     768   blocks.4.ffn.time_mix_r
2688  768         blocks.4.ffn.key.weight [scale 1.0]
768   768         blocks.4.ffn.receptance.weight [scale 0]
768   2688        blocks.4.ffn.value.weight [scale 0]
768               blocks.5.ln1.weight
768               blocks.5.ln1.bias
768               blocks.5.ln2.weight
768               blocks.5.ln2.bias
1     1     768   blocks.5.att.time_mix_k
1     1     768   blocks.5.att.time_mix_v
1     1     768   blocks.5.att.time_mix_r
1     1     768   blocks.5.att.time_mix_g
12    64          blocks.5.att.time_decay
12    64          blocks.5.att.time_faaaa
768   768         blocks.5.att.receptance.weight [scale 1.0]
768   768         blocks.5.att.key.weight [scale 0.1]
768   768         blocks.5.att.value.weight [scale 1.0]
768   768         blocks.5.att.output.weight [scale 0]
768   768         blocks.5.att.gate.weight [scale 0.1]
768               blocks.5.att.ln_x.weight
768               blocks.5.att.ln_x.bias
1     1     768   blocks.5.ffn.time_mix_k
1     1     768   blocks.5.ffn.time_mix_r
2688  768         blocks.5.ffn.key.weight [scale 1.0]
768   768         blocks.5.ffn.receptance.weight [scale 0]
768   2688        blocks.5.ffn.value.weight [scale 0]
768               blocks.6.ln1.weight
768               blocks.6.ln1.bias
768               blocks.6.ln2.weight
768               blocks.6.ln2.bias
1     1     768   blocks.6.att.time_mix_k
1     1     768   blocks.6.att.time_mix_v
1     1     768   blocks.6.att.time_mix_r
1     1     768   blocks.6.att.time_mix_g
12    64          blocks.6.att.time_decay
12    64          blocks.6.att.time_faaaa
768   768         blocks.6.att.receptance.weight [scale 1.0]
768   768         blocks.6.att.key.weight [scale 0.1]
768   768         blocks.6.att.value.weight [scale 1.0]
768   768         blocks.6.att.output.weight [scale 0]
768   768         blocks.6.att.gate.weight [scale 0.1]
768               blocks.6.att.ln_x.weight
768               blocks.6.att.ln_x.bias
1     1     768   blocks.6.ffn.time_mix_k
1     1     768   blocks.6.ffn.time_mix_r
2688  768         blocks.6.ffn.key.weight [scale 1.0]
768   768         blocks.6.ffn.receptance.weight [scale 0]
768   2688        blocks.6.ffn.value.weight [scale 0]
768               blocks.7.ln1.weight
768               blocks.7.ln1.bias
768               blocks.7.ln2.weight
768               blocks.7.ln2.bias
1     1     768   blocks.7.att.time_mix_k
1     1     768   blocks.7.att.time_mix_v
1     1     768   blocks.7.att.time_mix_r
1     1     768   blocks.7.att.time_mix_g
12    64          blocks.7.att.time_decay
12    64          blocks.7.att.time_faaaa
768   768         blocks.7.att.receptance.weight [scale 1.0]
768   768         blocks.7.att.key.weight [scale 0.1]
768   768         blocks.7.att.value.weight [scale 1.0]
768   768         blocks.7.att.output.weight [scale 0]
768   768         blocks.7.att.gate.weight [scale 0.1]
768               blocks.7.att.ln_x.weight
768               blocks.7.att.ln_x.bias
1     1     768   blocks.7.ffn.time_mix_k
1     1     768   blocks.7.ffn.time_mix_r
2688  768         blocks.7.ffn.key.weight [scale 1.0]
768   768         blocks.7.ffn.receptance.weight [scale 0]
768   2688        blocks.7.ffn.value.weight [scale 0]
768               blocks.8.ln1.weight
768               blocks.8.ln1.bias
768               blocks.8.ln2.weight
768               blocks.8.ln2.bias
1     1     768   blocks.8.att.time_mix_k
1     1     768   blocks.8.att.time_mix_v
1     1     768   blocks.8.att.time_mix_r
1     1     768   blocks.8.att.time_mix_g
12    64          blocks.8.att.time_decay
12    64          blocks.8.att.time_faaaa
768   768         blocks.8.att.receptance.weight [scale 1.0]
768   768         blocks.8.att.key.weight [scale 0.1]
768   768         blocks.8.att.value.weight [scale 1.0]
768   768         blocks.8.att.output.weight [scale 0]
768   768         blocks.8.att.gate.weight [scale 0.1]
768               blocks.8.att.ln_x.weight
768               blocks.8.att.ln_x.bias
1     1     768   blocks.8.ffn.time_mix_k
1     1     768   blocks.8.ffn.time_mix_r
2688  768         blocks.8.ffn.key.weight [scale 1.0]
768   768         blocks.8.ffn.receptance.weight [scale 0]
768   2688        blocks.8.ffn.value.weight [scale 0]
768               blocks.9.ln1.weight
768               blocks.9.ln1.bias
768               blocks.9.ln2.weight
768               blocks.9.ln2.bias
1     1     768   blocks.9.att.time_mix_k
1     1     768   blocks.9.att.time_mix_v
1     1     768   blocks.9.att.time_mix_r
1     1     768   blocks.9.att.time_mix_g
12    64          blocks.9.att.time_decay
12    64          blocks.9.att.time_faaaa
768   768         blocks.9.att.receptance.weight [scale 1.0]
768   768         blocks.9.att.key.weight [scale 0.1]
768   768         blocks.9.att.value.weight [scale 1.0]
768   768         blocks.9.att.output.weight [scale 0]
768   768         blocks.9.att.gate.weight [scale 0.1]
768               blocks.9.att.ln_x.weight
768               blocks.9.att.ln_x.bias
1     1     768   blocks.9.ffn.time_mix_k
1     1     768   blocks.9.ffn.time_mix_r
2688  768         blocks.9.ffn.key.weight [scale 1.0]
768   768         blocks.9.ffn.receptance.weight [scale 0]
768   2688        blocks.9.ffn.value.weight [scale 0]
768               blocks.10.ln1.weight
768               blocks.10.ln1.bias
768               blocks.10.ln2.weight
768               blocks.10.ln2.bias
1     1     768   blocks.10.att.time_mix_k
1     1     768   blocks.10.att.time_mix_v
1     1     768   blocks.10.att.time_mix_r
1     1     768   blocks.10.att.time_mix_g
12    64          blocks.10.att.time_decay
12    64          blocks.10.att.time_faaaa
768   768         blocks.10.att.receptance.weight [scale 1.0]
768   768         blocks.10.att.key.weight [scale 0.1]
768   768         blocks.10.att.value.weight [scale 1.0]
768   768         blocks.10.att.output.weight [scale 0]
768   768         blocks.10.att.gate.weight [scale 0.1]
768               blocks.10.att.ln_x.weight
768               blocks.10.att.ln_x.bias
1     1     768   blocks.10.ffn.time_mix_k
1     1     768   blocks.10.ffn.time_mix_r
2688  768         blocks.10.ffn.key.weight [scale 1.0]
768   768         blocks.10.ffn.receptance.weight [scale 0]
768   2688        blocks.10.ffn.value.weight [scale 0]
768               blocks.11.ln1.weight
768               blocks.11.ln1.bias
768               blocks.11.ln2.weight
768               blocks.11.ln2.bias
1     1     768   blocks.11.att.time_mix_k
1     1     768   blocks.11.att.time_mix_v
1     1     768   blocks.11.att.time_mix_r
1     1     768   blocks.11.att.time_mix_g
12    64          blocks.11.att.time_decay
12    64          blocks.11.att.time_faaaa
768   768         blocks.11.att.receptance.weight [scale 1.0]
768   768         blocks.11.att.key.weight [scale 0.1]
768   768         blocks.11.att.value.weight [scale 1.0]
768   768         blocks.11.att.output.weight [scale 0]
768   768         blocks.11.att.gate.weight [scale 0.1]
768               blocks.11.att.ln_x.weight
768               blocks.11.att.ln_x.bias
1     1     768   blocks.11.ffn.time_mix_k
1     1     768   blocks.11.ffn.time_mix_r
2688  768         blocks.11.ffn.key.weight [scale 1.0]
768   768         blocks.11.ffn.receptance.weight [scale 0]
768   2688        blocks.11.ffn.value.weight [scale 0]
768               ln_out.weight
768               ln_out.bias
65536 768         head.weight [scale 4.618802153517006]
model params 192807936
Save to out/L12-D768-x052/rwkv-init.pth...
Done. Now go for stage 2.






xl6yq@gpusrv10 (main)[RWKV-v5]$ ./demo-training-run.sh
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.16.0-unknown is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.1build1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 0.1.43ubuntu1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.16.0-unknown is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.1build1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 0.1.43ubuntu1 is an invalid version and will not be supported in a future release
  warnings.warn(
INFO:torch.distributed.nn.jit.instantiator:Created a temporary directory at /tmp/tmpfcs7h8ef
INFO:torch.distributed.nn.jit.instantiator:Writing /tmp/tmpfcs7h8ef/_remote_module_non_scriptable.py
INFO:pytorch_lightning.utilities.rank_zero:########## work in progress ##########
[2024-05-01 20:26:47,702] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Warning: The default cache directory for DeepSpeed Triton autotune, /u/xl6yq/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
 [WARNING]  NVIDIA Inference is only supported on Ampere and newer architectures
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.0
 [WARNING]  using untested triton version (2.0.0), only 1.0.0 is known to be compatible
INFO:numexpr.utils:Note: NumExpr detected 40 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
INFO:numexpr.utils:NumExpr defaulting to 8 threads.
INFO:pytorch_lightning.utilities.rank_zero:
############################################################################
#
# RWKV-5 BF16 on 1x4 GPU, bsz 1x4x8=32, deepspeed_stage_2 with grad_cp
#
# Data = data/minipile (binidx), ProjDir = out/L12-D768-x052
#
# Epoch = 0 to 71 (will continue afterwards), save every 10 epoch
#
# Each "epoch" = 1260 steps, 40320 samples, 20643840 tokens
#
# Model = 12 n_layer, 768 n_embd, 512 ctx_len
#
# Adam = lr 0.0006 to 6e-05, warmup 10 steps, beta (0.9, 0.99), eps 1e-08
#
# Found torch 2.0.1+cu117, recommend latest torch
# Found deepspeed 0.14.2, recommend latest deepspeed
# Found pytorch_lightning 1.9.5, recommend 1.9.5
#
############################################################################

INFO:pytorch_lightning.utilities.rank_zero:{'load_model': 'out/L12-D768-x052/rwkv-init.pth', 'wandb': 'rwkv', 'proj_dir': 'out/L12-D768-x052', 'random_seed': -1, 'train_type': '', 'data_file': 'data/minipile', 'data_type': 'binidx', 'vocab_size': 65536, 'ctx_len': 512, 'epoch_steps': 1260, 'epoch_count': 72, 'epoch_begin': 0, 'epoch_save': 10, 'micro_bsz': 8, 'n_layer': 12, 'n_embd': 768, 'dim_att': 768, 'dim_ffn': 2688, 'pre_ffn': 0, 'head_qk': 0, 'tiny_att_dim': 0, 'tiny_att_layer': -999, 'lr_init': 0.0006, 'lr_final': 6e-05, 'warmup_steps': 10, 'beta1': 0.9, 'beta2': 0.99, 'adam_eps': 1e-08, 'grad_cp': 1, 'dropout': 0, 'weight_decay': 0.001, 'weight_decay_final': -1, 'my_pile_version': 1, 'my_pile_stage': 3, 'my_pile_shift': 0, 'my_pile_edecay': 0, 'layerwise_lr': 1, 'ds_bucket_mb': 2, 'my_sample_len': 0, 'my_ffn_shift': 1, 'my_att_shift': 1, 'head_size_a': 64, 'head_size_divisor': 8, 'my_pos_emb': 0, 'load_partial': 0, 'magic_prime': 2926181, 'my_qa_mask': 0, 'my_random_steps': 0, 'my_testing': 'x052', 'my_exit': 99999999, 'my_exit_tokens': 1498226207, 'logger': False, 'enable_checkpointing': False, 'default_root_dir': None, 'gradient_clip_val': 1.0, 'gradient_clip_algorithm': None, 'num_nodes': 1, 'num_processes': None, 'devices': '4', 'gpus': None, 'auto_select_gpus': None, 'tpu_cores': None, 'ipus': None, 'enable_progress_bar': True, 'overfit_batches': 0.0, 'track_grad_norm': -1, 'check_val_every_n_epoch': 100000000000000000000, 'fast_dev_run': False, 'accumulate_grad_batches': None, 'max_epochs': -1, 'min_epochs': None, 'max_steps': -1, 'min_steps': None, 'max_time': None, 'limit_train_batches': None, 'limit_val_batches': None, 'limit_test_batches': None, 'limit_predict_batches': None, 'val_check_interval': None, 'log_every_n_steps': 100000000000000000000, 'accelerator': 'gpu', 'strategy': 'deepspeed_stage_2', 'sync_batchnorm': False, 'precision': 'bf16', 'enable_model_summary': True, 'num_sanity_val_steps': 0, 'resume_from_checkpoint': None, 'profiler': None, 'benchmark': None, 'reload_dataloaders_every_n_epochs': 0, 'auto_lr_find': False, 'replace_sampler_ddp': False, 'detect_anomaly': False, 'auto_scale_batch_size': False, 'plugins': None, 'amp_backend': None, 'amp_level': None, 'move_metrics_to_cpu': False, 'multiple_trainloader_mode': 'max_size_cycle', 'inference_mode': True, 'my_timestamp': '2024-05-01-20-27-27', 'betas': (0.9, 0.99), 'real_bsz': 32, 'run_name': '65536 ctx512 L12 D768'}

INFO:pytorch_lightning.utilities.rank_zero:Current vocab size = 65536 (make sure it's correct)
INFO:pytorch_lightning.utilities.rank_zero:Data has 1498226207 tokens.
INFO:pytorch_lightning.utilities.rank_zero:########## Pile 20b-tokenized stage 3 ##########
RWKV_MY_TESTING x052
Using /u/xl6yq/.cache/torch_extensions/py310_cu117 as PyTorch extensions root...
Detected CUDA files, patching ldflags
Emitting ninja build file /u/xl6yq/.cache/torch_extensions/py310_cu117/wkv5/build.ninja...
Building extension module wkv5...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module wkv5...
INFO:pytorch_lightning.utilities.rank_zero:########## Loading out/L12-D768-x052/rwkv-init.pth... ##########
INFO:pytorch_lightning.utilities.rank_zero:GPU available: True (cuda), used: True
INFO:pytorch_lightning.utilities.rank_zero:TPU available: False, using: 0 TPU cores
INFO:pytorch_lightning.utilities.rank_zero:IPU available: False, using: 0 IPUs
INFO:pytorch_lightning.utilities.rank_zero:HPU available: False, using: 0 HPUs
65536 768         emb.weight
768               blocks.0.ln1.weight
768               blocks.0.ln1.bias
768               blocks.0.ln2.weight
768               blocks.0.ln2.bias
768               blocks.0.ln0.weight
768               blocks.0.ln0.bias
768               blocks.0.att.time_mix_k
768               blocks.0.att.time_mix_v
768               blocks.0.att.time_mix_r
768               blocks.0.att.time_mix_g
12    64          blocks.0.att.time_decay
12    64          blocks.0.att.time_faaaa
384   768         blocks.0.att.receptance1.weight
768   384         blocks.0.att.receptance2.weight
384   768         blocks.0.att.key1.weight
768   384         blocks.0.att.key2.weight
384   768         blocks.0.att.value1.weight
768   384         blocks.0.att.value2.weight
384   768         blocks.0.att.output1.weight
768   384         blocks.0.att.output2.weight
384   768         blocks.0.att.gate1.weight
768   384         blocks.0.att.gate2.weight
768               blocks.0.att.ln_x.weight
768               blocks.0.att.ln_x.bias
768               blocks.0.ffn.time_mix_k
768               blocks.0.ffn.time_mix_r
2688  768         blocks.0.ffn.key.weight
768   768         blocks.0.ffn.receptance.weight
768   2688        blocks.0.ffn.value.weight
768               blocks.1.ln1.weight
768               blocks.1.ln1.bias
768               blocks.1.ln2.weight
768               blocks.1.ln2.bias
768               blocks.1.att.time_mix_k
768               blocks.1.att.time_mix_v
768               blocks.1.att.time_mix_r
768               blocks.1.att.time_mix_g
12    64          blocks.1.att.time_decay
12    64          blocks.1.att.time_faaaa
384   768         blocks.1.att.receptance1.weight
768   384         blocks.1.att.receptance2.weight
384   768         blocks.1.att.key1.weight
768   384         blocks.1.att.key2.weight
384   768         blocks.1.att.value1.weight
768   384         blocks.1.att.value2.weight
384   768         blocks.1.att.output1.weight
768   384         blocks.1.att.output2.weight
384   768         blocks.1.att.gate1.weight
768   384         blocks.1.att.gate2.weight
768               blocks.1.att.ln_x.weight
768               blocks.1.att.ln_x.bias
768               blocks.1.ffn.time_mix_k
768               blocks.1.ffn.time_mix_r
2688  768         blocks.1.ffn.key.weight
768   768         blocks.1.ffn.receptance.weight
768   2688        blocks.1.ffn.value.weight
768               blocks.2.ln1.weight
768               blocks.2.ln1.bias
768               blocks.2.ln2.weight
768               blocks.2.ln2.bias
768               blocks.2.att.time_mix_k
768               blocks.2.att.time_mix_v
768               blocks.2.att.time_mix_r
768               blocks.2.att.time_mix_g
12    64          blocks.2.att.time_decay
12    64          blocks.2.att.time_faaaa
384   768         blocks.2.att.receptance1.weight
768   384         blocks.2.att.receptance2.weight
384   768         blocks.2.att.key1.weight
768   384         blocks.2.att.key2.weight
384   768         blocks.2.att.value1.weight
768   384         blocks.2.att.value2.weight
384   768         blocks.2.att.output1.weight
768   384         blocks.2.att.output2.weight
384   768         blocks.2.att.gate1.weight
768   384         blocks.2.att.gate2.weight
768               blocks.2.att.ln_x.weight
768               blocks.2.att.ln_x.bias
768               blocks.2.ffn.time_mix_k
768               blocks.2.ffn.time_mix_r
2688  768         blocks.2.ffn.key.weight
768   768         blocks.2.ffn.receptance.weight
768   2688        blocks.2.ffn.value.weight
768               blocks.3.ln1.weight
768               blocks.3.ln1.bias
768               blocks.3.ln2.weight
768               blocks.3.ln2.bias
768               blocks.3.att.time_mix_k
768               blocks.3.att.time_mix_v
768               blocks.3.att.time_mix_r
768               blocks.3.att.time_mix_g
12    64          blocks.3.att.time_decay
12    64          blocks.3.att.time_faaaa
384   768         blocks.3.att.receptance1.weight
768   384         blocks.3.att.receptance2.weight
384   768         blocks.3.att.key1.weight
768   384         blocks.3.att.key2.weight
384   768         blocks.3.att.value1.weight
768   384         blocks.3.att.value2.weight
384   768         blocks.3.att.output1.weight
768   384         blocks.3.att.output2.weight
384   768         blocks.3.att.gate1.weight
768   384         blocks.3.att.gate2.weight
768               blocks.3.att.ln_x.weight
768               blocks.3.att.ln_x.bias
768               blocks.3.ffn.time_mix_k
768               blocks.3.ffn.time_mix_r
2688  768         blocks.3.ffn.key.weight
768   768         blocks.3.ffn.receptance.weight
768   2688        blocks.3.ffn.value.weight
768               blocks.4.ln1.weight
768               blocks.4.ln1.bias
768               blocks.4.ln2.weight
768               blocks.4.ln2.bias
768               blocks.4.att.time_mix_k
768               blocks.4.att.time_mix_v
768               blocks.4.att.time_mix_r
768               blocks.4.att.time_mix_g
12    64          blocks.4.att.time_decay
12    64          blocks.4.att.time_faaaa
384   768         blocks.4.att.receptance1.weight
768   384         blocks.4.att.receptance2.weight
384   768         blocks.4.att.key1.weight
768   384         blocks.4.att.key2.weight
384   768         blocks.4.att.value1.weight
768   384         blocks.4.att.value2.weight
384   768         blocks.4.att.output1.weight
768   384         blocks.4.att.output2.weight
384   768         blocks.4.att.gate1.weight
768   384         blocks.4.att.gate2.weight
768               blocks.4.att.ln_x.weight
768               blocks.4.att.ln_x.bias
768               blocks.4.ffn.time_mix_k
768               blocks.4.ffn.time_mix_r
2688  768         blocks.4.ffn.key.weight
768   768         blocks.4.ffn.receptance.weight
768   2688        blocks.4.ffn.value.weight
768               blocks.5.ln1.weight
768               blocks.5.ln1.bias
768               blocks.5.ln2.weight
768               blocks.5.ln2.bias
768               blocks.5.att.time_mix_k
768               blocks.5.att.time_mix_v
768               blocks.5.att.time_mix_r
768               blocks.5.att.time_mix_g
12    64          blocks.5.att.time_decay
12    64          blocks.5.att.time_faaaa
384   768         blocks.5.att.receptance1.weight
768   384         blocks.5.att.receptance2.weight
384   768         blocks.5.att.key1.weight
768   384         blocks.5.att.key2.weight
384   768         blocks.5.att.value1.weight
768   384         blocks.5.att.value2.weight
384   768         blocks.5.att.output1.weight
768   384         blocks.5.att.output2.weight
384   768         blocks.5.att.gate1.weight
768   384         blocks.5.att.gate2.weight
768               blocks.5.att.ln_x.weight
768               blocks.5.att.ln_x.bias
768               blocks.5.ffn.time_mix_k
768               blocks.5.ffn.time_mix_r
2688  768         blocks.5.ffn.key.weight
768   768         blocks.5.ffn.receptance.weight
768   2688        blocks.5.ffn.value.weight
768               blocks.6.ln1.weight
768               blocks.6.ln1.bias
768               blocks.6.ln2.weight
768               blocks.6.ln2.bias
768               blocks.6.att.time_mix_k
768               blocks.6.att.time_mix_v
768               blocks.6.att.time_mix_r
768               blocks.6.att.time_mix_g
12    64          blocks.6.att.time_decay
12    64          blocks.6.att.time_faaaa
384   768         blocks.6.att.receptance1.weight
768   384         blocks.6.att.receptance2.weight
384   768         blocks.6.att.key1.weight
768   384         blocks.6.att.key2.weight
384   768         blocks.6.att.value1.weight
768   384         blocks.6.att.value2.weight
384   768         blocks.6.att.output1.weight
768   384         blocks.6.att.output2.weight
384   768         blocks.6.att.gate1.weight
768   384         blocks.6.att.gate2.weight
768               blocks.6.att.ln_x.weight
768               blocks.6.att.ln_x.bias
768               blocks.6.ffn.time_mix_k
768               blocks.6.ffn.time_mix_r
2688  768         blocks.6.ffn.key.weight
768   768         blocks.6.ffn.receptance.weight
768   2688        blocks.6.ffn.value.weight
768               blocks.7.ln1.weight
768               blocks.7.ln1.bias
768               blocks.7.ln2.weight
768               blocks.7.ln2.bias
768               blocks.7.att.time_mix_k
768               blocks.7.att.time_mix_v
768               blocks.7.att.time_mix_r
768               blocks.7.att.time_mix_g
12    64          blocks.7.att.time_decay
12    64          blocks.7.att.time_faaaa
384   768         blocks.7.att.receptance1.weight
768   384         blocks.7.att.receptance2.weight
384   768         blocks.7.att.key1.weight
768   384         blocks.7.att.key2.weight
384   768         blocks.7.att.value1.weight
768   384         blocks.7.att.value2.weight
384   768         blocks.7.att.output1.weight
768   384         blocks.7.att.output2.weight
384   768         blocks.7.att.gate1.weight
768   384         blocks.7.att.gate2.weight
768               blocks.7.att.ln_x.weight
768               blocks.7.att.ln_x.bias
768               blocks.7.ffn.time_mix_k
768               blocks.7.ffn.time_mix_r
2688  768         blocks.7.ffn.key.weight
768   768         blocks.7.ffn.receptance.weight
768   2688        blocks.7.ffn.value.weight
768               blocks.8.ln1.weight
768               blocks.8.ln1.bias
768               blocks.8.ln2.weight
768               blocks.8.ln2.bias
768               blocks.8.att.time_mix_k
768               blocks.8.att.time_mix_v
768               blocks.8.att.time_mix_r
768               blocks.8.att.time_mix_g
12    64          blocks.8.att.time_decay
12    64          blocks.8.att.time_faaaa
384   768         blocks.8.att.receptance1.weight
768   384         blocks.8.att.receptance2.weight
384   768         blocks.8.att.key1.weight
768   384         blocks.8.att.key2.weight
384   768         blocks.8.att.value1.weight
768   384         blocks.8.att.value2.weight
384   768         blocks.8.att.output1.weight
768   384         blocks.8.att.output2.weight
384   768         blocks.8.att.gate1.weight
768   384         blocks.8.att.gate2.weight
768               blocks.8.att.ln_x.weight
768               blocks.8.att.ln_x.bias
768               blocks.8.ffn.time_mix_k
768               blocks.8.ffn.time_mix_r
2688  768         blocks.8.ffn.key.weight
768   768         blocks.8.ffn.receptance.weight
768   2688        blocks.8.ffn.value.weight
768               blocks.9.ln1.weight
768               blocks.9.ln1.bias
768               blocks.9.ln2.weight
768               blocks.9.ln2.bias
768               blocks.9.att.time_mix_k
768               blocks.9.att.time_mix_v
768               blocks.9.att.time_mix_r
768               blocks.9.att.time_mix_g
12    64          blocks.9.att.time_decay
12    64          blocks.9.att.time_faaaa
384   768         blocks.9.att.receptance1.weight
768   384         blocks.9.att.receptance2.weight
384   768         blocks.9.att.key1.weight
768   384         blocks.9.att.key2.weight
384   768         blocks.9.att.value1.weight
768   384         blocks.9.att.value2.weight
384   768         blocks.9.att.output1.weight
768   384         blocks.9.att.output2.weight
384   768         blocks.9.att.gate1.weight
768   384         blocks.9.att.gate2.weight
768               blocks.9.att.ln_x.weight
768               blocks.9.att.ln_x.bias
768               blocks.9.ffn.time_mix_k
768               blocks.9.ffn.time_mix_r
2688  768         blocks.9.ffn.key.weight
768   768         blocks.9.ffn.receptance.weight
768   2688        blocks.9.ffn.value.weight
768               blocks.10.ln1.weight
768               blocks.10.ln1.bias
768               blocks.10.ln2.weight
768               blocks.10.ln2.bias
768               blocks.10.att.time_mix_k
768               blocks.10.att.time_mix_v
768               blocks.10.att.time_mix_r
768               blocks.10.att.time_mix_g
12    64          blocks.10.att.time_decay
12    64          blocks.10.att.time_faaaa
384   768         blocks.10.att.receptance1.weight
768   384         blocks.10.att.receptance2.weight
384   768         blocks.10.att.key1.weight
768   384         blocks.10.att.key2.weight
384   768         blocks.10.att.value1.weight
768   384         blocks.10.att.value2.weight
384   768         blocks.10.att.output1.weight
768   384         blocks.10.att.output2.weight
384   768         blocks.10.att.gate1.weight
768   384         blocks.10.att.gate2.weight
768               blocks.10.att.ln_x.weight
768               blocks.10.att.ln_x.bias
768               blocks.10.ffn.time_mix_k
768               blocks.10.ffn.time_mix_r
2688  768         blocks.10.ffn.key.weight
768   768         blocks.10.ffn.receptance.weight
768   2688        blocks.10.ffn.value.weight
768               blocks.11.ln1.weight
768               blocks.11.ln1.bias
768               blocks.11.ln2.weight
768               blocks.11.ln2.bias
768               blocks.11.att.time_mix_k
768               blocks.11.att.time_mix_v
768               blocks.11.att.time_mix_r
768               blocks.11.att.time_mix_g
12    64          blocks.11.att.time_decay
12    64          blocks.11.att.time_faaaa
384   768         blocks.11.att.receptance1.weight
768   384         blocks.11.att.receptance2.weight
384   768         blocks.11.att.key1.weight
768   384         blocks.11.att.key2.weight
384   768         blocks.11.att.value1.weight
768   384         blocks.11.att.value2.weight
384   768         blocks.11.att.output1.weight
768   384         blocks.11.att.output2.weight
384   768         blocks.11.att.gate1.weight
768   384         blocks.11.att.gate2.weight
768               blocks.11.att.ln_x.weight
768               blocks.11.att.ln_x.bias
768               blocks.11.ffn.time_mix_k
768               blocks.11.ffn.time_mix_r
2688  768         blocks.11.ffn.key.weight
768   768         blocks.11.ffn.receptance.weight
768   2688        blocks.11.ffn.value.weight
768               ln_out.weight
768               ln_out.bias
65536 768         head.weight
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.16.0-unknown is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.1build1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 0.1.43ubuntu1 is an invalid version and will not be supported in a future release
  warnings.warn(
INFO:pytorch_lightning.strategies.deepspeed:initializing deepspeed distributed: GLOBAL_RANK: 0, MEMBER: 1/4
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.16.0-unknown is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.1build1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 0.1.43ubuntu1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.16.0-unknown is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.1build1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 0.1.43ubuntu1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.16.0-unknown is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.1build1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 0.1.43ubuntu1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.16.0-unknown is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.1build1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 0.1.43ubuntu1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.16.0-unknown is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.1build1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 0.1.43ubuntu1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.16.0-unknown is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.1build1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 0.1.43ubuntu1 is an invalid version and will not be supported in a future release
  warnings.warn(
INFO:torch.distributed.nn.jit.instantiator:Created a temporary directory at /tmp/tmpacbc1npl
INFO:torch.distributed.nn.jit.instantiator:Created a temporary directory at /tmp/tmpwnasdj26
INFO:torch.distributed.nn.jit.instantiator:Writing /tmp/tmpwnasdj26/_remote_module_non_scriptable.py
INFO:torch.distributed.nn.jit.instantiator:Writing /tmp/tmpacbc1npl/_remote_module_non_scriptable.py
INFO:torch.distributed.nn.jit.instantiator:Created a temporary directory at /tmp/tmpdd_4vuy1
INFO:torch.distributed.nn.jit.instantiator:Writing /tmp/tmpdd_4vuy1/_remote_module_non_scriptable.py
[2024-05-01 20:27:51,346] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-05-01 20:27:51,356] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-05-01 20:27:51,370] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Warning: The default cache directory for DeepSpeed Triton autotune, /u/xl6yq/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
Warning: The default cache directory for DeepSpeed Triton autotune, /u/xl6yq/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
Warning: The default cache directory for DeepSpeed Triton autotune, /u/xl6yq/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
 [WARNING]  NVIDIA Inference is only supported on Ampere and newer architectures
 [WARNING]  NVIDIA Inference is only supported on Ampere and newer architectures
 [WARNING]  NVIDIA Inference is only supported on Ampere and newer architectures
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.0
 [WARNING]  using untested triton version (2.0.0), only 1.0.0 is known to be compatible
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.0
 [WARNING]  using untested triton version (2.0.0), only 1.0.0 is known to be compatible
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.0
 [WARNING]  using untested triton version (2.0.0), only 1.0.0 is known to be compatible
INFO:numexpr.utils:Note: NumExpr detected 40 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
INFO:numexpr.utils:Note: NumExpr detected 40 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
INFO:numexpr.utils:NumExpr defaulting to 8 threads.
INFO:numexpr.utils:NumExpr defaulting to 8 threads.
INFO:numexpr.utils:Note: NumExpr detected 40 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
INFO:numexpr.utils:NumExpr defaulting to 8 threads.
RWKV_MY_TESTING x052
Using /u/xl6yq/.cache/torch_extensions/py310_cu117 as PyTorch extensions root...
RWKV_MY_TESTING x052
Using /u/xl6yq/.cache/torch_extensions/py310_cu117 as PyTorch extensions root...
RWKV_MY_TESTING x052
Using /u/xl6yq/.cache/torch_extensions/py310_cu117 as PyTorch extensions root...
Detected CUDA files, patching ldflags
Emitting ninja build file /u/xl6yq/.cache/torch_extensions/py310_cu117/wkv5/build.ninja...
Building extension module wkv5...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module wkv5...
Loading extension module wkv5...
Loading extension module wkv5...
INFO:pytorch_lightning.strategies.deepspeed:initializing deepspeed distributed: GLOBAL_RANK: 3, MEMBER: 4/4
INFO:torch.distributed.distributed_c10d:Added key: store_based_barrier_key:1 to store for rank: 3
INFO:pytorch_lightning.strategies.deepspeed:initializing deepspeed distributed: GLOBAL_RANK: 1, MEMBER: 2/4
INFO:torch.distributed.distributed_c10d:Added key: store_based_barrier_key:1 to store for rank: 1
INFO:pytorch_lightning.strategies.deepspeed:initializing deepspeed distributed: GLOBAL_RANK: 2, MEMBER: 3/4
INFO:torch.distributed.distributed_c10d:Added key: store_based_barrier_key:1 to store for rank: 2
INFO:torch.distributed.distributed_c10d:Added key: store_based_barrier_key:1 to store for rank: 0
INFO:torch.distributed.distributed_c10d:Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
INFO:pytorch_lightning.utilities.rank_zero:Enabling DeepSpeed BF16.
INFO:torch.distributed.distributed_c10d:Rank 1: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
INFO:torch.distributed.distributed_c10d:Rank 2: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
INFO:torch.distributed.distributed_c10d:Rank 3: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 2 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 3 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
decay ['blocks.0.att.gate1.weight', 'blocks.0.att.gate2.weight', 'blocks.0.att.key1.weight', 'blocks.0.att.key2.weight', 'blocks.0.att.output1.weight', 'blocks.0.att.output2.weight', 'blocks.0.att.receptance1.weight', 'blocks.0.att.receptance2.weight', 'blocks.0.att.value1.weight', 'blocks.0.att.value2.weight', 'blocks.0.ffn.key.weight', 'blocks.0.ffn.receptance.weight', 'blocks.0.ffn.value.weight', 'blocks.1.att.gate1.weight', 'blocks.1.att.gate2.weight', 'blocks.1.att.key1.weight', 'blocks.1.att.key2.weight', 'blocks.1.att.output1.weight', 'blocks.1.att.output2.weight', 'blocks.1.att.receptance1.weight', 'blocks.1.att.receptance2.weight', 'blocks.1.att.value1.weight', 'blocks.1.att.value2.weight', 'blocks.1.ffn.key.weight', 'blocks.1.ffn.receptance.weight', 'blocks.1.ffn.value.weight', 'blocks.10.att.gate1.weight', 'blocks.10.att.gate2.weight', 'blocks.10.att.key1.weight', 'blocks.10.att.key2.weight', 'blocks.10.att.output1.weight', 'blocks.10.att.output2.weight', 'blocks.10.att.receptance1.weight', 'blocks.10.att.receptance2.weight', 'blocks.10.att.value1.weight', 'blocks.10.att.value2.weight', 'blocks.10.ffn.key.weight', 'blocks.10.ffn.receptance.weight', 'blocks.10.ffn.value.weight', 'blocks.11.att.gate1.weight', 'blocks.11.att.gate2.weight', 'blocks.11.att.key1.weight', 'blocks.11.att.key2.weight', 'blocks.11.att.output1.weight', 'blocks.11.att.output2.weight', 'blocks.11.att.receptance1.weight', 'blocks.11.att.receptance2.weight', 'blocks.11.att.value1.weight', 'blocks.11.att.value2.weight', 'blocks.11.ffn.key.weight', 'blocks.11.ffn.receptance.weight', 'blocks.11.ffn.value.weight', 'blocks.2.att.gate1.weight', 'blocks.2.att.gate2.weight', 'blocks.2.att.key1.weight', 'blocks.2.att.key2.weight', 'blocks.2.att.output1.weight', 'blocks.2.att.output2.weight', 'blocks.2.att.receptance1.weight', 'blocks.2.att.receptance2.weight', 'blocks.2.att.value1.weight', 'blocks.2.att.value2.weight', 'blocks.2.ffn.key.weight', 'blocks.2.ffn.receptance.weight', 'blocks.2.ffn.value.weight', 'blocks.3.att.gate1.weight', 'blocks.3.att.gate2.weight', 'blocks.3.att.key1.weight', 'blocks.3.att.key2.weight', 'blocks.3.att.output1.weight', 'blocks.3.att.output2.weight', 'blocks.3.att.receptance1.weight', 'blocks.3.att.receptance2.weight', 'blocks.3.att.value1.weight', 'blocks.3.att.value2.weight', 'blocks.3.ffn.key.weight', 'blocks.3.ffn.receptance.weight', 'blocks.3.ffn.value.weight', 'blocks.4.att.gate1.weight', 'blocks.4.att.gate2.weight', 'blocks.4.att.key1.weight', 'blocks.4.att.key2.weight', 'blocks.4.att.output1.weight', 'blocks.4.att.output2.weight', 'blocks.4.att.receptance1.weight', 'blocks.4.att.receptance2.weight', 'blocks.4.att.value1.weight', 'blocks.4.att.value2.weight', 'blocks.4.ffn.key.weight', 'blocks.4.ffn.receptance.weight', 'blocks.4.ffn.value.weight', 'blocks.5.att.gate1.weight', 'blocks.5.att.gate2.weight', 'blocks.5.att.key1.weight', 'blocks.5.att.key2.weight', 'blocks.5.att.output1.weight', 'blocks.5.att.output2.weight', 'blocks.5.att.receptance1.weight', 'blocks.5.att.receptance2.weight', 'blocks.5.att.value1.weight', 'blocks.5.att.value2.weight', 'blocks.5.ffn.key.weight', 'blocks.5.ffn.receptance.weight', 'blocks.5.ffn.value.weight', 'blocks.6.att.gate1.weight', 'blocks.6.att.gate2.weight', 'blocks.6.att.key1.weight', 'blocks.6.att.key2.weight', 'blocks.6.att.output1.weight', 'blocks.6.att.output2.weight', 'blocks.6.att.receptance1.weight', 'blocks.6.att.receptance2.weight', 'blocks.6.att.value1.weight', 'blocks.6.att.value2.weight', 'blocks.6.ffn.key.weight', 'blocks.6.ffn.receptance.weight', 'blocks.6.ffn.value.weight', 'blocks.7.att.gate1.weight', 'blocks.7.att.gate2.weight', 'blocks.7.att.key1.weight', 'blocks.7.att.key2.weight', 'blocks.7.att.output1.weight', 'blocks.7.att.output2.weight', 'blocks.7.att.receptance1.weight', 'blocks.7.att.receptance2.weight', 'blocks.7.att.value1.weight', 'blocks.7.att.value2.weight', 'blocks.7.ffn.key.weight', 'blocks.7.ffn.receptance.weight', 'blocks.7.ffn.value.weight', 'blocks.8.att.gate1.weight', 'blocks.8.att.gate2.weight', 'blocks.8.att.key1.weight', 'blocks.8.att.key2.weight', 'blocks.8.att.output1.weight', 'blocks.8.att.output2.weight', 'blocks.8.att.receptance1.weight', 'blocks.8.att.receptance2.weight', 'blocks.8.att.value1.weight', 'blocks.8.att.value2.weight', 'blocks.8.ffn.key.weight', 'blocks.8.ffn.receptance.weight', 'blocks.8.ffn.value.weight', 'blocks.9.att.gate1.weight', 'blocks.9.att.gate2.weight', 'blocks.9.att.key1.weight', 'blocks.9.att.key2.weight', 'blocks.9.att.output1.weight', 'blocks.9.att.output2.weight', 'blocks.9.att.receptance1.weight', 'blocks.9.att.receptance2.weight', 'blocks.9.att.value1.weight', 'blocks.9.att.value2.weight', 'blocks.9.ffn.key.weight', 'blocks.9.ffn.receptance.weight', 'blocks.9.ffn.value.weight', 'emb.weight', 'head.weight']

1x ['blocks.0.att.ln_x.bias', 'blocks.0.att.ln_x.weight', 'blocks.0.att.time_faaaa', 'blocks.0.att.time_mix_g', 'blocks.0.att.time_mix_k', 'blocks.0.att.time_mix_r', 'blocks.0.att.time_mix_v', 'blocks.0.ffn.time_mix_k', 'blocks.0.ffn.time_mix_r', 'blocks.0.ln0.bias', 'blocks.0.ln0.weight', 'blocks.0.ln1.bias', 'blocks.0.ln1.weight', 'blocks.0.ln2.bias', 'blocks.0.ln2.weight', 'blocks.1.att.ln_x.bias', 'blocks.1.att.ln_x.weight', 'blocks.1.att.time_faaaa', 'blocks.1.att.time_mix_g', 'blocks.1.att.time_mix_k', 'blocks.1.att.time_mix_r', 'blocks.1.att.time_mix_v', 'blocks.1.ffn.time_mix_k', 'blocks.1.ffn.time_mix_r', 'blocks.1.ln1.bias', 'blocks.1.ln1.weight', 'blocks.1.ln2.bias', 'blocks.1.ln2.weight', 'blocks.10.att.ln_x.bias', 'blocks.10.att.ln_x.weight', 'blocks.10.att.time_faaaa', 'blocks.10.att.time_mix_g', 'blocks.10.att.time_mix_k', 'blocks.10.att.time_mix_r', 'blocks.10.att.time_mix_v', 'blocks.10.ffn.time_mix_k', 'blocks.10.ffn.time_mix_r', 'blocks.10.ln1.bias', 'blocks.10.ln1.weight', 'blocks.10.ln2.bias', 'blocks.10.ln2.weight', 'blocks.11.att.ln_x.bias', 'blocks.11.att.ln_x.weight', 'blocks.11.att.time_faaaa', 'blocks.11.att.time_mix_g', 'blocks.11.att.time_mix_k', 'blocks.11.att.time_mix_r', 'blocks.11.att.time_mix_v', 'blocks.11.ffn.time_mix_k', 'blocks.11.ffn.time_mix_r', 'blocks.11.ln1.bias', 'blocks.11.ln1.weight', 'blocks.11.ln2.bias', 'blocks.11.ln2.weight', 'blocks.2.att.ln_x.bias', 'blocks.2.att.ln_x.weight', 'blocks.2.att.time_faaaa', 'blocks.2.att.time_mix_g', 'blocks.2.att.time_mix_k', 'blocks.2.att.time_mix_r', 'blocks.2.att.time_mix_v', 'blocks.2.ffn.time_mix_k', 'blocks.2.ffn.time_mix_r', 'blocks.2.ln1.bias', 'blocks.2.ln1.weight', 'blocks.2.ln2.bias', 'blocks.2.ln2.weight', 'blocks.3.att.ln_x.bias', 'blocks.3.att.ln_x.weight', 'blocks.3.att.time_faaaa', 'blocks.3.att.time_mix_g', 'blocks.3.att.time_mix_k', 'blocks.3.att.time_mix_r', 'blocks.3.att.time_mix_v', 'blocks.3.ffn.time_mix_k', 'blocks.3.ffn.time_mix_r', 'blocks.3.ln1.bias', 'blocks.3.ln1.weight', 'blocks.3.ln2.bias', 'blocks.3.ln2.weight', 'blocks.4.att.ln_x.bias', 'blocks.4.att.ln_x.weight', 'blocks.4.att.time_faaaa', 'blocks.4.att.time_mix_g', 'blocks.4.att.time_mix_k', 'blocks.4.att.time_mix_r', 'blocks.4.att.time_mix_v', 'blocks.4.ffn.time_mix_k', 'blocks.4.ffn.time_mix_r', 'blocks.4.ln1.bias', 'blocks.4.ln1.weight', 'blocks.4.ln2.bias', 'blocks.4.ln2.weight', 'blocks.5.att.ln_x.bias', 'blocks.5.att.ln_x.weight', 'blocks.5.att.time_faaaa', 'blocks.5.att.time_mix_g', 'blocks.5.att.time_mix_k', 'blocks.5.att.time_mix_r', 'blocks.5.att.time_mix_v', 'blocks.5.ffn.time_mix_k', 'blocks.5.ffn.time_mix_r', 'blocks.5.ln1.bias', 'blocks.5.ln1.weight', 'blocks.5.ln2.bias', 'blocks.5.ln2.weight', 'blocks.6.att.ln_x.bias', 'blocks.6.att.ln_x.weight', 'blocks.6.att.time_faaaa', 'blocks.6.att.time_mix_g', 'blocks.6.att.time_mix_k', 'blocks.6.att.time_mix_r', 'blocks.6.att.time_mix_v', 'blocks.6.ffn.time_mix_k', 'blocks.6.ffn.time_mix_r', 'blocks.6.ln1.bias', 'blocks.6.ln1.weight', 'blocks.6.ln2.bias', 'blocks.6.ln2.weight', 'blocks.7.att.ln_x.bias', 'blocks.7.att.ln_x.weight', 'blocks.7.att.time_faaaa', 'blocks.7.att.time_mix_g', 'blocks.7.att.time_mix_k', 'blocks.7.att.time_mix_r', 'blocks.7.att.time_mix_v', 'blocks.7.ffn.time_mix_k', 'blocks.7.ffn.time_mix_r', 'blocks.7.ln1.bias', 'blocks.7.ln1.weight', 'blocks.7.ln2.bias', 'blocks.7.ln2.weight', 'blocks.8.att.ln_x.bias', 'blocks.8.att.ln_x.weight', 'blocks.8.att.time_faaaa', 'blocks.8.att.time_mix_g', 'blocks.8.att.time_mix_k', 'blocks.8.att.time_mix_r', 'blocks.8.att.time_mix_v', 'blocks.8.ffn.time_mix_k', 'blocks.8.ffn.time_mix_r', 'blocks.8.ln1.bias', 'blocks.8.ln1.weight', 'blocks.8.ln2.bias', 'blocks.8.ln2.weight', 'blocks.9.att.ln_x.bias', 'blocks.9.att.ln_x.weight', 'blocks.9.att.time_faaaa', 'blocks.9.att.time_mix_g', 'blocks.9.att.time_mix_k', 'blocks.9.att.time_mix_r', 'blocks.9.att.time_mix_v', 'blocks.9.ffn.time_mix_k', 'blocks.9.ffn.time_mix_r', 'blocks.9.ln1.bias', 'blocks.9.ln1.weight', 'blocks.9.ln2.bias', 'blocks.9.ln2.weight', 'ln_out.bias', 'ln_out.weight']

2x ['blocks.0.att.time_decay', 'blocks.1.att.time_decay', 'blocks.10.att.time_decay', 'blocks.11.att.time_decay', 'blocks.2.att.time_decay', 'blocks.3.att.time_decay', 'blocks.4.att.time_decay', 'blocks.5.att.time_decay', 'blocks.6.att.time_decay', 'blocks.7.att.time_decay', 'blocks.8.att.time_decay', 'blocks.9.att.time_decay']

3x []

Using /u/xl6yq/.cache/torch_extensions/py310_cu117 as PyTorch extensions root...
Using /u/xl6yq/.cache/torch_extensions/py310_cu117 as PyTorch extensions root...
Using /u/xl6yq/.cache/torch_extensions/py310_cu117 as PyTorch extensions root...
Using /u/xl6yq/.cache/torch_extensions/py310_cu117 as PyTorch extensions root...
Detected CUDA files, patching ldflags
Emitting ninja build file /u/xl6yq/.cache/torch_extensions/py310_cu117/fused_adam/build.ninja...
Building extension module fused_adam...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module fused_adam...
Loading extension module fused_adam...
Loading extension module fused_adam...
Loading extension module fused_adam...
Time to load fused_adam op: 0.4573025703430176 seconds
Time to load fused_adam op: 0.4575920104980469 seconds
Time to load fused_adam op: 0.45755457878112793 seconds
Time to load fused_adam op: 0.4576914310455322 seconds
INFO:torch.distributed.distributed_c10d:Added key: store_based_barrier_key:2 to store for rank: 2
INFO:torch.distributed.distributed_c10d:Added key: store_based_barrier_key:2 to store for rank: 0
INFO:torch.distributed.distributed_c10d:Added key: store_based_barrier_key:2 to store for rank: 1
INFO:torch.distributed.distributed_c10d:Added key: store_based_barrier_key:2 to store for rank: 3
INFO:torch.distributed.distributed_c10d:Rank 3: Completed store-based barrier for key:store_based_barrier_key:2 with 4 nodes.
INFO:torch.distributed.distributed_c10d:Rank 2: Completed store-based barrier for key:store_based_barrier_key:2 with 4 nodes.
INFO:torch.distributed.distributed_c10d:Rank 1: Completed store-based barrier for key:store_based_barrier_key:2 with 4 nodes.
INFO:torch.distributed.distributed_c10d:Rank 0: Completed store-based barrier for key:store_based_barrier_key:2 with 4 nodes.
INFO:pytorch_lightning.callbacks.model_summary:
  | Name   | Type       | Params
--------------------------------------
0 | emb    | Embedding  | 50.3 M
1 | blocks | ModuleList | 92.1 M
2 | ln_out | LayerNorm  | 1.5 K
3 | head   | Linear     | 50.3 M
--------------------------------------
192 M     Trainable params
0         Non-trainable params
192 M     Total params
771.232   Total estimated model params size (MB)
Epoch 0:   0%|                                                                                 | 0/1260 [00:00<?, ?it/s]
{'zero_allow_untested_optimizer': True, 'zero_optimization': {'stage': 2, 'contiguous_gradients': True, 'overlap_comm': True, 'allgather_partitions': True, 'reduce_scatter': True, 'allgather_bucket_size': 2000000, 'reduce_bucket_size': 2000000, 'sub_group_size': 1000000000000}, 'activation_checkpointing': {'partition_activations': False, 'cpu_checkpointing': False, 'contiguous_memory_optimization': False, 'synchronize_checkpoint_boundary': False}, 'aio': {'block_size': 1048576, 'queue_depth': 8, 'single_submit': False, 'overlap_events': True, 'thread_count': 1}, 'gradient_accumulation_steps': 1, 'train_micro_batch_size_per_gpu': 8, 'gradient_clipping': 1.0, 'bf16': {'enabled': True}}

Login to wandb...
wandb: Currently logged in as: felixlinatuva (xsel). Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.16.6
wandb: Run data is saved locally in /u/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/wandb/run-20240501_202819-wa69vrz9
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run 65536 ctx512 L12 D768 2024-05-01-20-27-27
wandb: ⭐️ View project at https://wandb.ai/xsel/rwkv
wandb: 🚀 View run at https://wandb.ai/xsel/rwkv/runs/wa69vrz9
Epoch 0:   0%|                 | 5/1260 [00:18<1:18:01,  3.73s/it, loss=10.20, lr=0.000312, REAL it/s=0.791, Kt/s=13.00]Epoch 0:   5%|▉                   | 61/1260 [01:31<29:55,  1.50s/it, loss=7.400, lr=0.0006, REAL it/s=0.755, Kt/s=12.40]











~~~~~~~~~~~~~~~~~~~~~~~~~

xl6yq@gpusrv13 (main)[RWKV-v5]$ ./demo-training-run.sh 

/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.16.0-unknown is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.1build1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 0.1.43ubuntu1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.16.0-unknown is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 1.1build1 is an invalid version and will not be supported in a future release
  warnings.warn(
/usr/lib/python3/dist-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning: 0.1.43ubuntu1 is an invalid version and will not be supported in a future release
  warnings.warn(
INFO:torch.distributed.nn.jit.instantiator:Created a temporary directory at /tmp/tmpptxjg_py
INFO:torch.distributed.nn.jit.instantiator:Writing /tmp/tmpptxjg_py/_remote_module_non_scriptable.py
INFO:pytorch_lightning.utilities.rank_zero:########## work in progress ##########
[2024-04-29 15:16:48,528] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Warning: The default cache directory for DeepSpeed Triton autotune, /u/xl6yq/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
 [WARNING]  NVIDIA Inference is only supported on Ampere and newer architectures
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.0
 [WARNING]  using untested triton version (2.0.0), only 1.0.0 is known to be compatible
INFO:numexpr.utils:Note: NumExpr detected 40 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
INFO:numexpr.utils:NumExpr defaulting to 8 threads.
INFO:pytorch_lightning.utilities.rank_zero:
############################################################################
#
# RWKV-5 BF16 on 1x1 GPU, bsz 1x1x8=8, deepspeed_stage_2 with grad_cp
#
# Data = data/minipile (binidx), ProjDir = out/L12-D768-x052
#
# Epoch = 0 to 71 (will continue afterwards), save every 10 epoch
#
# Each "epoch" = 5040 steps, 40320 samples, 20643840 tokens
#
# Model = 12 n_layer, 768 n_embd, 512 ctx_len
#
# Adam = lr 0.0006 to 6e-05, warmup 10 steps, beta (0.9, 0.99), eps 1e-08
#
# Found torch 2.0.1+cu117, recommend latest torch
# Found deepspeed 0.14.2, recommend latest deepspeed
# Found pytorch_lightning 1.9.5, recommend 1.9.5
#
############################################################################

INFO:pytorch_lightning.utilities.rank_zero:{'load_model': 'out/L12-D768-x052/rwkv-init.pth', 'wandb': 'Test', 'proj_dir': 'out/L12-D768-x052', 'random_seed': -1, 'train_type': '', 'data_file': 'data/minipile', 'data_type': 'binidx', 'vocab_size': 65536, 'ctx_len': 512, 'epoch_steps': 5040, 'epoch_count': 72, 'epoch_begin': 0, 'epoch_save': 10, 'micro_bsz': 8, 'n_layer': 12, 'n_embd': 768, 'dim_att': 768, 'dim_ffn': 2688, 'pre_ffn': 0, 'head_qk': 0, 'tiny_att_dim': 0, 'tiny_att_layer': -999, 'lr_init': 0.0006, 'lr_final': 6e-05, 'warmup_steps': 10, 'beta1': 0.9, 'beta2': 0.99, 'adam_eps': 1e-08, 'grad_cp': 1, 'dropout': 0, 'weight_decay': 0.001, 'weight_decay_final': -1, 'my_pile_version': 1, 'my_pile_stage': 3, 'my_pile_shift': 0, 'my_pile_edecay': 0, 'layerwise_lr': 1, 'ds_bucket_mb': 2, 'my_sample_len': 0, 'my_ffn_shift': 1, 'my_att_shift': 1, 'head_size_a': 64, 'head_size_divisor': 8, 'my_pos_emb': 0, 'load_partial': 0, 'magic_prime': 2926181, 'my_qa_mask': 0, 'my_random_steps': 0, 'my_testing': 'x052', 'my_exit': 99999999, 'my_exit_tokens': 1498226207, 'logger': False, 'enable_checkpointing': False, 'default_root_dir': None, 'gradient_clip_val': 1.0, 'gradient_clip_algorithm': None, 'num_nodes': 1, 'num_processes': None, 'devices': '1', 'gpus': None, 'auto_select_gpus': None, 'tpu_cores': None, 'ipus': None, 'enable_progress_bar': True, 'overfit_batches': 0.0, 'track_grad_norm': -1, 'check_val_every_n_epoch': 100000000000000000000, 'fast_dev_run': False, 'accumulate_grad_batches': None, 'max_epochs': -1, 'min_epochs': None, 'max_steps': -1, 'min_steps': None, 'max_time': None, 'limit_train_batches': None, 'limit_val_batches': None, 'limit_test_batches': None, 'limit_predict_batches': None, 'val_check_interval': None, 'log_every_n_steps': 100000000000000000000, 'accelerator': 'gpu', 'strategy': 'deepspeed_stage_2', 'sync_batchnorm': False, 'precision': 'bf16', 'enable_model_summary': True, 'num_sanity_val_steps': 0, 'resume_from_checkpoint': None, 'profiler': None, 'benchmark': None, 'reload_dataloaders_every_n_epochs': 0, 'auto_lr_find': False, 'replace_sampler_ddp': False, 'detect_anomaly': False, 'auto_scale_batch_size': False, 'plugins': None, 'amp_backend': None, 'amp_level': None, 'move_metrics_to_cpu': False, 'multiple_trainloader_mode': 'max_size_cycle', 'inference_mode': True, 'my_timestamp': '2024-04-29-15-16-58', 'betas': (0.9, 0.99), 'real_bsz': 8, 'run_name': '65536 ctx512 L12 D768'}

INFO:pytorch_lightning.utilities.rank_zero:Current vocab size = 65536 (make sure it's correct)
INFO:pytorch_lightning.utilities.rank_zero:Data has 1498226207 tokens.
INFO:pytorch_lightning.utilities.rank_zero:########## Pile 20b-tokenized stage 3 ##########
RWKV_MY_TESTING x052
Using /u/xl6yq/.cache/torch_extensions/py310_cu117 as PyTorch extensions root...
Detected CUDA files, patching ldflags
Emitting ninja build file /u/xl6yq/.cache/torch_extensions/py310_cu117/wkv5/build.ninja...
Building extension module wkv5...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module wkv5...
INFO:pytorch_lightning.utilities.rank_zero:########## Loading out/L12-D768-x052/rwkv-init.pth... ##########
INFO:pytorch_lightning.utilities.rank_zero:GPU available: True (cuda), used: True
INFO:pytorch_lightning.utilities.rank_zero:TPU available: False, using: 0 TPU cores
INFO:pytorch_lightning.utilities.rank_zero:IPU available: False, using: 0 IPUs
INFO:pytorch_lightning.utilities.rank_zero:HPU available: False, using: 0 HPUs
65536 768         emb.weight
768               blocks.0.ln1.weight
768               blocks.0.ln1.bias
768               blocks.0.ln2.weight
768               blocks.0.ln2.bias
768               blocks.0.ln0.weight
768               blocks.0.ln0.bias
768               blocks.0.att.time_mix_k
768               blocks.0.att.time_mix_v
768               blocks.0.att.time_mix_r
768               blocks.0.att.time_mix_g
12    64          blocks.0.att.time_decay
12    64          blocks.0.att.time_faaaa
768   768         blocks.0.att.receptance.weight
768   768         blocks.0.att.key.weight
768   768         blocks.0.att.value.weight
768   768         blocks.0.att.output.weight
768   768         blocks.0.att.gate.weight
768               blocks.0.att.ln_x.weight
768               blocks.0.att.ln_x.bias
768               blocks.0.ffn.time_mix_k
768               blocks.0.ffn.time_mix_r
2688  768         blocks.0.ffn.key.weight
768   768         blocks.0.ffn.receptance.weight
768   2688        blocks.0.ffn.value.weight
768               blocks.1.ln1.weight
768               blocks.1.ln1.bias
768               blocks.1.ln2.weight
768               blocks.1.ln2.bias
768               blocks.1.att.time_mix_k
768               blocks.1.att.time_mix_v
768               blocks.1.att.time_mix_r
768               blocks.1.att.time_mix_g
12    64          blocks.1.att.time_decay
12    64          blocks.1.att.time_faaaa
768   768         blocks.1.att.receptance.weight
768   768         blocks.1.att.key.weight
768   768         blocks.1.att.value.weight
768   768         blocks.1.att.output.weight
768   768         blocks.1.att.gate.weight
768               blocks.1.att.ln_x.weight
768               blocks.1.att.ln_x.bias
768               blocks.1.ffn.time_mix_k
768               blocks.1.ffn.time_mix_r
2688  768         blocks.1.ffn.key.weight
768   768         blocks.1.ffn.receptance.weight
768   2688        blocks.1.ffn.value.weight
768               blocks.2.ln1.weight
768               blocks.2.ln1.bias
768               blocks.2.ln2.weight
768               blocks.2.ln2.bias
768               blocks.2.att.time_mix_k
768               blocks.2.att.time_mix_v
768               blocks.2.att.time_mix_r
768               blocks.2.att.time_mix_g
12    64          blocks.2.att.time_decay
12    64          blocks.2.att.time_faaaa
768   768         blocks.2.att.receptance.weight
768   768         blocks.2.att.key.weight
768   768         blocks.2.att.value.weight
768   768         blocks.2.att.output.weight
768   768         blocks.2.att.gate.weight
768               blocks.2.att.ln_x.weight
768               blocks.2.att.ln_x.bias
768               blocks.2.ffn.time_mix_k
768               blocks.2.ffn.time_mix_r
2688  768         blocks.2.ffn.key.weight
768   768         blocks.2.ffn.receptance.weight
768   2688        blocks.2.ffn.value.weight
768               blocks.3.ln1.weight
768               blocks.3.ln1.bias
768               blocks.3.ln2.weight
768               blocks.3.ln2.bias
768               blocks.3.att.time_mix_k
768               blocks.3.att.time_mix_v
768               blocks.3.att.time_mix_r
768               blocks.3.att.time_mix_g
12    64          blocks.3.att.time_decay
12    64          blocks.3.att.time_faaaa
768   768         blocks.3.att.receptance.weight
768   768         blocks.3.att.key.weight
768   768         blocks.3.att.value.weight
768   768         blocks.3.att.output.weight
768   768         blocks.3.att.gate.weight
768               blocks.3.att.ln_x.weight
768               blocks.3.att.ln_x.bias
768               blocks.3.ffn.time_mix_k
768               blocks.3.ffn.time_mix_r
2688  768         blocks.3.ffn.key.weight
768   768         blocks.3.ffn.receptance.weight
768   2688        blocks.3.ffn.value.weight
768               blocks.4.ln1.weight
768               blocks.4.ln1.bias
768               blocks.4.ln2.weight
768               blocks.4.ln2.bias
768               blocks.4.att.time_mix_k
768               blocks.4.att.time_mix_v
768               blocks.4.att.time_mix_r
768               blocks.4.att.time_mix_g
12    64          blocks.4.att.time_decay
12    64          blocks.4.att.time_faaaa
768   768         blocks.4.att.receptance.weight
768   768         blocks.4.att.key.weight
768   768         blocks.4.att.value.weight
768   768         blocks.4.att.output.weight
768   768         blocks.4.att.gate.weight
768               blocks.4.att.ln_x.weight
768               blocks.4.att.ln_x.bias
768               blocks.4.ffn.time_mix_k
768               blocks.4.ffn.time_mix_r
2688  768         blocks.4.ffn.key.weight
768   768         blocks.4.ffn.receptance.weight
768   2688        blocks.4.ffn.value.weight
768               blocks.5.ln1.weight
768               blocks.5.ln1.bias
768               blocks.5.ln2.weight
768               blocks.5.ln2.bias
768               blocks.5.att.time_mix_k
768               blocks.5.att.time_mix_v
768               blocks.5.att.time_mix_r
768               blocks.5.att.time_mix_g
12    64          blocks.5.att.time_decay
12    64          blocks.5.att.time_faaaa
768   768         blocks.5.att.receptance.weight
768   768         blocks.5.att.key.weight
768   768         blocks.5.att.value.weight
768   768         blocks.5.att.output.weight
768   768         blocks.5.att.gate.weight
768               blocks.5.att.ln_x.weight
768               blocks.5.att.ln_x.bias
768               blocks.5.ffn.time_mix_k
768               blocks.5.ffn.time_mix_r
2688  768         blocks.5.ffn.key.weight
768   768         blocks.5.ffn.receptance.weight
768   2688        blocks.5.ffn.value.weight
768               blocks.6.ln1.weight
768               blocks.6.ln1.bias
768               blocks.6.ln2.weight
768               blocks.6.ln2.bias
768               blocks.6.att.time_mix_k
768               blocks.6.att.time_mix_v
768               blocks.6.att.time_mix_r
768               blocks.6.att.time_mix_g
12    64          blocks.6.att.time_decay
12    64          blocks.6.att.time_faaaa
768   768         blocks.6.att.receptance.weight
768   768         blocks.6.att.key.weight
768   768         blocks.6.att.value.weight
768   768         blocks.6.att.output.weight
768   768         blocks.6.att.gate.weight
768               blocks.6.att.ln_x.weight
768               blocks.6.att.ln_x.bias
768               blocks.6.ffn.time_mix_k
768               blocks.6.ffn.time_mix_r
2688  768         blocks.6.ffn.key.weight
768   768         blocks.6.ffn.receptance.weight
768   2688        blocks.6.ffn.value.weight
768               blocks.7.ln1.weight
768               blocks.7.ln1.bias
768               blocks.7.ln2.weight
768               blocks.7.ln2.bias
768               blocks.7.att.time_mix_k
768               blocks.7.att.time_mix_v
768               blocks.7.att.time_mix_r
768               blocks.7.att.time_mix_g
12    64          blocks.7.att.time_decay
12    64          blocks.7.att.time_faaaa
768   768         blocks.7.att.receptance.weight
768   768         blocks.7.att.key.weight
768   768         blocks.7.att.value.weight
768   768         blocks.7.att.output.weight
768   768         blocks.7.att.gate.weight
768               blocks.7.att.ln_x.weight
768               blocks.7.att.ln_x.bias
768               blocks.7.ffn.time_mix_k
768               blocks.7.ffn.time_mix_r
2688  768         blocks.7.ffn.key.weight
768   768         blocks.7.ffn.receptance.weight
768   2688        blocks.7.ffn.value.weight
768               blocks.8.ln1.weight
768               blocks.8.ln1.bias
768               blocks.8.ln2.weight
768               blocks.8.ln2.bias
768               blocks.8.att.time_mix_k
768               blocks.8.att.time_mix_v
768               blocks.8.att.time_mix_r
768               blocks.8.att.time_mix_g
12    64          blocks.8.att.time_decay
12    64          blocks.8.att.time_faaaa
768   768         blocks.8.att.receptance.weight
768   768         blocks.8.att.key.weight
768   768         blocks.8.att.value.weight
768   768         blocks.8.att.output.weight
768   768         blocks.8.att.gate.weight
768               blocks.8.att.ln_x.weight
768               blocks.8.att.ln_x.bias
768               blocks.8.ffn.time_mix_k
768               blocks.8.ffn.time_mix_r
2688  768         blocks.8.ffn.key.weight
768   768         blocks.8.ffn.receptance.weight
768   2688        blocks.8.ffn.value.weight
768               blocks.9.ln1.weight
768               blocks.9.ln1.bias
768               blocks.9.ln2.weight
768               blocks.9.ln2.bias
768               blocks.9.att.time_mix_k
768               blocks.9.att.time_mix_v
768               blocks.9.att.time_mix_r
768               blocks.9.att.time_mix_g
12    64          blocks.9.att.time_decay
12    64          blocks.9.att.time_faaaa
768   768         blocks.9.att.receptance.weight
768   768         blocks.9.att.key.weight
768   768         blocks.9.att.value.weight
768   768         blocks.9.att.output.weight
768   768         blocks.9.att.gate.weight
768               blocks.9.att.ln_x.weight
768               blocks.9.att.ln_x.bias
768               blocks.9.ffn.time_mix_k
768               blocks.9.ffn.time_mix_r
2688  768         blocks.9.ffn.key.weight
768   768         blocks.9.ffn.receptance.weight
768   2688        blocks.9.ffn.value.weight
768               blocks.10.ln1.weight
768               blocks.10.ln1.bias
768               blocks.10.ln2.weight
768               blocks.10.ln2.bias
768               blocks.10.att.time_mix_k
768               blocks.10.att.time_mix_v
768               blocks.10.att.time_mix_r
768               blocks.10.att.time_mix_g
12    64          blocks.10.att.time_decay
12    64          blocks.10.att.time_faaaa
768   768         blocks.10.att.receptance.weight
768   768         blocks.10.att.key.weight
768   768         blocks.10.att.value.weight
768   768         blocks.10.att.output.weight
768   768         blocks.10.att.gate.weight
768               blocks.10.att.ln_x.weight
768               blocks.10.att.ln_x.bias
768               blocks.10.ffn.time_mix_k
768               blocks.10.ffn.time_mix_r
2688  768         blocks.10.ffn.key.weight
768   768         blocks.10.ffn.receptance.weight
768   2688        blocks.10.ffn.value.weight
768               blocks.11.ln1.weight
768               blocks.11.ln1.bias
768               blocks.11.ln2.weight
768               blocks.11.ln2.bias
768               blocks.11.att.time_mix_k
768               blocks.11.att.time_mix_v
768               blocks.11.att.time_mix_r
768               blocks.11.att.time_mix_g
12    64          blocks.11.att.time_decay
12    64          blocks.11.att.time_faaaa
768   768         blocks.11.att.receptance.weight
768   768         blocks.11.att.key.weight
768   768         blocks.11.att.value.weight
768   768         blocks.11.att.output.weight
768   768         blocks.11.att.gate.weight
768               blocks.11.att.ln_x.weight
768               blocks.11.att.ln_x.bias
768               blocks.11.ffn.time_mix_k
768               blocks.11.ffn.time_mix_r
2688  768         blocks.11.ffn.key.weight
768   768         blocks.11.ffn.receptance.weight
768   2688        blocks.11.ffn.value.weight
768               ln_out.weight
768               ln_out.bias
65536 768         head.weight
INFO:pytorch_lightning.strategies.deepspeed:initializing deepspeed distributed: GLOBAL_RANK: 0, MEMBER: 1/1
INFO:torch.distributed.distributed_c10d:Added key: store_based_barrier_key:1 to store for rank: 0
INFO:torch.distributed.distributed_c10d:Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 1 nodes.
INFO:pytorch_lightning.utilities.rank_zero:Enabling DeepSpeed BF16.
INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [2,3]
decay ['blocks.0.att.gate.weight', 'blocks.0.att.key.weight', 'blocks.0.att.output.weight', 'blocks.0.att.receptance.weight', 'blocks.0.att.value.weight', 'blocks.0.ffn.key.weight', 'blocks.0.ffn.receptance.weight', 'blocks.0.ffn.value.weight', 'blocks.1.att.gate.weight', 'blocks.1.att.key.weight', 'blocks.1.att.output.weight', 'blocks.1.att.receptance.weight', 'blocks.1.att.value.weight', 'blocks.1.ffn.key.weight', 'blocks.1.ffn.receptance.weight', 'blocks.1.ffn.value.weight', 'blocks.10.att.gate.weight', 'blocks.10.att.key.weight', 'blocks.10.att.output.weight', 'blocks.10.att.receptance.weight', 'blocks.10.att.value.weight', 'blocks.10.ffn.key.weight', 'blocks.10.ffn.receptance.weight', 'blocks.10.ffn.value.weight', 'blocks.11.att.gate.weight', 'blocks.11.att.key.weight', 'blocks.11.att.output.weight', 'blocks.11.att.receptance.weight', 'blocks.11.att.value.weight', 'blocks.11.ffn.key.weight', 'blocks.11.ffn.receptance.weight', 'blocks.11.ffn.value.weight', 'blocks.2.att.gate.weight', 'blocks.2.att.key.weight', 'blocks.2.att.output.weight', 'blocks.2.att.receptance.weight', 'blocks.2.att.value.weight', 'blocks.2.ffn.key.weight', 'blocks.2.ffn.receptance.weight', 'blocks.2.ffn.value.weight', 'blocks.3.att.gate.weight', 'blocks.3.att.key.weight', 'blocks.3.att.output.weight', 'blocks.3.att.receptance.weight', 'blocks.3.att.value.weight', 'blocks.3.ffn.key.weight', 'blocks.3.ffn.receptance.weight', 'blocks.3.ffn.value.weight', 'blocks.4.att.gate.weight', 'blocks.4.att.key.weight', 'blocks.4.att.output.weight', 'blocks.4.att.receptance.weight', 'blocks.4.att.value.weight', 'blocks.4.ffn.key.weight', 'blocks.4.ffn.receptance.weight', 'blocks.4.ffn.value.weight', 'blocks.5.att.gate.weight', 'blocks.5.att.key.weight', 'blocks.5.att.output.weight', 'blocks.5.att.receptance.weight', 'blocks.5.att.value.weight', 'blocks.5.ffn.key.weight', 'blocks.5.ffn.receptance.weight', 'blocks.5.ffn.value.weight', 'blocks.6.att.gate.weight', 'blocks.6.att.key.weight', 'blocks.6.att.output.weight', 'blocks.6.att.receptance.weight', 'blocks.6.att.value.weight', 'blocks.6.ffn.key.weight', 'blocks.6.ffn.receptance.weight', 'blocks.6.ffn.value.weight', 'blocks.7.att.gate.weight', 'blocks.7.att.key.weight', 'blocks.7.att.output.weight', 'blocks.7.att.receptance.weight', 'blocks.7.att.value.weight', 'blocks.7.ffn.key.weight', 'blocks.7.ffn.receptance.weight', 'blocks.7.ffn.value.weight', 'blocks.8.att.gate.weight', 'blocks.8.att.key.weight', 'blocks.8.att.output.weight', 'blocks.8.att.receptance.weight', 'blocks.8.att.value.weight', 'blocks.8.ffn.key.weight', 'blocks.8.ffn.receptance.weight', 'blocks.8.ffn.value.weight', 'blocks.9.att.gate.weight', 'blocks.9.att.key.weight', 'blocks.9.att.output.weight', 'blocks.9.att.receptance.weight', 'blocks.9.att.value.weight', 'blocks.9.ffn.key.weight', 'blocks.9.ffn.receptance.weight', 'blocks.9.ffn.value.weight', 'emb.weight', 'head.weight'] 

1x ['blocks.0.att.ln_x.bias', 'blocks.0.att.ln_x.weight', 'blocks.0.att.time_faaaa', 'blocks.0.att.time_mix_g', 'blocks.0.att.time_mix_k', 'blocks.0.att.time_mix_r', 'blocks.0.att.time_mix_v', 'blocks.0.ffn.time_mix_k', 'blocks.0.ffn.time_mix_r', 'blocks.0.ln0.bias', 'blocks.0.ln0.weight', 'blocks.0.ln1.bias', 'blocks.0.ln1.weight', 'blocks.0.ln2.bias', 'blocks.0.ln2.weight', 'blocks.1.att.ln_x.bias', 'blocks.1.att.ln_x.weight', 'blocks.1.att.time_faaaa', 'blocks.1.att.time_mix_g', 'blocks.1.att.time_mix_k', 'blocks.1.att.time_mix_r', 'blocks.1.att.time_mix_v', 'blocks.1.ffn.time_mix_k', 'blocks.1.ffn.time_mix_r', 'blocks.1.ln1.bias', 'blocks.1.ln1.weight', 'blocks.1.ln2.bias', 'blocks.1.ln2.weight', 'blocks.10.att.ln_x.bias', 'blocks.10.att.ln_x.weight', 'blocks.10.att.time_faaaa', 'blocks.10.att.time_mix_g', 'blocks.10.att.time_mix_k', 'blocks.10.att.time_mix_r', 'blocks.10.att.time_mix_v', 'blocks.10.ffn.time_mix_k', 'blocks.10.ffn.time_mix_r', 'blocks.10.ln1.bias', 'blocks.10.ln1.weight', 'blocks.10.ln2.bias', 'blocks.10.ln2.weight', 'blocks.11.att.ln_x.bias', 'blocks.11.att.ln_x.weight', 'blocks.11.att.time_faaaa', 'blocks.11.att.time_mix_g', 'blocks.11.att.time_mix_k', 'blocks.11.att.time_mix_r', 'blocks.11.att.time_mix_v', 'blocks.11.ffn.time_mix_k', 'blocks.11.ffn.time_mix_r', 'blocks.11.ln1.bias', 'blocks.11.ln1.weight', 'blocks.11.ln2.bias', 'blocks.11.ln2.weight', 'blocks.2.att.ln_x.bias', 'blocks.2.att.ln_x.weight', 'blocks.2.att.time_faaaa', 'blocks.2.att.time_mix_g', 'blocks.2.att.time_mix_k', 'blocks.2.att.time_mix_r', 'blocks.2.att.time_mix_v', 'blocks.2.ffn.time_mix_k', 'blocks.2.ffn.time_mix_r', 'blocks.2.ln1.bias', 'blocks.2.ln1.weight', 'blocks.2.ln2.bias', 'blocks.2.ln2.weight', 'blocks.3.att.ln_x.bias', 'blocks.3.att.ln_x.weight', 'blocks.3.att.time_faaaa', 'blocks.3.att.time_mix_g', 'blocks.3.att.time_mix_k', 'blocks.3.att.time_mix_r', 'blocks.3.att.time_mix_v', 'blocks.3.ffn.time_mix_k', 'blocks.3.ffn.time_mix_r', 'blocks.3.ln1.bias', 'blocks.3.ln1.weight', 'blocks.3.ln2.bias', 'blocks.3.ln2.weight', 'blocks.4.att.ln_x.bias', 'blocks.4.att.ln_x.weight', 'blocks.4.att.time_faaaa', 'blocks.4.att.time_mix_g', 'blocks.4.att.time_mix_k', 'blocks.4.att.time_mix_r', 'blocks.4.att.time_mix_v', 'blocks.4.ffn.time_mix_k', 'blocks.4.ffn.time_mix_r', 'blocks.4.ln1.bias', 'blocks.4.ln1.weight', 'blocks.4.ln2.bias', 'blocks.4.ln2.weight', 'blocks.5.att.ln_x.bias', 'blocks.5.att.ln_x.weight', 'blocks.5.att.time_faaaa', 'blocks.5.att.time_mix_g', 'blocks.5.att.time_mix_k', 'blocks.5.att.time_mix_r', 'blocks.5.att.time_mix_v', 'blocks.5.ffn.time_mix_k', 'blocks.5.ffn.time_mix_r', 'blocks.5.ln1.bias', 'blocks.5.ln1.weight', 'blocks.5.ln2.bias', 'blocks.5.ln2.weight', 'blocks.6.att.ln_x.bias', 'blocks.6.att.ln_x.weight', 'blocks.6.att.time_faaaa', 'blocks.6.att.time_mix_g', 'blocks.6.att.time_mix_k', 'blocks.6.att.time_mix_r', 'blocks.6.att.time_mix_v', 'blocks.6.ffn.time_mix_k', 'blocks.6.ffn.time_mix_r', 'blocks.6.ln1.bias', 'blocks.6.ln1.weight', 'blocks.6.ln2.bias', 'blocks.6.ln2.weight', 'blocks.7.att.ln_x.bias', 'blocks.7.att.ln_x.weight', 'blocks.7.att.time_faaaa', 'blocks.7.att.time_mix_g', 'blocks.7.att.time_mix_k', 'blocks.7.att.time_mix_r', 'blocks.7.att.time_mix_v', 'blocks.7.ffn.time_mix_k', 'blocks.7.ffn.time_mix_r', 'blocks.7.ln1.bias', 'blocks.7.ln1.weight', 'blocks.7.ln2.bias', 'blocks.7.ln2.weight', 'blocks.8.att.ln_x.bias', 'blocks.8.att.ln_x.weight', 'blocks.8.att.time_faaaa', 'blocks.8.att.time_mix_g', 'blocks.8.att.time_mix_k', 'blocks.8.att.time_mix_r', 'blocks.8.att.time_mix_v', 'blocks.8.ffn.time_mix_k', 'blocks.8.ffn.time_mix_r', 'blocks.8.ln1.bias', 'blocks.8.ln1.weight', 'blocks.8.ln2.bias', 'blocks.8.ln2.weight', 'blocks.9.att.ln_x.bias', 'blocks.9.att.ln_x.weight', 'blocks.9.att.time_faaaa', 'blocks.9.att.time_mix_g', 'blocks.9.att.time_mix_k', 'blocks.9.att.time_mix_r', 'blocks.9.att.time_mix_v', 'blocks.9.ffn.time_mix_k', 'blocks.9.ffn.time_mix_r', 'blocks.9.ln1.bias', 'blocks.9.ln1.weight', 'blocks.9.ln2.bias', 'blocks.9.ln2.weight', 'ln_out.bias', 'ln_out.weight'] 

2x ['blocks.0.att.time_decay', 'blocks.1.att.time_decay', 'blocks.10.att.time_decay', 'blocks.11.att.time_decay', 'blocks.2.att.time_decay', 'blocks.3.att.time_decay', 'blocks.4.att.time_decay', 'blocks.5.att.time_decay', 'blocks.6.att.time_decay', 'blocks.7.att.time_decay', 'blocks.8.att.time_decay', 'blocks.9.att.time_decay'] 

3x [] 

Using /u/xl6yq/.cache/torch_extensions/py310_cu117 as PyTorch extensions root...
Detected CUDA files, patching ldflags
Emitting ninja build file /u/xl6yq/.cache/torch_extensions/py310_cu117/fused_adam/build.ninja...
Building extension module fused_adam...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module fused_adam...
Time to load fused_adam op: 0.27609705924987793 seconds
INFO:torch.distributed.distributed_c10d:Added key: store_based_barrier_key:2 to store for rank: 0
INFO:torch.distributed.distributed_c10d:Rank 0: Completed store-based barrier for key:store_based_barrier_key:2 with 1 nodes.
INFO:pytorch_lightning.callbacks.model_summary:
  | Name   | Type       | Params
--------------------------------------
0 | emb    | Embedding  | 50.3 M
1 | blocks | ModuleList | 92.1 M
2 | ln_out | LayerNorm  | 1.5 K 
3 | head   | Linear     | 50.3 M
--------------------------------------
192 M     Trainable params
0         Non-trainable params
192 M     Total params
771.232   Total estimated model params size (MB)
Epoch 0:   0%|                                                                                                                                                                                               | 0/5040 [00:00<?, ?it/s]
{'zero_allow_untested_optimizer': True, 'zero_optimization': {'stage': 2, 'contiguous_gradients': True, 'overlap_comm': True, 'allgather_partitions': True, 'reduce_scatter': True, 'allgather_bucket_size': 2000000, 'reduce_bucket_size': 2000000, 'sub_group_size': 1000000000000}, 'activation_checkpointing': {'partition_activations': False, 'cpu_checkpointing': False, 'contiguous_memory_optimization': False, 'synchronize_checkpoint_boundary': False}, 'aio': {'block_size': 1048576, 'queue_depth': 8, 'single_submit': False, 'overlap_events': True, 'thread_count': 1}, 'gradient_accumulation_steps': 1, 'train_micro_batch_size_per_gpu': 8, 'gradient_clipping': 1.0, 'bf16': {'enabled': True}}

Login to wandb...
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: wandb: Enter your choice: wandb: Enter your choice: 3
wandb: You chose "Don't visualize my results"
wandb: Tracking run with wandb version 0.16.6
wandb: W&B syncing is set to `offline` in this directory.  
wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
Epoch 0:   7%|████████▍                                                                                                                      | 334/5040 [07:18<1:42:51,  1.31s/it, loss=6.710, lr=0.0006, REAL it/s=0.768, Kt/s=3.150]
Epoch 0:  20%|█████████████████████████▎                                                                                                    | 1010/5040 [21:58<1:27:39,  1.31s/it, loss=6.050, lr=0.0006, REAL it/s=0.769, Kt/s=3.150]




