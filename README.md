# LoRA fork of RWKV-LM

A RWKV-LM fork, added with [LoRA](https://arxiv.org/abs/2106.09685) finetuning support.
Currently only RWKV-v4neo is supported.
The LoRA module is self-implemented to work with the TorchScript JIT.
Existing RWKV-v4neo models/checkpoints should work out of the box.
Now only LoRA-finetuned weights are checkpointed during training: it provides much smaller checkpoints, but you now need to specify the base model to use it.
See `args.MODEL_LOAD` and `args.MODEL_LORA` in `RWKV-v4neo/chat.py`.

To finetune an existing model with LoRA, just work like full finetuning but with the LoRA options, in the directory `RWKV-v4neo`:

```
python3 train.py \
  --load_model <pretrained base model> \
  --proj_dir <place to save checkpoints> \
  --data_file <data for finetune> \
  --data_type <data type for finetune> \
  --vocab_size 50277 --ctx_len 1024 --epoch_steps 1000 --epoch_count 1000 --epoch_begin 0 --epoch_save 5 --micro_bsz 2 --n_layer 24 --n_embd 1024 --pre_ffn 0 --head_qk 0 --lr_init 1e-4 --lr_final 1e-4 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0 \ # all your familiar options
  --lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.01 \
  --lora_load <lora checkpoint to continue training> \ # optional
  --lora_parts=att,ffn,time,ln # configure which parts to finetune
```

The `r`, `alpha` and `dropout` options are up to your choice.
The `att`, `ffn`, `time` and `ln` refers to the TimeMix, ChannelMix, time decay/first/mix parameters, and layernorm parameters; DON'T FORGET to add the set of parameters to be finetuned here.
I'm still experimenting with different configurations; your experience is also welcomed!

To use the finetuned model, use `chat.py` as usual with the checkpoints in your specified `proj_dir`, but **remember to align the LoRA-corresponded options** with what you have specified during training!

```
args.MODEL_LORA = 'your_lora_checkpoint.pth'
args.lora_r = 8
args.lora_alpha = 32
```

## TODOs

* Seperate model merging to allow LoRA pretrained models to be used with other RWKV inference implementation (especially [ChatRWKV](https://github.com/BlinkDL/ChatRWKV))
