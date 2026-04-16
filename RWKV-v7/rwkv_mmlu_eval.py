########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import os, json, datetime, random
from tqdm import tqdm
import numpy as np
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

from torch.nn import functional as F
from datasets import load_dataset, load_from_disk

# for HF_MODE, do these first:
# pip install flash_attn -U --force-reinstall
# pip install git+https://github.com/fla-org/flash-linear-attention -U --force-reinstall
#
# HF_MODE = True # currently HF_MODE only has 30% running speed. wait for optimizations

HF_MODE = False # you will get 44.87% for RWKV-x070-World-1.5B-v3-20250127-ctx4096

########################################################################################################

if not HF_MODE:
    # download from https://huggingface.co/BlinkDL/rwkv-7-world
    MODEL_NAME = "/mnt/e/RWKV-Runner/models/RWKV-x070-World-1.5B-v3-20250127-ctx4096"
    print(f"Loading model - {MODEL_NAME}")

    os.environ["RWKV_V7_ON"] = '1'
    os.environ["RWKV_JIT_ON"] = "1"
    os.environ["RWKV_CUDA_ON"] = "1"

    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE
    model = RWKV(model=MODEL_NAME, strategy="cuda fp16")
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
    tokenizer = pipeline.tokenizer
else:
    MODEL_NAME = 'fla-hub/rwkv7-1.5B-world'
    print(f"Loading model - {MODEL_NAME}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cuda:0", trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

########################################################################################################

mmlu_test = load_from_disk("mmlu_test_dataset")
mmlu_dev = load_from_disk("mmlu_dev_dataset")

TEMPLATE = '''User: You are a very talented expert in <SUBJECT>. Answer this question:
<Q>
A. <|A|>
B. <|B|>
C. <|C|>
D. <|D|>

Assistant: The answer is'''

CHOICES = [" A", " B", " C", " D"]

SHUFFLE = False
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

########################################################################################################

correct = 0
total = 0
pbar = tqdm(total=len(mmlu_test))

choices_token = [tokenizer.encode(x) for x in CHOICES]
assert all([len(x) == 1 for x in choices_token]), "Choices are not single token, use rwkv_mmlu.py instead"
choices_token = [x[0] for x in choices_token]

for idx, sample in enumerate(mmlu_test):
    question = sample["question"]
    choices = sample["choices"]
    subject = sample["subject"]
    gt = sample["answer"]

    if SHUFFLE and not any(["Both" in x for x in choices]):  # exclude choices like "Both A and B"
        original_gt_text = choices[gt]
        np.random.shuffle(choices)
        gt = choices.index(original_gt_text)

    all_prefix = (
        TEMPLATE.replace("<Q>", question)
        .replace("<|A|>", choices[0])
        .replace("<|B|>", choices[1])
        .replace("<|C|>", choices[2])
        .replace("<|D|>", choices[3])
        .replace("<SUBJECT>", subject.replace("_", " "))
    )

    if idx == 0:
        print(f"Format example:")
        print("-" * 100)
        print(all_prefix)
        print("-" * 100)
        format_example = all_prefix

    all_prefix_ids = [0] + tokenizer.encode(all_prefix.replace('\r\n','\n').strip())

    if HF_MODE:
        logits = model.forward(torch.tensor([all_prefix_ids]).cuda())[0][0][-1]
    else:
        logits, _ = model.forward(all_prefix_ids, None, full_output=False)
    
    neg_log_prob = F.log_softmax(logits, dim=-1)
    target_prob = neg_log_prob[choices_token]
    
    if torch.argmax(target_prob).item() == gt:
        correct += 1
    total += 1
    pbar.set_description(f"Correct: {correct} - Total: {total} - Accuracy: {correct / total:.5f}")
    pbar.update(1)
pbar.close()
