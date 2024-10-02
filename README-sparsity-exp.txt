README-sparsity-exp.txt

br: sparsity-exp

10/2/2024

MODEL_NAME='/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/04b-pre-x59-SPARSITY-EXP/rwkv-860-mlp'
gold acc -- openAI .47 

USING 4bit quant 
        thr = 0.7 # MLP threshold
        percent = .85 # percentile, we use to take quant as activated
        if (layer_id <= 4): 
            percent = .95
        if (layer_id >= 9): 
            percent = .80

{
    "lambada_openai": {
        "perplexity,none": 13.258916518502005,
        "perplexity_stderr,none": 0.4139389620023476,
        "acc,none": 0.4737046380749078,
        "acc_stderr,none": 0.006956337791536687,
        "alias": "lambada_openai"
    }
}

        thr = 0.7 # MLP threshold
        percent = .85 # percentile, we use to take quant as activated
        if (layer_id <= 4): 
            percent = .95
        if (layer_id >= 9): 
            percent = .85   ***** <<< this could imrpove sparsity, but not quite sensitive to acc. good

{
    "lambada_openai": {
        "perplexity,none": 13.180471374688842,
        "perplexity_stderr,none": 0.4114133735552594,
        "acc,none": 0.47603337861439937,        <<<<<<<<<<<<<  only minor loss. good 
        "acc_stderr,none": 0.0069579705549026,
        "alias": "lambada_openai"
    }
}

FFN original 16 bit (fp16)
2-bit FFN predictor --- 1/8 of that -- 12.5%

If our predicted sparsity is 70%, it means we will load 30% of the original weights.

That means our memory overhead is 40%-50% of the original fp16 weights.
That means we save FFN by 2x with negligible accuracy loss.

----
If it's 1-bit, 1/16 of the original, 6.25%
(Then this is stronger even with int8 quant for the whole model)


USING 2bit quant 
        thr = 0.7 # MLP threshold
        percent = .85 # percentile, we use to take quant as activated
        if (layer_id <= 4): 
            percent = .95
        if (layer_id >= 9): 
            percent = .80

{
    "lambada_openai": {
        "perplexity,none": 13.565993775552156,
        "perplexity_stderr,none": 0.42939556964770753,
        "acc,none": 0.47331651465165925,    <<<<<<<<<< too good??? need double chk
        "acc_stderr,none": 0.006956050915151117,
        "alias": "lambada_openai"
    }
}

        thr = 0.7 # MLP threshold
        percent = .90 # percentile, we use to take quant as activated
        if (layer_id <= 4): 
            percent = .95
        if (layer_id >= 16): 
            percent = .85
{
    "lambada_openai": {
        "perplexity,none": 13.655222718124183,
        "perplexity_stderr,none": 0.4354031987515853,
        "acc,none": 0.4696293421307976, <<<<<<<<<< still too good??? need double chk
        "acc_stderr,none": 0.006953115269071242,
        "alias": "lambada_openai"
    }
}

        thr = 0.7 # MLP threshold
        percent = .90 # percentile, we use to take quant as activated
        if (layer_id <= 8): 
            percent = .95
        if (layer_id >= 16): 
            percent = .90
{
    "lambada_openai": {
        "perplexity,none": 13.283044922111806,
        "perplexity_stderr,none": 0.424074693765596,
        "acc,none": 0.4694352804191733,
        "acc_stderr,none": 0.006952950213860609,
        "alias": "lambada_openai"
    }
}

############################################################


9/23/2024

### data collection:

code: 
rwkv/model.py
    ffn_one_v5_9

    FFN weight is saved as a single npy "weights.npy"
    each input is saved as a tensor, which is appended to "query.npy"

run: 
RWKV-v5/src/test-rwkv-chat.py
changes prmopt and do inference 
    (slow, b/c the way we write to query.npy is inefficient) 


### mlp training, validation 
code: 
RWKV-v5/src/test-sparsity-mlp.py

    change "outpath" for loading training data, & FFN weight


### PQ
code: 
RWKV-v5/src/test-sparsity-pq.py
