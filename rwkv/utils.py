########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.nn import functional as F

class PIPELINE_ARGS():
    def __init__(self, temperature=1.0, top_p=0.85, top_k=0, alpha_frequency=0.2, alpha_presence=0.2, alpha_decay=0.996, token_ban=[], token_stop=[], chunk_len=256):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alpha_frequency = alpha_frequency # Frequency Penalty (as in GPT-3)
        self.alpha_presence = alpha_presence # Presence Penalty (as in GPT-3)
        self.alpha_decay = alpha_decay # gradually decay the penalty
        self.token_ban = token_ban # ban the generation of some tokens
        self.token_stop = token_stop # stop generation whenever you see any token here
        self.chunk_len = chunk_len # split input into chunks to save VRAM (shorter -> slower)

class PIPELINE():
    def __init__(self, model, WORD_NAME):
        self.model = model
        if WORD_NAME == 'cl100k_base':
            import tiktoken
            self.tokenizer = tiktoken.get_encoding(WORD_NAME)
        elif WORD_NAME == 'rwkv_vocab_v20230424':
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from rwkv_tokenizer import TRIE_TOKENIZER
            self.tokenizer = TRIE_TOKENIZER(os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt')        
        else:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(WORD_NAME)

    def refine_context(self, context):
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        if context == '':
            context = '\n'
        return context

    def encode(self, x):
        if 'Tokenizer' in str(type(self.tokenizer)):
            return self.tokenizer.encode(x).ids
        else:
            return self.tokenizer.encode(x)
    
    def decode(self, x):
        return self.tokenizer.decode(x)

    def sample_logits(self, logits, temperature=1.0, top_p=0.85, top_k=0):
        if temperature == 0:
            temperature = 1.0
            top_p = 0
        probs = F.softmax(logits.float(), dim=-1)
        # breakpoint()
        top_k = int(top_k)
        # 'privateuseone' is the type of custom devices like `torch_directml.device()`
        if probs.device.type in ['cpu', 'privateuseone']:
            probs = probs.cpu().numpy()
            sorted_ids = np.argsort(probs)
            sorted_probs = probs[sorted_ids][::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            probs = probs / np.sum(probs)
            out = np.random.choice(a=len(probs), p=probs)
            return int(out)
        else:
            sorted_ids = torch.argsort(probs)
            sorted_probs = probs[sorted_ids]
            sorted_probs = torch.flip(sorted_probs, dims=(0,))
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return int(out)
    
    def generate(self, ctx, token_count=100, args=PIPELINE_ARGS(), callback=None, state=None):
        all_tokens = []
        out_last = 0
        out_str = ''
        occurrence = {}   # xzl: will decay over time 
        token_masks = []
        for i in range(token_count):

            # forward & adjust prob.
            tokens = self.encode(ctx) if i == 0 else [token]
            while len(tokens) > 0:
                out, state, token_mask = self.model.forward(tokens[:args.chunk_len], state)
                tokens = tokens[args.chunk_len:]

            #if token_mask[0] is not None:
            #    token_masks.append(torch.stack(token_mask))
                
            # xzl: out: logits over all possible tokens
            for n in args.token_ban:
                out[n] = -float('inf')
            # xzl: reduce a logit for a token as it recurs...
            for n in occurrence:
                out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)
            
            # sampler
            token = self.sample_logits(out, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
            if token in args.token_stop:
                break
            all_tokens += [token]
            for xxx in occurrence:
                occurrence[xxx] *= args.alpha_decay
            
            ttt = self.decode([token])
            www = 1
            if ttt in ' \t0123456789':
                www = 0
            # elif ttt in '\r\n,.;?!"\':+-*/=#@$%^&_`~|<>\\()[]{}，。；“”：？！（）【】':
            #     www = 0.5
            if token not in occurrence:
                occurrence[token] = www
            else:
                occurrence[token] += www
            # print(occurrence) # debug
            
            # output
            tmp = self.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp: # is valid utf-8 string?
                if callback:
                    callback(tmp)
                out_str += tmp
                out_last = i + 1

        #def _plot_and_save_weights(data, token_index, layer_name, output_dir="weights_plots"):
        #    # Ensure output directory exists
        #    if not os.path.exists(output_dir):
        #        os.makedirs(output_dir)
        #   
        #    #print(data)
        #    #print(data.shape)
        #    plt.imshow(data, aspect='auto', cmap='magma', interpolation='none')

        #    plt.colorbar(label="True/False")
        #    #plt.title(f"Weights for Token {token_index} in {layer_name}")
        #    plt.xlabel("Column")
        #    plt.ylabel("Layer")
        #    
        #    # Save the figure
        #    plot_filename = f"{output_dir}/token_{token_index}_{layer_name}.png"
        #    plt.savefig(plot_filename, dpi=300)
        #    plt.close()

        #def _plot_cdf_for_each_row(tensor):
        #    output_dir = "cdf_plots"
        #    if not os.path.exists(output_dir):
        #        os.makedirs(output_dir)

        #    for row_idx in range(tensor.shape[0]):
        #        plt.figure(figsize=(6, 4))
        #        num_cols = tensor.shape[1]
        #        row_data = tensor[row_idx, :num_cols]


        #        np.savetxt(f"{row_idx}.csv", row_data, delimiter=",")
        #        row_data = np.sort(row_data)
        #        1/0
        #            
        #        cdf = np.cumsum(row_data / row_data.sum())
        #         
        #        # Sort the data to compute the CDF
        #        plt.plot(np.arange(num_cols), cdf, alpha=0.75, color='blue')

        #        plt.title(f"Layer {row_idx}")
        #        plt.ylabel("Cumulative probability")
        #        plt.xlabel("# of Columns")
        #        plt.legend(loc='best')
        #        plt.grid(True)
        #        filename = f"{output_dir}/{row_idx}.png"
        #        plt.savefig(filename, dpi=300)
        #        plt.close()  # Close the plot to free memory


        #for n, layer_masks in enumerate(token_masks):
        #    _plot_and_save_weights(layer_masks.cpu().numpy(), n, f"FFN") 
        #new_tensor = torch.zeros(token_masks[0].shape[0], token_masks[1].shape[1])

        #for t in token_masks:
        #    new_tensor += t.cpu()

        #output_dir = "unimpt_layers"
        #for i, row in enumerate(new_tensor):
        #    layers = torch.where(row < torch.quantile(row, 0.1))[0].numpy()
        #    np.save(f"{output_dir}/{i}_layer.npy", layers)



        #_plot_and_save_weights(new_tensor.cpu().numpy(), "ALL", "FFN") 
        #_plot_cdf_for_each_row(new_tensor.cpu())

        return out_str
