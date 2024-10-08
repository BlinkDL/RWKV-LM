import torch
import nanopq
import statistics
import numpy as np
import io
import argparse
from time import time

import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser(description='Your script description here')
parser.add_argument('--layer', default=0, type=int, help='An integer parameter')
args = parser.parse_args()

# --- res: cf end of file --- 
in_model_file=None
# .1b
# outpath='/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-pre-x59-SPARSITY-EXP'
NLAYERS=12

# .4b
# outpath='/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/04b-pre-x59-SPARSITY-EXP'

#outpath='/home/bfr4xr/RWKV-LM/RWKV-v5/out/1b5-pre-x59-8x-sparsity'
#outpath='/home/bfr4xr/RWKV-LM/RWKV-v5/out/04b-pre-x59-8x-sparsity'
# save mlp to...  
#in_model_file = '/home/bfr4xr/RWKV-LM/RWKV-v5/out/3b-official-sparsity/official.pth'
#out_model_file = '/home/bfr4xr/RWKV-LM/RWKV-v5/out/3b-official-sparsity/official-mlp.pth'

#in_model_file = '/home/bfr4xr/RWKV-LM/RWKV-v5/out/04b-pre-x59-8x-sparsity/rwkv-2405-mlp.pth'
#out_model_file = '/home/bfr4xr/RWKV-LM/RWKV-v5/out/04b-pre-x59-8x-sparsity/rwkv-2405-mlp.pth'
#in_model_file = f'{outpath}/rwkv-385-mlp.pth'
#out_model_file = f'{outpath}/rwkv-385-mlp.pth'
# in_model_file = '/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/04b-pre-x59-SPARSITY-EXP/rwkv-860.pth'
# out_model_file = '/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/04b-pre-x59-SPARSITY-EXP/rwkv-860-mlp.pth'

###############################################
TEST_LAYERS = range(0, NLAYERS)
#TEST_LAYERS = [0, NLAYERS//2, NLAYERS-1]   # sample
#TEST_LAYERS = [args.layer]   # sample
###############################################
# TEST_THR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
#TEST_THR = [0.5, 0.6, 0.7, 0.8]
TEST_THR = [0.7]
###############################################

# Define a simple 2-layer MLP
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        # hidden dim
        #self.ddd = 256   # no much better
        self.ddd = 64  # balanced
        # self.ddd = 32       # no much worse than 32
        # Two fully connected layers, bias=None as in dejavu
        self.fc1 = nn.Linear(input_dim, self.ddd, bias=None)  
        self.fc2 = nn.Linear(self.ddd, output_dim, bias=None) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation after first layer
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for binary classification
        return x

# Define a weight initialization function
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier initialization for weights
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # Initialize biases to zero

# return: pq, X_code
def train_layer_pq(layer_id): 
    global weights, inputs, labels, model_dict

    # XXXX only takes 80% of it 
    X = weights[layer_id].cpu().numpy().T.astype(np.float32)         # (3.5D, D)
    Xt = X  # training data
    pq = nanopq.PQ(M=8, # sub-spaces
                   Ks=256, verbose=False, metric='dot')    
    # Train codewords
    pq.fit(Xt)
    # Encode to PQ-codes
    X_code = pq.encode(X)
    return pq, X_code

# Custom n-bit quantization function
def quantize(tensor, bit):
    factor = pow(2, bit) - 1
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / factor  # 2^2 - 1 = 3
    zero_point = min_val
    quantized = ((tensor - zero_point) / scale).round().clamp(0, factor).to(torch.int8)
    return quantized, scale, zero_point

# Custom n-bit dequantization function
def dequantize(quantized, scale, zero_point, bit):
    if bit == 4:
        return quantized.to(torch.half) * scale + zero_point
    else:
        return quantized.to(torch.float32) * scale + zero_point

    
def train_layer(layer_id):
    global weights, inputs, labels, model_dict, outputs_gt

    # weights[0] (D,3.5xD)
    D1 = weights[layer_id].shape[0]
    D2 = weights[layer_id].shape[1]   # # of cols -- # of vectors to be indexed
    batch_size = 4

    # train/val split 
    n_batches = labels[layer_id].shape[0] // batch_size
    n_batches_val = n_batches // 5    #20% for validation
    n_batches_train = n_batches - n_batches_val

    N_TRAIN = n_batches_train * batch_size

    # training data  ... T/F labels    
    mlpinput = inputs[layer_id][:N_TRAIN].view(-1,batch_size,D1).to(torch.float32)
    mlplabel = labels[layer_id][:N_TRAIN].view(-1,batch_size,D2).to(torch.float32)

    # val data ... 
    val_inputs =         inputs[layer_id][N_TRAIN:N_TRAIN+n_batches_val*batch_size].view(-1,batch_size,D1).to(torch.float32)
    val_labels =         labels[layer_id][N_TRAIN:N_TRAIN+n_batches_val*batch_size].view(-1,batch_size,D2).to(torch.float32)
    val_outputs_gt = outputs_gt[layer_id][N_TRAIN:N_TRAIN+n_batches_val*batch_size].view(-1,batch_size,D2).to(torch.float32)

    www = torch.numel(mlplabel) / torch.sum(mlplabel) - 1  # == #false/#true

    model = MLP(D1, D2).to(torch.float32).to('cuda') 
    class_weight = torch.tensor([1.0, www])  # Higher weight for the minority class
    # class_weight = torch.tensor([1.0, 1.0])  # no weight -- balanced classes, should be BAD (skewed classes)
    # class_weight = torch.tensor([www, 1.0])  # sanity check -- should be VERY bad
    # class_weight = torch.tensor([1.0, 5.0])  # fixed weight
    # loss_fn = nn.BCELoss(weight=class_weights.to('cuda'))  # Binary Cross-Entropy Loss for binary classification
    loss_fn = nn.BCELoss(reduction='none')  # Use no reduction initially to apply per-sample weights
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    model.apply(weights_init)

    # --------- Validation w/o training. acc~=50%----------
    # model.eval()  # Set model to evaluation mode
    # with torch.no_grad():  # Disable gradient computation for validation
    #     val_outputs = model(val_inputs)
    #     val_loss = loss_fn(val_outputs, val_labels)

    #     # Compute accuracy (considering outputs > 0.5 as True, else False)
    #     predicted = (val_outputs > 0.5).float()
    #     correct = (predicted == val_labels).float().sum()
    #     val_accuracy = correct / (val_labels.numel())

    #     print(f"Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy.item() * 100:.2f}%")

    epochs = 100
    for epoch in range(epochs):
        model.train()  # Set the model in training mode
        
        optimizer.zero_grad()  # Zero the gradients
        
        # Forward pass
        outputs = model(mlpinput)
        
        # breakpoint()
        # loss = loss_fn(outputs, mlplabel)

        per_sample_loss = loss_fn(outputs, mlplabel)
        weighted_loss = per_sample_loss * (mlplabel * class_weight[1] + (1 - mlplabel) * class_weight[0])
        loss = weighted_loss.mean()
        
        # Backward pass and optimization
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights
        
        # check progress
        if False: 
            if epoch % 5 ==0: 
                model.eval()  # Set model to evaluation mode
                with torch.no_grad():  # Disable gradient computation for validation
                    val_outputs = model(val_inputs)

                    # Compute accuracy (considering outputs > 0.5 as True, else False)
                    predicted = (val_outputs > 0.5).float()
                    # predicted = (val_outputs > 0.35).float()          # xzl: can play with this 

                    # Compute recall
                    true_positives = (predicted * val_labels).sum()  # Count of TP
                    false_negatives = ((1 - predicted) * val_labels).sum()  # Count of FN
                    recall = true_positives / (true_positives + false_negatives + 1e-10)  # Add epsilon to avoid division by zero

                    print(f'epoch {epoch:03d}: layer {layer_id} sparsity: true {1-torch.sum(val_labels)/torch.numel(val_labels):.3f} pred {1-torch.sum(predicted)/torch.numel(predicted):.3f}')
                    print(f"    Validation Recall: {recall.item() * 100:.2f}%")

        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # print("Training complete.")

    # ---  save this MLP to an existing model file ---- # 
    if model_dict:         
        # model_dict[f'blocks.{layer_id}.mlp'] = model.state_dict()
        # RWKV convention.... 
        model_dict[f'blocks.{layer_id}.mlp.fc1.weight'] = model.state_dict()['fc1.weight']
        model_dict[f'blocks.{layer_id}.mlp.fc2.weight'] = model.state_dict()['fc2.weight']

        torch.save(model_dict, out_model_file)   # OK to overwrite
        print(f'>>>>>>>>>> saved blocks.{layer_id}.mlp to:' + out_model_file)

    # --- final ----- #
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for validation
        for thr in TEST_THR:
            val_outputs = model(val_inputs)
            # val_loss = loss_fn(val_outputs, val_labels)
            val_loss = loss_fn(val_outputs, val_labels).mean()

            # Compute accuracy (considering outputs > 0.5 as True, else False)
            predicted = (val_outputs > thr).float()         # xzl: can play with this 
            # predicted: tensor shape (#batches, batch_size, D2)
            correct = (predicted == val_labels).float().sum()
            val_accuracy = correct / (val_labels.numel())

            # Compute recall
            true_positives = (predicted * val_labels).sum()  # Count of TP
            false_negatives_mask = ((1 - predicted) * val_labels)  # 0/1 masks, (#batches, batch_size, D2)
            
            # -- debug -- 
            false_neg_vals = val_outputs_gt[false_negatives_mask.bool()]  # Collect values from outputs_gt where mask is 1
            false_neg_vals1 = val_outputs_gt * false_negatives_mask # tensor version, not collecting values
            true_pos_vals = val_outputs_gt[(predicted * val_labels).bool()]  # Collect values from outputs_gt where mask is 1

            # Find top K values in false_neg_vals1 and their locations
            K = 20
            # topK logits across all inputs
            top_k_false_neg_vals1_all, top_k_indices1_all = torch.topk(false_neg_vals1.view(-1), K)  
            # topK logits per input, their indices
            top_k_false_neg_vals1, top_k_indices1 = torch.topk(false_neg_vals1, K)

            # Print top K values
            top_k_false_neg_vals, top_k_indices = torch.topk(false_neg_vals, K)
            top_k_true_pos_vals = torch.topk(true_pos_vals, K).values

            print(f'Top {K} false negative values: {top_k_false_neg_vals}')
            print(f'Top {K} true positive values: {top_k_true_pos_vals}')

            print(f'true_pos_vals: min {true_pos_vals.min()} max {true_pos_vals.max()} mean {true_pos_vals.mean()} median {torch.median(true_pos_vals)}')
            print(f'false_neg_vals: min {false_neg_vals.min()} max {false_neg_vals.max()} mean {false_neg_vals.mean()} median {torch.median(false_neg_vals)}')
            
            # breakpoint()

            false_negatives = false_negatives_mask.sum()  # Count of FN
            recall = true_positives / (true_positives + false_negatives + 1e-10)  # Add epsilon to avoid division by zero

            print(f'layer {layer_id} thr {thr} sparsity: true {1-torch.sum(val_labels)/torch.numel(val_labels):.3f} pred {1-torch.sum(predicted)/torch.numel(predicted):.3f}')
            # breakpoint()

            print(f"Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy.item() * 100:.2f}%, Recall: {recall.item() * 100:.2f}%")

            # --- add on: try pq ---- #
            if True: 
                pq, X_code = train_layer_pq(layer_id)
                query = val_inputs.cpu().numpy().astype(np.float32) # all val inputs. can only do float32
                dists = pq.dtable(query[0][0]).adist(X_code) # can only query with a single input 

                # --sanity check: run the orig model to get the ground truth output
                kw = weights[layer_id]
                kx = val_inputs[0][0].half()
                k = kx @ kw
                vx = torch.relu(k) ** 2
                #assert torch.equal(vx, val_outputs_gt[0][0].half())

                gt_output = val_outputs_gt[0][0].cpu().numpy()
                negs_idx = top_k_indices1[0][0].cpu().numpy()
                # print(f"gt_output at false neg -- {gt_output[negs_idx]}")
                # print(f"distance at false neg -- {dists[negs_idx]}")
                
                # --- run the orig model quantized, 8 bit
                # Define the quantization configuration  (quant only works float
                kw = kw.cpu().float()
                kx = kx.cpu().float()
                observer = torch.quantization.default_observer()
                observer(kw)
                scale_kw, zero_point_kw = observer.calculate_qparams()
                observer(kx)
                scale_kx, zero_point_kx = observer.calculate_qparams()
                # Quantize the weights and input tensors
                kw_quantized = torch.quantize_per_tensor(kw, scale_kw.item(), zero_point_kw.item(), dtype=torch.qint8)
                kx_quantized = torch.quantize_per_tensor(kx, scale_kx.item(), zero_point_kx.item(), dtype=torch.qint8)
                # Calculate the output scale and zero_point
                output_scale = scale_kx.item() * scale_kw.item()
                output_zero_point = 0  # Typically set to 0 for simplicity
                # Perform quantized matrix multiplication
                result_quantized = torch.ops.quantized.matmul(kx_quantized, kw_quantized, output_scale, output_zero_point)
                # Dequantize the result to get it back to floating-point
                result = result_quantized.dequantize()
                # print(f"quantgized res at false neg -- {result[negs_idx]}")
                # breakpoint()            
                '''
                ex: GT: 209 activated; quant: 169 activated (95% percentile), only 1 false positive

                percentile = torch.quantile(result, 0.95).item()
                msk=(result>percentile)
                masked=val_outputs_gt[0][0].cpu()[msk]
                (result>percentile_90).sum()  # num activated -- according to quantization 
                (val_outputs_gt[0][0].cpu()>0).sum()  # GT num activated
                (masked==0).sum()    # false positive
                '''

                #  --- 4 bit quantization
                start = time()
                kw_4bit, scale_kw_4bit, zero_kw_4bit = quantize(kw, 4)
                end = time()
                print(f'4bit kw quantization time: {end-start}')
                kx_4bit, scale_kx_4bit, zero_kx_4bit = quantize(kx, 4)
               
                start = time()
                kw_4bit_dequant = dequantize(kw_4bit, scale_kw_4bit, zero_kw_4bit, 4)
                end = time()
                print(f'4bit kw dequantization time: {end-start}')
                kx_4bit_dequant = dequantize(kx_4bit, scale_kx_4bit, zero_kx_4bit, 4)
                result_4bit = kx_4bit_dequant @ kw_4bit_dequant

                # --- 2-bit quantization
                start = time()
                kw_2bit, scale_kw_2bit, zero_kw_2bit = quantize(kw, 2)
                end = time()
                print(f'2bit kw quantization time: {end-start}')
                kx_2bit, scale_kx_2bit, zero_kx_2bit = quantize(kx, 2)

                start = time()
                kw_2bit_dequant = dequantize(kw_2bit, scale_kw_2bit, zero_kw_2bit, 2)
                end = time()
                print(f'2bit kw dequantization time: {end-start}')
                kx_2bit_dequant = dequantize(kx_2bit, scale_kx_2bit, zero_kx_2bit, 2)
                result_2bit = kx_2bit_dequant @ kw_2bit_dequant

                start = time()
                kw_1bit, scale_kw_1bit, zero_kw_1bit = quantize(kw, 1)
                end = time()
                print(f'1bit kw quantization time: {end-start}')
                kx_1bit, scale_kx_1bit, zero_kx_1bit = quantize(kx, 1)

                start = time()
                kw_1bit_dequant = dequantize(kw_1bit, scale_kw_1bit, zero_kw_1bit)
                end = time()
                print(f'1bit kw dequantization time: {end-start}')
                kx_1bit_dequant = dequantize(kx_1bit, scale_kx_1bit, zero_kx_1bit)
                result_1bit = kx_1bit_dequant @ kw_1bit_dequant

                '''
                ex: GT: 209 activated; quant: 139 activated (95% percentile), 24 false positive
                '''

                per = 0.85  # percetile we use to take quant as activated            
                result=result_1bit.float()
                percentile = torch.quantile(result, per).item()
                # msk=(result>percentile)
                # masked=val_outputs_gt[0][0].cpu()[msk]
                
                mlp_pred = predicted[0][0].cpu().int()
                quant_pred = (result > percentile).int()
                ensemble_pred = mlp_pred | quant_pred
                gt_labels = val_labels[0][0].cpu()  # GT labels
                gt_sparsity = 1 - torch.sum(gt_labels) / torch.numel(gt_labels)
                print(f'\033[91mGT sparsity {gt_sparsity:.2f}\033[0m')

                # -- mlp perf
                tp = (mlp_pred * gt_labels).sum()
                fn = ((1 - mlp_pred) * gt_labels).sum()
                sparsity = 1 - torch.sum(mlp_pred) / torch.numel(mlp_pred)
                print(f'\033[91mMLP thr {thr} tp {tp} fn {fn} sparsity {sparsity:.2f}\033[0m')

                # -- quant perf
                tp = (quant_pred * gt_labels).sum()
                fn = ((1 - quant_pred) * gt_labels).sum()
                sparsity = 1 - torch.sum(quant_pred) / torch.numel(result)
                print(torch.sum(quant_pred))
                print(torch.numel(quant_pred))
                print(torch.numel(result))
                print(f'\033[91mQUANT per {per} tp {tp} fn {fn} sparse {sparsity:.2f}\033[0m')

                # -- ensemble perf
                tp = (ensemble_pred * gt_labels).sum()
                fn = ((1 - ensemble_pred) * gt_labels).sum()
                sparsity = 1 - torch.sum(ensemble_pred) / torch.numel(ensemble_pred)
                print(f'\033[91mENSENBLE tp {tp} fn {fn} sparse {sparsity:.2f}\033[0m')

                # for K in [50]:
                #     largest_k_indices = np.argpartition(dists, -K)[-K:]
                #     largest_k_items = dists[largest_k_indices]            

        #breakpoint()

        if False: 
            pq, X_code = train_layer_pq(layer_id)
            # query = val_inputs.cpu().numpy().astype(np.float32) # can only do float32
            query = val_inputs.cpu().numpy().astype(np.float32)     
            dists = pq.dtable(query[0][0]).adist(X_code) # can only query single vector
            predicted = predicted[0][0]
            val_labels = val_labels[0][0]

            for K in [50, 100, 200, 400, 800]:
                largest_k_indices = np.argpartition(dists, -K)[-K:]
                # largest_k_items = dists[largest_k_indices]
                largest = torch.from_numpy(largest_k_indices)
                over = predicted[largest]
                comp = ((1 - predicted) * val_labels)[largest]   # select from false negative
                print(f'K {K} overlapped {over.sum()} complementary {comp.sum()}')


def load_a_tensor(file_path):
    """
    Load the a  tensor from the file.
    """
    try:
        data = torch.load(file_path, map_location=torch.device('cuda'),weights_only=True)
        return data
    except FileNotFoundError:
        print("File not found.")
        return None
    
def load_tensors(file_path):
    """
    Load the list of tensors from the file.
    """
    try:
        data = torch.load(file_path, map_location=torch.device('cuda'), weights_only=True)
        if isinstance(data, list):
            print(len(data))
            return data
    except FileNotFoundError:
        print("File not found.")
        return []
    
if __name__ == '__main__':
    # ---------- load from file 
    weights={}  # dict:layer_id->ffnkey (D,3.5xD)
    inputs={}   # dict: layer_id -> 2D tensors, (# inputs, D)]
    outputs_gt={}   # dict: layer_id -> 2D tensors (# inputs, 3.5D)  as from the orig model
    labels={}   # dict: layer_id -> 2D tensors (# inputs, 3.5D) True/False

    model_dict=None
    if in_model_file != None:
        model_dict = torch.load(in_model_file)

    for layer_id in TEST_LAYERS:    
        outpath_query=f'{outpath}/FFN.key-layer{layer_id}-query.npy'
        outpath_weights=f'{outpath}/FFN.key-layer{layer_id}-weights.npy'

        w=load_a_tensor(outpath_weights)
        weights[layer_id]=w

        print("##### ", layer_id)
        q=load_tensors(outpath_query)
        #for row in q:
        #    print(len(row))
        #continue
        inputs[layer_id]=torch.stack(q)

        ## gen T/F labels by running the actual matmul
        kw = weights[layer_id]
        kx = inputs[layer_id]
        k = kx @ kw
        vx = torch.relu(k) ** 2
        # nz_mask = ~torch.eq(vx, 0)
        # num_nzeros = torch.sum(nz_mask).item()
        # num_zeros = nz_mask.shape[-1] - num_nzeros
        outputs_gt[layer_id] = vx
        labels[layer_id] = (vx>0)  # one hot

        print(f"layer {layer_id} #inputs {len(q)} ========= ")

        # ------------ check for sparsity ....
        # kx = inputs[layer_id]
        # kw = weights[layer_id]
        # k = kx @ kw
        # vx = torch.relu(k) ** 2
        # onehot = (vx>0)
        # print(f'layer {layer_id} avg sparsity {1-torch.sum(onehot)/torch.numel(onehot)}')

        # train mlp    
        train_layer(layer_id)
        del q
        del inputs[layer_id]

    if model_dict:
        print(f'saved to:' + out_model_file)

'''
.1b, train 100 epochs, MLP d=64

(venv) xl6yq@xsel02 (sparsity-exp)[RWKV-v5]$ python3 src/test-sparsity-mlp.py 
layer 0 #inputs 2288
layer 0 sparsity: true 0.9410467147827148 pred 0.8273610472679138
Validation Loss: 0.3014, Validation Accuracy: 86.53%, Recall: 82.20%
layer 1 #inputs 2288
layer 1 sparsity: true 0.9484688639640808 pred 0.8546931743621826
Validation Loss: 0.2580, Validation Accuracy: 88.81%, Recall: 82.37%
layer 2 #inputs 2288
layer 2 sparsity: true 0.9355177879333496 pred 0.8382202982902527
Validation Loss: 0.2837, Validation Accuracy: 87.60%, Recall: 79.28%
layer 3 #inputs 2288
layer 3 sparsity: true 0.9217869639396667 pred 0.8204461932182312
Validation Loss: 0.2990, Validation Accuracy: 86.60%, Recall: 79.15%
layer 4 #inputs 2288
layer 4 sparsity: true 0.9073619246482849 pred 0.8172383308410645
Validation Loss: 0.3041, Validation Accuracy: 86.57%, Recall: 76.15%
layer 5 #inputs 2288
layer 5 sparsity: true 0.8718020915985107 pred 0.8005504012107849
Validation Loss: 0.3162, Validation Accuracy: 85.74%, Recall: 72.15%
layer 6 #inputs 2288
layer 6 sparsity: true 0.886154055595398 pred 0.7847186326980591
Validation Loss: 0.3449, Validation Accuracy: 84.09%, Recall: 74.67%
layer 7 #inputs 2288
layer 7 sparsity: true 0.8874586224555969 pred 0.7930302023887634
Validation Loss: 0.3368, Validation Accuracy: 84.51%, Recall: 73.15%
layer 8 #inputs 2288
layer 8 sparsity: true 0.8839700818061829 pred 0.811973512172699
Validation Loss: 0.3287, Validation Accuracy: 85.24%, Recall: 67.42%
layer 9 #inputs 2287
layer 9 sparsity: true 0.881123423576355 pred 0.8294370174407959
Validation Loss: 0.3401, Validation Accuracy: 84.83%, Recall: 57.92%
layer 10 #inputs 2287
layer 10 sparsity: true 0.8522251844406128 pred 0.8367496728897095
Validation Loss: 0.3645, Validation Accuracy: 83.93%, Recall: 50.87%
layer 11 #inputs 2287
layer 11 sparsity: true 0.7620068788528442 pred 0.7187275886535645
Validation Loss: 0.4065, Validation Accuracy: 81.29%, Recall: 69.79%

.4b, 100 epochs hidden=64   (hidden=128 not getting better
(venv) xl6yq@xsel02 (sparsity-exp)[RWKV-v5]$ python3 src/test-sparsity-mlp.py 
layer 0 #inputs 1497
layer 0 sparsity: true 0.9401903748512268 pred 0.82181316614151
Validation Loss: 0.3383, Validation Accuracy: 84.45%, Recall: 68.96%
layer 1 #inputs 1497
layer 1 sparsity: true 0.9535444974899292 pred 0.8428916335105896
Validation Loss: 0.3112, Validation Accuracy: 86.32%, Recall: 71.90%
layer 2 #inputs 1497
layer 2 sparsity: true 0.956179678440094 pred 0.8387799859046936
Validation Loss: 0.3165, Validation Accuracy: 85.79%, Recall: 71.79%
layer 3 #inputs 1497
layer 3 sparsity: true 0.9534757137298584 pred 0.8333420753479004
Validation Loss: 0.3253, Validation Accuracy: 85.23%, Recall: 70.37%
layer 4 #inputs 1497
layer 4 sparsity: true 0.925886869430542 pred 0.7641863226890564
Validation Loss: 0.4122, Validation Accuracy: 80.14%, Recall: 75.12%
layer 5 #inputs 1497
layer 5 sparsity: true 0.9078795313835144 pred 0.7220245599746704
Validation Loss: 0.4597, Validation Accuracy: 77.12%, Recall: 76.70%
layer 6 #inputs 1497
layer 6 sparsity: true 0.893843412399292 pred 0.719440758228302
Validation Loss: 0.4623, Validation Accuracy: 77.17%, Recall: 74.61%
layer 7 #inputs 1497
layer 7 sparsity: true 0.8816353678703308 pred 0.7002388834953308
Validation Loss: 0.4845, Validation Accuracy: 75.68%, Recall: 73.91%
layer 8 #inputs 1497
layer 8 sparsity: true 0.8587181568145752 pred 0.6895277500152588
Validation Loss: 0.4867, Validation Accuracy: 75.50%, Recall: 73.15%
layer 9 #inputs 1497
layer 9 sparsity: true 0.82523113489151 pred 0.6574174165725708
Validation Loss: 0.5101, Validation Accuracy: 73.97%, Recall: 73.55%
layer 10 #inputs 1497
layer 10 sparsity: true 0.862481951713562 pred 0.6909227967262268
Validation Loss: 0.4886, Validation Accuracy: 75.14%, Recall: 71.97%
layer 11 #inputs 1497
layer 11 sparsity: true 0.852275550365448 pred 0.6826190948486328
Validation Loss: 0.4895, Validation Accuracy: 75.32%, Recall: 73.90%
layer 12 #inputs 1497
layer 12 sparsity: true 0.858160138130188 pred 0.7043079137802124
Validation Loss: 0.4636, Validation Accuracy: 76.95%, Recall: 72.96%
layer 13 #inputs 1497
layer 13 sparsity: true 0.8531697392463684 pred 0.6977161169052124
Validation Loss: 0.4637, Validation Accuracy: 77.22%, Recall: 75.36%
layer 14 #inputs 1497
layer 14 sparsity: true 0.8969154953956604 pred 0.7631070613861084
Validation Loss: 0.4038, Validation Accuracy: 80.56%, Recall: 70.62%
layer 15 #inputs 1497
layer 15 sparsity: true 0.8887803554534912 pred 0.76138836145401
Validation Loss: 0.3972, Validation Accuracy: 81.08%, Recall: 72.22%
layer 16 #inputs 1497
layer 16 sparsity: true 0.8860715627670288 pred 0.7331891655921936
Validation Loss: 0.4195, Validation Accuracy: 79.59%, Recall: 77.54%
layer 17 #inputs 1497
layer 17 sparsity: true 0.8620896339416504 pred 0.7275031805038452
Validation Loss: 0.4185, Validation Accuracy: 79.87%, Recall: 75.81%
layer 18 #inputs 1497
layer 18 sparsity: true 0.8856326937675476 pred 0.7638278603553772
Validation Loss: 0.3767, Validation Accuracy: 82.13%, Recall: 75.10%
layer 19 #inputs 1497
layer 19 sparsity: true 0.8571283221244812 pred 0.7246161699295044
Validation Loss: 0.4104, Validation Accuracy: 80.76%, Recall: 79.05%
layer 20 #inputs 1497
layer 20 sparsity: true 0.8564433455467224 pred 0.7116166353225708
Validation Loss: 0.4125, Validation Accuracy: 80.44%, Recall: 82.31%
layer 21 #inputs 1497
layer 21 sparsity: true 0.8579556941986084 pred 0.73359215259552
Validation Loss: 0.3882, Validation Accuracy: 81.86%, Recall: 79.92%
layer 22 #inputs 1497
layer 22 sparsity: true 0.8434011936187744 pred 0.7280418872833252
Validation Loss: 0.3735, Validation Accuracy: 82.85%, Recall: 82.07%
layer 23 #inputs 1497
layer 23 sparsity: true 0.7246074080467224 pred 0.6626702547073364
Validation Loss: 0.3603, Validation Accuracy: 83.75%, Recall: 81.74%


hidden=32
(venv) xl6yq@xsel02 (sparsity-exp)[RWKV-v5]$ python3 src/test-sparsity-mlp.py 
layer 0 #inputs 1497
layer 0 sparsity: true 0.9401903748512268 pred 0.7829434871673584
Validation Loss: 0.4006, Validation Accuracy: 80.64%, Recall: 69.61%
layer 1 #inputs 1497
layer 1 sparsity: true 0.9535444974899292 pred 0.7977663278579712
Validation Loss: 0.3876, Validation Accuracy: 81.88%, Recall: 72.60%
layer 2 #inputs 1497
layer 2 sparsity: true 0.956179678440094 pred 0.7937883734703064
Validation Loss: 0.3939, Validation Accuracy: 81.42%, Recall: 73.28%
layer 3 #inputs 1497
layer 3 sparsity: true 0.9534757137298584 pred 0.7831207513809204
Validation Loss: 0.4089, Validation Accuracy: 80.37%, Recall: 72.11%
layer 4 #inputs 1497
layer 4 sparsity: true 0.925886869430542 pred 0.7267746925354004
Validation Loss: 0.4703, Validation Accuracy: 76.43%, Recall: 75.34%
layer 5 #inputs 1497
layer 5 sparsity: true 0.9078795313835144 pred 0.6776762008666992
Validation Loss: 0.5209, Validation Accuracy: 72.86%, Recall: 77.63%
layer 6 #inputs 1497
layer 6 sparsity: true 0.893843412399292 pred 0.672945499420166
Validation Loss: 0.5248, Validation Accuracy: 72.77%, Recall: 75.80%
layer 7 #inputs 1497
layer 7 sparsity: true 0.8816353678703308 pred 0.6530015468597412
Validation Loss: 0.5431, Validation Accuracy: 71.30%, Recall: 75.36%
layer 8 #inputs 1497
layer 8 sparsity: true 0.8587181568145752 pred 0.655600905418396
Validation Loss: 0.5309, Validation Accuracy: 72.22%, Recall: 73.57%
layer 9 #inputs 1497
layer 9 sparsity: true 0.82523113489151 pred 0.6265878677368164
Validation Loss: 0.5492, Validation Accuracy: 70.82%, Recall: 73.34%
layer 10 #inputs 1497
layer 10 sparsity: true 0.862481951713562 pred 0.650390625
Validation Loss: 0.5364, Validation Accuracy: 71.34%, Recall: 72.92%
layer 11 #inputs 1497
layer 11 sparsity: true 0.852275550365448 pred 0.6313669681549072
Validation Loss: 0.5490, Validation Accuracy: 70.73%, Recall: 75.69%
layer 12 #inputs 1497
layer 12 sparsity: true 0.858160138130188 pred 0.65576171875
Validation Loss: 0.5213, Validation Accuracy: 72.57%, Recall: 74.65%
layer 13 #inputs 1497
layer 13 sparsity: true 0.8531697392463684 pred 0.6641254425048828
Validation Loss: 0.5080, Validation Accuracy: 73.93%, Recall: 75.61%
layer 14 #inputs 1497
layer 14 sparsity: true 0.8969154953956604 pred 0.720112144947052
Validation Loss: 0.4622, Validation Accuracy: 76.52%, Recall: 71.86%
layer 15 #inputs 1497
layer 15 sparsity: true 0.8887803554534912 pred 0.7233731746673584
Validation Loss: 0.4502, Validation Accuracy: 77.57%, Recall: 73.54%
layer 16 #inputs 1497
layer 16 sparsity: true 0.8860715627670288 pred 0.7037498950958252
Validation Loss: 0.4612, Validation Accuracy: 76.68%, Recall: 77.66%
layer 17 #inputs 1497
layer 17 sparsity: true 0.8620896339416504 pred 0.6980668306350708
Validation Loss: 0.4581, Validation Accuracy: 76.93%, Recall: 75.83%
layer 18 #inputs 1497
layer 18 sparsity: true 0.8856326937675476 pred 0.7244533896446228
Validation Loss: 0.4327, Validation Accuracy: 78.41%, Recall: 76.07%
layer 19 #inputs 1497
layer 19 sparsity: true 0.8571283221244812 pred 0.6929117441177368
Validation Loss: 0.4570, Validation Accuracy: 77.62%, Recall: 79.16%
layer 20 #inputs 1497
layer 20 sparsity: true 0.8564433455467224 pred 0.6825348138809204
Validation Loss: 0.4539, Validation Accuracy: 77.57%, Recall: 82.46%
layer 21 #inputs 1497
layer 21 sparsity: true 0.8579556941986084 pred 0.703357458114624
Validation Loss: 0.4361, Validation Accuracy: 78.88%, Recall: 80.07%
layer 22 #inputs 1497
layer 22 sparsity: true 0.8434011936187744 pred 0.7147352695465088
Validation Loss: 0.4002, Validation Accuracy: 81.18%, Recall: 80.99%
layer 23 #inputs 1497
layer 23 sparsity: true 0.7246074080467224 pred 0.6535896062850952
Validation Loss: 0.3877, Validation Accuracy: 82.07%, Recall: 80.33%

'''
