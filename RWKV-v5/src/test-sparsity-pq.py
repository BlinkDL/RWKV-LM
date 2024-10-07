import torch
import nanopq
import statistics
import numpy as np


# --- res: cf end of file

outpath='/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-pre-x59-SPARSITY-EXP'

def load_a_tensor(file_path):
    """
    Load the a  tensor from the file.
    """
    try:
        data = torch.load(file_path, map_location=torch.device('cpu'))
        return data
    except FileNotFoundError:
        print("File not found.")
        return None
    
def load_tensors(file_path):
    """
    Load the list of tensors from the file.
    """
    try:
        data = torch.load(file_path, map_location=torch.device('cpu'))
        if isinstance(data, list):
            return data
    except FileNotFoundError:
        print("File not found.")
        return []
    
if __name__ == '__main__':
    # ---------- load from file 
    weights={}  # dict:layer_id->ffnkey (D,3.5xD)
    inputs={}   # dict: layer_id -> list of tensors, each shape (D)
    for layer_id in range(0,12):
        outpath_query=f'{outpath}/FFN.key-layer{layer_id}-query.npy'
        outpath_weights=f'{outpath}/FFN.key-layer{layer_id}-weights.npy'

        w=load_a_tensor(outpath_weights)
        weights[layer_id]=w

        q=load_tensors(outpath_query)
        inputs[layer_id]=q

    # ------------ test for sparsity ....
    if False: 
        for layer_id in range(0,12):
            sp = []
            kw = weights[layer_id]
            for kx in inputs[layer_id]: 
                k = kx @ kw
                vx = torch.relu(k) ** 2
                # count zeros... 
                zero_mask = torch.eq(vx, 0)
                num_zeros = torch.sum(zero_mask)
                sp.append(num_zeros.item()/zero_mask.shape[-1])
                # print(num_zeros.item()/zero_mask.shape[-1])
                # breakpoint()
            # breakpoint()
            print(f'layer {layer_id} avg sparisty {statistics.mean(sp):.2f}')
            
    # ------------ PQ ------------------------      
    # https://github.com/matsui528/nanopq
    # weights[0] (D,3.5xD)
    N = weights[0].shape[1]   # # of cols -- # of vectors to be indexed
    # training data? 
    D = weights[0].shape[0]
    
    input_id = 1

    X = weights[0].numpy().T.astype(np.float32)         # (3.5D, D)
    Xt = X  # training data
    query = inputs[0][input_id].numpy().astype(np.float32)   # (D)

    # pq = nanopq.PQ(M=4, # sub-spaces
    #                Ks=256, verbose=False, metric='dot')

    # pq = nanopq.PQ(M=8, # sub-spaces
    #                Ks=1024, verbose=False, metric='dot')

    pq = nanopq.PQ(M=16, # sub-spaces
                   Ks=256, verbose=False, metric='dot')
    
    # Train codewords
    pq.fit(Xt)

    # Encode to PQ-codes
    X_code = pq.encode(X)  

    # Results: create a distance table online, and compute Asymmetric Distance to each PQ-code 
    dists = pq.dtable(query).adist(X_code)  # (10000, ) 

    # compete distances ..... 
    for K in [50, 100, 200, 400, 800, 1200]:
        smallest_k_indices = np.argpartition(dists, K)[:K]
        smallest_k_items = dists[smallest_k_indices]
        # smallest_k_items = np.partition(dists, K)[:K]

        largest_k_indices = np.argpartition(dists, -K)[-K:]
        largest_k_items = dists[largest_k_indices]

        print(largest_k_items)
        largest = torch.from_numpy(largest_k_indices)

        #     check distances agains the actual output vx ... 
        layer_id = 0
        kw = weights[layer_id]
        kx = inputs[layer_id][input_id]
        k = kx @ kw
        vx = torch.relu(k) ** 2
        # zero_mask = torch.eq(vx, 0)
        nz_mask = ~torch.eq(vx, 0)
        num_nzeros = torch.sum(nz_mask).item()
        num_zeros = nz_mask.shape[-1] - num_nzeros

        # --- top K neurons --- 
        topk_values, topk_indices = torch.topk(vx, K)
        topk_mask = torch.zeros(vx.size(), dtype=torch.bool)
        topk_mask[topk_indices] = True

        # print(num_nzeros)   
        # sp.append(num_zeros.item()/zero_mask.shape[-1])

        pred_nz = torch.sum(nz_mask[largest])
        pred_topk = torch.sum(topk_mask[largest])

        print(f'#nerouns {nz_mask.shape[-1]} #nzeros {num_nzeros}, K {K}, pred_nz {pred_nz} pred_topk {pred_topk}')
    
    # breakpoint()

    '''
    https://nanopq.readthedocs.io/en/latest/source/tutorial.html
    you need to train this quantizer by running k-means clustering for each
    sub-space of the training vectors

    If you do not have training data, you can simply use the database vectors
    (or a subset of them) for training
    '''


'''
pq = nanopq.PQ(M=4, # sub-spaces
#nerouns 2688 #nzeros 307, K 50, pred_nz 17
#nerouns 2688 #nzeros 307, K 100, pred_nz 28
#nerouns 2688 #nzeros 307, K 200, pred_nz 49
#nerouns 2688 #nzeros 307, K 400, pred_nz 90
#nerouns 2688 #nzeros 307, K 800, pred_nz 143
#nerouns 2688 #nzeros 307, K 1200, pred_nz 189

pq = nanopq.PQ(M=8, # sub-spaces
                   Ks=256, verbose=False, metric='dot')
#nerouns 2688 #nzeros 307, K 50, pred_nz 21
#nerouns 2688 #nzeros 307, K 100, pred_nz 32
#nerouns 2688 #nzeros 307, K 200, pred_nz 56
#nerouns 2688 #nzeros 307, K 400, pred_nz 93
#nerouns 2688 #nzeros 307, K 800, pred_nz 171
#nerouns 2688 #nzeros 307, K 1200, pred_nz 220

pq = nanopq.PQ(M=8, # sub-spaces
                Ks=1024, verbose=False, metric='dot')
#nerouns 2688 #nzeros 307, K 50, pred_nz 43
#nerouns 2688 #nzeros 307, K 100, pred_nz 72
#nerouns 2688 #nzeros 307, K 200, pred_nz 112
#nerouns 2688 #nzeros 307, K 400, pred_nz 173
#nerouns 2688 #nzeros 307, K 800, pred_nz 240
#nerouns 2688 #nzeros 307, K 1200, pred_nz 276

pq = nanopq.PQ(M=16, # sub-spaces
                Ks=256, verbose=False, metric='dot')
#nerouns 2688 #nzeros 307, K 50, pred_nz 31
#nerouns 2688 #nzeros 307, K 100, pred_nz 53
#nerouns 2688 #nzeros 307, K 200, pred_nz 89
#nerouns 2688 #nzeros 307, K 400, pred_nz 138
#nerouns 2688 #nzeros 307, K 800, pred_nz 207
#nerouns 2688 #nzeros 307, K 1200, pred_nz 251
'''