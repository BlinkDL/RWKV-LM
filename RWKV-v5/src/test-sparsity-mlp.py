import torch
import nanopq
import statistics
import numpy as np

import torch.nn as nn
import torch.optim as optim

# --- res: cf end of file

outpath='/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-pre-x59-SPARSITY-EXP'
NLAYERS=12

# Define a simple 2-layer MLP
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.ddd = 64       # hidden dim
        # Two fully connected layers
        self.fc1 = nn.Linear(input_dim, self.ddd)  # First layer with 64 units
        self.fc2 = nn.Linear(self.ddd, output_dim)  # Second layer with output_dim units

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


def train_layer(layer_id):
    global weights, inputs, labels

    # weights[0] (D,3.5xD)
    D1 = weights[layer_id].shape[0]
    D2 = weights[layer_id].shape[1]   # # of cols -- # of vectors to be indexed
    batch_size = 16

    # train/val split 
    n_batches = labels[layer_id].shape[0] // batch_size
    n_batches_val = n_batches // 5    #20% for validation
    n_batches_train = n_batches - n_batches_val

    N_TRAIN = n_batches_train * batch_size

    # training data  ... T/F labels    
    mlpinput = inputs[layer_id][:N_TRAIN].view(-1,batch_size,D1).to(torch.float32)
    mlplabel = labels[layer_id][:N_TRAIN].view(-1,batch_size,D2).to(torch.float32)

    # val data ... 
    val_inputs = inputs[layer_id][N_TRAIN:N_TRAIN+n_batches_val*batch_size].view(-1,batch_size,D1).to(torch.float32)
    val_labels = labels[layer_id][N_TRAIN:N_TRAIN+n_batches_val*batch_size].view(-1,batch_size,D2).to(torch.float32)

    www = torch.numel(mlplabel) / torch.sum(mlplabel) - 1  # == #false/#true

    model = MLP(D1, D2).to(torch.float32).to('cuda') 
    class_weight = torch.tensor([1.0, www])  # Higher weight for the minority class
    # class_weight = torch.tensor([1.0, 5.0])  # Higher weight for the minority class  (worse)
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
        
        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # print("Training complete.")


    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for validation
        val_outputs = model(val_inputs)
        # val_loss = loss_fn(val_outputs, val_labels)
        val_loss = loss_fn(val_outputs, val_labels).mean()

        # Compute accuracy (considering outputs > 0.5 as True, else False)
        predicted = (val_outputs > 0.5).float()
        # predicted = (val_outputs > 0.35).float()          # xzl: can play with this 
        correct = (predicted == val_labels).float().sum()
        val_accuracy = correct / (val_labels.numel())

        # Compute recall
        true_positives = (predicted * val_labels).sum()  # Count of TP
        false_negatives = ((1 - predicted) * val_labels).sum()  # Count of FN
        recall = true_positives / (true_positives + false_negatives + 1e-10)  # Add epsilon to avoid division by zero

        print(f'layer {layer_id} sparsity: true {1-torch.sum(val_labels)/torch.numel(val_labels)} pred {1-torch.sum(predicted)/torch.numel(predicted)}')
        # breakpoint()

        # print(f"Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy.item() * 100:.2f}%")
        print(f"Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy.item() * 100:.2f}%, Recall: {recall.item() * 100:.2f}%")


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
        data = torch.load(file_path, map_location=torch.device('cuda'),weights_only=True)
        if isinstance(data, list):
            return data
    except FileNotFoundError:
        print("File not found.")
        return []
    
if __name__ == '__main__':
    # ---------- load from file 
    weights={}  # dict:layer_id->ffnkey (D,3.5xD)
    inputs={}   # dict: layer_id -> 2D tensors, (# inputs, D)
    labels={}   # dict: layer_id -> 2D tensors (# inputs, D) True/False
    for layer_id in range(0,NLAYERS):
        outpath_query=f'{outpath}/FFN.key-layer{layer_id}-query.npy'
        outpath_weights=f'{outpath}/FFN.key-layer{layer_id}-weights.npy'

        w=load_a_tensor(outpath_weights)
        weights[layer_id]=w

        q=load_tensors(outpath_query)
        inputs[layer_id]=torch.stack(q)

        ## gen T/F labels by running the actual matmul
        kw = weights[layer_id]
        kx = inputs[layer_id]
        k = kx @ kw
        vx = torch.relu(k) ** 2
        # nz_mask = ~torch.eq(vx, 0)
        # num_nzeros = torch.sum(nz_mask).item()
        # num_zeros = nz_mask.shape[-1] - num_nzeros
        labels[layer_id] = (vx>0)  # one hot

        print(f"layer {layer_id} #inputs {len(q)}")

        # ------------ check for sparsity ....
        # kx = inputs[layer_id]
        # kw = weights[layer_id]
        # k = kx @ kw
        # vx = torch.relu(k) ** 2
        # onehot = (vx>0)
        # print(f'layer {layer_id} avg sparsity {1-torch.sum(onehot)/torch.numel(onehot)}')

        # train mlp    
        train_layer(layer_id)
