import torch
import numpy as np
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from model import GCN

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def permute_graph(x, edge_index, mask=None, batch=None):
    """Permute node order in a graph."""
    n = x.size(0)
    perm = torch.randperm(n)
    print('perm ',perm)
    x_perm = x[perm]
    # Compute inverse permutation to map old indices -> new indices
    # perm maps new_pos -> old_index (x_perm[i] = x[perm[i]]).
    # We need inv_perm such that inv_perm[old_index] = new_pos.
    inv_perm = perm.argsort()
    # Apply inverse mapping to edge_index so edges refer to new positions
    edge_index_perm = inv_perm[edge_index]
    mask_perm = mask[perm] if mask is not None else None
    batch_perm = batch[perm] if batch is not None else None
    return x_perm, edge_index_perm, mask_perm, batch_perm, perm

def test_equivariance_node():
    """Test node-level permutation equivariance globally and per split."""
    # Load Cora dataset
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]
    x, edge_index = data.x, data.edge_index
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    # Initialize model
    model = GCN(dataset.num_features, 16, dataset.num_classes)
    model.eval()

    # Compute original logits
    with torch.no_grad():
        out = model(x, edge_index)

    masks = {'train': train_mask, 'val': val_mask, 'test': test_mask}

    # Try a few random permutations
    for _ in range(3):
        # Permute graph (we only need 'perm' to remap outputs;
        # masks themselves don't need to be permuted for this check)
        x_perm, edge_index_perm, _, _, perm = permute_graph(x, edge_index, None)

        with torch.no_grad():
            out_perm = model(x_perm, edge_index_perm)

        # ---- Global equivariance: f(PX, PAP^T) == P f(X, A)
        assert torch.allclose(out_perm, out[perm], atol=1e-6), "Node equivariance failed (global check)."

        # ---- Per-split equivariance
        # inv_perm maps old_index -> new_index
        inv_perm = perm.argsort()

        for split_name, mask in masks.items():
            idx_old = mask.nonzero(as_tuple=False).view(-1)
            if idx_old.numel() == 0:
                continue
            # Where did those same nodes go in the permuted graph?
            idx_new = inv_perm[idx_old]

            # Compare logits of the same physical nodes before/after permutation
            ok = torch.allclose(out[idx_old], out_perm[idx_new], atol=1e-6)
            assert ok, f"Node equivariance failed on '{split_name}' split."

    print("Node equivariance test (global + per split) passed!")


def test_invariance_graph():
    """Test graph-level permutation invariance on MUTAG."""
    # Load MUTAG dataset
    dataset = TUDataset(root='./data', name='MUTAG')
    data = dataset[0]  # Use one graph for simplicity
    x, edge_index, batch = data.x, data.edge_index, data.batch

    # Initialize model (output dimension set to 16 for graph embeddings)
    model = GCN(dataset.num_features, 16, 16)
    model.eval()

    # Define pooling functions
    pool_fns = {
        'sum': global_add_pool,
        'mean': global_mean_pool,
        'max': global_max_pool
    }

    # Compute original node embeddings and pooled embeddings
    with torch.no_grad():
        h = model(x, edge_index)
        pooled = {name: fn(h, batch) for name, fn in pool_fns.items()}

    # Test 3 random permutations
    for _ in range(3):
        x_perm, edge_index_perm, _, batch_perm, _ = permute_graph(x, edge_index, batch=batch)
        with torch.no_grad():
            h_perm = model(x_perm, edge_index_perm)
            for name, fn in pool_fns.items():
                pooled_perm = fn(h_perm, batch_perm)
                assert torch.allclose(pooled_perm, pooled[name], atol=1e-6), f"{name} pooling invariance failed"
    print("Graph invariance test passed for sum, mean, max pooling!")

    # Counterexample: Non-invariant readout (take embedding of node 0)
    with torch.no_grad():
        non_invariant = h[0]  # Embedding of node 0
        x_perm, edge_index_perm, _, batch_perm, _ = permute_graph(x, edge_index, batch=batch)
        h_perm = model(x_perm, edge_index_perm)
        non_invariant_perm = h_perm[0]  # Node 0 after permutation (different node)
        if not torch.allclose(non_invariant, non_invariant_perm, atol=1e-6):
            print("Counterexample: Taking embedding of node 0 is not permutation-invariant.")

def main():
    #test_equivariance_node()
    test_invariance_graph()

if __name__ == "__main__":
    main()