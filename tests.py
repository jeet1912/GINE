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
    #print('perm ',perm)
    x_perm = x[perm]
    # inverse permutation to map old indices -> new indices
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
    data = dataset[0]  # use a single graph for simplicity
    x, edge_index = data.x, data.edge_index
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    # Initialize model
    model = GCN(dataset.num_features, 16, dataset.num_classes)
    # model.eval() to disable dropout/bn randomness; equivariance requires determinism
    model.eval()

    # Compute original logits in original node order
    with torch.no_grad():
        # f(X, A) in original node order
        out = model(x, edge_index)

    masks = {'train': train_mask, 'val': val_mask, 'test': test_mask}

    # Try a few random permutations
    for _ in range(3):
        # Permute graph (we only need 'perm' to remap outputs;
        # masks themselves don't need to be permuted for this check)
        x_perm, edge_index_perm, _, _, perm = permute_graph(x, edge_index, None)
        # x_perm = P X, edge_index_perm corresponds to P A P^T (via index relabeling)

        with torch.no_grad():
            # Output in new node order
            out_perm = model(x_perm, edge_index_perm)

        # Global equivariance: f(PX, PAP^T) == P f(X, A); 
        # In code, left-multiplying by P is just row reindexing: out[perm]
        assert torch.allclose(out_perm, out[perm], atol=1e-6), "Node equivariance failed (global check)."

        # Per-split equivariance
        # inv_perm maps old_index -> new_index (positions after permutation)
        # So out[idx_old] should equal out_perm[idx_new] for the same physical nodes.
        inv_perm = perm.argsort()

        for split_name, mask in masks.items():
            idx_old = mask.nonzero(as_tuple=False).view(-1)
            if idx_old.numel() == 0:
                continue
            # new indexes in permuted graph
            idx_new = inv_perm[idx_old]

            # Comparing logits of the same physical nodes before/after permutation
            ok = torch.allclose(out[idx_old], out_perm[idx_new], atol=1e-6)
            assert ok, f"Node equivariance failed on '{split_name}' split."

    print("Node equivariance test (global + per split) passed!")


def test_invariance_graph():
    """
    Test graph-level permutation invariance on MUTAG by:
      1) Computing node embeddings with a GCN
      2) Pooling node embeddings into a single graph embedding with
         permutation-invariant readouts: sum, mean, max
      3) Randomly permuting node order (features + edges + batch ids)
      4) Verifying pooled graph embeddings do not change

    Also show a counterexample: taking "node 0" embedding is NOT invariant.
    
    Also note: adding any MLP after a symmetric pooling remains invariant (composition preserves invariance),
    because MLP acts on the pooled graph vector, not on node order.
    """
    # Load MUTAG (TUDataset packs multiple graphs with a 'batch' vector)
    dataset = TUDataset(root='./data', name='MUTAG')
    data = dataset[0]                 # use a single graph for simplicity
    x, edge_index, batch = data.x, data.edge_index, data.batch

    model = GCN(dataset.num_features, 16, dataset.num_classes)
    model.eval()  # turn off dropout/bn to keep deterministic outputs

    # Permutation-invariant pooling operators (symmetric over the node multiset)
    # - SUM:  sum({h_i}) is unchanged if you reorder nodes
    # - MEAN: mean({h_i}) = sum / count is also invariant to ordering
    # - MAX:  max({h_i}) across nodes is independent of order
    # Symmetric (order-agnostic) => permutation invariant.
    pool_fns = {
        'sum':  global_add_pool,
        'mean': global_mean_pool,
        'max':  global_max_pool,
    }

    # Brief comparison
    # SUM  : keeps total magnitude; sensitive to graph size; good when count/frequency matters.
    # MEAN : size-agnostic summary; good default when graphs vary in node count.
    # MAX  : detects presence of strong local patterns; sparse signal, ignores frequency.

    # Baseline (original order): compute node embeddings and pooled graph embeddings
    with torch.no_grad():
        h = model(x, edge_index) # f(X,A)                 
        pooled = {name: fn(h, batch) for name, fn in pool_fns.items()} # g(f(X,A))

    # Randomly permute node order 3 times and re-check invariance
    for _ in range(3):
        # x_perm = P X
        # edge_index_perm = P A P^T (implemented by remapping node ids)
        # batch_perm = P batch (each node keeps its graph id after permutation)
        x_perm, edge_index_perm, _, batch_perm, _ = permute_graph(x, edge_index, batch=batch)

        with torch.no_grad():
            h_perm = model(x_perm, edge_index_perm)  # f(PX, P A P_T) -> node embeddings in new order
            for name, fn in pool_fns.items():
                pooled_perm = fn(h_perm, batch_perm) # symmetric readout over nodes, g(f(PX, P A P_T))
                # Invariance check:
                # For symmetric pooling, pooled_perm should equal pooled[name] exactly (up to tolerance),
                # because reordering nodes within a graph does not affect sum/mean/max.
                # g(f(X,A)) == g(f(PX, P A P_T))
                assert torch.allclose(pooled_perm, pooled[name], atol=1e-6, rtol=1e-6), f"{name} pooling invariance failed"

    print("Graph invariance test passed for sum, mean, max pooling!")

    # Counterexample: a NON-invariant readout on purpose
    # Taking "the embedding of node 0" depends on the node ordering;
    # after permutation, index 0 refers to a DIFFERENT node, so the value changes.
    with torch.no_grad():
        non_invariant = h[0]  # 'node 0' in original order
        x_perm, edge_index_perm, _, batch_perm, _ = permute_graph(x, edge_index, batch=batch)
        h_perm = model(x_perm, edge_index_perm)
        non_invariant_perm = h_perm[0]  # 'node 0' AFTER permutation (almost surely a different node because of permutation)

        # We EXPECT mismatch -> this demonstrates non-invariance of index-based readouts.
        if not torch.allclose(non_invariant, non_invariant_perm, atol=1e-6, rtol=1e-6):
            print("Counterexample: Selecting node[0] is NOT permutation-invariant (as expected).")
    

def main():
    test_equivariance_node()
    test_invariance_graph()


if __name__ == "__main__":
    main()