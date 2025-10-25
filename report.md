# **Graph Neural Network Equivariance & Invariance Tests**

### **Goal**

Verify the theoretical guarantees that

* **Node-level equivariance:** *f(P X, P A Pᵀ) = P f(X, A)*
* **Graph-level invariance:** *g(f(P X, P A Pᵀ)) = g(f(X, A))*
  for permutation-invariant readouts *g ∈ {sum, mean, max}*.

---

### **1. Node-Level Equivariance (Cora)**

**Proof sketch**

A GCN layer propagates as

```
H' = σ(A_norm · H · W)
where  A_norm = D^(-1/2) · (A + I) · D^(-1/2)
```

Under a permutation **P**:

```
A_norm' = P · A_norm · Pᵀ
X' = P · X
```

Then,

```
σ(A_norm' · X' · W)
 = σ(P · A_norm · Pᵀ · P · X · W)
 = σ(P · A_norm · X · W)
 = P · σ(A_norm · X · W)
 = P · f(X, A)
```

Thus,
**f(P X, P A Pᵀ) = P f(X, A)**. GCNs are *permutation-equivariant*.

**Empirical test.**
For 3 random permutations P on Cora:

`out_perm == out[perm]` (global check)
Per-split (train/val/test) logits identical after mapping back with `inv_perm`.

**Result:**
`Node equivariance test (global + per split) passed!`

---

### **2. Graph-Level Invariance (MUTAG)**

**Proof sketch.**
If f is equivariant and g is a symmetric pooling (sum/mean/max) over node embeddings,
```
g(f(P X, P A Pᵀ))
= g(P f(X, A))
= g(f(X, A))
```
Permutation cancels because g ignores node order.

**Empirical test.**
After 3 random node permutations per graph:

`global_add_pool`, `global_mean_pool`, `global_max_pool` produce **identical graph embeddings** within 1e-6.

---

### **3. Counterexample (Readout ≠ Invariant)**

Readout = “take embedding of node 0”.
After permutation, index 0 points to a **different node** ⇒ output changes.

Observed:

```
Counterexample: Selecting node[0] is NOT permutation-invariant (as expected).
```

**Reason:** Taking "the embedding of node 0" depends on the node ordering. After permutation, index 0 refers to a DIFFERENT node, so the value changes.
Index-based readouts depend on node ordering → not symmetric. 

---

### **4. Pooling Comparison**

| Pool g(h) | Invariant? | Strengths                                              | Limitations               |
| --------- | ---------- | ------------------------------------------------------ | ------------------------- |
| **Sum**   | Yes          | Captures total magnitude; good when graph size matters | Sensitive to # nodes      |
| **Mean**  | Yes          | Size-normalized; default for variable-size graphs      | May dilute rare signals   |
| **Max**   | Yes          | Detects presence of strong local features              | Ignores frequency, sparse |

---

### **5. Results Screenshot**

![Tests output](https://raw.githubusercontent.com/jeet1912/GINE/main/tests.png)

---

**Conclusion:**
Empirical results match the theory—GCN layers are **node-equivariant**, and symmetric readouts (sum/mean/max) yield **graph-level invariance**, while index-based readouts fail by design.

---
