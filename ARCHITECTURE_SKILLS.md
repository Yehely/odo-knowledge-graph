# OpioidGNN: System Architecture & Coding Guidelines

This document serves as the absolute blueprint for the core architecture, data pipelines, and training constraints of the OpioidGNN (Opioid Drug Ontology) predictive model. Every script, module, and optimization step created must adhere strictly to these definitions.

---

## 1. Graph Structure & Feature Engineering

The full ODO knowledge graph must be flattened/projected into a clean **Heterogeneous Bipartite Graph**. Interactions happen strictly between two node types: `Compound` and `Target`, where the connecting `Edge` represents a specific bioassay experiment.

### A. Compound Node Features
* **Vector Dimension:** $2,060$ dimensions.
* **Composition:**
  1. **2,048 dimensions (Morgan ECFP4):** Spatial chemical structure fingerprints with a radius of up to 2 atoms.
  2. **10 dimensions (ADMET):** Continuous pharmacokinetic/physicochemical properties (e.g., molecular weight, solubility). **Must be scaled using `StandardScaler` (Z-score normalization)**.
  3. **2 dimensions (Clinical/Labeling):** Max clinical phase reached (normalized between 0 and 1) and a binary flag for radiolabeled entity.
* **Note:** Always use the specific chemical compound from the experiment itself, NOT its parent compound.

### B. Target (Receptor) Node Features
* **Vector Dimension:** $45$ dimensions total.
* **Static Components (13 dimensions encoded via One-Hot):**
  1. **4 dimensions:** Target type (single protein, protein family, etc.).
  2. **9 dimensions:** Biological species (human, mouse, rat, etc.).
* **Dynamic Component (32 dimensions):** A **Learned Embedding** module. These 32 weights must be initialized randomly at the start of training and remain fully updateable during the backpropagation loop to mathematically represent biological character.

### C. Experiment Connection (Edge Features)
* **Vector Dimension:** $16$ dimensions.
* **Composition:**
  1. **7 dimensions:** Measurement type ($K_i$, $IC_{50}$, $EC_{50}$, etc.) via One-Hot.
  2. **6 dimensions:** Result comparison operators ($>$, $=$, $<$, $\le$, $\ge$) via One-Hot.
  3. **3 dimensions:** Experimental setting (*in vitro*, *in vivo*, or other) via One-Hot.

---

## 2. Encoders & Processing Layers (Per-Node Processing)

Before performing graph mathematics, initial node feature vectors must be mapped onto a uniform **shared Latent Space of 256 dimensions**.

Each encoder pipeline must follow this sequential execution block:
$$\text{Linear Layer} \rightarrow \text{Batch Normalization (BatchNorm)} \rightarrow \text{ReLU} \rightarrow \text{Dropout(0.3)}$$

* **Compression Block:** Shrinks the Compound vector from $2,060 \rightarrow 256$ dimensions.
* **Expansion Block:** Expands the Target vector from $45 \rightarrow 256$ dimensions.
* **Regularization Constraints:** Ensure `Dropout` is set to exactly $30\%$ during training to avoid feature dependency and prevent overfitting.

---

## 3. Graph Message Passing (3-Hop GraphSAGE)

Message passing must be executed using the **GraphSAGE** algorithm across a defined depth of **3 layers/Hops**. 

* **Aggregation Scheme:** Use **Mean Aggregation** to average neighborhood representations.
* **Bipartite Execution:**
  * **Hop 0:** Base representations post-linear encoding ($256$ dims).
  * **Hop 1 (Direct Neighbors):** Targets aggregate connected Compounds; Compounds aggregate connected Targets. Aggregated means are concatenated to the node's current state and multiplied by layer weights.
  * **Hop 2 (Structural Similarity Learning):** Nodes absorb 2-hop context, forcing compounds with similar receptor activation behaviors closer together in the latent space.
  * **Hop 3 (Deep Topology Context):** Receptive fields cover 3 hops, capturing full graph topology. Output tensors retain a length of **256 dimensions**.

---

## 4. Vector Concatenation & Prediction Head (MLP)

After 3 layers of GNN enrichment, individual interactions are reconstructed for final affinity estimation.

### A. Concatenation Mechanism
Glue the final embedded outputs together sequentially to represent the full interaction context:
$$\text{Enriched Compound (256)} \ \vert\vert \ \text{Enriched Target (256)} \ \vert\vert \ \text{Original Experiment Edge (16)} = \mathbf{528\text{-dimensional unified vector}}$$

### B. Edge Head MLP (Prediction Funnel)
Pass the $528$-dimensional unified vector through dense linear layers that act as a mathematical funnel:
$$\text{528 dims} \rightarrow \text{256 dims} \rightarrow \text{128 dims} \rightarrow \mathbf{1\text{ continuous output (Predicted pChEMBL score)}}$$

---

## 5. Training, Evaluation, & Censored Data (Masking Strategy)

### A. Temporal Train/Test Validation Split
The dataset must be split inductively based on chronological metrics:
* **Training Set:** Experiments/papers published up to and including **2014**.
* **Testing Set:** Experiments/papers published between **2015 and 2020**.

### B. Data Censoring Handling (The Masking Core)
* **Topology Inclusion:** **Do NOT drop** edges/experiments that are missing raw `pchemblValue` figures or are flagged with threshold signs ($>$, $<$). These edges are vital for GraphSAGE message passing and neighborhood aggregation.
* **Loss Mask Configuration:**
  1. Generate a binary tensor array `train_mask` corresponding to edge indices.
  2. Map an index value of `1` if an edge possesses a clean, precise numeric ground-truth pChEMBL score.
  3. Map an index value of `0` if the pChEMBL entry is empty (N/A) or bounds-constrained ($>$, $<$).
* **Optimization Optimization:**
  * Calculate training errors using the Mean Squared Error (`MSELoss`) function.
  * Multiply loss elements by the `train_mask` tensor to zero-out errors stemming from censored/missing labels.
  * Execute backpropagation and weight optimization via the **Adam Optimizer**, ensuring that gradient updates affect weights based purely on the exact target references.