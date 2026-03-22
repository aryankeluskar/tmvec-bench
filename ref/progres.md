Bioinformatics Advances, 2025, 00, vbaf042
[https://doi.org/10.1093/bioadv/vbaf042](https://doi.org/10.1093/bioadv/vbaf042)
Advance Access Publication Date: 5 March 2024
Original Article
OXFORD

## Structural bioinformatics
# Fast protein structure searching using structure graph embeddings

Joe G. Greener¹, and Kiarash Jamali¹
¹Medical Research Council Laboratory of Molecular Biology, Cambridge, CB2 0QH, United Kingdom

---

### Abstract
Comparing and searching protein structures independent of primary sequence has proved useful for remote homology detection, function annotation, and protein classification. Fast and accurate methods to search with structures will be essential to make use of the vast databases that have recently become available, in the same way that fast protein sequence searching underpins much of bioinformatics. We train a simple graph neural network using supervised contrastive learning to learn a low-dimensional embedding of protein domains. The method, called Progres, is available as software at [https://github.com/greener-group/progres](https://github.com/greener-group/progres) and as a web server at [https://progres.mrc-lmb.cam.ac.uk](https://progres.mrc-lmb.cam.ac.uk). It has accuracy comparable to the best current methods and can search the AlphaFold database TED domains in a 10th of a second per query on CPU.

---

### 1 Introduction
A variety of methods have been developed to compare, align, and search with protein structures. Since structure is more conserved than sequence these methods have proved useful in remote homology detection, protein classification, inferring function from structure, clustering large databases, and assessing the accuracy of structure predictions. Global coordinate comparisons like TM-align provide interpretable scores that are comparable across protein size, with a challenge being how to align the residues independent of the primary sequence. Mathematical representations of 3D space such as 3D Zernike descriptors avoid this issue but are limited in accuracy. Other approaches include comparing residue-residue distances, which can access precise geometries conserved in the structural core, and considering local geometry. The highest accuracy methods tend to be careful comparisons based on coordinates like Dali, but searching large structural databases such as the AlphaFold Protein Structure Database or the ESM Metagenomic Atlas with these methods is slow.

Recently, Foldseek has addressed this problem by converting protein structure into a sequence of learned local tertiary motifs. It then uses the rich history of fast sequence searching in bioinformatics to dramatically reduce the pairwise comparison time of the query with each member of the database. It follows that to further reduce search time, the pairwise comparison step should be made even faster.

Inspired by the impressive performance of simple graph neural networks (GNNs) using coordinate information for a variety of molecular tasks, we decided to train a model to embed protein domains into a low-dimensional representation. Two embeddings can be compared very quickly by cosine similarity and a query can be compared to each member of a pre-embedded database in a vectorized manner on CPU or GPU. It makes sense to use expertly-curated classifications of protein structures when training such an embedding; we use supervised contrastive learning to allow the embedding to be learned in a manner that reflects such an understanding of protein structure space and returns search results consistent with it.

A number of recent methods have used protein structure graph embeddings and contrastive learning. Embedding protein folds has also been done using residue-level features, and GNNs acting on protein structure have been used for function prediction. Other studies have used unsupervised contrastive learning on protein structures and show that the representations are useful for downstream prediction tasks including protein structural similarity. Contrastive learning using protein classifications has also improved language models for protein sequences, showing clustering that better preserves protein structure space. Protein structure has been incorporated into language models more broadly, often with the intention of searching for remote homology. Progres provides a fast and accurate alternative to these methods, with the ability to search the AlphaFold TED domains, that is available as a web server and as software.

---

### 2 Methods
#### 2.1 Training
Structures in the Astral 2.08 95% sequence identity set including discontinuous domains were used for training. We chose 400 domains randomly from the Astral 2.08 40% sequence identity set to use as a test set (see below) and another 200 domains to use as a validation set to monitor training. We removed domains with 30% or greater sequence identity to these 600 domains using MMseqs2, and also removed domains with fewer than 20 or more than 500 residues. This left 30,549 domains in 4,862 families for training.

mmCIF files were downloaded and processed with Biopython. Some processing was also carried out with BioStructures.jl. $C\alpha$ atoms were extracted for the residues corresponding to the domain. Each $C\alpha$ atom is treated as a node with the following features:
* Number of $C\alpha$ atoms within 10 Å divided by the largest such number in the protein.
* Whether the $C\alpha$ atom is at the N-terminus.
* Whether the $C\alpha$ atom is at the C-terminus.
* The $\tau$ torsion angle between $C\alpha_{i-1}/C\alpha_i/C\alpha_{i+1}/C\alpha_{i+2}$.
* A 64D sinusoidal positional encoding for the residue number in the domain.

PyTorch was used for training. The neural network architecture was similar to the E(n)-equivariant GNN. We used a configuration similar to the molecular data prediction task, i.e. not updating the particle position. In this case, the model is analogous to a standard GNN with relative squared norms inputted to the edge operation. Edges are sparse and are between $C\alpha$ atoms within 10 Å of each other. Six such layers with residual connections are preceded by a one-layer multilayer perceptron (MLP) acting on node features and followed by a two-layer MLP acting on node features. Node features are then sum-pooled and a two-layer MLP generates the output embedding, which is normalized. Each hidden layer has 128 dimensions and uses the Swish/SiLU activation function, apart from the edge MLP in the GNN which has a hidden layer with 256 dimensions and 64D output. The final embedding has 128 dimensions.

Supervised contrastive learning is used for training. Each epoch cycles over the 4,862 training families. For each family, five other families are chosen randomly. For each of these six families, six domains from the family present in the training set are chosen randomly. If there are fewer than six domains in the family, duplicates are added to give six. This set of 36 domains with six unique labels is embedded with the model and the embeddings are used to calculate the supervised contrastive loss with a temperature of 0.1. During training only, Gaussian noise with variance 1.0 Å is added to the $x, y,$ and $z$ coordinates of each $C\alpha$ atom. Training was carried out with the Adam optimizer with learning rate $5 \times 10^{-5}$ and weight decay $1 \times 10^{-16}$. Each set of 36 domains was treated as one batch. Training was stopped after 500 epochs and the epoch with the best family sensitivity on the validation set was used as the final model. Training took around a week on one RTX A6000 GPU.

#### 2.2 Testing
For testing, a similar approach to Foldseek was adopted. The 15,177 Astral 2.08 40% sequence identity set domains were embedded with the model. The embeddings are stored as Float16 to reduce the size of large databases on disk, but this has no effect on search performance. Four hundred of these domains were chosen randomly and held out of the training data as described previously. Like Foldseek, we only chose domains with at least one other family, superfamily, and fold member. For each of these 400 domains, the cosine similarity of embeddings to each of the 15,177 domains was calculated and the domains ranked by similarity with the query domain included.

For each domain, we measured the fraction of TPs (True Positives) detected up to the first incorrect fold detected. TPs are same family in the case of family-level recognition, same superfamily and not same family in the case of superfamily-level recognition, and same fold and not same superfamily in the case of fold-level recognition. We also report the mean TM-align score and the fraction of hits with the same fold for the top 20 hits for each query.

All CPU methods were run on an Intel i9-10980XE CPU and with 256 GB RAM. Progres, Foldseek, and MMseqs2 were run on 16 threads. The GPU methods were run on a RTX A6000 GPU. Progres was run with PyTorch 1.11. For TM-align, we used the fast mode. ESM-2 embeddings used the esm2_t36_3B_UR50D model which has a 2560D embedding. The mean of the per-residue representations was normalized and comparison between sequences was carried out with cosine similarity. For MMseqs2, easy-search with a sensitivity of 7.5 was used.

For contact order, all residue pairs with $C\beta$ atoms ($C\alpha$ for glycine) within 8 Å are considered. The contact order of a structure is then defined as:
$$\text{Contact Order} = \frac{1}{LN} \sum_{i=1}^{N} S_i$$
where $S_i$ is the sequence separation of the residues in contacting pair $i$, $N$ is the number of contacting pairs, and $L$ is the sequence length of the protein.

#### 2.3 Databases
The AlphaFold database domain embeddings were prepared from the TED set of domains using cluster representatives from clustering at 50% sequence identity. The FAISS index was prepared using "IndexFlatIP(128)", which carries out exhaustive searching using the same cosine similarity as Progres. Query structures may be automatically split into domains before searching using Chainsaw, with each domain searched separately.

#### 2.4 Web server
The web server was implemented in Django and uses 3Dmol.js for visualization.

---

### 3 Results
We trained a simple GNN, called Progres (PROtein GRaph Embedding Search), to embed a protein structure independent of its sequence. Since we use distance and torsion angle features based on coordinates the embedding is SE(3)-invariant, i.e. it does not change with translation or rotation of the input structure. Supervised contrastive learning on SCOPe domains is used to train the model, moving domains closer or further apart in the embedding space depending on whether they are in the same SCOPe family or not. Sinusoidal position encoding is also used to allow the model to effectively use information on the sequence separation of residues. The main intended use of such an embedding is fast searching for similar structures by comparing the embedding of a query structure to the pre-computed embeddings of a database of structures. Our model does not give structural alignments, but if these are required they can be computed with tools like Dali after fast initial filtering with Progres.

---

### Figure 1: Description and Explanation
This figure illustrates the Progres workflow and the training logic.

* **Panel (a): The Pipeline.** It starts with a 3D visualization of a protein domain (green ribbons). This structure is converted into a "$C\alpha$ graph," where each residue's central carbon atom is a purple node, and lines (edges) connect nodes that are within 10 Å of each other. This graph is fed into a Graph Neural Network (GNN). The GNN outputs a "128 dim embedding," represented as a single vertical purple bar. This bar is then compared against a "pre-embedded database," shown as a large blue square made of many horizontal lines (other embeddings).
* **Panel (b): Supervised Contrastive Learning.** A square represents the "Embedding space." Inside, two purple dots (representing a family) are connected by a dark arrow labeled "moved together," while two green dots (a different family) are also moved together. Lighter grey arrows point between purple and green dots, labeled "moved apart." This explains that the AI learns by grouping similar structural families together and pushing different ones away.
* **Panel (c): Application Example.** A multi-colored protein (3FRH) is shown. It is "Split with Chainsaw" into two domains. The smaller blue domain finds "helical bundles" (CATH superfamily 1.10.8.10), and the larger orange domain finds over 45,000 matches in the TED database (CATH 3.40.50.150). This shows the tool works on complex proteins by breaking them down.

---

### Table 1: Comparison of ability to retrieve homologous proteins from SCOPe

| Software | Fold Sensitivity | Super-family Sensitivity | Family Sensitivity | Mean TM-align | Fraction correct folds | Run time (Single) | Run time (All-v-all) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Progres (this work)** | 0.177 | 0.706 | 0.877 | 0.621 | 0.853 | 1.3 s (CPU) | 163 s (CPU) |
| Dali | 0.168 | 0.709 | 0.885 | 0.673 | 0.920 | 508 s | > 1 month |
| Foldseek-TM | 0.158 | 0.666 | 0.859 | 0.662 | 0.898 | 4.8 s | 2 h 47 m |
| Foldseek | 0.111 | 0.644 | 0.850 | 0.656 | 0.889 | 2.3 s | 250 s |
| TM-align fast | 0.100 | 0.594 | 0.806 | 0.688 | 0.847 | 390 s | ~23 days |
| 3D-SURFER | 0.046 | 0.140 | 0.349 | 0.511 | 0.560 | 7.2 s | 24 h |
| EAT | 0.101 | 0.615 | 0.843 | 0.627 | 0.825 | 34 s (GPU) | 4 h 37 m (GPU) |
| ESM-2 | 0.014 | 0.221 | 0.477 | 0.546 | 0.598 | 28 s (GPU) | 590 s (GPU) |
| MMseqs2 | 0.001 | 0.165 | 0.433 | 0.488 | 0.390 | 0.9 s | 17.1 s |

---

### Figure 2: Description and Explanation
This figure consists of four bar/line charts showing how Progres performs compared to other tools across different protein characteristics.

* **Panel (a): Performance by SCOPe Class.** A bar chart shows "Sensitivity for fold searching." Progres (teal bar) is the highest or among the highest in most categories like "All $\beta$," "$\alpha/\beta$," and "$\alpha+\beta$." Notably, Dali (orange) and Foldseek (pink) perform better on "Membrane" proteins.
* **Panel (b): Performance by Sequence Length.** This chart shows sensitivity based on the "Number of residues." Progres is particularly strong for small proteins (<100 residues) but its lead narrows as proteins get larger (>=300 residues).
* **Panel (c): Performance by Contact Order.** This measures how "entwined" the protein is. Progres shows a massive advantage in proteins with a high contact order (>0.175), where the teal bar is nearly double the height of Foldseek. This suggests Progres is excellent at finding similarities in complex 3D folds where the linear sequence doesn't help.
* **Panel (d): Embedding Size.** A line graph shows how accuracy improves as the embedding size increases from 4 to 128. The accuracy climbs steeply and starts to plateau at 32 dimensions, reaching its peak at 128, which is the size chosen for the model.

---

### Figure 3: Description and Explanation
This figure uses t-SNE plots (visual maps where similar items are clustered together) to show how the model "sees" the universe of proteins.

* **Panel (a) & (b): SCOPe Domains.** Large clusters of thousands of dots. In (a), they are colored by type; you can see distinct "islands" for all-alpha (green) and all-beta (red) proteins. In (b), the same map is colored by protein size (dark purple for small, yellow for large), showing a smooth color gradient across the clusters, meaning the model naturally understands size.
* **Panel (c) & (d): The Protein Universe.** These plots compare the small "known" set of proteins (blue dots) against the massive AlphaFold database (orange dots). The orange area is much larger and denser, illustrating just how many new structures Progres can now search.
* **Panel (e): Progres vs TM-align.** A density plot (heatmap) comparing Progres scores to TM-align scores. Most of the data is concentrated in a dark blue blob. A dashed green line at TM-align 0.5 and a purple line at Progres 0.8 intersect, showing that a Progres score of 0.8 is a very reliable indicator that two proteins share the same fold.

---

In order to assess the accuracy of the model for structure searching, we follow a similar procedure to Foldseek. Since our model is trained on SCOPe domains, it is important not to use domains for training that appear in the test set. We select a random set of 400 domains from the Astral 2.08 40% sequence identity set for testing. No domains in the training set have a sequence identity of 30% or more to these 400 domains. This represents the realistic use case that the query structure has not been seen during training—e.g. it is a predicted or new experimental structure—but other domains in the family may have been seen during training.

As shown in Table 1, our model has sensitivity comparable to Dali and Foldseek-TM for recovering domains in SCOPe from the same fold, superfamily, and family. Its strong performance at the fold-level indicates an ability to find remote homologs. Progres is more sensitive than the EAT and ESM-2 protein language model embeddings, and also the baseline sequence searching method of MMseqs2. This indicates the benefits of comparing structures rather than just sequences for detecting homology.

Progres does particularly well on all-$\beta$ domains, smaller domains and domains with higher contact order. This ability to do well in cases where residues separate in sequence form contacts is possibly due to the lack of primary sequence information in the embedding, compared to a method like Foldseek that retains the sequence order for searching. It has lower performance on membrane proteins and larger domains. Performance drops when the number of embedding dimensions is below 32.

For searching a single structure against SCOPe on CPU the model is faster than Foldseek with most run time in Python module loading. For example, going from 1 to 100 query structures increases run time from 1.3 to 2.4 s. When searching with multiple structures, most run time is in generating the query structure embeddings. Consequently, the speed benefits of the method arise when searching a structure or structures against the pre-computed embeddings of a huge database such as the AlphaFold database. The recent TED study split the whole AlphaFold database into domains using a consensus-based approach. We embed the TED domains clustered at 50% sequence identity and use FAISS to considerably speed up the search time against the resultant database of 53 million structures. This allows a search time of a tenth of a second per query on CPU, after an initial data loading time of around a minute.

---

### 4 Discussion
The model presented here is trained and validated on protein domains; due to the domain-specific nature of the training it is not expected to work without modification on protein chains containing multiple domains, long disordered regions, or complexes. Fortunately, there are a number of tools such as Chainsaw, Merizo, and SWORD2 that can split query structures into domains. We integrate Chainsaw into Progres to allow automated splitting of query structures into domains, with each domain then searched separately. This can overcome issues that arise from searching with multiple domains at the same time, such as missing related proteins due to differing orientations of the domains.

One issue with supervised learning on domains is whether performance drops when searching with domains that the model has not seen anything similar to during training. We trained an identical model on a different dataset where 200 domains were used for testing and domains were removed from the training set if they were from the same SCOPe superfamily as any of the testing domains. The fold, superfamily, and family sensitivities are 0.190, 0.383, and 0.546, respectively. This indicates similar performance at finding distantly related folds, the main use of structure searching over sequence searching, though there is a drop in performance at finding closely-related domains.

Aside from searching for similar structures, an accurate protein structure embedding has a number of uses. Fast protein comparison is useful for clustering large sets of structures, e.g. to identify novel folds in the AlphaFold database. The embedding of a structure is just a set of numbers, and therefore can be targeted by differentiable approaches for applications like protein design. A decoder could be trained to generate structures from the embedding space, and a diffusion model to move through the embedding space. Properties of proteins such as evolution, topological classification, the completeness of protein fold space, the continuity of fold space, function, and dynamics could also be explored in the context of the low-dimensional fold space. We believe that the extremely fast pairwise comparison allowed by structural embeddings is an effective way to take advantage of the opportunities provided by the million-structure era.