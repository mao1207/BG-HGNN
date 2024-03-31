# BG-HGNN: Toward Scalable and Efficient Heterogeneous Graph Neural Network

## Abstract
Many computer vision and machine learning problems are modeled as learning tasks on heterogeneous graphs, featuring a wide array of relations from diverse types of nodes and edges. Heterogeneous Graph Neural Networks (HGNNs) stand out as a promising neural model class designed for heterogeneous graphs. Built on traditional GNNs, existing HGNNs employ different parameter spaces to model the varied relationships. However, the practical effectiveness of existing HGNNs is often limited to simple heterogeneous graphs with few relation types. This paper first highlights and demonstrates that the standard approach employed by existing HGNNs inevitably leads to parameter explosion and relation collapse, making HGNNs less effective or impractical for complex heterogeneous graphs with numerous relation types. To overcome this issue, we introduce a novel framework, Blend&Grind-HGNN (BG-HGNN), which effectively tackles the challenges by carefully integrating different relations into a unified feature space manageable by a single set of parameters. This results in a refined HGNN method that is more efficient and effective in learning from heterogeneous graphs, especially when the number of relations grows. Our empirical studies illustrate that BG-HGNN significantly surpasses existing HGNNs in terms of parameter efficiency (up to 28.96×), training throughput (up to 8.12×), and accuracy (up to 1.07×).

![BG-HGNN Framework](https://raw.githubusercontent.com/mao1207/BG-HGNN/main/images/framework.png)

## Environment Setup

To run BG-HGNN, ensure that you have the following environments:

- PyTorch 2.0.0+
- Python versions: 3.8, 3.9, 3.10, 3.11, 3.12
- DGL

We recommend installing the DGL library, which is required for BG-HGNN. For installation commands, please visit [DGL Start Page](https://www.dgl.ai/pages/start.html).

## Experiments

### Experiment 1: Node Classification

We have used 8 different datasets for node classification:

- AIFB, AM, MUTAG, BGS:
  You can download these datasets using DGL's data loading feature:
  ```python
  from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset
  ```
- FREEBASE, ACM, DBLP, IMDB:These datasets can be downloaded from the [HGB](https://github.com/THUDM/HGB) repository.

To test BG-HGNN and other baselines for node classification, run:
  ```shell
  python NC/main.py -model your_model_name -dataset your_dataset
  ```

### Experiment 2: Link Prediction
For link prediction, we used LastFM, Amazon, and Youtube datasets available at the [HGB](https://github.com/THUDM/HGB) repository.
To run the link prediction experiments, execute:
  ```shell
  python LP/your_model_fold/main.py --device 0 --use_norm True --dataset your_dataset

  ```