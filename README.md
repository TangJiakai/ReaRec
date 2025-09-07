<h2 align="center">Think Before Recommend: Unleashing the Latent Reasoning Power for Sequential Recommendation</h2>

## 🌟 Overview
Sequential Recommendation (SeqRec) aims to predict the next item by capturing sequential patterns from users' historical interactions, playing a crucial role in many real-world recommender systems.
However, existing approaches predominantly adopt a direct forward computation paradigm, where the final hidden state of the sequence encoder serves as the user representation. We argue that this inference paradigm, due to its limited computational depth, struggles to model the complex evolving nature of user preferences and lacks a nuanced understanding of long-tail items, leading to suboptimal performance.
To address this issue, we propose **ReaRec**, the first inference-time computing framework for recommender systems, 
which enhances user representations through implicit multi-step reasoning.
Specifically, ReaRec autoregressively feeds the sequence's last hidden state into the sequential recommender while incorporating special reasoning position embeddings to decouple the original item encoding space from the multi-step reasoning space.
Moreover, we introduce two lightweight reasoning-based learning methods, Ensemble Reasoning Learning (ERL) and Progressive Reasoning Learning (PRL), to further effectively exploit ReaRec's reasoning potential.
Extensive experiments on five public real-world datasets and different SeqRec architectures demonstrate the generality and effectiveness of our proposed ReaRec.
Remarkably, post-hoc analyses reveal that ReaRec significantly elevates the performance ceiling of multiple sequential recommendation backbones by approximately 30%-50%.
Thus, we believe this work can open a new and promising avenue for future research in inference-time computing for sequential recommendation. 


## 🚀 Getting Started

### Requirements
```text
numpy==1.24.3
openai==1.70.0
pandas==2.0.3
Requests==2.32.3
torch==2.4.0
tqdm==4.66.4
```

### Installation
```bash
git clone https://github.com/TangJiakai/ReaRec.git
cd ReaRec
```

### Datasets
We conduct experiments on five public datasets: **Yelp**, **Amazon** (including **Video_Games**, **Software**, **CDs_Vinyl**, and **Baby_Products**).

- [**Yelp**](https://business.yelp.com/data/resources/open-dataset/): 
This dataset originates from a well-known business review website, providing rich multidimensional data support for studying user behaviors and business attributes. We treat interactions with ratings greater than 3 as positive samples and apply 20-core filtering to preprocess the data. For textual encoding, we retain the name, location (city and state), and business categories as item information. The dataset is chronologically split into training, validation, and test sets based on two timestamp thresholds: September 4, 2018 and May 12, 2020.

- [**Amazon2023**](https://amazon-reviews-2023.github.io/):
This dataset is derived from Amazon, a leading global e-commerce platform. We select datasets from four domains: Video & Games, Software, CDs & Vinyl, and Baby & Products. For textual features, we retain the product attributes like title, description, and price. Similarly, we treat user-item interactions with user ratings greater than 3 as positive samples. To ensure data quality, we filter out users with fewer than 5 interactions for Video & Games, Software, Baby & Products, and fewer than 10 interactions for CDs & Vinyl. For dataset splitting, we follow [the official absolute timestamps to partition item sequences](https://amazon-reviews-2023.github.io/data_processing/5core.html). This aligns well with real-world scenarios and facilitates fair performance comparisons within the recommendation research community.


### Implementation Details
We conduct all experiments on 8 NVIDIA A100 GPUs. To ensure a fair comparison, we set the embedding size and batch size for all methods to 256 and 2048, respectively. We optimize all models using the Adam optimizer with a learning rate of 0.001 and follow BERT4Rec by adopting GeLU as the activation function.
We truncate user sequences to a maximum length of 50 across all datasets. Since our framework is model-agnostic, it can be seamlessly integrated into various sequential recommendation models. In particular, for BERT4Rec's bidirectional Transformer, we employ a *Prefix Masking* strategy, where the item sequence part utilizes bidirectional attention, while the reasoning adopts unidirectional attention. Early stopping is triggered if the metrics on the validation set do not improve over 10 consecutive epochs.
For item-based methods, we use *LLaMA-3.1-8B* to encode item textual features. In particular, we apply \textit*Principle Component Analysis (PCA)* to the averaged hidden states from the last layer, preserving core features and distilling 768-dimensional model representations.
For ERL method, we search for the KL regularization hyperparameter $\lambda$ within $\{0.001, 0.005, 0.01, 0.05, 0.1\}$.
For PRL method, we set the noise strength $\gamma=0.01$ and tune the base temperature $\tau$ and temperature decay rate $\alpha$ over the ranges $\{0.05, 0.1, 0.5, 1.0, 2.0, 5.0\}$ and $\{1.0, 1.2, 1.5, 2.0, 5.0, 10.0\}$, respectively.


### Usage
```bash
cd src
python main.py --model_name PRL
```

## 📝 Citation

If you find this repository useful, please cite our paper:

```bibtex
@misc{tang2025thinkrecommendunleashinglatent,
      title={Think Before Recommend: Unleashing the Latent Reasoning Power for Sequential Recommendation}, 
      author={Jiakai Tang and Sunhao Dai and Teng Shi and Jun Xu and Xu Chen and Wen Chen and Wu Jian and Yuning Jiang},
      year={2025},
      eprint={2503.22675},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2503.22675}, 
}
```