# Unsupervised Text-to-Sound Mapping

# Paper

## 1. Introduction

This research will focus on developing a framework for unsupervised text-to-sound map-
ping by mapping two distinct embedding spaces: one for text and one for sound. The
goal is to create a system that can convert a textual input into a corresponding sound
output, without relying on labeled data. The most challenging aspect of this research
lies in finding the appropriate criteria for mapping the two spaces. Ultimately, the sound
generated from a sentence should reflect the meaning and inherent properties of the input
text, as well as the chosen sound dataset.

The task: Create an artistic/creative tool that can generate mappings between words and sounds. 
I type a sentence, this sentence is represented musically, where each word becomes one sound. 
We do not seek an objective mapping in which meaning is recreated with sound. There is no semantic link between the a word and its sonic mapping, but there is a semantic relationship as a whole, in the relationships between sounds. An example on the relationship between words and sounds: evaluation is not that ‘red’ maps to a certain sound that relates to the concept of ‘red’, but that the relationship between the ‘red’ sound and the ‘blue’ sound is somehow similar to the relationship between the (semantic meaning of the) words ‘red’ and ‘blue’. Since ‘red’ and ‘blue’ are both colors, they belong to the same semantic category, yet they are not similar colors. Therefore, a possible sound result could be two sounds that are from a similar category, such as:  string instrument samples, long sounds, harmonic sounds, or low frequency sounds; yet have a parameter that is distinct, such as: loudness, pitch, or timbre. 

In the context of natural language processing (NLP), word embeddings, such as word2vec,
have proven to be highly effective in capturing semantic relationships between words.
In a similar vein, deep learning techniques have been applied to audio data to extract
meaningful representations. However, the challenge arises when trying to map these two
embedding spaces—text and sound—into a common framework. Unsupervised learning
techniques can offer the necessary flexibility to explore this mapping without relying on
labeled data, which is often scarce in audio datasets (citation needed). The objectivity of labeling musical data can be questioned, as it may result from personal biases: the same music may sound different to two different people, based on their preferences and listening history (citation needed). For example, a listener entrained in Western music and sounds may label an inharmonic sound as sad or scary, which could result from cultural and historical uses of inharmonic sounds. 

This research will attempt to create a system that maps a given text input into a corresponding sound output by:

- Mapping a sound corpus and a text input into distinct embedding spaces using pre-trained models
- Addressing the complex task of aligning these embeddings into a meaningful text-
to-sound mapping using unsupervised methods
- Creating a mapping between the two embedding spaces that can convert an input text into an output audio file
- Evaluating the given mapping both quantitatively and qualitatively

Short summary of results, including quantitative and qualitative results. Comment on whether the quantitative and qualitative evaluations align: Do we like the sounds that result from the best mapping more than the sounds that come from the other mappings? 

## 2. Related Work

Different systems for text-to-sound mapping have been created (give examples). CLAP is an example of a multi-modal embedding space that connects sound and text in the same space. 

Various sound models exist. We can take advantage of existing models that feature an encoder in their architecture and use them to create embeddings of sound. Some examples of audio encoders are: MuQ, EnCodec, Wav2Vec. 

There are many available text embedding models, including word2vec, fastText, BERT, GloVe, and T5. These models can be static or contextual. 

Multi-modal models such as Wav2Vec2-BERT, AudioCLIP, CLAP, MERT, and MuLan, and MuQ-MuLan.

Find and discuss art pieces that do something similar to this.

## 3. Methodology

We use MuQ for sound embedding.

We use three text embeddings: two are static: word2vec, fastText and one is contextual: RoBERTa.

We use normalization techniques (StandardScaler from sklearn) and PCA. 

We define three mapping strategies. Each strategy starts by applying normalization and PCA with the same parameters for the sound and text spaces. The strategies are: 

1. Identity: the mapping matrix is the identity, so we simply find the closest sound embedding to the input text embedding
2. Cluster: we cluster the sound and text emebeddings separately. Then for each text input, we find the sound cluster whose centroid is closest to the centroid of the text cluster that the input belongs to. Then we find the sound embedding in that cluster that is closest to our text embedding.
3. Iterative Closest Point (ICP): defined in a different paper, provide summary here

We evaluate our system in different ways.

We evaluate mappings with three distance metrics:

1. Pairwise distance: given a random pair of text embeddings, we compute the distance between the text embeddings. Then we compute the distance between the two sound embeddings that those texts map to. Then we calculate the difference between the distances.
2. Wasserstein distance: same as the pairwise distance except we compare the distributions of distances between text pairs and sound pairs by calculating the Wasserstein (Earth Mover’s) distance
3. CLAP distance: we create the same pairs but the distance between pairs is calculated in CLAP feature space instead of the transformed embedding space that the text and sound embeddings are in

The pairwise approach is used because of our motivating example presented in the Introduction: we wish for the relationship between the input words ‘red’ and ‘blue’ to be similar to the relationship between the sounds that result from these words. Therefore, the distance between the two text embeddings and the distance between the two sound embeddings should be similar.

We choose these three metrics because

We also evaluate our clustering in the following way:

After clustering each space separately, we calculate the following three clustering metrics on the combined space, meaning a single space in which all clusters for both text and sound are present:

1. Silhouette score
2. Calinski-Harabasz score
3. Davies Bouldin score

We use these three metrics because

## 4. Experiments and Results

We define one experiment to be a single run of our system with a unique set of parameters. The possible parameters of an experiment are:

- sound corpus
- text input
- sound embedding
- text embedding
- sound pre-processing
- normalization method
- PCA dimensionality
- mapping type
    - number of clusters (k) if using clustering method

We evaluate our system quantitatively with the metrics presented in Section 3: Methodology. Here we present the results:

Discuss which mappings performed best with different distance metrics

Discuss qualitative evaluation, which is subjective.

## 5. Discussion

Present the analysis of the results. Tradeoffs of different distance metrics and clustering metrics. Discuss challenges and insights. 

## 6. Conclusion and Future Work

Summarize paper: the task, the methods, the experiments, the results, and analysis.

Reflect on the usefulness of such an artistic tool, specifically the advantages and disadvantages of using an supervised system.

Discuss various possibilities for future work, which should be robust and include: 

- more sound embeddings
- the use of a sound VAE that allows for the generation of new sounds
- the ability to train/create a mapping from a large text corpus and then input a new text input that is what is sonified. Could this be real time? What would it do if a word was input that it hadn’t seen before?
- polyphony


## Results

For euclidean distance metric:

As clustering improves, pairwise distance decreases, as shown by the Pearson coefficients:
- combined_silhouette_score: 0.0931 (weak positive correlation) (bad: better clustering correlates with worse mapping)
- combined_calinski_harabasz_score: -0.4150 (moderate negative correlation)
- combined_davies_bouldin_score: 0.0227 (weak positive correlation) (good: lower DB score correlates with better mapping)

Correlations with k=2 with wasserstein_distance:
- combined_silhouette_score: -0.9763 (strong negative correlation) (good: better clustering correlates with better mapping)
- combined_calinski_harabasz_score: -0.9996 (strong negative correlation)
- combined_davies_bouldin_score: 0.8378 (strong positive correlation) (good: lower DB score correlates with better mapping)


The task: Create an artistic/creative tool that can generate mappings between words and sounds. 
I type a sentence, this sentence is represented musically, where each word becomes one sound. 
The evaluation is not that ‘red’ maps to a certain sound that relates to the concept of ‘red’, 
but that the relationship between the ‘red’ sound and the ‘blue’ sound is somehow similar to the relationship between 
the (semantic meaning of the) words ‘red’ and ‘blue’.

The task could be either retrieval OR generation. In some ways, generation and retrieval are similar tasks, 
depending on the time granularity. When you separate the sounds into small grains (40ms), then even retrieval is essentially generation.

The user provides a sound corpus. They may also provide a text corpus, otherwise a generic text corpus is used.

Text corpus is either:
1. a generic text corpus provided by me
2. a user provided text corpus
3. there is no corpus but instead the generalized space of the text embedding, which could be transformed by sampling the space and applying a transform to each sampled point

The input to the system is a word or string of words. The output is a sound. The length of the sound may be arbitrary
or determined by the user. Each word leads to one sound. 

The mapping is unsupervised, but pre-trained supervised methods can be used in the system.

# Experiments

## Next steps
- [x] Sound corpus input: slice into chunks/grains and add option to remove silence
- [ ] Evaluation: how can we evaluate the system?
- [ ] Try with different granularities of sounds (full sounds, onsets, grains)
- [ ] Implement other mapping methods
- [ ] Implement other audio and text encoders
  - I think I should use word2vec or any static encoder that is not context dependent so that the same word maps to the same place. Start simple. 

## Methodology

Input sound corpus is embedded in a sound feature space $S$ as a result of a pre-trained audio encoder.

Text corpus is embedded in a text feature space $T$ as a result of a pre-trained text encoder.

A mapping $M$ is created between $S$ and $T$. 

At test time, the user inputs a string of words $t$. Each word $t_i$ is mapped to the sound space and a single sound $s_i$ is 
returned. These sounds are concatenated and returned to the user.

Parameters:
1. mapping function (or types of normalization)
2. distance metric

### Text encoders
- word2vec
- BERT
- RoBERTa
- GloVe
- T5
  - Flan-T5-Large: https://huggingface.co/google/flan-t5-large

Contextual vs. Static Embeddings: RoBERTa provides contextual embeddings, meaning the representation of a word depends on its context within the sentence. If you require static embeddings (e.g., for individual words without context), models like Word2Vec or GloVe might be more appropriate.

In transformer-based language models like BERT and RoBERTa, the [CLS] (classification) token is a special token appended to the beginning of every input sequence. During training, the model learns to aggregate the contextual information of the entire sequence into the embedding of this [CLS] token. This embedding is then utilized for various downstream tasks, such as classification or regression. Extracting the [CLS] Token Embedding: To obtain the embedding corresponding to the [CLS] token, you can access the first token's embedding from the model's output. In the context of the provided code, this can be achieved as follows: `cls_embedding = outputs.last_hidden_state[:, 0, :]`

To derive a single embedding representing the entire sentence, you can apply a pooling strategy to the token embeddings. A common approach is to average the embeddings of all tokens, considering the attention mask to ignore padding tokens:

For tasks like clustering or semantic search, you might prefer using pre-trained models optimized for generating sentence embeddings. The sentence-transformers library offers such models:

### Audio encoders
- a custom trained VAE
- Wav2Vec2
- COLA
- EnCodec
- MuQ https://arxiv.org/pdf/2501.01108v2
- HuBERT
  - uses K-means clustering labels to guide self-supervised learning in Masked Language Model (MLM) style

#### MuQ
Seems to be trained to work at sampling rate of 24kHz, but perhaps it could work at other sampling rates. 
It outputs embeddings of size (t, 1024) where t is related to the length of the sample
t = 55 for input length 6s
t = 728 for input 29s

The frame size is 2048 samples, so t * 2048 = length of the input in samples 

### Mapping strategies

So the question is: what is a good mapping $W$ between the two spaces? 

**Any mapping procedure that can be applied to points could also be applied to clusters**

First approach: scale and normalize the spaces to align them. For each feature, subtract the mean and divide by the variance,
or standard deviation, etc. 

Normalization techniques:
1. subtract the mean and divide by the variance or standard deviation, etc. 
2. Apply dimensionality reduction (PCA, NMF, t-SNE)

Once the spaces have been normalized, a first approach is to directly map the points. 

1. Essentially just take a point $t$ and find the nearest neighbor $s$. This would be an identity mapping $W = I$ 

Another idea is to use clustering:

2. By clustering: cluster each space into the same number of clusters and randomly associate each text cluster to a sound cluster.
Then calculate the distance between each text cluster and the origin. For the associated sound cluster, force it to also
be the same distance from the origin. Now you have a mapping.

Guide for selecting number of clusters: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py

Another possibility is to calculate the distance from each cluster and force that to be the same between the two spaces. 
For an input $t$, we calculate the distance to each text cluster, then we find the point in $S$ with the same distances 
to each cluster in $S$.

Both text and audio embedding spaces are clustered independently, then mappings are established between clusters rather than individual embeddings. For new inputs, the system identifies the appropriate cluster in the source domain and maps to the corresponding cluster in the target domain.

3. By comparing two points: for any two points $t_1, t_2$ in the text space, the distance $d_t$ between them should be the same
as the distance between their mappings in the sound space, $s_1, s_2$: $d(t_1, t_2) == d(s_1, s_2)$. Can we somehow 
optimize this? What if we cluster the sound space and then transform the text space so that this is true? That would
involve training a small network I think.

#### An adversarial approach

Train a generator $W$ to transform text embeddings into sound embeddings, while a discriminator 
attempts to distinguish between these generated embeddings and real sound embeddings. Through this adversarial process, 
the generator learns to produce increasingly convincing sound embeddings from text inputs.

#### Contrastive loss

Training a network with contrastive loss: create an arbitrary mapping between text and sound and use contrastive loss to push things farther apart

#### Iterative Closest Point Method: 

This approach uses **Mini-Batch Cycle Iterative Closest Point (MBC-ICP)** which is defined in [Non-Adversarial Unsupervised Word Translation](https://arxiv.org/pdf/1801.06126): 
given two spaces $S$ and $T$, use PCA to reduce the dimensionality. Now the spaces have the same dimensionality.
Given two transformational matrices $W_S$ which maps from $S$ to $T$ and $W_T$ which maps from $T$ to $S$.

Now perform an iterative process:
1. For each $s \in S$, find $t^*$, the nearest $W_T t$ to $s$, which is the best text encoding for $s$
2. For each $t \in T$, find $s^*$, the nearest $W_S s$ to $t$, which is the best sound encoding for $t$
3. Optimize $W_S$ and $W_T$ by minimizing: distance $W_S s$ and $t*$ + distance $W_T t$ and $s*$ + cycle constraints (ex: minimize distance between $s$ and $W_S W_T s$)

We could also use CLAP in a (semi-)supervised way, where you replace $t*$ with $CLAP(t)$ or you find $t*$ by finding the nearest neighbor in the CLAP encoding ($t*=$ nearest $CLAP(t)$ to $s$). 
But then we would really just be using CLAP to align our spaces, so what's the reason we don't just use CLAP for the entire thing?

Assumption they make: intializing the matrices to the identity is valid because the two spaces share some similarity since they both represent language data,
and PCA can align these spaces to some degree.

This may or may not be true with our feature spaces. 

Possible initialization methods:

1. Use Optimal Transport (OT) instead of PCA to get an initial alignment. Unlike PCA, OT does not assume the spaces are already structurally similar. It finds the best possible alignment between two distributions even if they are not naturally aligned.
2. Use Adversarial method to get an initialized $W_T$ that is then further refined by ICP
3. Use CLAP to get initial relationships (Would CLAP be considered a teacher model here?)
4. Use domain knowledge: compute standard audio descriptors (brightness, loudness, noisiness) and use that for initial alignment

## Evaluation

### Cluster evaluation

Cluster evaluation can tell us whether our embedding is good or not, as well as the parameters we chose for dimensionality reduction, 
number of clusters, etc.

We can use measures that evaluate the clustering of a space, such as Akaike information criterion (AIC) or Bayesian 
information criterion (BIC). Compute the intra-cluster distance (for each point in the cluster, the distance between
the point and that cluster's centroid) and normalize by the inter-cluster distance ...? I'm not sure. You want this to equal 1: $|aic1 - aic2| - |AIC1 - AIC2|$

https://www.geeksforgeeks.org/ml-intercluster-and-intracluster-distance/

If we use a sound corpus that has inherent clusters (like TinySOL we would expect clustering by instrument or family) 
then we could check to see if the words in a given cluster map to different samples of the same instrument.

Other metrics:
- Hausdorff distance: measure how well a learned mapping between sound embeddings and text embeddings aligns the two spaces. Since Hausdorff distance measures the worst-case closest distance between two sets, it’s ideal for assessing whether every point in one space has a close and meaningful counterpart in the other.
- Calinski–Harabasz Index (Variance Ratio Criterion): Evaluates cluster validity by considering the ratio of between-cluster dispersion to within-cluster dispersion. Higher values indicate better-defined clusters
- Silhouette Score: Measures how similar an object is to its own cluster compared to other clusters. Scores range from -1 to 1, with higher values indicating better clustering
- Davies–Bouldin Index: Represents the average similarity ratio of each cluster with its most similar cluster. Lower values signify better clustering.

### Mapping evaluation

For each $t_i,t_j$ that are in the same cluster, calculate the difference between the distance between them and the distance 
between their sounds: $|d(t_i, t_j) - d(s^*_i, s^*_j)|$. The smaller the value the better.

Variation: If we want to compare it for every pair of text inputs, we need to consider the fact that the distance between
text clusters should not affect the measurement -- well actually I'm not sure about that. Without alignment, it wouldn't matter that the 


## Notes

Things to look into
- https://en.wikipedia.org/wiki/Persistent_homology
- https://en.wikipedia.org/wiki/Gromov%E2%80%93Hausdorff_convergence / https://en.wikipedia.org/wiki/Hausdorff_distance

### Distance metrics

The lower the dimension, the more distance makes sense. The higher the dimension, the more variance makes more sense.

In low dimensional spaces (1-10 dims), distance metrics (Euclidean distance) align with our ideas of closeness. This is not true in higher dimensional spaces (hundreds of dims) , as data points become equidistant from each other and sparsity is high. 
k-NN and clustering makes sense in low dimensional spaces. 

We can apply PCA/NMF/t-SANE to reduce dimensions or consider other distance metrics:
- Cosine similarity (often better than Euclidean in high-dimensional spaces).
- Mahalanobis distance (accounts for correlations between variables).
- Kernel methods (e.g., using an RBF kernel to transform data before computing distances).

### Normalization methods

See: https://en.wikipedia.org/wiki/Feature_scaling, https://scikit-learn.org/stable/api/sklearn.preprocessing.html

Normalizing embeddings—points in a neural network's feature space—is crucial for ensuring consistent scales, improving model performance, and facilitating effective training. Several methods are commonly employed:

1. **Standardization (Z-score Normalization):** This technique adjusts the embedding's components to have a mean of zero and a standard deviation of one. For an embedding vector \( x \), each component \( x_i \) is transformed as:

   $
   x_i' = \frac{x_i - \mu}{\sigma}
   $

   where $ \mu $ is the mean and $ \sigma $ is the standard deviation of the components. This process ensures that the embedding has a standardized scale, which can be beneficial for various machine learning algorithms.

2. **Min-Max Normalization:** This method scales the embedding's components to a specific range, typically [0, 1]. The transformation for each component \( x_i \) is:

   $
   x_i' = \frac{x_i - \min(x)}{\max(x) - \min(x)}
   $

   where $ \min(x) $ and $ \max(x) $ are the minimum and maximum values of the components, respectively. This scaling is useful when the embedding's range needs to be confined within specific bounds.

3. **Unit Vector Normalization (Vector Scaling):** This approach scales the entire embedding vector to have a unit norm (typically L2 norm of 1), preserving its direction but standardizing its magnitude. For an embedding vector \( x \), the normalized vector \( x' \) is:

   $
   x' = \frac{x}{\|x\|}
   $

   where $ \|x\| $is the L2 norm of $ x $. This method is particularly effective when the direction of the embedding carries more significance than its magnitude.

4. **Batch Normalization:** Commonly used during neural network training, batch normalization normalizes the inputs of each layer to have a consistent distribution across mini-batches. This technique stabilizes and accelerates training by reducing internal covariate shift. [Source: Wikipedia - Batch normalization](https://en.wikipedia.org/wiki/Batch_normalization)

5. **Layer Normalization:** Unlike batch normalization, layer normalization normalizes the inputs across the features instead of the batch dimension. This method is particularly useful in recurrent neural networks and transformer architectures, where batch sizes can be small or vary. [Source: Wikipedia - Normalization (machine learning)](https://en.wikipedia.org/wiki/Normalization_%28machine_learning%29)

6. **Weight Normalization:** This technique reparameterizes the weight vectors in neural networks to decouple their direction and magnitude, which can lead to more stable and efficient training. [Source: Wikipedia - Normalization (machine learning)](https://en.wikipedia.org/wiki/Normalization_%28machine_learning%29)

7. **Spectral Normalization:** Primarily used in Generative Adversarial Networks (GANs), spectral normalization controls the spectral norm of weight matrices, promoting stability during training by ensuring that the discriminator's Lipschitz constraint is satisfied. [Source: Wikipedia - Normalization (machine learning)](https://en.wikipedia.org/wiki/Normalization_%28machine_learning%29)

The choice of normalization technique depends on the specific application, the architecture of the neural network, and the nature of the embeddings. Proper normalization ensures that embeddings are on a comparable scale, which is essential for the stability and performance of machine learning models.


### Multi-modal embeddings
- Wav2Vec2-BERT: https://huggingface.co/docs/transformers/en/model_doc/wav2vec2-bert
- AudioCLIP: https://arxiv.org/pdf/2106.13043
  - Contrastive learning is a type of self-supervised learning
  - Cross-modal querying: input a meow and get a picture of a cat
- CLAP: https://arxiv.org/pdf/2206.04769
  - Supervised, two encoders and a contrastive learning to create a joint embedding space between text and audio
- LAION-CLAP: https://arxiv.org/pdf/2211.06687
  - Supervised text-to-audio retrieval, potentially better than CLAP, also has a high dim feature space
- MERT: Music understanding model: https://arxiv.org/pdf/2306.00107
  - uses teacher models: one for signal understanding (acoustic teacher) and one for musical understanding (pitch, harmony)
  - RVQ-VAE
- MuLan: https://arxiv.org/pdf/2208.12415
  - Shared text-music embedding, trained on 44 million recordings (370,000 hours). Uses contrastive loss on supervised data. 
  - They connect a random FFT frame with one word of the associated audio tag. They found one word worked better than multiple
  - Audio encoder: Resnet-50 and Audio Spectrogram Transformer (AST)
  - Text encoder: BERT
  - not open source apparently?!
- MuQ-MuLan: https://arxiv.org/pdf/2501.01108v2
  - even better than MuLan
  - self-supervised approach, contrastive learning
  - LAION-CLAP with their own (MuQ) audio encoder
  - Audio encoder: MuQ
  - Text enocder: RoBERTa

### Audio datasets
Find more datasets here: https://paperswithcode.com/datasets?page=1
- Million Song Dataset: http://millionsongdataset.com/


### Text datasets
- https://ics.uci.edu/~smyth/courses/cs175/text_data_sets.html


### Audio-text datasets
These can be used for evaluation
- Clotho: https://arxiv.org/pdf/1910.09387
- AudioCaps: https://audiocaps.github.io/
- MusicCaps: https://www.kaggle.com/datasets/googleai/musiccaps
- AudioSet: https://research.google.com/audioset/
- MagnaTagATune: https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset
- MTG-Jamendo: https://mtg.github.io/mtg-jamendo-dataset/
- SoundDescs: https://github.com/akoepke/audio-retrieval-benchmark?tab=readme-ov-file

# Install instructions
## Windows
```
conda install conda-forge::transformers

# install fastText for Windows
pip install fasttext-wheel
mkdir fastText
cd fastText
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gunzip cc.en.300.bin.gz
```