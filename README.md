## Genre Classification: A Transductive, Inductive and Deep Approach

## Main Dependencies
- Python3.7
- PyGSP
- Networkx
- PyTorch
- Numpy
- Sklearn
- Pandas
- Matplotlib

## Installation
Clone this repository and install all required packages.
```
pip install -r requirements.txt
```
## How to use?
The different arguments that can be passed to `main.py`:
- `--recalculate-features` : Calculates and saves features and then continues classification (Default: False)
- `--only-features` : Only calculates and saves features (Default: False)
- `--graph-statistics`: Report Graph Statistics (Default:None)
- `--genres` : List of genres used for classification (Default: None)
- `--num-classes` : Specify the number of top genres used for classification (Default: None)
- `--threshold` : Threshold for the weights of edges (Default: 0.95)
- `--plot-graph`: Plot the Graph using PyGSP (Default: False)
- `--dataset-size`: Size of dataset used (Default: 'small')
- `--distance-metric`: Metric used to measure distance between features (Default: 'correlation')
- `--with-PCA`: "Apply PCA to features (Default:False)
- `--use-eigenmaps`: Use eigenmaps (Default:False)
- `--transductive-learning`: Apply Transductive Learning (Default:False)
- `--inductive-learning`: Apply Inductive Learning (Default:False)
- `--gcn`: Run GCN (Default: False)
- `--gcn_khop`: Run GCN KHOP (Default: False)
- `--mlp-nn`: Run MLP using Pytorch (Default: False)
- `--use-mlp-features`: Use output of MLP as features to build adjacency (Default: False)
- `--additional-models`: Run SVM, RF, KNN
- `--remove-disconnected`: Removes disconnected nodes (Default: False)
- `--train`: Train the chosen models (Default: False)
- `--prefix`: Add prefix to file names (Default: "None")
- `--PCA-dim`: PCA dimensions (Default: 10)
- `--use-cpu`: Use CPU instead of cuda (Default: False)

Note that you should apply at least one of the two types of learning (Transductive or Inductive)

## Graph Statistics
* Basic: Total number of edges, nodes, number of connected components, number of nodes in giant component, average degree, sparsity

* Advanced: Total number of edges, nodes, number of connected components, number of nodes in giant component, average degree, sparsity, degree distribution in log and linear scale + Plots

* All: Total number of edges, nodes, number of connected components, number of nodes in giant component, average degree, sparsity, degree distribution in log and linear scale, diameter of the graph, average clustering coefficients + Plots

## Training and Testing
To only calculate and save the features and labels with default arguments:
```
python main.py --only-features --threshold 0.9  --dataset-size medium --remove-disconnected --num-classes 8 --with-PCA --PCA-dim 120
```

To run genre classification only by loading previously saved features:
```
python main.py --threshold 0.9  --dataset-size medium --inductive-learning --remove-disconnected --num-classes 8 --with-PCA --PCA-dim 120 --additional-models --train
```

To calculate and save the features and then train classifiers:
```
python main.py --threshold 0.9  --dataset-size medium --inductive-learning --remove-disconnected --num-classes 8 --with-PCA --PCA-dim 120 --additional-models --train --recalculate-features --gcn --gcn_khop
```

To test trained classifiers using test set:
```
python main.py --threshold 0.9  --dataset-size medium --inductive-learning --remove-disconnected --num-classes 8 --with-PCA --PCA-dim 120 --additional-models --gcn --gcn_khop
```

## Using MLP Features to get better features and adjacency

First, train an MLP on the training dataset:
```
python main.py --threshold 0.9  --dataset-size large --inductive-learning --mlp-nn --train --remove-disconnected --num-classes 8 --prefix mlpFeatures --recalculate-features --with-PCA --PCA-dim 120
```

Test your trained MLP:
```
python main.py --threshold 0.9  --dataset-size large --inductive-learning --mlp-nn --remove-disconnected --num-classes 8 --prefix mlpFeatures --with-PCA --PCA-dim 120
```

Second, get new features by passing them through trained MLP and train the classification using them
```
python main.py --threshold 0.9  --dataset-size large --inductive-learning --mlp-nn --remove-disconnected --num-classes 8 --prefix mlpFeatures --with-PCA --PCA-dim 120 --train --recalculate-features --additional-models --gcn --gcn_khop
```

Test the trained methods using test set:
```
python main.py --threshold 0.9  --dataset-size large --inductive-learning --mlp-nn --remove-disconnected --num-classes 8 --prefix mlpFeatures --with-PCA --PCA-dim 120 --additional-models --gcn --gcn_khop
```

## Transductive Learning Methods
This type of learning contains 'Graph-Based Semi-Supervised Learning' algorithms

### Algorithms

* Harmonic Function (HMN) [ICML03][[paper](http://mlg.eng.cam.ac.uk/zoubin/papers/zgl.pdf)]
<!--* Local and Global Consistency (LGC) [NIPS04][[paper](https://papers.nips.cc/paper/2506-learning-with-local-and-global-consistency.pdf)]
* Modified Adsorption (MAD) [PKDD09][[paper](http://talukdar.net/papers/adsorption_ecml09.pdf)]-->
* Partially Absorbing Random Walk (PARW) [NIPS12][[paper](https://papers.nips.cc/paper/4833-learning-with-partially-absorbing-random-walks.pdf)]
* OMNI-Prop (OMNIProp) [AAAI15][[paper](https://pdfs.semanticscholar.org/f217/1ea6e028fb5c2eb1d0256639b4e732764ab4.pdf)]
<!--* Confidence-Aware Modulated Label Propagation (CAMLP) [SDM16][[paper](https://epubs.siam.org/doi/pdf/10.1137/1.9781611974348.58)]-->

## License
Please, see the [license](LICENSE) for further details.
