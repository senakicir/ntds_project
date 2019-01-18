## Genre Classification for playlist recommendation
A little info about your project and/ or overview that explains **what** the project is about.

## Motivation
A short description of the motivation behind the creation and maintenance of the project. This should explain **why** the project exists.

## Screenshots
Include logo/demo screenshot etc.

## Features
What makes your project stand out?

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
- `--threshold` : Threshold for the weights of edges (Default: 0.66)
- `--plot-graph`: Plot the Graph using PyGSP (Default: False)
- `--dataset-size`: Size of dataset used (Default: 'small')
- `--distance-metric`: Metric used to measure distance between features (Default: 'correlation')
- `--with-PCA`: "Apply PCA to features (Default:False)
- `--use-eigenmaps`: Use eigenmaps (Default:False)
- `--transductive-learning`: Apply Transductive Learning (Default:False)
- `--inductive-learning`: Apply Inductive Learning (Default:False)
- `--gcn`: Run GCN (Default: False)
- `--PCA-dim`: PCA dimensions (Default: 10)
-

Note that you should apply at least one of the two types of learning (Transductive or Inductive)

To only calculate and save the features and labels with default arguments:
```
python main.py --only-features
```

To run genre classification only by loading previously saved features:
```
python main.py --inductive-learning --transductive-learning
```

To calculate and save the features and then continue classification:
```
python main.py --recalculate-features --genres 'Hip-Hop' 'Rock' --threshold 0.66 --inductive-learning --transductive-learning
```
## Transductive Learning Methods
This type of learning contains 'Graph-Based Semi-Supervised Learning' algorithms

### Algorithms

* Harmonic Function (HMN) [ICML03][[paper](http://mlg.eng.cam.ac.uk/zoubin/papers/zgl.pdf)]
* Local and Global Consistency (LGC) [NIPS04][[paper](https://papers.nips.cc/paper/2506-learning-with-local-and-global-consistency.pdf)]
* Modified Adsorption (MAD) [PKDD09][[paper](http://talukdar.net/papers/adsorption_ecml09.pdf)]
* Partially Absorbing Random Walk (PARW) [NIPS12][[paper](https://papers.nips.cc/paper/4833-learning-with-partially-absorbing-random-walks.pdf)]
* OMNI-Prop (OMNIProp) [AAAI15][[paper](https://pdfs.semanticscholar.org/f217/1ea6e028fb5c2eb1d0256639b4e732764ab4.pdf)]
* Confidence-Aware Modulated Label Propagation (CAMLP) [SDM16][[paper](https://epubs.siam.org/doi/pdf/10.1137/1.9781611974348.58)]

## Credits
Reference papers used


## License
Please, see the [license](LICENSE) for further details.
