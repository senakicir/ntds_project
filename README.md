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
- `--genres` : List of genres used for classification (Default: None)
- `--num-classes` : Number of classes randomly chosen for classification (Default: None)
- `--threshold` : Threshold for the weights of edges (Default: 0.66)
- `--plot-graph`: Plot the Graph using PyGSP (Default: False)

To only calculate and save the features and labels with default arguments:
```
python main.py --only-features
```

To run genre classification only by loading previously saved features:
```
python main.py
```

To calculate and save the features and then continue classification:
```
python main.py --recalculate-features --genres 'Hip-Hop' 'Rock' --threshold 0.66
```

## Credits
Reference papers used


## License
Please, see the [license](LICENSE) for further details.
