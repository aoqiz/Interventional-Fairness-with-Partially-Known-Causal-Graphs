## Overview
This repository contrains the code and instructions for conducting and repeating the experiments in the paper [Interventional Fairness on Partially Known Causal Graphs: A Constrained Optimization Approach](https://openreview.net/forum?id=SKulT2VX9p) (ICLR 2024). The experiments including synthetic data and real data (Student and Credit) are conducted to demonstrate the effectiveness of the proposed method.

## Installation and Setup

### Prerequisites
- Python 3.9
- R 4.2.1

Before running the scripts, ensure that you have the required environments set up You can install Python and R from their official websites.

- Python: https://www.python.org/downloads/
- R: https://cran.r-project.org/

### Dependencies
Install the necessary Python packages using:
```shell
pip install numpy==1.24.2 pandas==1.5.3 networkx==2.8.4 torch==2.0.0 python-igraph==0.10.4 scipy==1.10.1 scikit-learn==0.24.2
```
For R, install the required packages by running the following in your R console:
```R
install.packages(c("reshape", "ggplot2", "ggpubr", "e1071", "condMVNorm", "latex2exp"))
```

### Data and Scripts
For the real-world data, download the Student and Credit datasets from the UCI Machine Learning Repository and place them in the `./data` folder. The Student dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/student+performance) and the Credit dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).

For the convenience for the reader, we provide the processed data in the `./data` folder.

## Usage
### Generating Data and Results
Run the provided shell script to generate data  and results. Modify parameters within `run.sh` as necessary:
```shell
bash run.sh
```
Parameters explanation (as seen in ```run.sh```):
- `adm`: Number of admissible variables
- `s`: Number of edges
- `j`: Number of nodes, iterates from 10 to 40 in steps of 10.
- `i`: Graph number, iterates from 0 to 9.

### Generating Plots
Tradeoff Plots

To reproduce the tradeoff plots, execute `plot_tradeoff.R`:
1. Change the working directory in the script to where your data is located.
2. Set the variable `i` to one of the four settings (different combinations of nodes and edges) you wish to plot.
```R
setwd('PATH_TO_DATA')
...
i =3  # one of the four settings that you want to get
```
The output will be saved in the `./Repository_adm={adm}` folder as `Tradeoff_MnodesNedges_truth.pdf` and `Tradeoff_MnodesNedges.pdf`, where M and N denote the number of nodes and edges respectively.

Density Plots

For density plots, run plot_density.R after setting the working directory and parameters:
```R
setwd('PATH_TO_DATA')
...
dd = 5  # Number of nodes
ss = 8  # Number of edges
adm = 0 # Number of admissible variables
```
Output is stored in the `./Repository_adm={adm}` folder as `Density_MnodesNedgesKgraph.pdf`, where K is the number of graphs, and M and N denote the number of nodes and edges respectively.

### Real Data Analysis (Student and Credit)
Generate interventional data and results as follows: (Credit dataset is used as an example)
```shell
cd student  # or 'cd credit'
python gen_data_contingency.py    # generate data
python separate_data.py     # align input format
python soft_constraint.py   # process
```
Tradeoff plots can be reproduced using Simu_results_filter_and_processor_real.R within the respective folders. Remember to set the working directory appropriately:
```R
setwd('PATH_TO_DATA')
...
```

## Contributing
We welcome contributions to this repository. If you find any issues or have suggestions for improvements, please open an issue or a pull request.

