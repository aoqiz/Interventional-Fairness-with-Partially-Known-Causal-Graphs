## Requirement

This experiment code relies on both python and R environment. To be specific, it was conducted with `python=3.9` and `R=4.2.1` with the following libraries required apart from popular libraries:

### Python

numpy=1.24.2
pandas=1.5.3
networkx=2.8.4
pytorch=2.0.0
**python-igraph=0.10.4**
scipy=1.10.1
scikit-learn=0.24.2

### R

reshape
ggplot2
ggpubr
e1081
condMVNorm
latex2exp (for plotting)



## Get generated data and results

One can simply get the generated data and the results with

```shell
bash run.sh
```

Remember to change the corresponding parameters. The meanings of the parameters are shown as follow (scripts in the `run.sh` file):

```shell
# adm
adm=0
# number of edges
s=20
# number of nodes
for((j=10; j<=40; j+=10))
do
    # graph number
    for((i=0; i<10; i++))
    do
        # Generate synthetic observational data and the ground-truth interventional data.
        python data_generator.py $j $s $i $adm
        # Learn MPDAG from DAG.
        Rscript dag2mpdag.R $j $s $i $adm
        # Fitting conditional densities and generate the interventional data according to the causal effect identification formula.
        Rscript gene_interventions.R $j $s $i $adm
        # Fitting models.
        python soft_constraint.py $j $s $i $adm
    done
    # Rscript Simu_results_filter_and_processor.R $j $s
done
```



## Get plots in the paper

### Tradeoff

It’s easy to reproduce the tradeoff plot with the script `plot_tradeoff.R`. Explicitly, after changing the work directory and parameter `i`:

```R
setwd('PATH_TO_DATA')

...

# one of the four settings that you want to get
i =3
```

and the result plot will be found in `./Repository_adm={adm}` folder with the name `Tradeoff_MnodesNedges_truth.pdf` and `Tradeoff_MnodesNedges.pdf` where M and N denotes the number of nodes and edges correspondingly.

### Density

And to reproduce the density plot with the script `plot_density.R`. Explicitly, after changing the work directory and parameters `dd, ss, adm`:

```R
setwd('PATH_TO_DATA')

...

# number of nodes
dd=5
# number of edges
ss = 8
# number of admissible variables
adm=0
```

and the result plot will be found in `./Repository_adm={adm}` folder with the name `Density_MnodesNedgesKgraph.pdf`  where K stands for the number of graph, M and N denotes the number of nodes and edges correspondingly.

For Student and Credit dataset, run `draw.py` in their folders separately to gain the same results.



## Real data - Student / Credit

One can simply get the generated data in student/credit folder with

```shell
python gen_data_contingency.py
```

and get the results with 

```shell
python soft_constraint_contingency.py
```

Remember to change the corresponding working path. 

It’s easy to reproduce the tradeoff plot with the script in student/credit folder `Simu_results_filter_and_processor_realR`. 

```R
setwd('PATH_TO_DATA')
...
```

and the result plot will be found in `./data` folder with the name `Tradeoff.pdf`.
