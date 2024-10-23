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