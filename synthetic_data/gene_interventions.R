rm(list=ls())
setwd('PATH_TO_DATA')
source('utils.R')

suppressPackageStartupMessages(library(pcalg))
library(ggpubr, quietly = TRUE)
library(e1071, quietly = TRUE)
library(condMVNorm, quietly = TRUE)
options(warn = -1)

## 4. --- Generate interventionals ---
############
# ifRMSE_list = rep(0,m) #interventional fairness
## Generate interventions.
### learn MPDAG from the dataset including Y^ to get the definite descendants and non-descendants of A.
### learn MPDAG from the dataset without Y^ to decompose buckets and generate interventions.
args = commandArgs(trailingOnly = TRUE)
# number of nodes
d = args[1]; d = as.numeric(d)
# number of edges
s = args[2]; s = as.numeric(s)
# which graph
k = args[3]; k = as.numeric(k)
# how many admissible variables
adm = args[4]; adm = as.numeric(adm)

filename = paste0("./Repository_adm=", adm, "/", d, "nodes", s, "edges")

# load data
observational_data = read.csv(paste0(filename, "/observational_data_", k, ".csv"), header=FALSE)
B_bin = read.csv(paste0(filename, "/adjacency_matrix_", k, ".csv"), header=FALSE)
config <- read.csv(paste0(filename, "/config_", k, ".txt"))
protected <- config$protected
outcome <- config$outcome
m <- config$num_sample
n <- config$sample_size
nn <- n
# load bk
bk_file <- paste0(filename, "/bk_", k, ".csv")
bk_file_info <- file.info(bk_file)
bk_file_size <- bk_file_info$size
if(bk_file_size == 0){
  bk = matrix(nrow = 0, ncol = 2)
}else{
  bk = read.table(bk_file, header=FALSE, sep = ",")  
}

# DAG2CPDAG
B_bin_woY = B_bin[-outcome, -outcome]
amat.dag_woY = t(B_bin_woY); rownames(amat.dag_woY) = colnames(amat.dag_woY) = NULL
rDAG_woY = as(t(amat.dag_woY), "graphNEL")
CPDAG_woY <- dag2cpdag(rDAG_woY)

# CPDAG2MPDAG
amat_woY = t(as(CPDAG_woY, 'matrix'))
bk_woY = bk
bk_woY[bk_woY>outcome] = bk_woY[bk_woY>outcome]-1
amat.mpdag2_woY = addBgKnowledge(gInput = amat_woY, x=bk_woY[,1], y=bk_woY[,2])

### Generate interventions V_{A<-a} and V_{A<-a'}
# read admissible variables and values
admissible = read.csv(paste0(filename, "/admissible_", k, ".csv"), header=TRUE)
admissible_vars = admissible$var
admissible_vals = admissible$val

set.seed(2023)
protected_woY = ifelse(protected>outcome, protected-1, protected)
buckets <- getBucketDecomp(x=protected_woY, y=seq(1,d-1)[seq(1,d-1)!=protected_woY], amat=amat.mpdag2_woY)
mu = colMeans(observational_data[,-outcome])
var = cov(observational_data[,-outcome])
# set interventions!
gene_interventional_data0 = matrix(, nrow=nn, ncol=d-1)
gene_interventional_data0[,protected_woY] = rep(0, nn)
gene_interventional_data1 = matrix(, nrow=nn, ncol=d-1)
gene_interventional_data1[,protected_woY] = rep(1, nn)

# write in parent and buckets info
wfilename <- paste0(filename, '/interventional0_bp_', k, '.txt')
# cat( length(buckets),protected)
cat( length(buckets),protected, file=wfilename)

for (i in 1:length(buckets)) {
  # iterated over buckets
  b <- buckets[[i]]
  p <- getParents(amat = amat.mpdag2_woY, b)
  cat('\nb', b, '\tp', p)
  cat( c('\nb', b, '\tp', p), file=wfilename,append=TRUE)
  if (length(intersect(admissible_vars, b))>0) {
    index = match(b, admissible_vars)
    gene_interventional_data0[, b] = rep(admissible_vals[index], nn)
    gene_interventional_data1[, b] = rep(admissible_vals[index], nn) 
  }
  else{
    for (j in 1:nn){
      gene_interventional_data0[j, b] <- rcmvnorm(n=1, mean=mu, sigma=var, dep=b,
                                                  given=p, X=gene_interventional_data0[j, p], method="eigen")
      gene_interventional_data1[j, b] <- rcmvnorm(n=1, mean=mu, sigma=var, dep=b,
                                                  given=p, X=gene_interventional_data1[j, p], method="eigen")
    }
  }
}
interventional0_data_file = paste0(filename, '/interventional0_data_gene_', k, '.csv')
interventional1_data_file = paste0(filename, '/interventional1_data_gene_', k, '.csv')
write.table(gene_interventional_data0, interventional0_data_file, sep = ",", col.names = FALSE, row.names = FALSE)
write.table(gene_interventional_data1, interventional1_data_file, sep = ",", col.names = FALSE, row.names = FALSE)

