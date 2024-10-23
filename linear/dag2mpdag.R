rm(list=ls())
setwd('PATH_TO_DATA')

suppressPackageStartupMessages(library(pcalg))
library(ggpubr, quietly = TRUE)
library(e1071, quietly = TRUE)
options(warn = -1)

args = commandArgs(trailingOnly = TRUE)
# number of nodes
d = args[1]; d = as.numeric(d)
# number of edges
s = args[2]; s = as.numeric(s)
# which graph
k = args[3]; k = as.numeric(k)
# how many admissible variables
adm = args[4]; adm = as.numeric(adm)

cat(k, ":\n")

bk_prop = 0.1
set.seed(2022)
filename = paste0("./Repository_adm=", adm, "/", d, "nodes", s, "edges")

# 1.  --- DAG2CPDAG ---
B_bin = read.csv(paste0(filename, "/adjacency_matrix_", k, ".csv"), header=FALSE)
amat.dag = t(B_bin)
rownames(amat.dag) = NULL
rDAG = as(t(amat.dag), "graphNEL")
CPDAG <- dag2cpdag(rDAG)


## 2. --- Add background knowledge and get MPDAG ----
config <- read.csv(paste0(filename, "/config_", k, ".txt"))
protected <- config$protected
protected_classes <- config$protected_classes
outcome <- config$outcome

amat = t(as(CPDAG, 'matrix'))
diff_dag2cpdag = amat-ifelse(amat.dag!=0,1,0)
undirected_edge = which(diff_dag2cpdag==1, arr.ind = TRUE, useNames = FALSE)
n_undirected = nrow(undirected_edge)
bk_random = matrix(undirected_edge[sample(1:n_undirected, ceiling(bk_prop*n_undirected)),], ncol=2)

bk_protected_out = matrix(undirected_edge[undirected_edge[,1]==protected,], ncol=2)
bk_protected_in = matrix(undirected_edge[undirected_edge[,2]==protected,], ncol=2)
bk_protected = rbind(bk_protected_out, bk_protected_in)

## the background knowledge can help meet the identification criterion (no Adm->V)  
admissible = read.csv(paste0(filename, "/admissible_", k, ".csv"), header=TRUE)
admissible_vars = admissible$var
# Define the set V as the difference of 1:(d-1) and admissible_vars
V <- setdiff(1:(d-1), admissible_vars)
# Select the rows that satisfy the conditions
selected_rows <- which((undirected_edge[, 1] %in% admissible_vars & undirected_edge[, 2] %in% V) | (undirected_edge[, 1] %in% V & undirected_edge[, 2] %in% admissible_vars))
bk_identification <- undirected_edge[selected_rows, ]
# Give the case where Adm-Adm can exists in the graph.
exist = which(undirected_edge[, 1] %in% admissible_vars & undirected_edge[, 2] %in% admissible_vars)
if (length(exist) != 0){cat('\nHere exists Adm-Adm in the graph!!!')}

# background knowledge
bk = rbind(bk_random, bk_protected, bk_identification)
bk <- bk[!apply(bk, 1, function(row) any(row == outcome)), ]
if(nrow(bk)>1){bk = matrix(bk[!duplicated(bk[,]),], ncol=2)}

bk_file = paste0(filename, '/bk_', k, '.csv')
write.table(data.frame(bk), file = bk_file, sep = ",", row.names = FALSE, col.names = FALSE)

## 3.  --- Identify causal relationships ---
Des <- possDe(m = amat.dag, x = protected, possible = FALSE, ds = FALSE, type = "dag")
NonDes <- setdiff(1:length(amat.dag[1,]), Des)
# write out
relation <- list('defNonDes' = NonDes, 'possDes' = c(), 'defDes' = Des, 'NonDes' = NonDes, 'Des' = Des)
relation_file = paste0(filename, '/relation_', k, '.txt')
out = lapply(relation, write, relation_file, append=TRUE, ncolumns=1000)

