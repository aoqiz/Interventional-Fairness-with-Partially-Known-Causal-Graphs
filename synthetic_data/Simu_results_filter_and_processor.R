rm(list=ls())
setwd('PATH_TO_DATA')
options(warn = -1)
library(reshape)
library(ggplot2)

#############################################################################################
## Compare the results on the generated interventionals with the ground-truth interventionals
#############################################################################################
# number of nodes
dd = c(5,10,20,30)
# number of edges
ss = c(8,20,40,60)
adm=0
filename = paste0("./Repository_adm=", adm)
k=c(0:9)

df_tradeoff = as.data.frame(matrix(nrow=0, ncol = 4))
dfBase_tradeoff = as.data.frame(matrix(nrow=0, ncol = 4))
for(i in 3:length(dd))
{
  d=dd[i]
  s=ss[i]
  rawRMSE <- read.csv(paste0(filename, "/", d, "nodes", s, "edges", "/RMSE.csv", sep=""))[k+1,]
  rawUnfairness <- read.csv(paste0(filename, "/", d, "nodes", s, "edges", "/Unfairness.csv", sep=""))[k+1,]
  rawRMSEIF <- read.csv(paste0(filename, "/", d, "nodes", s, "edges", "/RMSEIF.csv", sep=""))[k+1,]
  rawUnfairnessIF <- read.csv(paste0(filename, "/", d, "nodes", s, "edges", "/UnfairnessIF.csv", sep=""))[k+1,]
  rawRMSEIF_truth <- read.csv(paste0(filename, "/", d, "nodes", s, "edges", "/RMSEIF_truth.csv", sep=""))[k+1,]
  rawUnfairnessIF_truth <- read.csv(paste0(filename, "/", d, "nodes", s, "edges", "/UnfairnessIF_truth.csv", sep=""))[k+1,]
  
  RMSE <- na.omit(rawRMSE); Unfairness <- na.omit(rawUnfairness)
  RMSEIF <- na.omit(rawRMSEIF); UnfairnessIF <- na.omit(rawUnfairnessIF)
  RMSEIF_truth <- na.omit(rawRMSEIF_truth); UnfairnessIF_truth <- na.omit(rawUnfairnessIF_truth)
  
  cutoff_gene = intersect(which(RMSEIF$X20IF<5), which(UnfairnessIF$X0IF<0.075))
  cutoff_truth = intersect(which(RMSEIF_truth$X20IF<5), which(UnfairnessIF_truth$X0IF<0.075))
  
  cutoff = intersect(cutoff_gene, cutoff_truth)
  
  RMSE = RMSE[cutoff,]; Unfairness = Unfairness[cutoff,]
  RMSE_mean <- colMeans(RMSE); Unfairness_mean <- colMeans(Unfairness)
  
  RMSEIF = RMSEIF[cutoff,]; UnfairnessIF = UnfairnessIF[cutoff,]
  RMSEIF_mean <- colMeans(RMSEIF); UnfairnessIF_mean <- colMeans(UnfairnessIF)
  
  RMSEIF_truth = RMSEIF_truth[cutoff,]; UnfairnessIF_truth = UnfairnessIF_truth[cutoff,]
  RMSEIF_truth_mean <- colMeans(RMSEIF_truth); UnfairnessIF_truth_mean <- colMeans(UnfairnessIF_truth)
  
  df = data.frame(Unfairness = UnfairnessIF_mean, RMSE = RMSEIF_mean, type = paste(d, "nodes", s, "edges", sep = ''), source = "Gene")
  dftruth = data.frame(Unfairness = UnfairnessIF_truth_mean, RMSE = RMSEIF_truth_mean, type = paste(d, "nodes", s, "edges", sep = ''), source = "Real")
  dfBase = data.frame(Unfairness = Unfairness_mean[c(2,3)], RMSE = RMSE_mean[c(2,3)], type = paste(d, "nodes", s, "edges", sep = ''), source = "Real")
  df_tradeoff=rbind(df_tradeoff, df, dftruth)
  dfBase_tradeoff=rbind(dfBase_tradeoff, dfBase)
}


pdf(paste0(filename, "/RMSE.pdf", sep=""), width = 14, height = 7)
# pdf(paste0(filename,"/", d, "nodes", s, "edges", "/RMSE.pdf", sep=""), width = 14, height = 7)
ggplot(data = df_tradeoff, mapping = aes(x = RMSE, y = Unfairness, color = type, linetype = source)) + geom_line() + geom_point() +
  geom_point(data = dfBase_tradeoff, mapping = aes(x = RMSE, y = Unfairness, color = type, linetype = source)) +
  theme(legend.position = "top",
        legend.title = element_blank(),
        legend.text=element_text(size = 12),
        axis.title=element_text(size = 20))
dev.off()

