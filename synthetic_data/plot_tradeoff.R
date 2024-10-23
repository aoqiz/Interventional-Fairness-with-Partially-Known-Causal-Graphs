# ****************************************
# gene plot
# ****************************************
rm(list=ls())
setwd('PATH_TO_DATA')
library(ggplot2)
library(ggbreak)
library(latex2exp)

# # number of nodes
dd=c(5,10,20,30)
# number of edges
ss = c(8,20,40,60)
# number of admissible variables
adm=0
# one of the four settings that you want to get
i =3
filename = paste0("./Repository_adm=", adm)
k=c(0:9)
colors <- c("#f9ca24", "#6ab04c", "#7ed6df", "#ea8685")


df_tradeoff = as.data.frame(matrix(nrow=0, ncol = 4))
dfBase_tradeoff = as.data.frame(matrix(nrow=0, ncol = 4))


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

RMSE_mean <- colMeans(RMSE); Unfairness_mean <- colMeans(Unfairness)
RMSEIF_mean <- colMeans(RMSEIF); UnfairnessIF_mean <- colMeans(UnfairnessIF)
RMSEIF_truth_mean <- colMeans(RMSEIF_truth); UnfairnessIF_truth_mean <- colMeans(UnfairnessIF_truth)

df = data.frame(Unfairness = UnfairnessIF_mean, RMSE = RMSEIF_mean, type = paste(d, "nodes", s, "edges", sep = ''), source = "Gene")
dfBase = data.frame(Unfairness = Unfairness_mean, RMSE = RMSE_mean, type = paste(d, "nodes", s, "edges", sep = ''), source = "Real")
dfBase$Model = rownames(dfBase)
df_tradeoff=rbind(df_tradeoff, df)
df_tradeoff$Model = 'General'
dfBase_tradeoff=rbind(dfBase_tradeoff, dfBase)


# for 20nodes40edges and for 30nodes60edges
df_tradeoff = df_tradeoff[-4,]  # drop too near lambda

pdf(paste0(filename,"/Tradeoff_", d, "nodes", s, "edges", "_truth.pdf", sep=""), width = 9, height = 6)

ggplot(data = df_tradeoff, mapping = aes(x = Unfairness, y = RMSE, shape = Model, color = type, linetype = source)) +
  geom_line(size=1.7, color=colors[i]) + 
  geom_point(size=9, color=colors[i]) +
  geom_point(data = dfBase_tradeoff, mapping = aes(x = Unfairness, y = RMSE, color = type), size=12, color=colors[i]) +
  scale_shape_manual(values = c("General" = 16, "Full" = 15, "Unaware" = 17, "IFair" = 18),
                     labels = c("General" = TeX("$\\epsilon$-IFair"))) +
  # scale_x_break(c(0.1085, 0.25)) + scale_x_break(c(0.065, 0.104)) + scale_x_break(c(0.038, 0.059)) +
  theme(
        legend.position = "top",
        legend.title = element_blank(),
        legend.text=element_text(size = 30),
        axis.title=element_text(size = 36),
        axis.text=element_text(size = 32))

dev.off()


# ****************************************
# gene-real plot
# ****************************************

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
# one of the four settings that you want to get
i =3
filename = paste0("./Repository_adm=", adm)
k=c(0:9)
colors <- c("#f9ca24", "#6ab04c", "#7ed6df", "#ea8685")

df_tradeoff = as.data.frame(matrix(nrow=0, ncol = 4))
dfBase_tradeoff = as.data.frame(matrix(nrow=0, ncol = 4))

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

cutoff_gene = intersect(which(RMSEIF$X20IF<5), which(UnfairnessIF$X0IF<5))
cutoff_truth = intersect(which(RMSEIF_truth$X20IF<5), which(UnfairnessIF_truth$X0IF<5))

cutoff = intersect(cutoff_gene, cutoff_truth)

RMSE = RMSE[cutoff,]; Unfairness = Unfairness[cutoff,]
RMSE_mean <- colMeans(RMSE); Unfairness_mean <- colMeans(Unfairness)

RMSEIF = RMSEIF[cutoff,]; UnfairnessIF = UnfairnessIF[cutoff,]
RMSEIF_mean <- colMeans(RMSEIF); UnfairnessIF_mean <- colMeans(UnfairnessIF)

RMSEIF_truth = RMSEIF_truth[cutoff,]; UnfairnessIF_truth = UnfairnessIF_truth[cutoff,]
RMSEIF_truth_mean <- colMeans(RMSEIF_truth); UnfairnessIF_truth_mean <- colMeans(UnfairnessIF_truth)

df = data.frame(Unfairness = UnfairnessIF_mean, RMSE = RMSEIF_mean, type = paste(d, "nodes", s, "edges", sep = ''), source = "Gene")
dftruth = data.frame(Unfairness = UnfairnessIF_truth_mean, RMSE = RMSEIF_truth_mean, type = paste(d, "nodes", s, "edges", sep = ''), source = "Real")
dfBase = data.frame(Unfairness = Unfairness_mean, RMSE = RMSE_mean, type = paste(d, "nodes", s, "edges", sep = ''), source = "Real")
rownames(dfBase)[ rownames(dfBase) == "Fair" ] = 'IFair'
dfBase$Model = rownames(dfBase)
df_tradeoff=rbind(df_tradeoff, df, dftruth)
df_tradeoff$Model = 'General'
dfBase_tradeoff=rbind(dfBase_tradeoff, dfBase)

pdf(paste0(filename,"/Tradeoff_", d, "nodes", s, "edges", "_truth.pdf", sep=""), width = 12.5, height = 8.9)
ggplot(data = df_tradeoff, mapping = aes(x = Unfairness, y = RMSE, shape = Model, color = type, linetype = source)) +
  scale_linetype(guide = guide_legend(direction="horizontal")) + 
  geom_path(size=2.1, color=colors[i]) + 
  geom_point(size=11, color=colors[i]) +
  geom_point(data = dfBase_tradeoff, mapping = aes(x = Unfairness, y = RMSE, shape=Model, color = type), size=12, color=colors[i]) +
  scale_shape_manual(values = c("General" = 16, "Full" = 15, "Unaware" = 17, "IFair" = 18),
                     labels = c("General" = TeX("$\\epsilon$-IFair")), 
                     guide = guide_legend(direction = "horizontal")
  ) +
  theme(
    legend.position = c(0.42, 1.09),
    legend.spacing.x = unit(0.08, 'cm'),
    legend.box = "horizontal",
    legend.key.width = unit(1.5, "cm"),
    legend.title = element_blank(),
    legend.text=element_text(size = 40),
    axis.title=element_text(size = 48),
    axis.text=element_text(size = 40),
    plot.margin = margin(2.5, 0.3, 0.4, 0.4, "cm"),
    plot.background = element_rect(
      fill = "white",
      colour = "white",
      linewidth = 0
    )
  )

dev.off()

