###### Density plot ######
rm(list=ls())
setwd('PATH_TO_DATA')
library(ggplot2)
library(ggpubr)
library(latex2exp)

# number of nodes
dd=5
# number of edges
ss = 8
# number of admissible variables
adm=0
filename = paste0("./Repository_adm=", adm, "/", dd, "nodes", ss, "edges", "/y_pred", sep="")


Full_y0 <- read.csv(paste0(filename, "/Full_y0.csv", sep=""), header = FALSE)
Full_y1 <- read.csv(paste0(filename, "/Full_y1.csv", sep=""), header = FALSE)
sample_size = dim(Full_y0)[2]
Unaware_y0 <- read.csv(paste0(filename, "/Unaware_y0.csv", sep=""), header = FALSE)
Unaware_y1 <- read.csv(paste0(filename, "/Unaware_y1.csv", sep=""), header = FALSE)
Fair_y0 <- read.csv(paste0(filename, "/Fair_y0.csv", sep=""), header = FALSE)
Fair_y1 <- read.csv(paste0(filename, "/Fair_y1.csv", sep=""), header = FALSE)
Fair5_y0 <- read.csv(paste0(filename, "/e-IFair_5_y0_real.csv", sep=""), header = FALSE)
Fair5_y1 <- read.csv(paste0(filename, "/e-IFair_5_y1_real.csv", sep=""), header = FALSE)
Fair60_y0 <- read.csv(paste0(filename, "/e-IFair_60_y0_gene.csv", sep=""), header = FALSE)
Fair60_y1 <- read.csv(paste0(filename, "/e-IFair_60_y1_gene.csv", sep=""), header = FALSE)
Fair60_y0 <- read.csv(paste0(filename, "/e-IFair_100_y0_real.csv", sep=""), header = FALSE)
Fair60_y1 <- read.csv(paste0(filename, "/e-IFair_100_y1_real.csv", sep=""), header = FALSE)

k=5
pdf(paste0("./Repository_adm=", adm,"/Density_", dd, "nodes", ss, "edges", k, 'graph',  ".pdf", sep=""), width = 16, height = 3.5)
df_Full <- data.frame(fullPred = c(t(Full_y0[k+1,]), t(Full_y1[k+1,])))
df_Unaware <- data.frame(unPred = c(t(Unaware_y0[k+1,]), t(Unaware_y1[k+1,])))
df_Fair <- data.frame(fairPred = c(t(Fair_y0[k+1,]), t(Fair_y1[k+1,])), group = rep(c("y0", "y1"), each=sample_size))
df_Fair5 <- data.frame(fair5Pred = c(t(Fair5_y0[k+1,]), t(Fair5_y1[k+1,])))
df_Fair60 <- data.frame(fair60Pred = c(t(Fair60_y0[k+1,]), t(Fair60_y1[k+1,])))
df <- cbind(df_Full, df_Unaware, df_Fair, df_Fair5, df_Fair60)


legend_size = 34
title_size = 22
axis_size = 16

# Create the plot
fullplot <- ggplot(df, aes(x = fullPred, fill = group)) +
  geom_density(alpha = 0.5) +
  labs(x=expression(~hat(Y)~(Full)), y='density', color="") +
  scale_fill_manual(values = c("#F4A582","#92C5DE"), labels = c( "y0" = expression(~hat(Y)[A %<-% a]), "y1" = expression(~hat(Y)[A %<-% a^bold("'")]) ) ) +
  theme(legend.position = "none",
        legend.text=element_text(size=legend_size),
        legend.title = element_blank(),
        axis.title=element_text(size = title_size)
  )
unplot <- ggplot(df, aes(x = unPred, fill = group)) +
  geom_density(alpha = 0.5) +
  labs(x=expression(~hat(Y)~(Unaware)), y='density', color="") +
  scale_fill_manual(values = c("#F4A582","#92C5DE")) +
  theme(legend.position = "none",
        legend.text=element_text(size=legend_size),
        legend.title = element_blank(),
        axis.title=element_text(size = title_size))
fairplot <- ggplot(df, aes(x = fairPred, fill = group)) +
  geom_density(alpha = 0.5) +
  labs(x=expression(~hat(Y)~(IFair)), y='density', color="") +
  scale_fill_manual(values = c("#F4A582","#92C5DE")) +
  theme(legend.position = "none",
        legend.text=element_text(size=legend_size),
        legend.title = element_blank(),
        axis.title=element_text(size = title_size)
  )
fair5plot <- ggplot(df, aes(x = fair5Pred, fill = group)) +
  geom_density(alpha = 0.5) +
  labs(x=expression(~hat(Y)~(epsilon-IFair~","~lambda == 5)), y='density', color="") +
  scale_fill_manual(values = c("#F4A582","#92C5DE")) +
  theme(legend.position = "none",
        legend.text=element_text(size=legend_size),
        legend.title = element_blank(),
        axis.title=element_text(size = title_size))
fair60plot <- ggplot(df, aes(x = fair60Pred, fill = group)) +
  geom_density(alpha = 0.5) +
  labs(x=expression(~hat(Y)~(epsilon-IFair~","~lambda == 60)), y='density', color="") +
  scale_fill_manual(values = c("#F4A582","#92C5DE")) +
  theme(legend.position = "none",
        legend.text=element_text(size=legend_size),
        legend.title = element_blank(),
        axis.title=element_text(size = title_size))
ggarrange(fullplot, unplot, fairplot, fair5plot, fair60plot, nrow=1, ncol=5, common.legend = TRUE)
dev.off()




