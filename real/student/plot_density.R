###### Density plot ######
rm(list=ls())
setwd('PATH_TO_DATA')
library(ggplot2)
library(ggpubr)
library(latex2exp)

filename = paste0("./student/results/save/0509/y_pred")
# filename = paste0("./credit/results/y_pred")

Full_y0 <- read.csv(paste0(filename, "/Full_y0.csv", sep=""), header = FALSE)
Full_y1 <- read.csv(paste0(filename, "/Full_y1.csv", sep=""), header = FALSE)
# To draw the density plot, y0 and y1 should have the same sample size.
sample_size = min(dim(Full_y0)[2], dim(Full_y1)[2])
sample_Full_y0 <- Full_y0[, sample(ncol(Full_y0), sample_size, replace=FALSE)]
sample_Full_y1 <- Full_y1[, sample(ncol(Full_y1), sample_size, replace=FALSE)]

Unaware_y0 <- read.csv(paste0(filename, "/Unaware_y0.csv", sep=""), header = FALSE)
Unaware_y1 <- read.csv(paste0(filename, "/Unaware_y1.csv", sep=""), header = FALSE)
sample_Unaware_y0 <- Unaware_y0[, sample(ncol(Unaware_y0), sample_size, replace=FALSE)]
sample_Unaware_y1 <- Unaware_y1[, sample(ncol(Unaware_y1), sample_size, replace=FALSE)]

Fair_y0 <- read.csv(paste0(filename, "/Fair_y0.csv", sep=""), header = FALSE)
Fair_y1 <- read.csv(paste0(filename, "/Fair_y1.csv", sep=""), header = FALSE)
sample_Fair_y0 <- Fair_y0[, sample(ncol(Fair_y0), sample_size, replace=FALSE)]
sample_Fair_y1 <- Fair_y1[, sample(ncol(Fair_y1), sample_size, replace=FALSE)]

Fair_250_y0 <- read.csv(paste0(filename, "/e-IFair_250_y0.csv", sep=""), header = FALSE)
Fair_250_y1 <- read.csv(paste0(filename, "/e-IFair_250_y1.csv", sep=""), header = FALSE)
sample_Fair250_y0 <- Fair_250_y0[, sample(ncol(Fair_250_y0), sample_size, replace=FALSE)]
sample_Fair250_y1 <- Fair_250_y1[, sample(ncol(Fair_250_y1), sample_size, replace=FALSE)]

k=8
# png(paste0(filename,"/density_", k, 'graph',  ".png", sep=""), width = 800, height = 175)
pdf(paste0(filename,"/Student_density.pdf", sep=""), width = 18, height = 4)
df_Full <- data.frame(fullPred = c(t(sample_Full_y0[k+1,]), t(sample_Full_y1[k+1,])))
df_Unaware <- data.frame(unPred = c(t(sample_Unaware_y0[k+1,]), t(sample_Unaware_y1[k+1,])))
df_Fair <- data.frame(fairPred = c(t(sample_Fair_y0[k+1,]), t(sample_Fair_y1[k+1,])), group = rep(c("y0", "y1"), each=sample_size))
df_Fair250 <- data.frame(fair250Pred = c(t(sample_Fair250_y0[k+1,]), t(sample_Fair250_y1[k+1,])))
df <- cbind(df_Full, df_Unaware, df_Fair, df_Fair250)

legend_size = 34
title_size = 30
axis_size = 18

# Create the plot
fullplot <- ggplot(df, aes(x = fullPred, fill = group)) +
  geom_density(alpha = 0.5) +
  labs(x=expression(~hat(Y)~(Full)), y='density', color="") +
  scale_fill_manual(values = c("#F4A582","#92C5DE"), labels = c( "y0" = expression(~hat(Y)[A %<-% a]), "y1" = expression(~hat(Y)[A %<-% a^bold("'")]) ) ) +
  theme(legend.position = "none",
        legend.text=element_text(size=legend_size),
        legend.title = element_blank(),
        axis.title=element_text(size = title_size)
        # axis.text=element_text(size = axis_size)
        )
unplot <- ggplot(df, aes(x = unPred, fill = group)) +
  geom_density(alpha = 0.5) +
  labs(x=expression(~hat(Y)~(Unaware)), y='density', color="") +
  scale_fill_manual(values = c("#F4A582","#92C5DE")) +
  theme(legend.position = "none",
        legend.text=element_text(size=legend_size),
        legend.title = element_blank(),
        axis.title=element_text(size = title_size)
        # axis.text=element_text(size = axis_size)
        )
fairplot <- ggplot(df, aes(x = fairPred, fill = group)) +
  geom_density(alpha = 0.5) +
  labs(x=expression(~hat(Y)~(IFair)), y='density', color="") +
  scale_fill_manual(values = c("#F4A582","#92C5DE")) +
  theme(legend.position = "none",
        legend.text=element_text(size=legend_size),
        legend.title = element_blank(),
        axis.title=element_text(size = title_size)
        # axis.text=element_text(size = axis_size)
        )
fair250plot <- ggplot(df, aes(x = fair250Pred, fill = group)) +
  geom_density(alpha = 0.5) +
  labs(x=expression(~hat(Y)~(epsilon-IFair~","~lambda == 250)), y='density', color="") +
  scale_fill_manual(values = c("#F4A582","#92C5DE")) +
  theme(legend.position = "none",
        legend.text=element_text(size=legend_size),
        legend.title = element_blank(),
        axis.title=element_text(size = title_size)
        # axis.text=element_text(size = axis_size)
        )
ggarrange(fullplot, unplot, fairplot, fair250plot, nrow=1, ncol=4, common.legend = TRUE)
dev.off()

