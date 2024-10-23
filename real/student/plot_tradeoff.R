##### Trade-off plot #####
rm(list=ls())
setwd('PATH_TO_DATA')
library(ggplot2)
library(ggbreak)
library(grDevices)
library(latex2exp)

filename = paste0("./student/results/save/results_0")
k=c(0:19) # on train set
# k=c(20:39) # on test set
my_color = "#F4A582"
RMSE <- read.csv(paste0(filename, "/", "/RMSE.csv", sep=""))[k+1,]
Unfairness <- read.csv(paste0(filename, "/", "/Unfairness.csv", sep=""))[k+1,]
RMSEIF <- read.csv(paste0(filename, "/", "/RMSEIF.csv", sep=""))[k+1,]
UnfairnessIF <- read.csv(paste0(filename, "/", "/UnfairnessIF.csv", sep=""))[k+1,]

RMSE_mean <- colMeans(RMSE); Unfairness_mean <- colMeans(Unfairness)
RMSEIF_mean <- colMeans(RMSEIF); UnfairnessIF_mean <- colMeans(UnfairnessIF)

df = data.frame(Unfairness = UnfairnessIF_mean, RMSE = RMSEIF_mean)
df$Model = 'General'
dfBase = data.frame(Unfairness = Unfairness_mean, RMSE = RMSE_mean)
dfBase$Model = rownames(dfBase)

pdf(paste0(filename, "/Student_tradeoff_test_0516.pdf"), width = 9, height = 6)

# ggplot(data = df, mapping = aes(x = Unfairness, y = RMSE, shape = Model)) + 
#   geom_line(size=0.7) + 
#   geom_point(size=3) +
#   geom_point(data = dfBase, mapping = aes(x = Unfairness, y = RMSE), size=3) +
#   scale_shape_manual(values = c("General" = 16, "Full" = 15, "Unaware" = 17, "IFair" = 18),
#                      labels = c("General" = TeX("$\\epsilon$-IFair"))) +
#   # scale_x_break(c(0.025, 0.065)) +
#   theme(legend.position = c(0.9, 0.9),
#         legend.title = element_blank(),
#         legend.text=element_text(size = 12),
#         axis.title=element_text(size = 20)
#         # axis.text=element_text(size = 22)
#         )
# 
# ggplot(data = df, mapping = aes(x = RMSE, y = Unfairness, shape = Model)) +
#   geom_line(size=0.7) +
#   geom_point(size=3) +
#   geom_point(data = dfBase, mapping = aes(x = RMSE, y = Unfairness), size=3) +
#   scale_shape_manual(values = c("General" = 16, "Full" = 15, "Unaware" = 17, "Fair" = 18),
#                      labels = c("General" = TeX("$\\epsilon$-IFair"))) +
#   # scale_x_break(c(3.5, 3.65)) +
#   theme(legend.position = c(0.9, 0.9),
#         legend.title = element_blank(),
#         legend.text=element_text(size = 12),
#         axis.title=element_text(size = 20))


ggplot(data = df, mapping = aes(x = Unfairness, y = RMSE, shape = Model)) +
  geom_line(size=1.7, color=my_color) +
  geom_point(size=9, color=my_color) +
  geom_point(data = dfBase, mapping = aes(x = Unfairness, y = RMSE), size=12, color=my_color) +
  scale_shape_manual(values = c("General" = 16, "Full" = 15, "Unaware" = 17, "IFair" = 18),
                     labels = c("General" = TeX("$\\epsilon$-IFair"))) +
  # scale_x_break(c(3.5, 3.65)) +
  theme(legend.position = "top", #c(0.8, 0.76),
        legend.title = element_blank(),
        legend.text=element_text(size = 30),
        axis.title=element_text(size = 36),
        axis.text=element_text(size = 32))

dev.off()


