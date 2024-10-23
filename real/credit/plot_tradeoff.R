################ Credit dataset ###################
rm(list=ls())
setwd('PATH_TO_DATA')
library(ggplot2)
library(ggbreak)
library(latex2exp)

filename = paste0("./credit/results")
k=c(0:9) # on test set
my_color = "#F4A582"

Accuracy <- read.csv(paste0(filename, "/", "Accuracy.csv", sep=""))[k+1,]
AccuracyIF <- read.csv(paste0(filename, "/", "AccuracyIF.csv", sep=""))[k+1,]
Unfairness <- read.csv(paste0(filename, "/", "Unfairness.csv", sep=""))[k+1,]
UnfairnessIF <- read.csv(paste0(filename, "/", "UnfairnessIF.csv", sep=""))[k+1,]

Accuracy <- 1-Accuracy
AccuracyIF <- 1-AccuracyIF

# Accuracy <- Accuracy[-c(3,5,7,8,9),]
# AccuracyIF <- AccuracyIF[,-c(3,5,7,8,9)]
# Unfairness <- Unfairness[-c(3,5,7,8,9),]
# UnfairnessIF <- UnfairnessIF[,-c(3,5,7,8,9)]

Unfairness_mean <- colMeans(Unfairness); Accuracy_mean <- colMeans(Accuracy)
UnfairnessIF_mean <- colMeans(UnfairnessIF); AccuracyIF_mean <- colMeans(AccuracyIF)

df = data.frame(Unfairness = UnfairnessIF_mean, Accuracy = AccuracyIF_mean)
df$Model = 'General'
dfBase = data.frame(Unfairness = Unfairness_mean, Accuracy = Accuracy_mean)
dfBase$Model = rownames(dfBase)

df <- df[-c(3,5,7,8,9),]


# pdf(paste0(filename, "/Credit_tradeoff.pdf"), width = 9, height = 6)
plot_credit <- ggplot(df, aes(x = Unfairness, y = Accuracy, shape = Model)) + 
  geom_line(size=1.7, color=my_color) + 
  geom_point(size=9, color=my_color) +
  geom_point(data = dfBase, aes(x = Unfairness, y = Accuracy), size=12, color=my_color) +
  scale_shape_manual(values = c("General" = 16, "Full" = 15, "Unaware" = 17),
                     labels = c("General" = TeX("$\\epsilon$-IFair"))) +
  theme(legend.position = "top", #c(0.15,0.83),
        legend.title = element_blank(),
        legend.text=element_text(size = 30),
        axis.title=element_text(size = 36),
        axis.text=element_text(size = 32))+
  ylab("1-Accuracy")
plot_credit
# dev.off()


#***************************************************
#* AUC

# 导入相关包
library(pROC)
library(ROCR)

## 绘制ROC曲线 & 计算AUC值
plot.roc(trainData$churn, trainYhat, print.auc=TRUE, col=4, lty=1, xlab = "特异度",ylab = "敏感度",main="训练集和测试集的ROC曲线")
plot.roc(testData$churn, testYhat, print.auc=TRUE, add=TRUE, col=8, lty=1, print.auc.y=0.3)
legend("topleft", c("训练集", "测试集"), lty=c(1,1), col=c(4,8))

# 计算AUC值
auc(trainData$churn, trainYhat) 
auc(testData$churn, testYhat) 

