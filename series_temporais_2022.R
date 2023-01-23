library(glmnet)
library(caret)
library(Metrics)

#Definição do diretório de trabalho
setwd("C:/Users/Diego/Desktop")
dir()

#Importanto a base de dados
dados<-read.table("ML_series.txt", head=T)
dados = dados[,-1]

#Separando a amostra em treino e teste
indice_treino = createDataPartition(y=dados$V1, p=0.7, list=FALSE)
treino = dados[indice_treino, ]
teste = dados[-indice_treino, ]

#Definindo a variável de resposta e os preditores
y = treino[,1]
x = treino[,-1]

y_teste = teste[,1]
x_teste = teste[,-1]

#GLMNET trabalha com matrizes e não com dataframes

y = as.matrix(y)
x = as.matrix(x)

y_teste = as.matrix(y_teste)
x_teste = as.matrix(x_teste)

#Estimação de um modelo Elastic Net
fit <- glmnet(x, y, alpha = 0.5)

plot(fit)

print(fit)

#Obtendo os coeficientes para um intervalo de lambda

coef(fit, s = 0.1)
coef(fit, s = c(0.1, 0.05))
coef(fit)

#Cross-Validation

cvfit <- cv.glmnet(x, y, alpha = 0.5, nfolds = 10)

plot(cvfit)

#Lambda mínimo

cvfit$lambda.min

coef(cvfit, s = "lambda.min")

#Predição usando o valor de lambda mínimo

prev1 = predict(cvfit, newx = x_teste, s = "lambda.min")

#Acurácia da Previsão (RMSE)

rmse1 = rmse(y_teste,prev1)
rmse1


