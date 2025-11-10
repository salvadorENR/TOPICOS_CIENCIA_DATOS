#################################################################################
## Códigos - Auto                                                            ##
## Asignatura: Tópicos en Ciencia de Datos                                    ##
## Profesor: Ricardo Felipe Ferreira                                          ##
## Año: 2025/2                                                                ##
#################################################################################

############################################################
## 1) Criando a base
############################################################
#install.packages("ISLR2")   # ou ISLR, dependendo da versão do livro
library(ISLR2)

data(Auto)
auto <- na.omit(Auto)

## Ponto de corte: primeiro quartil de mpg
q3_mpg <- quantile(auto$mpg, 0.75)

auto$economico <- factor(ifelse(auto$mpg >= q3_mpg, "sim", "nao"))

prop.table(table(auto$economico))

write.csv(auto,
          file = "auto_economico_q1.csv",
          row.names = FALSE)

dados <- auto

############################################################
## 2) Divisão em treino e teste (estratificada)
############################################################
dados <- Dados_Auto

# install.packages("rsample")   # se ainda não tiver
library(rsample)

set.seed(2026)  # para reprodutibilidade

# 70% treino, 30% teste, estratificando pela variável de classe y
split <- initial_split(
  dados,
  prop   = 0.7,   # proporção no treino
  strata = "economico"    # estratificação pela classe
)

treino <- training(split)
teste  <- testing(split)

# Conferir tamanhos e proporções em cada conjunto
nrow(treino); nrow(teste)
prop.table(table(treino$economico))
prop.table(table(teste$economico))

############################################################
## 3) Ajustar SVM linear com C grande (≈ hard margin)
############################################################

# install.packages("e1071")  # se ainda não tiver
library(e1071)

svm_hard <- svm(
  as.factor(economico) ~ horsepower + weight + displacement,
  data   = treino,
  kernel = "linear",          # sem kernel não linear
  type   = "C-classification",
  cost   = 1e6,               # C bem grande ≈ margem dura
  scale  = FALSE              # NÃO padroniza x1 e x2 (mantém escala original)
)

w <- t(svm_hard$coefs) %*% svm_hard$SV   # vetor (1 x 2): (w1, w2)
b <- -svm_hard$rho                       # intercepto

w
b
#boxplot(dados$mpg~as.factor(dados$economico))

############################################################
## 4) Avaliar desempenho no conjunto de teste (SVM hard)
############################################################

# Predições no teste com o modelo de margem "dura"
pred_hard <- predict(svm_hard, newdata = teste)

# Matriz de confusão (positivo = cliente premium, y = 1)
tab_hard <- table(
  Verdadeiro = teste$economico,
  Predito    = pred_hard
)
tab_hard

TP <- tab_hard["sim", "sim"]  # verdadeiros positivos
TN <- tab_hard["nao", "nao"]  # verdadeiros negativos
FP <- tab_hard["nao", "sim"]  # falsos positivos
FN <- tab_hard["sim", "nao"]  # falsos negativos


# Métricas para o SVM hard margin
acc_hard   <- (TP + TN) / (TP + TN + FP + FN)       # acurácia
sens_hard  <- TP / (TP + FN)                        # sensibilidade (recall, TPR)
spec_hard  <- TN / (TN + FP)                        # especificidade (TNR)
ppv_hard   <- TP / (TP + FP)                        # valor preditivo positivo (precisão)
npv_hard   <- TN / (TN + FN)                        # valor preditivo negativo
gmean_hard <- sqrt(sens_hard * spec_hard)           # g-média
f1_hard    <- 2 * ppv_hard * sens_hard / (ppv_hard + sens_hard)  # F1-score

# MCC
mcc_num <- (TP * TN - FP * FN)
mcc_den <- sqrt( (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) )
mcc_hard <- mcc_num / mcc_den

cat("Desempenho no conjunto de teste – SVM linear (C grande, margem 'dura')\n")
cat(sprintf("Acurácia          : %.4f\n", acc_hard))
cat(sprintf("Sensibilidade     : %.4f\n", sens_hard))
cat(sprintf("Especificidade    : %.4f\n", spec_hard))
cat(sprintf("VPP (precisão)    : %.4f\n", ppv_hard))
cat(sprintf("VPN               : %.4f\n", npv_hard))
cat(sprintf("G-média           : %.4f\n", gmean_hard))
cat(sprintf("F1-score          : %.4f\n", f1_hard))
cat(sprintf("MCC               : %.4f\n", mcc_hard))

############################################################
## 5) Ajustar SVM linear com C moderado (≈ soft margin)
############################################################

svm_soft <- svm(
  as.factor(economico) ~ horsepower + weight + displacement,
  data   = treino,
  kernel = "linear",
  type   = "C-classification",
  cost   = 1,        # C moderado → permite folga (margem "suave")
  scale  = FALSE
)

w <- t(svm_soft$coefs) %*% svm_soft$SV   # vetor (1 x 2): (w1, w2)
b <- -svm_soft$rho                       # intercepto

w
b

############################################################
## 6) Avaliar desempenho no conjunto de teste (SVM soft)
############################################################

# Predições no teste com o modelo de margem "dura"
pred_soft <- predict(svm_soft, newdata = teste)

# Matriz de confusão (positivo = cliente premium, y = 1)
tab_soft <- table(
  Verdadeiro = teste$economico,
  Predito    = pred_hard
)
tab_soft

TP <- tab_soft["sim", "sim"]  # verdadeiros positivos
TN <- tab_soft["nao", "nao"]  # verdadeiros negativos
FP <- tab_soft["nao", "sim"]  # falsos positivos
FN <- tab_soft["sim", "nao"]  # falsos negativos


# Métricas para o SVM hard margin
acc_soft   <- (TP + TN) / (TP + TN + FP + FN)       # acurácia
sens_soft  <- TP / (TP + FN)                        # sensibilidade (recall, TPR)
spec_soft  <- TN / (TN + FP)                        # especificidade (TNR)
ppv_soft   <- TP / (TP + FP)                        # valor preditivo positivo (precisão)
npv_soft   <- TN / (TN + FN)                        # valor preditivo negativo
gmean_soft <- sqrt(sens_soft * spec_soft)           # g-média
f1_soft    <- 2 * ppv_soft * sens_soft / (ppv_soft + sens_soft)  # F1-score

# MCC
mcc_num <- (TP * TN - FP * FN)
mcc_den <- sqrt( (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) )
mcc_soft <- mcc_num / mcc_den

cat("Desempenho no conjunto de teste – SVM linear (C pequeno, margem 'suave')\n")
cat(sprintf("Acurácia          : %.4f\n", acc_soft))
cat(sprintf("Sensibilidade     : %.4f\n", sens_soft))
cat(sprintf("Especificidade    : %.4f\n", spec_soft))
cat(sprintf("VPP (precisão)    : %.4f\n", ppv_soft))
cat(sprintf("VPN               : %.4f\n", npv_soft))
cat(sprintf("G-média           : %.4f\n", gmean_soft))
cat(sprintf("F1-score          : %.4f\n", f1_soft))
cat(sprintf("MCC               : %.4f\n", mcc_soft))

#################################################################
## 7) Ajustar SVM linear tunado
################################################################

library(e1071)

## grades candidatas de C (ajuste como quiser)
grid_C <- 10^seq(-3, 3, length.out = 5)
grid_C

set.seed(123)

tune_lin <- tune.svm(
  as.factor(economico) ~ horsepower + weight + displacement,
  data   = treino,
  kernel = "linear",
  type   = "C-classification",
  scale  = FALSE,
  cost   = grid_C,
  tunecontrol = tune.control(cross = 3)  # 10-fold CV
)

summary(tune_lin)          # mostra o C ótimo e os erros médios
best_C     <- tune_lin$best.parameters$cost
best_model <- tune_lin$best.model
best_C

#################################################################
## 8) Avaliar desempenho no conjunto de teste (SVM linear tunado)
################################################################

pred_cv <- predict(best_model, newdata = teste)

tab_cv <- table(
  Verdadeiro = teste$economico,
  Predito    = pred_cv
)
tab_cv

TP <- tab_cv["sim", "sim"]
TN <- tab_cv["nao", "nao"]
FP <- tab_cv["nao", "sim"]
FN <- tab_cv["sim", "nao"]

# Métricas para o SVM linear com C escolhido por CV
acc_cv   <- (TP + TN) / (TP + TN + FP + FN)       # acurácia
sens_cv  <- TP / (TP + FN)                        # sensibilidade (recall, TPR)
spec_cv  <- TN / (TN + FP)                        # especificidade (TNR)
ppv_cv   <- TP / (TP + FP)                        # valor preditivo positivo (precisão)
npv_cv   <- TN / (TN + FN)                        # valor preditivo negativo
gmean_cv <- sqrt(sens_cv * spec_cv)               # g-média
f1_cv    <- 2 * ppv_cv * sens_cv / (ppv_cv + sens_cv)  # F1-score

# MCC
mcc_num <- (TP * TN - FP * FN)
mcc_den <- sqrt( (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) )
mcc_cv  <- mcc_num / mcc_den

cat("Desempenho no conjunto de teste – SVM linear (C escolhido por validação cruzada)\n")
cat(sprintf("Acurácia          : %.4f\n", acc_cv))
cat(sprintf("Sensibilidade     : %.4f\n", sens_cv))
cat(sprintf("Especificidade    : %.4f\n", spec_cv))
cat(sprintf("VPP (precisão)    : %.4f\n", ppv_cv))
cat(sprintf("VPN               : %.4f\n", npv_cv))
cat(sprintf("G-média           : %.4f\n", gmean_cv))
cat(sprintf("F1-score          : %.4f\n", f1_cv))
cat(sprintf("MCC               : %.4f\n", mcc_cv))

#################################################################
## 9) Ajustar SVM polinomial tunado
################################################################

library(e1071)

## grades candidatas de C, gamma e degree
grid_C      <- 10^seq(-3, 3, length.out = 5)
#grid_gamma  <- 10^seq(-3, 1, length.out = 4)
#grid_degree <- c(2, 3)

set.seed(123)

tune_poly <- tune.svm(
  as.factor(economico) ~ horsepower + weight + displacement,
  data   = treino,
  kernel = "polynomial",
  type   = "C-classification",
  scale  = FALSE,
  cost   = grid_C,
  #gamma  = grid_gamma,
  #degree = grid_degree,
  tunecontrol = tune.control(cross = 3)  # 3-fold CV
)

summary(tune_poly)                 # mostra C, gamma e degree ótimos
best_params_poly <- tune_poly$best.parameters
best_model_poly  <- tune_poly$best.model
best_params_poly

#################################################################
## 10) Avaliar desempenho no conjunto de teste (SVM polinomial tunado)
################################################################

pred_poly <- predict(best_model_poly, newdata = teste)

tab_poly <- table(
  Verdadeiro = teste$economico,
  Predito    = pred_poly
)
tab_poly

TP_poly <- tab_poly["sim", "sim"]
TN_poly <- tab_poly["nao", "nao"]
FP_poly <- tab_poly["nao", "sim"]
FN_poly <- tab_poly["sim", "nao"]

# Métricas para o SVM polinomial com hiperparâmetros escolhidos por CV
acc_poly   <- (TP_poly + TN_poly) / (TP_poly + TN_poly + FP_poly + FN_poly)
sens_poly  <- TP_poly / (TP_poly + FN_poly)
spec_poly  <- TN_poly / (TN_poly + FP_poly)
ppv_poly   <- TP_poly / (TP_poly + FP_poly)
npv_poly   <- TN_poly / (TN_poly + FN_poly)
gmean_poly <- sqrt(sens_poly * spec_poly)
f1_poly    <- 2 * ppv_poly * sens_poly / (ppv_poly + sens_poly)

mcc_num_poly <- (TP_poly * TN_poly - FP_poly * FN_poly)
mcc_den_poly <- sqrt( (TP_poly + FP_poly) * (TP_poly + FN_poly) *
                        (TN_poly + FP_poly) * (TN_poly + FN_poly) )
mcc_poly  <- mcc_num_poly / mcc_den_poly

cat("Desempenho no conjunto de teste – SVM polinomial (hiperparâmetros via validação cruzada)\n")
cat(sprintf("Acurácia          : %.4f\n", acc_poly))
cat(sprintf("Sensibilidade     : %.4f\n", sens_poly))
cat(sprintf("Especificidade    : %.4f\n", spec_poly))
cat(sprintf("VPP (precisão)    : %.4f\n", ppv_poly))
cat(sprintf("VPN               : %.4f\n", npv_poly))
cat(sprintf("G-média           : %.4f\n", gmean_poly))
cat(sprintf("F1-score          : %.4f\n", f1_poly))
cat(sprintf("MCC               : %.4f\n", mcc_poly))
