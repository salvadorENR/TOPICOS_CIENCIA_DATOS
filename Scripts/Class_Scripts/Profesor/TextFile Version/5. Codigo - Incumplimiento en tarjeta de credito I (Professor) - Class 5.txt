#################################################################################
## Códigos - Incumplimiento en tarjeta de crédito                             ##
## Asignatura: Topicos en Ciencias de Datos                                   ##
## Profesor: Ricardo Felipe Ferreira                                          ##
## Año: 2025/2                                                                ##
#################################################################################

############################################################
## 0) Pacotes
############################################################

# install.packages("mRMRe")
# install.packages("infotheo")
# install.packages("e1071")
# install.packages("dplyr")
# install.packages("rsample")

library(mRMRe)
library(infotheo)
library(e1071)
library(dplyr)
library(rsample)

#################################################################################
## Base para seleção via informação mútua (mRMR)
## 10 preditoras (todas com nome de negócio) e 1 resposta (inadimplente)
## V1, V2, V3, V4 (abaixo com nomes reais) FORTEMENTE relacionados com Y
## V2 e V4: relação NÃO LINEAR (quadrática em torno do meio)
## Demais: ruído, independentes de Y
#################################################################################

set.seed(2026)

n <- 800

############################################################
## 1) Preditoras realmente relevantes
############################################################
# 1) freq_atraso: frequência de atrasos recentes (1=quase nunca ... 4=muito frequente)
freq_atraso <- sample(1:4, n, replace = TRUE,
                      prob = c(0.15, 0.25, 0.35, 0.25))

# 2) indice_endividamento: grau de comprometimento da renda (1=muito baixo ... 5=muito alto)
#    vamos usar depois para gerar uma variável não linear
indice_endividamento <- sample(1:5, n, replace = TRUE,
                               prob = c(0.12, 0.18, 0.40, 0.18, 0.12))

# 3) qtd_parcelas_aberto: quantas dívidas/parc. o cliente mantém ao mesmo tempo (1=poucas ... 5=muitas)
qtd_parcelas_aberto <- sample(1:5, n, replace = TRUE,
                              prob = c(0.10, 0.20, 0.30, 0.25, 0.15))

# 4) perfil_pagamento: indicador comportamental derivado do endividamento (não linear)
#    ideia: cliente com endividamento "médio" (3) costuma girar mais crédito -> risco maior
#    endividamento 2 ou 4 -> risco alto, mas menor que o pico
#    muito baixo (1) ou muito alto (5) -> risco mais baixo neste indicador
#perfil_pagamento <- ifelse(indice_endividamento == 3, 5,
#                          ifelse(indice_endividamento %in% c(2,4), 4, 2))
perfil_pagamento <- freq_atraso^2
# isso cria exatamente o "U invertido" em torno do meio

############################################################
## 2) Preditoras irrelevantes (ruído, mas com cara de banco)
############################################################
# 5) canal_contratacao: 1=app, 2=agência, 3=corresp., 4=site, 5=telefone
canal_contratacao <- sample(1:5, n, replace = TRUE)

# 6) qtd_produtos: qtde de produtos no banco (0=conta, 1=+cartão, ..., 6=muitos)
qtd_produtos <- sample(0:6, n, replace = TRUE)

# 7) regiao_cliente: 1=sul/sudeste, 2=centro, 3=norte/nordeste, 4=interior
regiao_cliente <- sample(1:4, n, replace = TRUE)

# 8) tipo_conta: 1=salário, 2=individual, 3=conjunta, 4=MEI, 5=PF-premium, 6=universitário, 7=outros
tipo_conta <- sample(1:7, n, replace = TRUE)

# 9) uso_app: 1=baixo, 2=médio, 3=alto
uso_app <- sample(1:3, n, replace = TRUE)

# 10) campanha: 1..4
campanha <- sample(1:4, n, replace = TRUE)

############################################################
## 3) Variável resposta: inadimplente (0/1)
##    Depende FORTEMENTE das 4 primeiras
##    NÃO depende das demais
############################################################
eta <- -7 +
  1.3 * (freq_atraso >= 3) +          # cliente que atrasa muito
  4.3 * (indice_endividamento >= 3) + # muito comprometido
  1.3 * (qtd_parcelas_aberto >= 3) +  # muitas parcelas
  1.3 * (perfil_pagamento >= 16)       # perfil de risco (não linear)

prob_Y <- 1 / (1 + exp(-eta))

inadimplente <- rbinom(n, size = 1, prob = prob_Y)

############################################################
## 4) Montar base final
############################################################
dados <- data.frame(
  inadimplente,
  freq_atraso,             # era X1
  indice_endividamento,    # era X2
  qtd_parcelas_aberto,     # era X3
  perfil_pagamento,        # era X4 (não linear)
  canal_contratacao,       # era X5
  qtd_produtos,            # era X6
  regiao_cliente,          # era X7
  tipo_conta,              # era X8
  uso_app,                 # era X9
  campanha                 # era X10
)

str(dados)
summary(dados)

#write.csv(dados, "dados_mrmr.csv", row.names = FALSE)

dados <- Dados_Incumplimiento_en_tarjeta_de_cre_dito

# 1) dizer que TUDO, menos a resposta, é fator
cols_cat <- names(dados)[names(dados) != "inadimplente"]
dados[cols_cat] <- lapply(dados[cols_cat], factor)

############################################################
## 2) Treino e teste com rsample (estratificado)
############################################################
set.seed(2026)
split_obj <- initial_split(dados,
                           prop  = 0.70,             # 70% p/ treino
                           strata = "inadimplente")  # estratifica na resposta

treino <- training(split_obj)
teste  <- testing(split_obj)

# conferir proporções
prop.table(table(dados$inadimplente))
prop.table(table(treino$inadimplente))
prop.table(table(teste$inadimplente))


############################################################
## 3) Aplicar mRMR
############################################################

############################################################
## (A) Garantir que as colunas são do tipo aceito pelo mRMRe
############################################################

# treino é o teu data.frame com variáveis discretas
str(treino)  # aqui você vai ver que várias estão como "int"

# converter tudo para numeric (mRMRe aceita)
treino_num <- as.data.frame(lapply(treino, function(x) as.numeric(x)))

str(treino_num)  # agora deve aparecer "num" para todo mundo

############################################################
## (B) Rodar mRMR de novo
############################################################

# criar objeto especial
dados_mrmr <- mRMR.data(data = treino_num)

# selecionar, por exemplo, 4 variáveis (sabemos que são 4 as relevantes)
res_mrmr <- mRMR.classic(
  data = dados_mrmr,
  target_indices = 1,   # 1a coluna = inadimplente
  feature_count = 3
)

res_mrmr@filters
vars_selecionadas <- colnames(treino_num)[res_mrmr@filters[[1]] ]
vars_selecionadas

############################################################
## 4) Modelos: todas as variáveis vs mRMR
##    (usando REGRESSÃO LOGÍSTICA)
############################################################

## 4.1) Modelo com TODAS as variáveis
modelo_todas <- glm(inadimplente ~ .,
                    data = treino,
                    family = binomial)

summary(modelo_todas)

# probabilidade prevista para o teste
prob_todas <- predict(modelo_todas,
                      newdata = teste,
                      type = "response")

# classe prevista (limiar 0.5, pode mudar se quiser)
pred_todas <- ifelse(prob_todas >= 0.28, 1, 0)


## 4.2) Modelo APENAS com variáveis selecionadas pelo mRMR
# montar fórmula: inadimplente ~ var1 + var2 + ...
form_mrmr <- as.formula(
  paste(
    "inadimplente ~",
    paste(vars_selecionadas[vars_selecionadas != "inadimplente"],
          collapse = " + ")
  )
)

modelo_mrmr <- glm(form_mrmr,
                   data = treino,
                   family = binomial)

summary(modelo_mrmr)

prob_mrmr <- predict(modelo_mrmr,
                     newdata = teste,
                     type = "response")

pred_mrmr <- ifelse(prob_mrmr >= 0.28, 1, 0)

## 4.3) Métricas para cenário desbalanceado

# função auxiliar: recebe vetores de 0/1
calc_metrics <- function(y_true, y_pred) {
  # matriz 2x2
  tab <- table(Pred = y_pred, Real = y_true)
  
  # garantir que tem todas as entradas
  TP <- ifelse("1" %in% rownames(tab) & "1" %in% colnames(tab), tab["1","1"], 0)
  TN <- ifelse("0" %in% rownames(tab) & "0" %in% colnames(tab), tab["0","0"], 0)
  FP <- ifelse("1" %in% rownames(tab) & "0" %in% colnames(tab), tab["1","0"], 0)
  FN <- ifelse("0" %in% rownames(tab) & "1" %in% colnames(tab), tab["0","1"], 0)
  
  # métricas
  acc <- ifelse((TP + TN + FP + FN) > 0, (TP + TN)/(TP + TN + FP + FN), NA)
  sens <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)  # recall
  esp  <- ifelse((TN + FP) > 0, TN / (TN + FP), NA)
  ppv  <- ifelse((TP + FP) > 0, TP / (TP + FP), NA)  # precision
  npv  <- ifelse((TN + FN) > 0, TN / (TN + FN), NA)
  gmean <- ifelse(!is.na(sens) & !is.na(esp), sqrt(sens * esp), NA)
  f1    <- ifelse((ppv + sens) > 0, 2 * ppv * sens / (ppv + sens), NA)
  
  # MCC
  denom <- sqrt( (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) )
  mcc <- ifelse(denom > 0, (TP*TN - FP*FN) / denom, NA)
  
  data.frame(
    Acuracia = acc,
    Sensibilidade = sens,
    Especificidade = esp,
    PPV = ppv,
    NPV = npv,
    Gmedia = gmean,
    F1 = f1,
    MCC = mcc
  )
}

met_todas <- calc_metrics(teste$inadimplente, pred_todas)
met_mrmr  <- calc_metrics(teste$inadimplente, pred_mrmr)

cat("\n== Métricas - Modelo com TODAS ==\n")
print(met_todas)

cat("\n== Métricas - Modelo com mRMR ==\n")
print(met_mrmr)

############################################################
## 4) Modelos: todas as variáveis vs mRMR
##    (usando kNN com dummies via model.matrix)
############################################################

library(class)

## 4.1) kNN com TODAS as variáveis

# model.matrix cria automaticamente as dummies para TODOS os fatores
X_train_all <- model.matrix(inadimplente ~ . - 1, data = treino)
X_test_all  <- model.matrix(inadimplente ~ . - 1, data = teste)

y_train <- as.factor(treino$inadimplente)
y_test  <- teste$inadimplente  # numérico 0/1

# escolher k
k_val <- 3

pred_todas_knn_fac <- knn(train = X_train_all,
                          test  = X_test_all,
                          cl    = y_train,
                          k     = k_val)

pred_todas_knn <- as.numeric(as.character(pred_todas_knn_fac))

## 4.2) kNN APENAS com variáveis selecionadas pelo mRMR

# montar fórmula só com as selecionadas
form_mrmr_knn <- as.formula(
  paste("inadimplente ~", paste(vars_selecionadas[vars_selecionadas != "inadimplente"],
                                collapse = " + "))
)

X_train_mrmr <- model.matrix(form_mrmr_knn, data = treino)[ , -1, drop = FALSE]
X_test_mrmr  <- model.matrix(form_mrmr_knn,  data = teste)[  , -1, drop = FALSE]

pred_mrmr_knn_fac <- knn(train = X_train_mrmr,
                         test  = X_test_mrmr,
                         cl    = y_train,
                         k     = k_val)

pred_mrmr_knn <- as.numeric(as.character(pred_mrmr_knn_fac))


## 4.3) Métricas

## 4.0) Função de métricas (a mesma que você já tem)
calc_metrics <- function(y_true, y_pred) {
  tab <- table(Pred = y_pred, Real = y_true)
  
  TP <- ifelse("1" %in% rownames(tab) & "1" %in% colnames(tab), tab["1","1"], 0)
  TN <- ifelse("0" %in% rownames(tab) & "0" %in% colnames(tab), tab["0","0"], 0)
  FP <- ifelse("1" %in% rownames(tab) & "0" %in% colnames(tab), tab["1","0"], 0)
  FN <- ifelse("0" %in% rownames(tab) & "1" %in% colnames(tab), tab["0","1"], 0)
  
  acc  <- ifelse((TP + TN + FP + FN) > 0, (TP + TN)/(TP + TN + FP + FN), NA)
  sens <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)
  esp  <- ifelse((TN + FP) > 0, TN / (TN + FP), NA)
  ppv  <- ifelse((TP + FP) > 0, TP / (TP + FP), NA)
  npv  <- ifelse((TN + FN) > 0, TN / (TN + FN), NA)
  gmean <- ifelse(!is.na(sens) & !is.na(esp), sqrt(sens * esp), NA)
  f1    <- ifelse((ppv + sens) > 0, 2 * ppv * sens / (ppv + sens), NA)
  
  denom <- sqrt( (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) )
  mcc   <- ifelse(denom > 0, (TP*TN - FP*FN) / denom, NA)
  
  data.frame(
    Acuracia = acc,
    Sensibilidade = sens,
    Especificidade = esp,
    PPV = ppv,
    NPV = npv,
    Gmedia = gmean,
    F1 = f1,
    MCC = mcc
  )
}

met_todas_knn <- calc_metrics(y_test, pred_todas_knn)
met_mrmr_knn  <- calc_metrics(y_test, pred_mrmr_knn)


cat("\n== Métricas - kNN com TODAS ==\n")
print(met_todas_knn)

cat("\n== Métricas - kNN com mRMR ==\n")
print(met_mrmr_knn)


############################################################
## 5) Comparar mRMR x forward stepwise (AIC)
############################################################

library(stats)  # step() já está aqui; se quiser stepAIC, usar MASS

## 5.1) Modelos base
# modelo nulo: só intercepto
mod_null <- glm(inadimplente ~ 1,
                data = treino,
                family = binomial)

# modelo cheio: todas as variáveis
mod_full <- glm(inadimplente ~ .,
                data = treino,
                family = binomial)

## 5.2) Forward stepwise
# vai entrando variável a variável, escolhendo a que mais reduz o AIC
modelo_step <- step(mod_null,
                    scope = formula(mod_full),
                    direction = "forward",
                    trace = FALSE)

summary(modelo_step)

## 5.3) Predição no teste
prob_step <- predict(modelo_step,
                     newdata = teste,
                     type = "response")

# pode usar o mesmo limiar que você usou nos outros (0.28)
pred_step <- ifelse(prob_step >= 0.28, 1, 0)

## 5.4) Métricas
met_step <- calc_metrics(teste$inadimplente, pred_step)

cat("\n== Métricas - Modelo com forward stepwise ==\n")
print(met_step)

#### Sumarizar todas aqui
print(met_todas)
print(met_mrmr)
print(met_step)
print(met_todas_knn)
print(met_mrmr_knn)


