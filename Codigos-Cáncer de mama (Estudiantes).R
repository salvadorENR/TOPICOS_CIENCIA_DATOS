################################################################################
## Códigos - Câncer de Mama                                                   ##
## Asignatura: Tópicos en Ciencia de Datos                                    ##
## Profesor: Ricardo Felipe Ferreira                                          ##
## Año: 2025/2                                                                ##
################################################################################

############################################################
## 0) Pacotes
############################################################
install.packages(c("readr", "dplyr", "tidyr", "rsample", "nnet", "tidymodels", "themis", "class"))

library(readr)      # Leitura de arquivos de dados (CSV etc.) de forma rápida e amigável
library(dplyr)      # Manipulação de dados (filter, select, mutate, summarize, %>%)
library(tidyr)      # Organização/reestruturação de dados (pivot_longer, pivot_wider, etc.)
library(rsample)    # Divisão dos dados em treino/teste, validação, reamostragem
library(nnet)       # Redes neurais simples (MLP) e regressão logística multinomial
library(tidymodels) # Framework unificado para modelagem (recipes, parsnip, workflows, yardstick...)
library(themis)     # Passos de pré-processamento para dados desbalanceados (SMOTE, ROSE, etc.) em recipes
library(class)      # Implementação clássica do kNN (classificação e regressão k-vizinhos)

############################################################
## 1) Ler base WDBC da UCI
############################################################

url_wdbc <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

wdbc_cols <- c(
  "id", "diagnosis",
  "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
  "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
  "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
  "compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
  "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
  "compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"
)

wdbc <- read_csv(
  file = url_wdbc,
  col_names = wdbc_cols,
  show_col_types = FALSE
)

# Diagnóstico como fator (B = Benign, M = Malignant)
wdbc <- wdbc %>%
  mutate(
    diagnosis = factor(diagnosis,
                       levels = c("B","M"),
                       labels = c("Benigno","Maligno"))
  ) %>%
  select(-id)  # remover id

glimpse(wdbc)  
# Muestra una vista general del conjunto de datos 'wdbc':
# número de filas, número de columnas, nombres de las variables,
# tipos de datos (numéricos, factores, caracteres, etc.) y algunos valores de ejemplo por variable.

############################################################
## 2) Divisão treino / teste (estratificada)
############################################################

set.seed(2026) # para garantir reprodutibilidad e
split <- initial_split(wdbc, prop = 0.7, strata = "diagnosis")

treino_wdbc <- training(split) # 70% para el entrenamiento
teste_wdbc  <- testing(split)  # 30% para la prueba

# Proporciones absolutas
table(wdbc$diagnosis)
table(treino$diagnosis)
table(teste$diagnosis)

# Verificando proporción de clases
prop.table(table(wdbc$diagnosis))
prop.table(table(treino$diagnosis))
prop.table(table(teste$diagnosis))

# ORIENTAÇÃO: Utilize a função initial_split da biblioteca rsample para fazer a divisão em treino e tese de maneira estratificada.
# Em seguida construa o conjunto de treino e teste com as funções training() e testing(), respectivamente. Por fim, faça as tabelas
# de frequência absolutra e as tabelas de frequência relativa para verificar se estamos ou não em um problema de classes desbalanceadas.

############################################################
## 3) Padronizar preditoras numéricas com recipes
############################################################

# Receita: diagnosis como resposta, demais como preditoras
rec_wdbc <- recipe(diagnosis ~ ., data = treino_wdbc) |>
  step_normalize(all_predictors())   # padroniza todas as preditoras numéricas

# Estimar parâmetros de normalização com base no treino
prep_wdbc <- prep(rec_wdbc)

# Aplicar ao treino (padronizado)
treino_wdbc_norm <- bake(prep_wdbc, new_data = NULL)

# Aplicar ao teste (mesma transformação)
teste_wdbc_norm  <- bake(prep_wdbc, new_data = teste_wdbc)


############################################################
## 4) Ajustar rede neural simples (1 camada oculta)
############################################################

set.seed(2026)  # fixa a semente para tornar os resultados reprodutíveis

modelo_nn_wdbc <- nnet(          # ajusta uma rede neural do tipo MLP
  formula = diagnosis ~ .,       # fórmula: diagnosis como resposta, todas as outras variáveis como preditoras
  data    = treino_wdbc_norm,    # usa o conjunto de treino já pré-processado
  size    = 5,                   # número de neurônios na camada oculta (arquitetura da rede)
  decay   = 0.01,                # parâmetro de regularização L2 (controla o tamanho dos pesos)
  maxit   = 500,                 # número máximo de iterações do algoritmo de otimização
  trace   = TRUE                 # exibe o progresso do ajuste (erro ao longo das iterações)
)

# modelo_nn_wdbc é a MLP simples ajustada para a base WDBC

# PERGUNTA: Você não sentiu falta de um argumento que deveria ser escolhido?
#Respuesta: La función nnet no es flexible en el número de capas y tampoco permite normalizar. 

#Será que el desbalanceamiento está afectando el desempeño de una red neuronal, sensibilidad y especificidad
############################################################
## 5) Avaliar desempenho no conjunto de teste (MLP – WDBC)
############################################################

# Predições no teste com a rede neural
pred_wdbc <- predict(modelo_nn_wdbc, 
  newdata = teste_wdbc_norm, 
    type = "class")
    
    # Matriz de confusão
    tab_wdbc <- table(
      Verdadeiro = teste_wdbc_norm$diagnosis  ,
        Predito    =pred_wdbc
    )
  tab_wdbc
  
  # Vamos considerar "Maligno" como classe positiva
  TP <- tab_wdbc["Maligno", "Maligno"]  # verdadeiros positivos
  TN <- tab_wdbc["Benigno", "Benigno"]  # verdadeiros negativos
  FP <- tab_wdbc["Benigno", "Maligno"]  # falsos positivos
  FN <- tab_wdbc["Maligno", "Benigno"]  # falsos negativos
  
  # Métricas para a MLP na base WDBC
   acc_wdbc   <- (TP + TN)/sum(tab_wdbc) #COMPLETE                 # acurácia
    sens_wdbc  <- TP / (TP + FN) #COMPLETE                                  # sensibilidade (recall, TPR)
    spec_wdbc  <- TN / (TN + FP) #COMPLETE                                  # especificidade (TNR)
    ppv_wdbc   <- TP / (TP + FP)  #COMPLETE                                 # VPP (precisão)
    npv_wdbc   <-  TN / (TN + FN) #COMPLETE                                  # VPN
    gmean_wdbc <- sqrt( sens_wdbc*spec_wdbc) #COMPLETE                     # g-média
    f1_wdbc    <- 2 * (ppv_wdbc * sens_wdbc) / (ppv_wdbc + sens_wdbc) #COMPLETE  # F1-score
    
    # MCC
  mcc_num <- (TP * TN - FP * FN)
  mcc_den <- sqrt( (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) )
  mcc_wdbc <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    
  cat("Desempenho no conjunto de teste – MLP (positivo = Maligno)\n")
  cat(sprintf("Acurácia          : %.4f\n", acc_wdbc))
  cat(sprintf("Sensibilidade     : %.4f\n", sens_wdbc))
  cat(sprintf("Especificidade    : %.4f\n", spec_wdbc))
  cat(sprintf("VPP (precisão)    : %.4f\n", ppv_wdbc))
  cat(sprintf("VPN               : %.4f\n", npv_wdbc))
  cat(sprintf("G-média           : %.4f\n", gmean_wdbc))
  cat(sprintf("F1-score          : %.4f\n", f1_wdbc))
  cat(sprintf("MCC               : %.4f\n", mcc_wdbc))
  
  ############################################################
  ## 6) Receita com SMOTE + padronização
  ############################################################
  rec_wdbc <- recipe(diagnosis ~ ., data = treino_wdbc) |>
    step_normalize(all_predictors()) |>   # padronizar preditoras numéricas
    step_smote(diagnosis)            # balancear classes no treino
  
  # Preparar a receita (aprende médias, desvios etc.)
  prep_wdbc <- prep(rec_wdbc) #COMPLETE
    
    # Aplicar transformação ao treino (com SMOTE e normalização)
    treino_wdbc_smote <- bake(prep_wdbc, new_data = NULL) #COMPLETE
    
    # Aplicar transformação ao teste (SOMENTE normalização, sem SMOTE)
    teste_wdbc_norm <- bake(prep_wdbc, new_data = teste_wdbc) #COMPLETE
    
    ############################################################
  ## 7) Ajustar kNN com dados tratados pela receita
  ############################################################
  # Matriz de preditores numéricos
  cols_num_wdbc <- sapply(treino_wdbc_smote, is.numeric)
  
  X_train_knn <- as.matrix(treino_wdbc_smote[, cols_num_wdbc, drop = FALSE])
  y_train_knn <- treino_wdbc_smote$diagnosis
  
  X_test_knn  <- as.matrix(teste_wdbc_norm[,  cols_num_wdbc, drop = FALSE])
  y_test_knn  <- teste_wdbc_norm$diagnosis
  
  # Escolher o número de vizinhos k
  k_val <- 5
  
  # Predições no teste com kNN
  set.seed(2026)
  pred_knn_smote <- knn(
    train = X_train_knn ,
      test  = X_test_knn,
      cl    = y_train_knn,
      k     = k_val #COMPLETE
  )
  
  ############################################################
  ## 8) Medidas de desempenho (positivo = Maligno)
  ############################################################
  # Predições no teste com a rede neural
  pred_knn_smote <- predict(modelo_nn_wdbc, 
                       newdata = teste_wdbc_norm, 
                       type = "class")
  
  # Matriz de confusão
  tab_wdbc <- table(
    Verdadeiro = teste_wdbc_norm$diagnosis  ,
    Predito    =pred_wdbc
  )
  tab_wdbc
  
  # Vamos considerar "Maligno" como classe positiva
  TP <- tab_wdbc["Maligno", "Maligno"]  # verdadeiros positivos
  TN <- tab_wdbc["Benigno", "Benigno"]  # verdadeiros negativos
  FP <- tab_wdbc["Benigno", "Maligno"]  # falsos positivos
  FN <- tab_wdbc["Maligno", "Benigno"]  # falsos negativos
  
  # Métricas para a MLP na base WDBC
  acc_wdbc   <- (TP + TN)/sum(tab_wdbc) #COMPLETE                 # acurácia
  sens_wdbc  <- TP / (TP + FN) #COMPLETE                                  # sensibilidade (recall, TPR)
  spec_wdbc  <- TN / (TN + FP) #COMPLETE                                  # especificidade (TNR)
  ppv_wdbc   <- TP / (TP + FP)  #COMPLETE                                 # VPP (precisão)
  npv_wdbc   <-  TN / (TN + FN) #COMPLETE                                  # VPN
  gmean_wdbc <- sqrt( sens_wdbc*spec_wdbc) #COMPLETE                     # g-média
  f1_wdbc    <- 2 * (ppv_wdbc * sens_wdbc) / (ppv_wdbc + sens_wdbc) #COMPLETE  # F1-score
  
  # MCC
  mcc_num <- (TP * TN - FP * FN)
  mcc_den <- sqrt( (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) )
  mcc_wdbc <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  
  cat("Desempenho no conjunto de teste – MLP (positivo = Maligno)\n")
  cat(sprintf("Acurácia          : %.4f\n", acc_wdbc))
  cat(sprintf("Sensibilidade     : %.4f\n", sens_wdbc))
  cat(sprintf("Especificidade    : %.4f\n", spec_wdbc))
  cat(sprintf("VPP (precisão)    : %.4f\n", ppv_wdbc))
  cat(sprintf("VPN               : %.4f\n", npv_wdbc))
  cat(sprintf("G-média           : %.4f\n", gmean_wdbc))
  cat(sprintf("F1-score          : %.4f\n", f1_wdbc))
  cat(sprintf("MCC               : %.4f\n", mcc_wdbc))
  
  
  # Repita o item 5) mas utilizando o sufixo _knn_smote ao invés do sufixo _wdbc
  
  
  ############################################################
  ## 9) Compare os resultados
  ############################################################