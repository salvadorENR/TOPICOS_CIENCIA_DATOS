#################################################################################
## Códigos - Incumplimiento en tarjeta de crédito                              ##
## Asignatura: Tópicos en Ciencia de Datos                                    ##
## Profesor: Ricardo Felipe Ferreira                                          ##
## Año: 2025/2                                                                ##
#################################################################################

############################################################
## 0) Paquetes
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

############################################################
## 1) Cargar los datos
############################################################

# IMPORTE LA BASE DE DATOS EN R. PUEDE USAR GOOGLE SI NO RECUERDA CÓMO IMPORTAR
# UNA BASE DE DATOS DESDE LA COMPUTADORA AL RSTUDIO.

############################################################
## 2) Entrenamiento y prueba con rsample (estratificado)
############################################################

# COMPLETE LOS ESPACIOS QUE FALTAN. SI ES NECESARIO, USE EL CÓDIGO DE LA CLASE PASADA SOBRE DIABETES.

set.seed(2026) #para mantener la reproducibilidad

split_obj <- initial_split(dados,
                           prop  = ,             # X% para entrenamiento
                           strata = "")  # estratifica en la variable respuesta

treino <- training()
teste  <- testing()

# verificar proporciones

table() #tabla de frecuencia absoluta de incumplidores en la base original
table() #tabla de frecuencia absoluta de incumplidores en la base de entrenamiento
table() #tabla de frecuencia absoluta de incumplidores en la base de prueba

prop.table() #tabla de frecuencia relativa de incumplidores en la base original
prop.table() #tabla de frecuencia relativa de incumplidores en la base de entrenamiento
prop.table() #tabla de frecuencia relativa de incumplidores en la base de prueba

############################################################
## 3) Aplicar mRMR
############################################################

###############################################################
## (3A) Garantizar que las columnas son del tipo aceptado por mRMRe
###############################################################

# treino es tu data.frame con variables discretas
str(treino)  #muestra el tipo de tus variables

# convertir todo a numérico (mRMRe lo acepta)
treino_num <- as.data.frame(lapply(treino, function(x) as.numeric(x)))

str(treino_num)  # ahora todas las variables deben aparecer como "num"

# mRMR también acepta ordinal.

# PENSAR: ¿NO DEBERÍAMOS CONSIDERAR ORDINARIA EN VEZ DE NUMÉRICA? ¿NO TIENE MÁS SENTIDO PARA NUESTRAS VARIABLES?

############################################################
## (3B) Volver a ejecutar mRMR
############################################################

# crear objeto especial
dados_mrmr <- mRMR.data(data = treino_num)

# seleccionar, por ejemplo, 3 variables
res_mrmr <- mRMR.classic(
  data = dados_mrmr,
  target_indices = 1,   # 1ª columna = incumplimiento
  feature_count = 3     # número de variables a seleccionar
)

res_mrmr@filters
vars_selecionadas <- colnames(treino_num)[res_mrmr@filters[[1]] ]
vars_selecionadas

# PENSAR: AQUÍ FIJAMOS EL NÚMERO DE VARIABLES A SELECCIONAR COMO 3. ¿DE DÓNDE VINO ESE NÚMERO? ¿DEL CORAZÓN?
# ¿QUÉ PODRÍA HACERSE PARA QUE ESTA ELECCIÓN FUERA MENOS SUBJETIVA?

############################################################
## 4) Modelos: todas las variables vs mRMR
##    (usando REGRESIÓN LOGÍSTICA)
############################################################

## 4.1) Modelo con TODAS las variables
modelo_todas <- glm(inadimplente ~ ., 
                    data = , # datos donde se va a ajustar
                    family =) # ¿DE QUÉ TIPO ES ESTE GLM?

# probabilidad predicha para el conjunto de prueba
prob_todas <- predict(, # modelo ajustado que vamos a usar para predecir
                      newdata = , # qué conjunto de datos vamos a predecir
                      type = "response")

# clase predicha (¿QUÉ PUNTO DE CORTE?)
pred_todas <- ifelse(prob_todas >= , 1, 0)


## 4.2) Modelo SOLO con las variables seleccionadas por mRMR

modelo_mrmr <- glm(, # COLOCAR LA FÓRMULA
                   data = , # datos donde se va a ajustar
                   family =) # ¿DE QUÉ TIPO ES ESTE GLM?

# probabilidad predicha para el conjunto de prueba
prob_todas <- predict(, # modelo ajustado que vamos a usar para predecir
                      newdata = , # qué conjunto de datos vamos a predecir
                      type = "response")

# clase predicha (¿QUÉ PUNTO DE CORTE?)
pred_todas <- ifelse(prob_todas >= , 1, 0)

## 4.3) Métricas para escenario desbalanceado

# función auxiliar: recibe vectores de 0/1
calc_metrics <- function(y_true, y_pred) {
  # matriz 2x2
  tab <- table(Pred = y_pred, Real = y_true)
  
  # garantizar que tiene todas las entradas
  TP <- ifelse("1" %in% rownames(tab) & "1" %in% colnames(tab), tab["1","1"], 0)
  TN <- ifelse("0" %in% rownames(tab) & "0" %in% colnames(tab), tab["0","0"], 0)
  FP <- ifelse("1" %in% rownames(tab) & "0" %in% colnames(tab), tab["1","0"], 0)
  FN <- ifelse("0" %in% rownames(tab) & "1" %in% colnames(tab), tab["0","1"], 0)
  
  
  #ATENCIÓN: AQUÍ DEBE CAMBIAR LA PALABRA "FÓRMULA" EN EL CÓDIGO POR LA FÓRMULA DE LA MEDIDA
  
  # métricas
  acc  <- ifelse((TP + TN + FP + FN) > 0, FÓRMULA, NA)
  sens <- ifelse((TP + FN) > 0, FÓRMULA, NA)  # recall
  esp  <- ifelse((TN + FP) > 0, FÓRMULA, NA)
  ppv  <- ifelse((TP + FP) > 0, FÓRMULA, NA)  # precision
  npv  <- ifelse((TN + FN) > 0, FÓRMULA, NA)
  gmean <- ifelse(!is.na(sens) & !is.na(esp), FÓRMULA, NA)
  f1    <- ifelse((ppv + sens) > 0, FÓRMULA, NA)
  
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

met_todas <- calc_metrics(teste$inadimplente, pred_todas) #cálculo de las métricas en las predicciones del modelo con todas
met_mrmr  <- calc_metrics(, ) #AQUÍ APLIQUE EN EL MODELO EN EL QUE SE APLICÓ EL MRMR

#Imprimir las métricas del caso con todas las variables
cat("\n== Métricas - Modelo con TODAS ==\n")
print(met_todas)

#Imprimir las métricas del caso en que hubo selección por mRMR
cat("\n== Métricas - Modelo con mRMR ==\n")
print(met_mrmr)



#Aquí va el siguiente código
#################################################################################
## Códigos - Incumplimiento en tarjeta de crédito                              ##
## Asignatura: Tópicos en Ciencia de Datos                                     ##
## Profesor: Ricardo Felipe Ferreira                                           ##
## Año: 2025/2                                                                 ##
#################################################################################

############################################################
## 0) Paquetes
############################################################

# install.packages(c("mRMRe","infotheo","e1071","dplyr","rsample"))
library(mRMRe)
library(infotheo)
library(e1071)
library(dplyr)
library(rsample)

############################################################
## 1) Cargar los datos
############################################################
## OPCIÓN A: si tienes un CSV local:
## dados <- read.csv("incumplimiento_tarjeta.csv", stringsAsFactors = TRUE)

## OPCIÓN B: si ya lo tienes en el entorno, asegúrate de que:
## - La variable respuesta se llama 'inadimplente' y es factor/0-1
## - El data.frame completo se llama 'dados'
stopifnot(exists("dados"))

## Coherencia de la respuesta:
if (!("inadimplente" %in% names(dados))) {
  stop("No encuentro la columna 'inadimplente' en 'dados'. Renómbrala o ajusta el código.")
}
## Asegurar tipo factor con niveles 0/1
if (!is.factor(dados$inadimplente)) {
  dados$inadimplente <- factor(dados$inadimplente, levels = sort(unique(dados$inadimplente)))
}
## Si los niveles no son 0/1, los reetiquetamos a 0/1 conservando el orden:
if (!all(levels(dados$inadimplente) %in% c("0","1"))) {
  levels(dados$inadimplente) <- c("0","1")
}

############################################################
## 2) Entrenamiento y prueba con rsample (estratificado)
############################################################

set.seed(2026) # reproducibilidad

split_obj <- initial_split(dados,
                           prop   = 0.7,          # 70% para entrenamiento
                           strata = "inadimplente")  # estratifica en la respuesta

treino <- training(split_obj)
teste  <- testing(split_obj)

# verificar proporciones
cat("\n=== Frecuencias absolutas ===\n")
print(table(dados$inadimplente))      # base original
print(table(treino$inadimplente))     # entrenamiento
print(table(teste$inadimplente))      # prueba

cat("\n=== Frecuencias relativas ===\n")
print(prop.table(table(dados$inadimplente)))
print(prop.table(table(treino$inadimplente)))
print(prop.table(table(teste$inadimplente)))

############################################################
## 3) Aplicar mRMR
############################################################

###############################################################
## (3A) Garantizar que las columnas son del tipo aceptado por mRMRe
###############################################################
# mRMRe requiere numéricas. Pondremos la respuesta como PRIMERA columna numérica (0/1).
# Guardamos una copia original de treino:
treino_orig <- treino

# mover 'inadimplente' al frente
treino <- treino %>% relocate(inadimplente)

# convertir a numérico (respuesta 0/1, predictores numéricos/ordinales codificados)
treino_num <- treino %>%
  mutate(inadimplente = as.numeric(as.character(inadimplente))) %>%
  mutate(across(-inadimplente, ~ as.numeric(as.character(.))))

# chequeo
# str(treino_num)

############################################################
## (3B) Ejecutar mRMR (relevancia - redundancia)
############################################################
dados_mrmr <- mRMR.data(data = treino_num)

# Número de variables a seleccionar (ajústalo si quieres):
k_features <- min(5, ncol(treino_num) - 1)  # por defecto 5 o menos si hay pocas

res_mrmr <- mRMR.classic(
  data = dados_mrmr,
  target_indices = 1,      # 1ª columna = inadimplente
  feature_count = k_features
)

# índices seleccionados (respecto a treino_num)
sel_idx <- res_mrmr@filters[[1]]
vars_selecionadas <- colnames(treino_num)[sel_idx]
cat("\nVariables seleccionadas por mRMR:\n")
print(vars_selecionadas)

############################################################
## 4) Modelos: todas las variables vs mRMR (Regresión Logística)
############################################################

## 4.1) Modelo con TODAS las variables
modelo_todas <- glm(inadimplente ~ .,
                    data   = treino_orig,     # usamos treino con tipos originales (factores OK)
                    family = binomial(link = "logit"))  # GLM binomial (logístico)

# probabilidad predicha en prueba
prob_todas <- predict(modelo_todas,
                      newdata = teste,
                      type = "response")

# clase predicha con punto de corte 0.5 (ajústalo si hay desbalance severo)
pred_todas <- ifelse(prob_todas >= 0.5, 1, 0)

## 4.2) Modelo SOLO con variables mRMR
# armamos fórmula: respuesta ~ vars mRMR
form_mrmr <- as.formula(
  paste("inadimplente ~", paste(vars_selecionadas[vars_selecionadas != "inadimplente"], collapse = " + "))
)

modelo_mrmr <- glm(form_mrmr,
                   data   = treino_orig,   # usa los tipos originales
                   family = binomial(link = "logit"))

# predicciones en prueba para mRMR
prob_mrmr <- predict(modelo_mrmr,
                     newdata = teste,
                     type = "response")

pred_mrmr <- ifelse(prob_mrmr >= 0.5, 1, 0)

## 4.3) Métricas para escenario desbalanceado
calc_metrics <- function(y_true, y_pred) {
  # asegurar vectores 0/1
  y_true <- as.numeric(as.character(y_true))
  if (any(is.na(y_true))) y_true <- as.numeric(factor(y_true)) - 1L
  
  tab <- table(Pred = y_pred, Real = y_true)
  
  # completar ausentes
  get <- function(r, c) ifelse(r %in% rownames(tab) & c %in% colnames(tab), tab[r, c], 0)
  TP <- get("1","1"); TN <- get("0","0"); FP <- get("1","0"); FN <- get("0","1")
  
  acc   <- ifelse((TP + TN + FP + FN) > 0, (TP + TN) / (TP + TN + FP + FN), NA)
  sens  <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)            # recall
  esp   <- ifelse((TN + FP) > 0, TN / (TN + FP), NA)
  ppv   <- ifelse((TP + FP) > 0, TP / (TP + FP), NA)            # precision
  npv   <- ifelse((TN + FN) > 0, TN / (TN + FN), NA)
  gmean <- ifelse(!is.na(sens) & !is.na(esp), sqrt(sens * esp), NA)
  f1    <- ifelse((ppv + sens) > 0, 2 * (ppv * sens) / (ppv + sens), NA)
  
  denom <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  mcc   <- ifelse(denom > 0, (TP * TN - FP * FN) / denom, NA)
  
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

cat("\n== Métricas - Modelo con TODAS ==\n"); print(met_todas)
cat("\n== Métricas - Modelo con mRMR ==\n");  print(met_mrmr)

############################################################
## Comentarios
############################################################
# - mRMR (Clases 04–06): usa información mutua para medir relevancia y redundancia
#   (entropía/MI: H, I(X;Y)) — teoría de la información. 
# - Regresión logística (ESL, Cap. 4): GLM binomial con enlace logit.
# - k_features (número de variables mRMR) puede elegirse por:
#     * validación cruzada del rendimiento por k (grid de k) y escoger el que maximiza F1/G-mean/MCC,
#     * o usando un criterio de información mínima (MDL/BIC) como guía conceptual (ESL Cap. 7).


























