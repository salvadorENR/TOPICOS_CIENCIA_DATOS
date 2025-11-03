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


