#################################################################################
## Códigos - Detección de diabetes                                            ##
## Asignatura: Topicos en Ciencias de Datos                                   ##
## Profesor: Ricardo Felipe Ferreira                                          ##
## Año: 2025/2                                                                ##
#################################################################################

################################### ANÁLISIS DESCRIPTIVO #############################################

library(mlbench)  # biblioteca que contiene la base de datos
library(ggplot2)  # para crear gráficos de medio-violín y otros necesarios
library(gghalves) # para el gráfico de medio-violín

# Acceder a la base de datos PimaIndiansDiabetes dentro de la biblioteca cargada
data(PimaIndiansDiabetes)

# Almacenar la base de datos en el objeto "datos"
datos <- PimaIndiansDiabetes

# Verificar si la asignación fue exitosa
datos

# Observar las primeras observaciones
head(datos)

# Verificar si existe algún NA en la base
anyNA(datos)

# Proporción de la variable respuesta en toda la base de datos
table(datos$diabetes) 
# devuelve una tabla con la frecuencia absoluta de cada clase

prop.table(table(datos$diabetes))
# transforma la tabla de frecuencias absolutas en una tabla de frecuencias relativas


################################### DESBALANCEADO #############################################

library(rsample)   # funciones para dividir los datos en entrenamiento/prueba (split estratificado)
set.seed(125)

# División estratificada por la variable Y
split <- initial_split(datos, prop = 0.7, strata = "diabetes")

entrenamiento <- training(split) # 70% para entrenamiento
prueba        <- testing(split)  # 30% para prueba

# Verificando la proporción de clases
prop.table(table(datos$diabetes))
prop.table(table(entrenamiento$diabetes))
prop.table(table(prueba$diabetes))

# Ajuste del árbol de clasificación
library(rpart)
arbol <- rpart(diabetes ~ ., data = entrenamiento, method = "class")

# Predicción en el conjunto de prueba
pred <- predict(arbol, newdata = prueba, type = "class")

# Matriz de confusión
cm <- table(Predicho = pred, Observado = prueba$diabetes)
cm

TP <- cm["pos","pos"]
TN <- cm["neg","neg"]
FP <- cm["pos","neg"]
FN <- cm["neg","pos"]

exactitud       <- (TP + TN) / sum(cm)
sensibilidad    <- TP / (TP + FN)
especificidad   <- TN / (TN + FP)
VPP             <- TP / (TP + FP)   # Valor predictivo positivo (precisión)
VPN             <- TN / (TN + FN)   # Valor predictivo negativo
gmedia          <- sqrt(sensibilidad * especificidad) # media geométrica
f1              <- 2 * (VPP * sensibilidad) / (VPP + sensibilidad) # media armónica
MCC             <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)) # coeficiente de Matthews

# Imprimir resultados
exactitud; sensibilidad; especificidad; VPP; VPN; gmedia; f1; MCC


################################### SMOTE + PADRONIZACIÓN #############################################

library(themis)
library(recipes)

# Recipe con normalización + SMOTE (solo sobre el conjunto de entrenamiento)
rec <- recipe(diabetes ~ ., data = entrenamiento) %>%
  step_normalize(all_predictors()) %>%
  step_smote(diabetes)

rec_prep <- prep(rec)

entrenamiento_smote <- bake(rec_prep, new_data = NULL)
prueba_norm         <- bake(rec_prep, new_data = prueba)

# Ajustar árbol con datos balanceados
arbol <- rpart(diabetes ~ ., data = entrenamiento_smote, method = "class")

# Predicción en el conjunto de prueba
pred <- predict(arbol, newdata = prueba_norm, type = "class")

# Matriz de confusión
cm <- table(Predicho = pred, Observado = prueba_norm$diabetes)
cm

TP <- cm["pos","pos"]
TN <- cm["neg","neg"]
FP <- cm["pos","neg"]
FN <- cm["neg","pos"]

exactitud       <- (TP + TN) / sum(cm)
sensibilidad    <- TP / (TP + FN)
especificidad   <- TN / (TN + FP)
VPP             <- TP / (TP + FP)
VPN             <- TN / (TN + FN)
gmedia          <- sqrt(sensibilidad * especificidad)
f1              <- 2 * (VPP * sensibilidad) / (VPP + sensibilidad)
MCC             <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))

# Resultados
exactitud; sensibilidad; especificidad; VPP; VPN; gmedia; f1; MCC


############################# ENN + PADRONIZACIÓN ###########################################

library(FNN)
library(tidymodels)

# Función ENN adaptada
ENN_manual <- function(data, target, k = 3, majority_class) {
  X <- data[, setdiff(names(data), target)]
  y <- data[[target]]
  nn <- knnx.index(as.matrix(X), as.matrix(X), k = k + 1)[, -1]
  remove_idx <- sapply(1:nrow(X), function(i) {
    if (y[i] == majority_class) {
      neigh_classes <- y[nn[i, ]]
      maj_class <- names(sort(table(neigh_classes), decreasing = TRUE))[1]
      return(maj_class != y[i])
    } else {
      return(FALSE)
    }
  })
  return(data[!remove_idx, ])
}

set.seed(125)

split   <- initial_split(datos, prop = 0.7, strata = "diabetes")
entrenamiento  <- training(split)
prueba   <- testing(split)

rec <- recipe(diabetes ~ ., data = entrenamiento) %>%
  step_normalize(all_numeric_predictors())

rec_prep <- prep(rec)
entrenamiento_norm <- bake(rec_prep, new_data = entrenamiento)
prueba_norm  <- bake(rec_prep, new_data = prueba)

entrenamiento_ENN <- ENN_manual(entrenamiento_norm, target = "diabetes", k = 3, majority_class = "neg")

arbol <- rpart(diabetes ~ ., data = entrenamiento_ENN, method = "class")
pred <- predict(arbol, newdata = prueba_norm, type = "class")

cm <- table(Predicho = pred, Observado = prueba_norm$diabetes)
cm

TP <- cm["pos","pos"]
TN <- cm["neg","neg"]
FP <- cm["pos","neg"]
FN <- cm["neg","pos"]

exactitud       <- (TP + TN) / sum(cm)
sensibilidad    <- TP / (TP + FN)
especificidad   <- TN / (TN + FP)
VPP             <- TP / (TP + FP)
VPN             <- TN / (TN + FN)
gmedia          <- sqrt(sensibilidad * especificidad)
f1              <- 2 * (VPP * sensibilidad) / (VPP + sensibilidad)
MCC             <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))

exactitud; sensibilidad; especificidad; VPP; VPN; gmedia; f1; MCC


######################## SMOTE + ENN + PADRONIZACIÓN ##################################

library(themis)

split   <- initial_split(datos, prop = 0.7, strata = "diabetes")
entrenamiento  <- training(split)
prueba   <- testing(split)

rec <- recipe(diabetes ~ ., data = entrenamiento) %>%
  step_normalize(all_numeric_predictors())

rec_prep <- prep(rec)
entrenamiento_proc <- bake(rec_prep, new_data = NULL)
prueba_proc  <- bake(rec_prep, new_data = prueba)

entrenamiento_ENN <- ENN_manual(entrenamiento_proc, target = "diabetes", k = 3, majority_class = "neg")

rec_smote <- recipe(diabetes ~ ., data = entrenamiento_ENN) %>%
  step_smote(diabetes)

rec_smote_prep <- prep(rec_smote)
entrenamiento_final <- bake(rec_smote_prep, new_data = NULL)

arbol <- rpart(diabetes ~ ., data = entrenamiento_final, method = "class")
pred <- predict(arbol, newdata = prueba_proc, type = "class")

cm <- table(Predicho = pred, Observado = prueba_proc$diabetes)
cm

TP <- cm["pos","pos"]
TN <- cm["neg","neg"]
FP <- cm["pos","neg"]
FN <- cm["neg","pos"]

exactitud       <- (TP + TN) / sum(cm)
sensibilidad    <- TP / (TP + FN)
especificidad   <- TN / (TN + FP)
VPP             <- TP / (TP + FP)
VPN             <- TN / (TN + FN)
gmedia          <- sqrt(sensibilidad * especificidad)
f1              <- 2 * (VPP * sensibilidad) / (VPP + sensibilidad)
MCC             <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))

exactitud; sensibilidad; especificidad; VPP; VPN; gmedia; f1; MCC

####################### DESBALANCEADO + PADRONIZACIÓN + IMPUTACIÓN ######################################

library(mlbench)
library(tidymodels)
library(rpart)

# Cargar la base de datos
data(PimaIndiansDiabetes)
datos <- PimaIndiansDiabetes

# Reemplazar ceros por NA en variables que no pueden ser cero
datos$glucose[datos$glucose == 0] <- NA
datos$pressure[datos$pressure == 0] <- NA
datos$triceps[datos$triceps == 0] <- NA
datos$insulin[datos$insulin == 0] <- NA
datos$mass[datos$mass == 0] <- NA

# División estratificada
set.seed(125)
split <- initial_split(datos, prop = 0.7, strata = "diabetes")
entrenamiento <- training(split)
prueba <- testing(split)

# Recipe con imputación KNN + normalización
rec <- recipe(diabetes ~ ., data = entrenamiento) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_impute_knn(all_numeric_predictors(), neighbors = 5)

rec_prep <- prep(rec)

entrenamiento_imp <- bake(rec_prep, new_data = NULL)
prueba_imp <- bake(rec_prep, new_data = prueba)

# Ajustar árbol de decisión
arbol <- rpart(diabetes ~ ., data = entrenamiento_imp, method = "class")

# Predicción en el conjunto de prueba imputado
pred <- predict(arbol, newdata = prueba_imp, type = "class")

# Matriz de confusión
cm <- table(Predicho = pred, Observado = prueba_imp$diabetes)
cm

TP <- cm["pos", "pos"]
TN <- cm["neg", "neg"]
FP <- cm["pos", "neg"]
FN <- cm["neg", "pos"]

exactitud <- (TP + TN) / sum(cm)
sensibilidad <- TP / (TP + FN)
especificidad <- TN / (TN + FP)
VPP <- TP / (TP + FP)
VPN <- TN / (TN + FN)
gmedia <- sqrt(sensibilidad * especificidad)
f1 <- 2 * (VPP * sensibilidad) / (VPP + sensibilidad)
MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

# Resultados
exactitud; sensibilidad; especificidad; VPP; VPN; gmedia; f1; MCC


######################## SMOTE + PADRONIZACIÓN + IMPUTACIÓN ###################################

library(themis)
library(tidymodels)
library(rpart)

data(PimaIndiansDiabetes)
datos <- PimaIndiansDiabetes

# Reemplazar ceros por NA
datos$glucose[datos$glucose == 0] <- NA
datos$pressure[datos$pressure == 0] <- NA
datos$triceps[datos$triceps == 0] <- NA
datos$insulin[datos$insulin == 0] <- NA
datos$mass[datos$mass == 0] <- NA

# División estratificada
set.seed(125)
split <- initial_split(datos, prop = 0.7, strata = "diabetes")
entrenamiento <- training(split)
prueba <- testing(split)

# Recipe con imputación KNN + normalización + SMOTE
rec <- recipe(diabetes ~ ., data = entrenamiento) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_impute_knn(all_numeric_predictors(), neighbors = 5) %>%
  step_smote(diabetes)

rec_prep <- prep(rec)
entrenamiento_smote <- bake(rec_prep, new_data = NULL)
prueba_proc <- bake(rec_prep, new_data = prueba)

# Ajustar árbol
arbol <- rpart(diabetes ~ ., data = entrenamiento_smote, method = "class")

# Predicción
pred <- predict(arbol, newdata = prueba_proc, type = "class")

# Matriz de confusión
cm <- table(Predicho = pred, Observado = prueba_proc$diabetes)
cm

TP <- cm["pos", "pos"]
TN <- cm["neg", "neg"]
FP <- cm["pos", "neg"]
FN <- cm["neg", "pos"]

exactitud <- (TP + TN) / sum(cm)
sensibilidad <- TP / (TP + FN)
especificidad <- TN / (TN + FP)
VPP <- TP / (TP + FP)
VPN <- TN / (TN + FN)
gmedia <- sqrt(sensibilidad * especificidad)
f1 <- 2 * (VPP * sensibilidad) / (VPP + sensibilidad)
MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

# Resultados
exactitud; sensibilidad; especificidad; VPP; VPN; gmedia; f1; MCC


######################## ENN + PADRONIZACIÓN + IMPUTACIÓN ###################################

library(FNN)
library(tidymodels)
library(rpart)

data(PimaIndiansDiabetes)
datos <- PimaIndiansDiabetes

# Reemplazar ceros por NA
datos$glucose[datos$glucose == 0] <- NA
datos$pressure[datos$pressure == 0] <- NA
datos$triceps[datos$triceps == 0] <- NA
datos$insulin[datos$insulin == 0] <- NA
datos$mass[datos$mass == 0] <- NA

# Función ENN
ENN_manual <- function(data, target, k = 3, majority_class) {
  X <- data[, setdiff(names(data), target)]
  y <- data[[target]]
  nn <- knnx.index(as.matrix(X), as.matrix(X), k = k + 1)[, -1]
  remove_idx <- sapply(1:nrow(X), function(i) {
    if (y[i] == majority_class) {
      neigh_classes <- y[nn[i, ]]
      maj_class <- names(sort(table(neigh_classes), decreasing = TRUE))[1]
      return(maj_class != y[i])
    } else {
      return(FALSE)
    }
  })
  return(data[!remove_idx, ])
}

set.seed(125)
split <- initial_split(datos, prop = 0.7, strata = "diabetes")
entrenamiento <- training(split)
prueba <- testing(split)

rec <- recipe(diabetes ~ ., data = entrenamiento) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_impute_knn(all_numeric_predictors(), neighbors = 5)

rec_prep <- prep(rec)
entrenamiento_proc <- bake(rec_prep, new_data = NULL)
prueba_proc <- bake(rec_prep, new_data = prueba)

entrenamiento_ENN <- ENN_manual(entrenamiento_proc, target = "diabetes", k = 3, majority_class = "neg")

# Ajustar árbol
arbol <- rpart(diabetes ~ ., data = entrenamiento_ENN, method = "class")

# Predicción
pred <- predict(arbol, newdata = prueba_proc, type = "class")

# Matriz de confusión
cm <- table(Predicho = pred, Observado = prueba_proc$diabetes)
cm

TP <- cm["pos", "pos"]
TN <- cm["neg", "neg"]
FP <- cm["pos", "neg"]
FN <- cm["neg", "pos"]

exactitud <- (TP + TN) / sum(cm)
sensibilidad <- TP / (TP + FN)
especificidad <- TN / (TN + FP)
VPP <- TP / (TP + FP)
VPN <- TN / (TN + FN)
gmedia <- sqrt(sensibilidad * especificidad)
f1 <- 2 * (VPP * sensibilidad) / (VPP + sensibilidad)
MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

# Resultados
exactitud; sensibilidad; especificidad; VPP; VPN; gmedia; f1; MCC


######################## SMOTE + ENN + PADRONIZACIÓN + IMPUTACIÓN #############################

library(themis)
library(FNN)
library(tidymodels)
library(rpart)

data(PimaIndiansDiabetes)
datos <- PimaIndiansDiabetes

# Reemplazar ceros por NA
datos$glucose[datos$glucose == 0] <- NA
datos$pressure[datos$pressure == 0] <- NA
datos$triceps[datos$triceps == 0] <- NA
datos$insulin[datos$insulin == 0] <- NA
datos$mass[datos$mass == 0] <- NA

# División estratificada
set.seed(125)
split <- initial_split(datos, prop = 0.7, strata = "diabetes")
entrenamiento <- training(split)
prueba <- testing(split)

# Recipe: imputación + normalización
rec <- recipe(diabetes ~ ., data = entrenamiento) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_impute_knn(all_numeric_predictors(), neighbors = 5)

rec_prep <- prep(rec)
entrenamiento_proc <- bake(rec_prep, new_data = NULL)
prueba_proc <- bake(rec_prep, new_data = prueba)

# ENN manual
entrenamiento_ENN <- ENN_manual(entrenamiento_proc, target = "diabetes", k = 3, majority_class = "neg")

# SMOTE posterior
rec_smote <- recipe(diabetes ~ ., data = entrenamiento_ENN) %>%
  step_smote(diabetes)

rec_smote_prep <- prep(rec_smote)
entrenamiento_final <- bake(rec_smote_prep, new_data = NULL)

# Ajustar árbol
arbol <- rpart(diabetes ~ ., data = entrenamiento_final, method = "class")

# Predicción
pred <- predict(arbol, newdata = prueba_proc, type = "class")

# Matriz de confusión
cm <- table(Predicho = pred, Observado = prueba_proc$diabetes)
cm

TP <- cm["pos", "pos"]
TN <- cm["neg", "neg"]
FP <- cm["pos", "neg"]
FN <- cm["neg", "pos"]

exactitud <- (TP + TN) / sum(cm)
sensibilidad <- TP / (TP + FN)
especificidad <- TN / (TN + FP)
VPP <- TP / (TP + FP)
VPN <- TN / (TN + FN)
gmedia <- sqrt(sensibilidad * especificidad)
f1 <- 2 * (VPP * sensibilidad) / (VPP + sensibilidad)
MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

# Resultados finales
exactitud; sensibilidad; especificidad; VPP; VPN; gmedia; f1; MCC
