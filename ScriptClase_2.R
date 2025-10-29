################################### ANÁLISIS DESCRIPTIVO #############################################

library(mlbench) # es la biblioteca donde se encuentra la base de datos
library(ggplot2) # para crear el gráfico de medio violín y otros según sea necesario
library(gghalves) # para crear el gráfico de medio violín 

# Accediendo a la base de datos PimaIndiansDiabetes dentro de la biblioteca cargada
data(PimaIndiansDiabetes)

# Asignando la base de datos al objeto "dados"
dados <- PimaIndiansDiabetes

# Verificando si la asignación fue exitosa
dados

# Observando las primeras observaciones
head(dados)

# Verificar si hay algún NA en la base
anyNA(dados)

# Proporción de la variable respuesta en toda la base de datos
table(dados$diabetes) 
# devuelve una tabla con la frecuencia absoluta de cada clase

prop.table(table(dados$diabetes))
# transforma la tabla de frecuencia absoluta en una tabla de frecuencia relativa

# Nombres de las variables
vars <- colnames(dados)
# accede al nombre de las variables

vars
# verifica si los nombres fueron seleccionados

# Diseño: 3 filas x 3 columnas (9 boxplots)
par(mfrow = c(3, 3))
# para construir 9 gráficos dispuestos en 3 filas y 3 columnas

# Bucle para generar los boxplots
for (var in vars) {
  boxplot(dados[[var]] ~ dados$diabetes,
          col = c("steelblue", "tomato"),
          names = c("Negativo", "Positivo"),
          ylab = var,
          main = paste(var))
}


################################### DESBALANCEADO #############################################

# Paquetes necesarios en este caso
library(mlbench) # es la biblioteca donde se encuentra la base de datos
library(ggplot2) # para crear el gráfico de medio violín y otros según sea necesario
library(gghalves) # para crear el gráfico de medio violín 

# Accediendo a la base de datos PimaIndiansDiabetes dentro de la biblioteca cargada
data(PimaIndiansDiabetes)

# Asignando la base de datos al objeto "dados"
dados <- PimaIndiansDiabetes

# Verificando si la asignación fue exitosa
dados

# Observando las primeras observaciones
head(dados)

# Verificar si hay algún NA en la base
anyNA(dados)

# Proporción de la variable respuesta en toda la base de datos
table(dados$diabetes) 
# devuelve una tabla con la frecuencia absoluta de cada clase

prop.table(table(dados$diabetes))
# transforma la tabla de frecuencia absoluta en una tabla de frecuencia relativa

# Aplicando árboles de clasificación
library(rsample)   # funciones para dividir los datos en entrenamiento/prueba (split estratificado, bootstrap, etc.)
set.seed(125)

# realiza la división estratificada por la variable Y
split <- initial_split(dados, prop = 0.7, strata = "diabetes")

treino <- training(split) # 70% para el entrenamiento
teste  <- testing(split)  # 30% para la prueba

# Verificando proporción de clases
prop.table(table(dados$diabetes))
prop.table(table(treino$diabetes))
prop.table(table(teste$diabetes))

# Ajustar nuevamente el árbol
library(rpart)
arvore <- rpart(diabetes ~ ., data = treino, method = "class")

# Predicción en el conjunto de prueba
pred <- predict(arvore, newdata = teste, type = "class")

# crea la matriz de confusión
cm <- table(Previsto = pred, Observado = teste$diabetes)
cm

TP <- cm["pos","pos"]
TN <- cm["neg","neg"]
FP <- cm["pos","neg"]
FN <- cm["neg","pos"]

acuracia       <- (TP + TN) / sum(cm)
sensibilidade  <- TP / (TP + FN)   
especificidade <- TN / (TN + FP)
VPP <- TP / (TP + FP)   # Valor predictivo positivo (precisión)
VPN <- TN / (TN + FN)   # Valor predictivo negativo
gmedia <- sqrt(sensibilidade*especificidade)
# media geométrica entre sensibilidad y especificidad
f1 <- 2 * (VPP * sensibilidade) / (VPP + sensibilidade)
# media armónica entre sensibilidad y precisión

mcc <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

# Imprimiendo los valores
acuracia
sensibilidade
especificidade
VPP
VPN
gmedia
f1
mcc

################################### SMOTE + ESTANDARIZACIÓN #############################################

library(mlbench) # es la biblioteca donde se encuentra la base de datos
library(ggplot2) # para crear el gráfico de medio violín y otros según sea necesario
library(gghalves) # para crear el gráfico de medio violín 

# Accediendo a la base de datos PimaIndiansDiabetes dentro de la biblioteca cargada
data(PimaIndiansDiabetes)

# Asignando la base de datos al objeto "dados"
dados <- PimaIndiansDiabetes

# Verificando si la asignación fue exitosa
dados

# Observando las primeras observaciones
head(dados)

# Verificar si hay algún NA en la base
anyNA(dados)

# Proporción de la variable respuesta en toda la base de datos
table(dados$diabetes) 
# devuelve una tabla con la frecuencia absoluta de cada clase

prop.table(table(dados$diabetes))
# transforma la tabla de frecuencia absoluta en una tabla de frecuencia relativa

# Paquetes
library(rsample)   # funciones para dividir los datos en entrenamiento/prueba (split estratificado, bootstrap, etc.)
library(recipes)   # preprocesamiento de datos (imputación, estandarización, normalización, creación de variables dummy, etc.)
library(themis)    # métodos para tratar el desbalanceo (SMOTE, ROSE, ENN, Tomek links, entre otros)
library(rpart)     # ajuste de árboles de clasificación y regresión

# REALIZAR EL SPLIT ESTRATIFICADO EN LOS DATOS (Hazlo tú mismo con base en lo anterior)

# Para garantizar la reproducibilidad
set.seed(125)

# USAR prop.table() PARA VERIFICAR LAS PROPORCIONES DE LAS CLASES ANTES DE SMOTE 
# (Hazlo tú mismo con base en lo anterior)

# USANDO LA BIBLIOTECA RECIPES PARA EL PREPROCESAMIENTO DE DATOS (Diapositivas)

# Receta con estandarizaci\'{o}n + SMOTE (solo en el conjunto de entrenamiento)
rec <- recipe(diabetes ~ ., data = treino) %>%
  step_normalize(all_predictors()) %>%   # estandariza todas las variables predictoras
  step_smote(diabetes) # aplica SMOTE

# Prepara la receta con base en el conjunto de entrenamiento
rec_prep <- prep(rec)

# Aplica en el conjunto de entrenamiento imputado + estandarizado + balanceado
treino_smote <- bake(rec_prep, new_data = NULL)

# AJUSTAR ÁRBOL EN treino_smote (Hazlo tú mismo con base en lo anterior)

# PREDECIR teste_norm CON EL ÁRBOL AJUSTADO EN treino_smote (Hazlo tú mismo con base en lo anterior)

# MATRIZ DE CONFUSIÓN + MÉTRICAS DE DESEMPEÑO (Hazlo tú mismo con base en lo anterior)

############################# ENN + ESTANDARIZACI\'{O}N ###########################################

library(mlbench) # es la biblioteca donde se encuentra la base de datos
library(ggplot2) # para crear el gráfico de medio violín y otros según sea necesario
library(gghalves) # para crear el gráfico de medio violín 

# Accediendo a la base de datos PimaIndiansDiabetes dentro de la biblioteca cargada
data(PimaIndiansDiabetes)

# Asignando la base de datos al objeto "dados"
dados <- PimaIndiansDiabetes

# Verificando si la asignación fue exitosa
dados

# Observando las primeras observaciones
head(dados)

# Verificar si hay algún NA en la base
anyNA(dados)

# Proporción de la variable respuesta en toda la base de datos
table(dados$diabetes) 
# devuelve una tabla con la frecuencia absoluta de cada clase

prop.table(table(dados$diabetes))
# transforma la tabla de frecuencia absoluta en una tabla de frecuencia relativa

# Paquetes
library(rsample)   # funciones para dividir los datos en entrenamiento/prueba (split estratificado, bootstrap, etc.)
library(FNN)       # Algoritmos de vecinos más cercanos (usado en KNN, imputación y ENN manual)
library(tidymodels)  # Conjunto de paquetes para modelado estadístico y aprendizaje automático en R (incluye rsample, recipes, yardstick, etc.)
library(rpart)       # Ajuste de árboles de clasificación y regresión

# Función ENN
ENN_manual <- function(data, target, k = 3, majority_class) {
  X <- data[, setdiff(names(data), target)]
  y <- data[[target]]
  
  # Índices de los k vecinos (no incluye el propio punto)
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

# REALIZAR EL SPLIT ESTRATIFICADO EN LOS DATOS (Hazlo tú mismo con base en lo anterior)

# Para garantizar la reproducibilidad
set.seed(125)

# USAR prop.table() PARA VERIFICAR LAS PROPORCIONES DE LAS CLASES ANTES DEL ENN 
# (Hazlo tú mismo con base en lo anterior)

# USANDO LA BIBLIOTECA RECIPES PARA EL PREPROCESAMIENTO DE DATOS (solo estandarización)
# (Hazlo tú mismo con base en lo anterior)

# APLICAR ENN AL CONJUNTO DE ENTRENAMIENTO ESTANDARIZADO (Diapositiva)

# AJUSTAR ÁRBOL EN treino_ENN (Hazlo tú mismo con base en lo anterior)

# PREDECIR teste_norm CON EL ÁRBOL AJUSTADO EN treino_ENN (Hazlo tú mismo con base en lo anterior)

# MATRIZ DE CONFUSIÓN + MÉTRICAS DE DESEMPEÑO (Hazlo tú mismo con base en lo anterior)


####################### DESBALANCEADO + ESTANDARIZACI\'{O}N + IMPUTACI\'{O}N ######################################

library(mlbench) # es la biblioteca donde se encuentra la base de datos
library(ggplot2) # para crear el gráfico de medio violín y otros según sea necesario
library(gghalves) # para crear el gráfico de medio violín 

# Accediendo a la base de datos PimaIndiansDiabetes dentro de la biblioteca cargada
data(PimaIndiansDiabetes)

# Asignando la base de datos al objeto "dados"
dados <- PimaIndiansDiabetes

# Verificando si la asignación fue exitosa
dados

# Observando las primeras observaciones
head(dados)

# Verificar si hay algún NA en la base
anyNA(dados)

# Proporción de la variable respuesta en toda la base de datos
table(dados$diabetes) 
# devuelve una tabla con la frecuencia absoluta de cada clase

prop.table(table(dados$diabetes))
# transforma la tabla de frecuencia absoluta en una tabla de frecuencia relativa

# REEMPLAZANDO DATOS INCONSISTENTES POR NAs (Diapositivas)

# Paquetes
library(rsample)   # funciones para dividir los datos en entrenamiento/prueba (split estratificado, bootstrap, etc.)
library(tidymodels)  # Conjunto de paquetes para modelado estadístico y aprendizaje automático en R (incluye rsample, recipes, yardstick, etc.)
library(rpart)       # Ajuste de árboles de clasificación y regresión

# REALIZAR EL SPLIT ESTRATIFICADO EN LOS DATOS (Hazlo tú mismo con base en lo anterior)

# Para garantizar la reproducibilidad
set.seed(125)

# USAR prop.table() PARA VERIFICAR LAS PROPORCIONES DE LAS CLASES 
# (Hazlo tú mismo con base en lo anterior)

# USANDO LA BIBLIOTECA RECIPES PARA EL PREPROCESAMIENTO DE DATOS - IMPUTACIÓN (Diapositivas)

# AJUSTAR ÁRBOL EN treino_imp (Hazlo tú mismo con base en lo anterior)

# PREDECIR teste_imp CON EL ÁRBOL AJUSTADO EN treino_imp (Hazlo tú mismo con base en lo anterior)

# MATRIZ DE CONFUSIÓN + MÉTRICAS DE DESEMPEÑO (Hazlo tú mismo con base en lo anterior)


######################## SMOTE + ESTANDARIZACI\'{O}N + IMPUTACI\'{O}N ###################################

library(mlbench) # es la biblioteca donde se encuentra la base de datos
library(ggplot2) # para crear el gráfico de medio violín y otros según sea necesario
library(gghalves) # para crear el gráfico de medio violín 

# Accediendo a la base de datos PimaIndiansDiabetes dentro de la biblioteca cargada
data(PimaIndiansDiabetes)

# Asignando la base de datos al objeto "dados"
dados <- PimaIndiansDiabetes

# Verificando si la asignación fue exitosa
dados

# Observando las primeras observaciones
head(dados)

# Verificar si hay algún NA en la base
anyNA(dados)

# Proporción de la variable respuesta en toda la base de datos
table(dados$diabetes) 
# devuelve una tabla con la frecuencia absoluta de cada clase

prop.table(table(dados$diabetes))
# transforma la tabla de frecuencia absoluta en una tabla de frecuencia relativa

# REEMPLAZANDO DATOS INCONSISTENTES POR NAs (Hazlo tú mismo con base en lo anterior)

# Paquetes
library(rsample)   # funciones para dividir los datos en entrenamiento/prueba (split estratificado, bootstrap, etc.)
library(recipes)   # preprocesamiento de datos (imputación, estandarización, normalización, creación de variables dummy, etc.)
library(themis)    # métodos para tratar el desbalanceo (SMOTE, ROSE, ENN, Tomek links, entre otros)
library(rpart)     # ajuste de árboles de clasificación y regresión

# REALIZAR EL SPLIT ESTRATIFICADO EN LOS DATOS (Hazlo tú mismo con base en lo anterior)

# Para garantizar la reproducibilidad
set.seed(125)

# USAR prop.table() PARA VERIFICAR LAS PROPORCIONES DE LAS CLASES ANTES DE SMOTE
# (Hazlo tú mismo con base en lo anterior)

# USANDO LA BIBLIOTECA RECIPES PARA EL PREPROCESAMIENTO DE DATOS - ESTANDARIZACIÓN + IMPUTACIÓN (Diapositivas)

# AJUSTAR ÁRBOL EN treino_knn_norm_smote (Hazlo tú mismo con base en lo anterior)

# PREDECIR teste_knn_norm CON EL ÁRBOL AJUSTADO EN treino_knn_norm_smote (Hazlo tú mismo con base en lo anterior)

# MATRIZ DE CONFUSIÓN + MÉTRICAS DE DESEMPEÑO (Hazlo tú mismo con base en lo anterior)


######################## ENN + ESTANDARIZACI\'{O}N + IMPUTACI\'{O}N ###################################

library(mlbench) # es la biblioteca donde se encuentra la base de datos
library(ggplot2) # para crear el gráfico de medio violín y otros según sea necesario
library(gghalves) # para crear el gráfico de medio violín 

# Accediendo a la base de datos PimaIndiansDiabetes dentro de la biblioteca cargada
data(PimaIndiansDiabetes)

# Asignando la base de datos al objeto "dados"
dados <- PimaIndiansDiabetes

# Verificando si la asignación fue exitosa
dados

# Observando las primeras observaciones
head(dados)

# Verificar si hay algún NA en la base
anyNA(dados)

# Proporción de la variable respuesta en toda la base de datos
table(dados$diabetes) 
# devuelve una tabla con la frecuencia absoluta de cada clase

prop.table(table(dados$diabetes))
# transforma la tabla de frecuencia absoluta en una tabla de frecuencia relativa

# Paquetes
library(rsample)   # funciones para dividir los datos en entrenamiento/prueba (split estratificado, bootstrap, etc.)
library(FNN)         # Algoritmos de vecinos más cercanos (usado en KNN, imputación y ENN manual)
library(tidymodels)  # Conjunto de paquetes para modelado estadístico y aprendizaje automático en R (incluye rsample, recipes, yardstick, etc.)
library(rpart)       # Ajuste de árboles de clasificación y regresión

# Función ENN
ENN_manual <- function(data, target, k = 3, majority_class) {
  X <- data[, setdiff(names(data), target)]
  y <- data[[target]]
  
  # Índices de los k vecinos (no incluye el propio punto)
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

# REALIZAR EL SPLIT ESTRATIFICADO EN LOS DATOS (Hazlo tú mismo con base en lo anterior)

# Para garantizar la reproducibilidad
set.seed(125)

# USAR prop.table() PARA VERIFICAR LAS PROPORCIONES DE LAS CLASES ANTES DEL ENN 
# (Hazlo tú mismo con base en lo anterior)

# USANDO LA BIBLIOTECA RECIPES PARA EL PREPROCESAMIENTO DE DATOS - ESTANDARIZACIÓN E IMPUTACIÓN
# (Hazlo tú mismo con base en lo anterior)

# APLICAR ENN EN treino_imp_norm

# AJUSTAR ÁRBOL EN treino_ENN (Hazlo tú mismo con base en lo anterior)

# PREDECIR teste_imp_norm CON EL ÁRBOL AJUSTADO EN treino_ENN (Hazlo tú mismo con base en lo anterior)

# MATRIZ DE CONFUSIÓN + MÉTRICAS DE DESEMPEÑO (Hazlo tú mismo con base en lo anterior)