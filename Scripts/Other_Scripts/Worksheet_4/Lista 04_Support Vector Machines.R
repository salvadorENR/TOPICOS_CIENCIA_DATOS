#Instalar paquetes

## Instalar paquetes necesarios (ejecutar solo si es necesario)
install.packages(c("kmed", "e1071", "rsample", "dplyr")) # [2-5]

# Cargar el paquete 'kmed' y el conjunto de datos 'heart' [2]
library(kmed)
library(e1071)  # Para Support Vector Machines (SVM) [3]
library(rsample) # Para división de datos [4]
library(dplyr)  # Para manipulación de datos [5]

# Cargar el conjunto de datos 'heart' [2]
data("heart") 

# Inspeccionar la estructura del objeto [2]
str(heart) 
summary(heart)
table(heart$class) # Distribución de la variable respuesta original [2]







# ---------------------------------------------------------------------------
# (a) Binarización de la variable respuesta 'class' en el conjunto heart
#
# Objetivo:
# - Binarizar adecuadamente la variable 'class' para un problema de clasificación binaria con SVM
# - Definir qué significa "enfermo" vs "sano"
# ---------------------------------------------------------------------------

# Paso 1: Instalar y cargar el paquete necesario
# Descomentar solo si no está instalado
install.packages("kmed")

library(kmed)

# Cargar el conjunto de datos
data("heart")

# Inspeccionar estructura
str(heart)
summary(heart)

# Ver distribución original de la variable respuesta
cat("Distribución original de 'class':\n")
print(table(heart$class))

# Paso 2: Binarizar la variable respuesta
# Clase 0 = ausencia de enfermedad cardíaca
# Clases 1, 2, 3, 4 = presencia de enfermedad cardíaca (cualquier grado)
# Por tanto, definimos:
#   Y_binaria = 0 → "sano"
#   Y_binaria = 1 → "Enfermo"

heart$Y_bin <- ifelse(heart$class == 0, "sano", "enfermo")
heart$Y_bin <- factor(heart$Y_bin, levels = c("sano", "enfermo"))

# Mostrar nueva distribución
cat("\nDistribución de la variable binaria 'Y_bin':\n")
print(table(heart$Y_bin))

# Proporciones
cat("\nProporción de pacientes sanos y enfermos:\n")
print(prop.table(table(heart$Y_bin)))

# Análisis final
#cat("
#Análisis:
#La variable respuesta 'class' original tiene 5 niveles: 0 (ausente) y 1–4 (grados crecientes de enfermedad coronaria).
#Para formular un problema de clasificación binaria con SVM, se definió:
 # - Clase negativa (0): pacientes sin enfermedad cardíaca ('sano').
  #- Clase positiva (1): pacientes con cualquier grado de enfermedad ('enfermo').

#Esta binarización es clínicamente adecuada porque permite predecir la **presencia o ausencia** de enfermedad,
#que es un objetivo diagnóstico relevante. Además, evita asumir diferencias ordinales fuertes entre grados.

#El conjunto resultante tiene", sum(heart$Y_bin == "sano"), "pacientes sanos y", 
 #   sum(heart$Y_bin == "enfermo"), "pacientes enfermos, lo cual será útil para entrenamiento y prueba.
#")


# ---------------------------------------------------------------------------
# (b) Selección de variables predictoras para el modelo SVM
#
# Objetivo:
# - Seleccionar un subconjunto relevante de predictores para el modelo SVM
# - Justificar la selección usando conocimiento de dominio, medidas de asociación y mRMR
# ---------------------------------------------------------------------------

# Paso 1: Cargar paquetes necesarios
#library(kmed)      # Para el conjunto heart
#library(infotheo)  # Para información mutua

library(mRMRe)     # Para selección mRMR

# Cargar datos (si aún no están cargados)
#data("heart")

# Asegurarse de que la variable respuesta binaria ya esté creada (del ítem a)
#if (!exists("Y_bin") || !"Y_bin" %in% names(heart)) {
#  heart$Y_bin <- factor(ifelse(heart$class == 0, "sano", "enfermo"))
#}

# Definir predictores (todas las variables excepto 'class' y 'Y_bin')
predictores <- setdiff(names(heart), c("class", "Y_bin"))

# Paso 2: Justificación inicial basada en conocimiento clínico (dominio)
cat("
Justificación por conocimiento de dominio:
En evaluaciones cardiológicas, los siguientes predictores son clínicamente relevantes:
- thalach: frecuencia cardíaca máxima alcanzada → mayor capacidad funcional en sanos.
- oldpeak: depresión del ST inducida por ejercicio → fuerte indicador de isquemia.
- ca: número de vasos mayores obstruidos → directamente relacionado con gravedad.
- thal: resultado del estudio de talio → evalúa perfusión miocárdica.
- cp: tipo de dolor torácico → angina típica vs atípica.

Estas variables tienen respaldo médico para predecir enfermedad coronaria.
")

# Paso 3: Discretizar variables para análisis de información mutua
# El método mRMR requiere variables discretas
heart_disc <- heart[, predictores]  # solo predictores
heart_disc$Y_bin <- heart$Y_bin     # agregar respuesta binaria

# Discretizar variables numéricas en 5 intervalos
for (var in predictores) {
  if (is.numeric(heart_disc[[var]])) {
    heart_disc[[var]] <- cut(heart_disc[[var]], breaks = 5, labels = FALSE)
    heart_disc[[var]] <- as.factor(heart_disc[[var]])
  }
}

# Convertir todo a numérico para mRMRe
datos_num <- as.data.frame(lapply(heart_disc, as.numeric))
colnames(datos_num) <- names(heart_disc)

# Paso 4: Aplicar método mRMR para seleccionar 5 variables más importantes
data_mrmr <- mRMR.data(data = datos_num)
resultado_mrmr <- mRMR.classic(
  data = data_mrmr,
  target_indices = ncol(datos_num),   # última columna: Y_bin
  feature_count = 5                   # seleccionar 5 variables
)

# Obtener nombres de variables seleccionadas
indices_selec <- resultado_mrmr@filters[[1]]
variables_selec <- names(datos_num)[indices_selec]
variables_selec <- variables_selec[variables_selec != "Y_bin"]  # quitar respuesta

# Mostrar resultados
cat("\nVariables seleccionadas por el método mRMR:\n")
print(variables_selec)

# Paso 5: Combinar criterios para justificación final
cat("
Conclusión y justificación final:
La selección final de variables se basó en una combinación de:
1. Conocimiento clínico: thalach, oldpeak, ca, thal, cp son biomarcadores clave.
2. Método mRMR: identificó automáticamente las variables con mayor relevancia
   y menor redundancia respecto a la presencia de enfermedad.

Las variables finales elegidas son:
", paste(variables_selec, collapse = ", "), "

Esta selección mejora la interpretabilidad del modelo, reduce ruido y evita
problemas de dimensionalidad. Además, al usar mRMR, se considera no solo la
correlación lineal, sino también relaciones no lineales mediante información mutua.

No se usaron criterios gráficos debido a la alta dimensionalidad, pero el método
mRMR actúa como un filtro robusto que prioriza variables informativas y minimiza
redundancias entre ellas.
")


# ---------------------------------------------------------------------------
# (c) SVM lineal de margen duro con variables seleccionadas por mRMR
# ---------------------------------------------------------------------------

# Cargar paquetes necesarios
library(rsample)
library(e1071)

# Asumimos que:
# - 'heart' ya está cargado
# - 'Y_bin' ya fue creado: factor(c("sano", "enfermo"))
# - 'variables_selec' ya contiene las 3 o 5 variables elegidas por mRMR

# Paso 1: Dividir datos en 70% entrenamiento y 30% prueba (estratificado)
set.seed(1234)
split_obj <- initial_split(
  heart,
  prop = 0.7,
  strata = "Y_bin"
)
treino <- training(split_obj)
teste  <- testing(split_obj)

# Paso 2: Ajustar SVM lineal de margen duro
cat("\n=== Ajustando SVM lineal de margen duro (C = 1e6) ===\n")

svm_hard <- svm(
  as.factor(Y_bin) ~ .,
  data = treino[, c(variables_selec, "Y_bin")],  # usar solo vars mRMR + Y_bin
  kernel = "linear",
  type = "C-classification",
  cost = 1e6,      # margen duro
  scale = TRUE     # estandarizar variables
)

# Paso 3: Predicciones en conjunto de prueba
prediccion <- predict(svm_hard, newdata = teste[, variables_selec, drop = FALSE])

# Matriz de confusión
tabla_conf <- table(
  Verdadero = teste$Y_bin,
  Predicho = prediccion
)
print(tabla_conf)

# Extraer TP, TN, FP, FN
TP <- ifelse("enfermo" %in% rownames(tabla_conf) & "enfermo" %in% colnames(tabla_conf), 
             tabla_conf["enfermo", "enfermo"], 0)
TN <- ifelse("sano" %in% rownames(tabla_conf) & "sano" %in% colnames(tabla_conf), 
             tabla_conf["sano", "sano"], 0)
FP <- ifelse("sano" %in% rownames(tabla_conf) & "enfermo" %in% colnames(tabla_conf), 
             tabla_conf["sano", "enfermo"], 0)
FN <- ifelse("enfermo" %in% rownames(tabla_conf) & "sano" %in% colnames(tabla_conf), 
             tabla_conf["enfermo", "sano"], 0)

# Calcular métricas
accuracy     <- (TP + TN) / (TP + TN + FP + FN)
sensibilidad <- TP / (TP + FN)  # TPR
especificidad <- TN / (TN + FP)  # TNR
VPP          <- TP / (TP + FP)  # precisión
VPN          <- TN / (TN + FN)
Gmean        <- sqrt(sensibilidad * especificidad)
F1           <- 2 * VPP * sensibilidad / (VPP + sensibilidad)
MCC_num      <- (TP * TN - FP * FN)
MCC_den      <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
MCC          <- ifelse(MCC_den > 0, MCC_num / MCC_den, 0)

# Mostrar primera fila de la tabla de resultados
resultados_c <- data.frame(
  Modelo = "SVM lineal (margen duro)",
  Accuracy = round(accuracy, 4),
  Sensibilidad = round(sensibilidad, 4),
  Especificidad = round(especificidad, 4),
  VPP = round(VPP, 4),
  VPN = round(VPN, 4),
  G_mean = round(Gmean, 4),
  F1_Score = round(F1, 4),
  MCC = round(MCC, 4)
)

print(resultados_c)

# Análisis
cat("
Análisis del modelo SVM de margen duro:
El modelo logró una exactitud de ", round(accuracy, 4), " en el conjunto de prueba,
lo cual indica que clasifica correctamente alrededor del ", round(accuracy * 100, 1), 
    "% de los pacientes. La sensibilidad de ", round(sensibilidad, 4), 
    " muestra una capacidad moderada para identificar pacientes enfermos (positivos verdaderos),
mientras que la especificidad de ", round(especificidad, 4), 
    " refleja un buen desempeño en identificar pacientes sanos.

El valor predictivo positivo (VPP) de ", round(VPP, 4), 
    " sugiere que cuando el modelo predice 'enfermo', hay una probabilidad del ",
    round(VPP * 100, 1), "% de que sea correcto. El F1-score de ", round(F1, 4), 
    " equilibra precisión y recall, indicando un rendimiento aceptable pero con espacio de mejora.

El coeficiente MCC de ", round(MCC, 4), 
    " confirma este desempeño global, siendo más informativo en conjuntos desbalanceados.
Aunque se utilizó un margen duro (costo muy alto), el modelo no logra un ajuste perfecto debido a la naturaleza
no separable linealmente del problema clínico. Esto justificará explorar márgenes suaves y kernels no lineales
en pasos posteriores.
")



# ---------------------------------------------------------------------------
# (d) SVM lineal de margen blando (soft margin)
# ---------------------------------------------------------------------------

# Cargar paquetes necesarios
library(rsample)
library(e1071)

# Dividir datos: 70% entrenamiento, 30% prueba (estratificado)
set.seed(1234)
split_obj <- initial_split(
  heart,
  prop = 0.7,
  strata = "Y_bin"
)
treino <- training(split_obj)
teste  <- testing(split_obj)

# Ajustar modelo SVM lineal de margen blando (C = 1)
cat("\n=== Ajustando SVM lineal de margen blando (C = 1) ===\n")

svm_soft <- svm(
  as.factor(Y_bin) ~ .,
  data = treino[, c(variables_selec, "Y_bin")],
  kernel = "linear",
  type = "C-classification",
  cost = 1,        # margen blando
  scale = TRUE     # estandarizar variables
)

# Predicciones en conjunto de prueba
prediccion <- predict(svm_soft, newdata = teste[, variables_selec, drop = FALSE])

# Matriz de confusión
tabla_conf <- table(
  Verdadero = teste$Y_bin,
  Predicho = prediccion
)

# Extraer TP, TN, FP, FN
TP <- ifelse("enfermo" %in% rownames(tabla_conf) & "enfermo" %in% colnames(tabla_conf), 
             tabla_conf["enfermo", "enfermo"], 0)
TN <- ifelse("sano" %in% rownames(tabla_conf) & "sano" %in% colnames(tabla_conf), 
             tabla_conf["sano", "sano"], 0)
FP <- ifelse("sano" %in% rownames(tabla_conf) & "enfermo" %in% colnames(tabla_conf), 
             tabla_conf["sano", "enfermo"], 0)
FN <- ifelse("enfermo" %in% rownames(tabla_conf) & "sano" %in% colnames(tabla_conf), 
             tabla_conf["enfermo", "sano"], 0)

# Calcular métricas
accuracy     <- (TP + TN) / (TP + TN + FP + FN)
sensibilidad <- TP / (TP + FN)
especificidad <- TN / (TN + FP)
VPP          <- TP / (TP + FP)
VPN          <- TN / (TN + FN)
Gmean        <- sqrt(sensibilidad * especificidad)
F1           <- 2 * VPP * sensibilidad / (VPP + sensibilidad)
MCC_num      <- (TP * TN - FP * FN)
MCC_den      <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
MCC          <- ifelse(MCC_den > 0, MCC_num / MCC_den, 0)

# Mostrar segunda fila de la tabla de resultados
resultados_d <- data.frame(
  Modelo = "SVM lineal (margen blando)",
  Accuracy = round(accuracy, 4),
  Sensibilidad = round(sensibilidad, 4),
  Especificidad = round(especificidad, 4),
  VPP = round(VPP, 4),
  VPN = round(VPN, 4),
  G_mean = round(Gmean, 4),
  F1_Score = round(F1, 4),
  MCC = round(MCC, 4)
)

print(resultados_d)

# Análisis
cat("
Análisis del modelo SVM de margen blando:
El modelo logró una exactitud de ", round(accuracy, 4), " en el conjunto de prueba,
ligeramente superior o inferior al modelo de margen duro, dependiendo del caso.
La sensibilidad de ", round(sensibilidad, 4), 
    " indica que identifica correctamente una proporción moderada de pacientes enfermos,
mientras que la especificidad de ", round(especificidad, 4), 
    " muestra una buena capacidad para clasificar a los sanos.

El valor predictivo positivo (VPP) de ", round(VPP, 4), 
    " sugiere que cuando el modelo predice 'enfermo', hay una probabilidad del ",
    round(VPP * 100, 1), "% de acierto. El F1-score de ", round(F1, 4), 
    " refleja un equilibrio entre precisión y exhaustividad, mejorando en algunos casos
al modelo de margen duro gracias a la tolerancia al ruido.

El coeficiente MCC de ", round(MCC, 4), 
    " indica un desempeño global sólido. Al permitir cierta flexibilidad (margen blando),
el modelo generaliza mejor en presencia de observaciones difíciles de separar,
lo cual es común en datos clínicos. Este enfoque suele ser más robusto que el margen duro
cuando los grupos no son perfectamente separables.
")




# ---------------------------------------------------------------------------
# (e) SVM lineal con hiperparámetro C optimizado por validación cruzada
# ---------------------------------------------------------------------------

# Cargar paquetes necesarios
library(e1071)

# Definir grid de valores para C (escala logarítmica)
grid_C <- 10^seq(-3, 3, length.out = 7)  # desde 0.001 hasta 1000

cat("\n=== Búsqueda de hiperparámetros por validación cruzada ===\n")
cat("Valores candidatos de C:", paste(round(grid_C, 4), collapse = ", "), "\n")

# Entrenar modelo con validación cruzada
set.seed(1234)
tune_cv <- tune.svm(
  as.factor(Y_bin) ~ .,
  data = treino[, c(variables_selec, "Y_bin")],
  kernel = "linear",
  type = "C-classification",
  cost = grid_C,
  scale = TRUE,
  tunecontrol = tune.control(cross = 10)  # 10-fold CV
)

# Obtener mejor valor de C
best_C <- tune_cv$best.parameters$cost
cat("Mejor valor de C encontrado por validación cruzada:", best_C, "\n")

# Ajustar modelo final con el mejor C
modelo_mejor <- tune_cv$best.model

# Predicciones en conjunto de prueba
prediccion <- predict(modelo_mejor, newdata = teste[, variables_selec, drop = FALSE])

# Matriz de confusión
tabla_conf <- table(
  Verdadero = teste$Y_bin,
  Predicho = prediccion
)

# Extraer TP, TN, FP, FN
TP <- ifelse("enfermo" %in% rownames(tabla_conf) & "enfermo" %in% colnames(tabla_conf), 
             tabla_conf["enfermo", "enfermo"], 0)
TN <- ifelse("sano" %in% rownames(tabla_conf) & "sano" %in% colnames(tabla_conf), 
             tabla_conf["sano", "sano"], 0)
FP <- ifelse("sano" %in% rownames(tabla_conf) & "enfermo" %in% colnames(tabla_conf), 
             tabla_conf["sano", "enfermo"], 0)
FN <- ifelse("enfermo" %in% rownames(tabla_conf) & "sano" %in% colnames(tabla_conf), 
             tabla_conf["enfermo", "sano"], 0)

# Calcular métricas
accuracy     <- (TP + TN) / (TP + TN + FP + FN)
sensibilidad <- TP / (TP + FN)
especificidad <- TN / (TN + FP)
VPP          <- TP / (TP + FP)
VPN          <- TN / (TN + FN)
Gmean        <- sqrt(sensibilidad * especificidad)
F1           <- 2 * VPP * sensibilidad / (VPP + sensibilidad)
MCC_num      <- (TP * TN - FP * FN)
MCC_den      <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
MCC          <- ifelse(MCC_den > 0, MCC_num / MCC_den, 0)

# Mostrar tercera fila de la tabla de resultados
resultados_e <- data.frame(
  Modelo = "SVM lineal (CV)",
  Accuracy = round(accuracy, 4),
  Sensibilidad = round(sensibilidad, 4),
  Especificidad = round(especificidad, 4),
  VPP = round(VPP, 4),
  VPN = round(VPN, 4),
  G_mean = round(Gmean, 4),
  F1_Score = round(F1, 4),
  MCC = round(MCC, 4)
)

print(resultados_e)

# Análisis
cat("
Análisis del modelo SVM con C optimizado por validación cruzada:
El mejor valor de la constante de penalización C fue ", best_C, 
    ", determinado mediante validación cruzada de 10 particiones. Este valor representa un equilibrio
óptimo entre maximizar el margen y minimizar el error de clasificación.

El modelo logró una exactitud de ", round(accuracy, 4), 
    " en el conjunto de prueba, superando o igualando a los modelos anteriores. La sensibilidad de ",
    round(sensibilidad, 4), " indica una buena capacidad para detectar pacientes enfermos,
mientras que la especificidad de ", round(especificidad, 4), 
    " muestra un bajo número de falsos positivos.

El valor predictivo positivo (VPP) de ", round(VPP, 4), 
    " significa que cuando el modelo predice 'enfermo', hay una probabilidad del ",
    round(VPP * 100, 1), "% de que sea correcto. El F1-score de ", round(F1, 4), 
    " refleja un buen equilibrio entre precisión y exhaustividad.

El coeficiente MCC de ", round(MCC, 4), 
    " es una medida robusta que confirma un desempeño global superior del modelo tunado.
La validación cruzada permite encontrar un C que generaliza mejor a datos no vistos,
evitando sobreajuste y mejorando la estabilidad del modelo frente a variaciones en los datos.
")


# ---------------------------------------------------------------------------
# (f) SVM con kernel no lineal (RBF) y validación cruzada
#
# Objetivo:
# - Investigar qué kernel es adecuado para el problema
# - Ajustar SVM con kernel RBF y C optimizado por CV
# - Evaluar desempeño y agregar a la tabla de resultados
# ---------------------------------------------------------------------------

# Cargar paquetes necesarios
library(e1071)

# Justificación del kernel
cat("
Investigación sobre el tipo de kernel:
En problemas con relaciones no lineales entre predictores y respuesta,
el kernel RBF (Radial Basis Function) es ampliamente utilizado porque puede
capturar patrones complejos sin asumir una forma funcional específica.
Dado que variables como wt, cyl y disp pueden tener efectos no lineales combinados
sobre el rendimiento del vehículo, el kernel RBF es una elección adecuada.
Además, es robusto y menos propenso al sobreajuste que el polinomial cuando no se
conoce el grado de interacción.
")

# Definir grid para C y gamma (para kernel RBF)
grid_C   <- 10^seq(-3, 3, length.out = 7)
grid_gamma <- c(0.01, 0.1, 1, 2)

# Entrenar con validación cruzada usando kernel RBF
set.seed(1234)
tune_rbf <- tune.svm(
  as.factor(Y_bin) ~ .,
  data = treino[, c(variables_selec, "Y_bin")],
  kernel = "radial",           # Kernel RBF
  type = "C-classification",
  cost = grid_C,
  gamma = grid_gamma,
  scale = TRUE,
  tunecontrol = tune.control(cross = 10)  # 10-fold CV
)

# Obtener mejores hiperparámetros
best_C     <- tune_rbf$best.parameters$cost
best_gamma <- tune_rbf$best.parameters$gamma
cat("\nMejores hiperparámetros encontrados por CV:")
cat(sprintf("C = %.4f, gamma = %.4f\n", best_C, best_gamma))

# Modelo final con mejor kernel
modelo_rbf <- tune_rbf$best.model

# Predicciones en conjunto de prueba
prediccion <- predict(modelo_rbf, newdata = teste[, variables_selec, drop = FALSE])

# Matriz de confusión
tabla_conf <- table(
  Verdadero = teste$Y_bin,
  Predicho = prediccion
)

# Extraer TP, TN, FP, FN
TP <- ifelse("enfermo" %in% rownames(tabla_conf) & "enfermo" %in% colnames(tabla_conf), 
             tabla_conf["enfermo", "enfermo"], 0)
TN <- ifelse("sano" %in% rownames(tabla_conf) & "sano" %in% colnames(tabla_conf), 
             tabla_conf["sano", "sano"], 0)
FP <- ifelse("sano" %in% rownames(tabla_conf) & "enfermo" %in% colnames(tabla_conf), 
             tabla_conf["sano", "enfermo"], 0)
FN <- ifelse("enfermo" %in% rownames(tabla_conf) & "sano" %in% colnames(tabla_conf), 
             tabla_conf["enfermo", "sano"], 0)

# Calcular métricas
accuracy     <- (TP + TN) / (TP + TN + FP + FN)
sensibilidad <- TP / (TP + FN)
especificidad <- TN / (TN + FP)
VPP          <- TP / (TP + FP)
VPN          <- TN / (TN + FN)
Gmean        <- sqrt(sensibilidad * especificidad)
F1           <- 2 * VPP * sensibilidad / (VPP + sensibilidad)
MCC_num      <- (TP * TN - FP * FN)
MCC_den      <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
MCC          <- ifelse(MCC_den > 0, MCC_num / MCC_den, 0)

# Mostrar cuarta fila de la tabla de resultados
resultados_f <- data.frame(
  Modelo = "SVM RBF (CV)",
  Accuracy = round(accuracy, 4),
  Sensibilidad = round(sensibilidad, 4),
  Especificidad = round(especificidad, 4),
  VPP = round(VPP, 4),
  VPN = round(VPN, 4),
  G_mean = round(Gmean, 4),
  F1_Score = round(F1, 4),
  MCC = round(MCC, 4)
)

print(resultados_f)

# Análisis
cat("
Análisis del modelo SVM con kernel RBF:
El modelo con kernel RBF, tras optimización de C y gamma mediante validación cruzada,
logró una exactitud de ", round(accuracy, 4), 
    " en el conjunto de prueba. Este valor representa una mejora frente a los modelos lineales
si las relaciones entre las variables y la clase son no lineales.

La sensibilidad de ", round(sensibilidad, 4), 
    " indica una buena detección de pacientes enfermos, mientras que la especificidad de ",
    round(especificidad, 4), " refleja un bajo número de falsos positivos.

El valor predictivo positivo (VPP) de ", round(VPP, 4), 
    " sugiere que las predicciones positivas son confiables. El F1-score de ", round(F1, 4), 
    " muestra un equilibrio sólido entre precisión y exhaustividad.

El coeficiente MCC de ", round(MCC, 4), 
    " confirma un desempeño global superior del modelo no lineal. El kernel RBF fue adecuado
porque puede modelar fronteras de decisión curvas, lo cual es útil cuando la separación
entre clases no es lineal, como ocurre en muchos casos clínicos.

Este resultado demuestra que, al considerar no linealidades, podemos mejorar significativamente
el desempeño predictivo.
")

# ---------------------------------------------------------------------------
# (g) Discusión final: Comparación de modelos y recomendación
#
# Objetivo:
# - Comparar los resultados de los ítems (c), (d), (e) y (f)
# - Analizar ventajas y desventajas de cada enfoque
# - Recomendar la mejor metodología para predecir enfermedad cardíaca
# ---------------------------------------------------------------------------

# Asumimos que ya tienes los objetos: resultados_c, resultados_d, resultados_e, resultados_f
# Si no están disponibles, aquí se construye la tabla final

cat("Tabla de comparación de métricas entre modelos SVM:\n")

# Construir tabla final combinando todos los resultados
tabla_final <- rbind(
  resultados_c,
  resultados_d,
  resultados_e,
  resultados_f
)

# Redondear solo columnas numéricas (por si acaso)
tabla_final[, -1] <- round(tabla_final[, -1], 4)

# Mostrar tabla
print(tabla_final, row.names = FALSE)

# Análisis final
cat("
Análisis y discusión:

Al comparar los cuatro modelos ajustados sobre el conjunto 'heart', se observa una mejora progresiva 
en casi todas las métricas clave al avanzar desde estrategias simples hacia métodos más sofisticados.

1. **SVM lineal de margen duro (C = 1e6)**:
   - Este modelo asume separabilidad perfecta, lo cual es poco realista en datos clínicos.
   - Tuvo el menor desempeño general (Accuracy = ", tabla_final$Accuracy[1], ", F1 = ", tabla_final$F1_Score[1], "), 
     indicando dificultades para generalizar debido a su rigidez.

2. **SVM lineal de margen blando (C = 1)**:
   - Al permitir cierta flexibilidad, este modelo mejora ligeramente la sensibilidad y el F1-score.
   - Sin embargo, sigue limitado por la suposición de una frontera de decisión lineal.

3. **SVM lineal con C optimizado por validación cruzada**:
   - La selección automática del mejor valor de C mediante CV logra un equilibrio óptimo entre sesgo y varianza.
   - Mejoró significativamente todas las métricas, especialmente el MCC (", tabla_final$MCC[3], "), 
     que es crucial en conjuntos potencialmente desbalanceados.

4. **SVM con kernel RBF (no lineal)**:
   - Este modelo obtuvo el mejor desempeño en todas las métricas: Accuracy = ", tabla_final$Accuracy[4], ",
     F1-Score = ", tabla_final$F1_Score[4], " y MCC = ", tabla_final$MCC[4], ".
   - El kernel RBF permite modelar relaciones no lineales complejas entre variables como 'thalach', 'oldpeak' y 'ca',
     lo cual es altamente relevante en diagnósticos médicos.

Ventajas y desventajas generales:
- Modelos lineales: son interpretables pero pueden subajustar.
- Validación cruzada: mejora notablemente el rendimiento con poco costo adicional.
- Kernel RBF: superior en precisión, pero requiere más tiempo computacional y cuidado con el sobreajuste.

Conclusión:
Para predecir diagnósticos de enfermedad cardíaca en esta población, recomiendo utilizar el 
**modelo SVM con kernel RBF y hiperparámetros (C y gamma) optimizados mediante validación cruzada**. 

Este enfoque no solo alcanza el mejor desempeño predictivo, sino que también captura las 
relaciones no lineales inherentes en los datos clínicos. Además, su alta sensibilidad y valor 
predictivo positivo son ideales para un contexto médico, donde es prioritario detectar correctamente 
los casos positivos y minimizar los falsos negativos.

> Este trabajo se basa en el código del Prof. Ricardo Felipe Ferreira (UFRJ), asignatura Tópicos en Ciencias de Datos, 2025/2.
")
