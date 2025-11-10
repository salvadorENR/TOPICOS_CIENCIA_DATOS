#################################################################################
## Códigos - Visualización: Boxplot + Dispersión                              ##
## Asignatura: Tópicos en Ciencia de Datos                                    ##                                   ##
#################################################################################

############################################################
## 0) Carga del conjunto mtcars tal como pide el profe
############################################################
library(datasets)
data("mtcars")
mtg <- mtcars  # renombrar a 'mtg' para este estudio

############################################################
## 1) Paquetes para graficar y combinar paneles
############################################################
library(ggplot2)
library(dplyr)
library(patchwork)  # para unir múltiples gráficos en una grilla

############################################################
## 2) Definir qué predictores son categóricos y cuáles numéricos
##    (mantener 'carb' como dispersión, es decir, numérico)
############################################################
predictores <- setdiff(names(mtg), "mpg")

cat_vars <- c("am", "vs", "cyl", "gear")        # tratar como categóricas
num_vars <- setdiff(predictores, cat_vars)      # el resto numéricas (incluye 'carb')

############################################################
## 3) Función que crea un gráfico por predictor:
##    - Si es categórico: boxplot + jitter
##    - Si es numérico : dispersión + recta lm
############################################################
graficar_pred <- function(var){
  if (var %in% cat_vars) {
    ggplot(mtg, aes(x = factor(.data[[var]]), y = mpg)) +
      geom_boxplot(width = 0.65, alpha = 0.7, outlier.alpha = 0) +
      geom_jitter(width = 0.15, alpha = 0.6, size = 1.5) +
      labs(title = var, x = var, y = "mpg") +
      theme_minimal(base_size = 11)
  } else {
    ggplot(mtg, aes(x = .data[[var]], y = mpg)) +
      geom_point(alpha = 0.7, size = 1.8) +
      geom_smooth(method = "lm", se = FALSE, linewidth = 0.7) +
      labs(title = var, x = var, y = "mpg") +
      theme_minimal(base_size = 11)
  }
}

############################################################
## 4) Construir la lista de gráficos y combinarlos
############################################################
plots <- lapply(predictores, graficar_pred)

# Armar una grilla (ajusta ncol a tu gusto)
p_final <- wrap_plots(plots, ncol = 3) +
  plot_annotation(
    title = "mtcars: mpg vs cada predictor",
    subtitle = "Boxplot + jitter (categóricas) / Dispersión + recta (numéricas; 'carb' incluido como dispersión)"
  )

p_final

############################################################
## (b) Cálculo de la relevancia entre mpg y cada predictor
############################################################

library(infotheo)

# Crear una versión discreta de mpg (para medidas de información)
mtg_discr <- mtg %>%
  mutate(mpg_disc = discretize(mpg, disc = "equalfreq", nbins = 4)) # discretiza mpg en 4 grupos

# Inicializar vector para almacenar resultados
relevancia <- data.frame(
  variable = setdiff(names(mtg), "mpg"),
  tipo = NA,
  medida = NA
)

# Calcular relevancia dependiendo del tipo de variable
for (var in relevancia$variable) {
  if (is.numeric(mtg[[var]])) {
    # Correlación absoluta (Pearson)
    relevancia[relevancia$variable == var, "tipo"] <- "Numérica"
    relevancia[relevancia$variable == var, "medida"] <- abs(cor(mtg$mpg, mtg[[var]], method = "pearson"))
  } else {
    # Información mutua (para categóricas)
    relevancia[relevancia$variable == var, "tipo"] <- "Categórica"
    relevancia[relevancia$variable == var, "medida"] <- mutinformation(mtg_discr$mpg_disc, mtg[[var]])
  }
}

# Ordenar de mayor a menor relevancia
relevancia <- relevancia %>% arrange(desc(medida))

relevancia

############################################################
## (c) División entrenamiento/prueba y modelo de regresión
############################################################

library(rsample)  # para dividir los datos

# Fijar semilla para reproducibilidad
set.seed(1234)

# Dividir 70% entrenamiento, 30% prueba
split_obj <- initial_split(mtg, prop = 0.7)

train_data <- training(split_obj)
test_data  <- testing(split_obj)

############################################################
## Ajuste del modelo de regresión lineal múltiple
############################################################

modelo_lm <- lm(mpg ~ ., data = train_data)  # todas las variables predictoras

# Resumen del modelo (opcional para interpretar)
summary(modelo_lm)

############################################################
## Predicciones sobre el conjunto de prueba
############################################################

predicciones <- predict(modelo_lm, newdata = test_data)

############################################################
## Cálculo del Error Cuadrático Medio (ECM)
############################################################

ecm <- mean((test_data$mpg - predicciones)^2)
cat("Error Cuadrático Medio (ECM) en el conjunto de prueba:", ecm, "\n")

############################################################
## Evaluación adicional (opcional)
############################################################

# También puedes calcular R² en prueba si deseas comparar desempeño
sst <- sum((test_data$mpg - mean(test_data$mpg))^2)
sse <- sum((test_data$mpg - predicciones)^2)
r2_prueba <- 1 - sse / sst
cat("R² en conjunto de prueba:", r2_prueba, "\n")

############################################################
## (d) Modelo de regresión lineal múltiple con 3 variables
##     seleccionadas por relevancia (mRMR)
############################################################

# Ajustar el modelo solo con las tres variables más relevantes
modelo_mrmr <- lm(mpg ~ wt + cyl + disp, data = train_data)

# Resumen del modelo (opcional para interpretación)
summary(modelo_mrmr)

############################################################
## Predicciones sobre el conjunto de prueba
############################################################

pred_mrmr <- predict(modelo_mrmr, newdata = test_data)

############################################################
## Cálculo del Error Cuadrático Medio (ECM)
############################################################

ecm_mrmr <- mean((test_data$mpg - pred_mrmr)^2)
cat("Error Cuadrático Medio (ECM) con mRMR (3 variables):", ecm_mrmr, "\n")

############################################################
## Evaluación adicional (opcional)
############################################################

# Cálculo del R² en el conjunto de prueba
sst_mrmr <- sum((test_data$mpg - mean(test_data$mpg))^2)
sse_mrmr <- sum((test_data$mpg - pred_mrmr)^2)
r2_mrmr <- 1 - sse_mrmr / sst_mrmr
cat("R² en conjunto de prueba (mRMR):", r2_mrmr, "\n")

############################################################
## (e) Comparación entre el modelo completo y el modelo mRMR
############################################################

# Crear un resumen comparativo
comparacion <- data.frame(
  Modelo = c("Completo (todas las variables)", "Reducido (3 variables mRMR)"),
  ECM = c(ecm, ecm_mrmr),
  R2 = c(r2_prueba, r2_mrmr)
)

cat("\n== Comparación de desempeño entre modelos ==\n")
print(comparacion)

############################################################
## Interpretación automática (básica)
############################################################

if (ecm_mrmr < ecm) {
  cat("\n➡️ El modelo reducido tiene un ECM menor, por lo tanto predice con menor error promedio.\n")
} else {
  cat("\n➡️ El modelo completo tiene un ECM menor, indicando mejor capacidad predictiva.\n")
}

if (r2_mrmr > r2_prueba) {
  cat("➡️ Además, el modelo reducido explica más varianza (R² mayor), siendo más eficiente.\n")
} else {
  cat("➡️ El modelo completo explica más varianza, aunque con mayor complejidad.\n")
}
