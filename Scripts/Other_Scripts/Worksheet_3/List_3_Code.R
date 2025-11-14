#################################################################################
## Código - Lista de ejercicios 03                                            ##
## Asignatura: Tópicos en Ciencias de Datos                                   ##
## Estudiantes:								      ##
## - Víctor Mauricio Ochoa García					      ##
## - Salvador Enrique Rodríguez Hernández                                     ##
## Año: 2025/2                                                                ##
#################################################################################

#-------------------------------------------------------------------------------
# 1. CARGAR PAQUETES Y BASE DE DATOS
#-------------------------------------------------------------------------------
# (1) Cargar paquetes requeridos
library(datasets)   # conjuntos de ejemplo incluidos en R (ej. mtcars)
library(ggplot2)    # creación de gráficas 
library(dplyr)      # manipulación de datos 
library(patchwork)  # combinar varias gráficas 
library(infotheo)   # discretización e información mutua 
library(rsample)    # particionado entrenamiento/prueba 

# (2) Cargar la base de datos
data("mtcars")      # carga el conjunto mtcars del paquete base
mtg <- mtcars       # renombrar a 'mtg' para uso interno 

#-------------------------------------------------------------------------------
# 2. ANÁLISIS DESCRIPTIVO 
#-------------------------------------------------------------------------------
# predictores: todas las variables excepto la respuesta 'mpg'
predictores <- setdiff(names(mtg), "mpg")

# Definición explícita de variables categóricas
cat_vars <- c("am", "vs", "cyl", "gear")        # tratar como categóricas
num_vars <- setdiff(predictores, cat_vars)     # el resto se consideran numéricas

#-------------------------------------------------------------------------------
# Ajustar márgenes y cuadrícula para las gráficas
#-------------------------------------------------------------------------------
# Ajustar márgenes: abajo, izquierda, arriba, derecha (en líneas)
# Valores pequeños para ahorrar espacio y mostrar todas las gráficas en una cuadrícula
par(mfrow = c(3, 4),          # 3 filas, 4 columnas
    mar = c(2.5, 2.5, 1.5, 1), # márgenes internos pequeños
    oma = c(1, 1, 1, 1),       # márgenes externos uniformes
    mgp = c(1.5, 0.5, 0))      # posición de etiquetas y título

#-------------------------------------------------------------------------------
# Generar gráficas exploratorias para todas las variables predictoras
#-------------------------------------------------------------------------------
# Este bloque crea:
# - diagramas de caja (boxplots) para variables categóricas (mpg por categoría)
# - gráficas de dispersión con línea de regresión para variables numéricas
for (var in predictores) {
  x <- mtg[[var]]
  if (var %in% c("vs", "am", "cyl", "gear", "carb")) {
    # Variables categóricas: diagrama de caja de mpg por niveles de la variable
    boxplot(mpg ~ x, data = mtg,
            main = var,
            xlab = var,
            ylab = "mpg",
            col = "lightblue",
            border = "darkblue",
            cex.main = 0.9,
            cex.lab = 0.8,
            cex.axis = 0.7)
  } else {
    # Variables numéricas: gráfica de dispersión (mpg vs predictor) + línea de regresión
    plot(mpg ~ x, data = mtg,
         main = var,
         xlab = var,
         ylab = "mpg",
         pch = 16,
         col = rgb(0, 0, 1, 0.6),
         cex = 0.7,
         cex.main = 0.9,
         cex.lab = 0.8,
         cex.axis = 0.7)
    abline(lm(mpg ~ x, data = mtg), col = "red", lwd = 1) # Línea de regresión
  }
}

# Gráfica vacía para completar la cuadrícula (solo efecto visual)
plot(1, type = "n", axes = FALSE, xlab = "", ylab = "")
text(1, 1, " ", cex = 0.1)

#-------------------------------------------------------------------------------
# 3. CÁLCULO DE LA RELEVANCIA 
#-------------------------------------------------------------------------------

# Crear versión discreta de mpg (necesaria para calcular información mutua)
# discretize(..., disc = "equalfreq", nbins = 4) crea 4 grupos con frecuencias similares
mtg_discr <- mtg %>%
  mutate(mpg_disc = discretize(mpg, disc = "equalfreq", nbins = 4))

# Preparar tabla para almacenar medidas de relevancia de cada variable
relevancia <- data.frame(
  variable = setdiff(names(mtg), "mpg"),
  tipo = NA,
  medida = NA
)

# Para cada variable:
# - si es numérica: se calcula la correlación absoluta de Pearson con mpg
# - si es categórica: se calcula la información mutua con mpg discretizado
for (var in relevancia$variable) {
  if (is.numeric(mtg[[var]])) {
    relevancia[relevancia$variable == var, "tipo"] <- "Numérica"
    relevancia[relevancia$variable == var, "medida"] <- abs(cor(mtg$mpg, mtg[[var]], method = "pearson"))
  } else {
    relevancia[relevancia$variable == var, "tipo"] <- "Categórica"
    relevancia[relevancia$variable == var, "medida"] <- mutinformation(mtg_discr$mpg_disc, mtg[[var]])
  }
}

# Ordenar las variables de mayor a menor relevancia
relevancia <- relevancia %>% arrange(desc(medida))
# Mostrar la tabla de las relevancias de las variables
relevancia
#-------------------------------------------------------------------------------
#  4. AJUSTE DEL MODELO COMPLETO
#-------------------------------------------------------------------------------

# División entrenamiento/prueba
# Se usa set.seed para garantizar reproducibilidad de la partición
set.seed(1234)
split_obj <- initial_split(mtg, prop = 0.7)
train_data <- training(split_obj)
test_data  <- testing(split_obj)

# Ajustar modelo de regresión lineal múltiple con todas las variables predictoras
modelo_lm <- lm(mpg ~ ., data = train_data)
# Mostrar el modelo
summary(modelo_lm) 

# Predicciones y métricas de evaluación
predicciones <- predict(modelo_lm, newdata = test_data)
ecm <- mean((test_data$mpg - predicciones)^2)  # Error Cuadrático Medio (ECM)
sst <- sum((test_data$mpg - mean(test_data$mpg))^2)  # Suma total de cuadrados
sse <- sum((test_data$mpg - predicciones)^2)         # Suma de errores al cuadrado
r2_prueba <- 1 - sse / sst                            # R² en el conjunto de prueba

# Tabla de coeficientes (modelo completo)
sm <- summary(modelo_lm)
coef_full_tbl <- as.data.frame(sm$coefficients)
coef_full_tbl$Termino <- rownames(coef_full_tbl)
rownames(coef_full_tbl) <- NULL
names(coef_full_tbl) <- c("Estimacion","Error.Estd","t","p.value","Termino")
coef_full_tbl <- coef_full_tbl[, c("Termino","Estimacion","Error.Estd","t","p.value")]
coef_full_tbl[, -1] <- lapply(coef_full_tbl[, -1], function(x) round(x,6))

# Tabla de medidas de ajuste (modelo completo)
fit_full_tbl <- data.frame(
  `R^2 (train)`        = round(sm$r.squared, 4),
  `R^2 ajustado (train)` = round(sm$adj.r.squared, 4),
  `ECM (test)`         = round(ecm, 4),
  `R^2 (test)`         = round(r2_prueba, 4),
  `Sigma resid.`       = round(sm$sigma, 4),
  check.names = FALSE
)
fit_full_tbl

#-------------------------------------------------------------------------------
#  5. AJUSTE DEL MODELO CON LAS TRES VARIABLES MÁS RELEVANTES
#-------------------------------------------------------------------------------
# Ajuste con las 3 variables más relevantes según mRMR (wt, cyl, disp)
modelo_mrmr <- lm(mpg ~ wt + cyl + disp, data = train_data)
# Mostrar el modelo
summary(modelo_mrmr) 

# Predicciones y métricas del modelo reducido
pred_mrmr <- predict(modelo_mrmr, newdata = test_data)
ecm_mrmr  <- mean((test_data$mpg - pred_mrmr)^2)
sst_mrmr  <- sum((test_data$mpg - mean(test_data$mpg))^2)
sse_mrmr  <- sum((test_data$mpg - pred_mrmr)^2)
r2_mrmr   <- 1 - sse_mrmr / sst_mrmr

# Tabla de coeficientes (modelo mRMR)
sm2 <- summary(modelo_mrmr)
coef_mrmr_tbl <- as.data.frame(sm2$coefficients)
coef_mrmr_tbl$Termino <- rownames(coef_mrmr_tbl)
rownames(coef_mrmr_tbl) <- NULL
names(coef_mrmr_tbl) <- c("Estimacion","Error.Estd","t","p.value","Termino")
coef_mrmr_tbl <- coef_mrmr_tbl[, c("Termino","Estimacion","Error.Estd","t","p.value")]
coef_mrmr_tbl[, -1] <- lapply(coef_mrmr_tbl[, -1], function(x) round(x,6))

# Tabla de medidas de ajuste (modelo mRMR)
fit_mrmr_tbl <- data.frame(
  `R^2 (train)`          = round(sm2$r.squared, 4),
  `R^2 ajustado (train)` = round(sm2$adj.r.squared, 4),
  `ECM (test)`           = round(ecm_mrmr, 4),
  `R^2 (test)`           = round(r2_mrmr, 4),
  `Sigma resid.`         = round(sm2$sigma, 4),
  check.names = FALSE
)
fit_mrmr_tbl

#-------------------------------------------------------------------------------
#  6.  COMPARACIÓN DE LOS MODELOS
#-------------------------------------------------------------------------------

# Comparación entre modelos completo y reducido
comparacion <- data.frame(
  Modelo = c("Completo (todas las variables)", "Reducido (3 variables mRMR)"),
  ECM    = c(round(ecm, 4), round(ecm_mrmr, 4)),
  `R^2`  = c(round(r2_prueba, 4), round(r2_mrmr, 4)),
  check.names = FALSE
)
# Mostrar la comparación
comparacion



