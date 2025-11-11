# Cargar paquetes requeridos
library(datasets)
library(ggplot2)
library(dplyr)
library(patchwork)  
library(infotheo)
library(rsample)

# Cargar la base de datos
data("mtcars")
mtg <- mtcars  

predictores <- setdiff(names(mtg), "mpg")

cat_vars <- c("am", "vs", "cyl", "gear")        # tratar como categóricas
num_vars <- setdiff(predictores, cat_vars)      # el resto numéricas 

# Función para graficar
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

# Crear lista de gráficas
plots <- lapply(predictores, graficar_pred)

# Mostrar las gráficas de 2 en 2 para mejor visibilidad
for (i in seq(1, length(plots), 2)) {
  patch <- wrap_plots(plots[i:min(i+1, length(plots))], ncol = 2)
  print(patch)
}

# Crear versión discreta de mpg (para medidas de información)
mtg_discr <- mtg %>%
  mutate(mpg_disc = discretize(mpg, disc = "equalfreq", nbins = 4))

# Calcular relevancia
relevancia <- data.frame(
  variable = setdiff(names(mtg), "mpg"),
  tipo = NA,
  medida = NA
)

for (var in relevancia$variable) {
  if (is.numeric(mtg[[var]])) {
    relevancia[relevancia$variable == var, "tipo"] <- "Numérica"
    relevancia[relevancia$variable == var, "medida"] <- abs(cor(mtg$mpg, mtg[[var]], method = "pearson"))
  } else {
    relevancia[relevancia$variable == var, "tipo"] <- "Categórica"
    relevancia[relevancia$variable == var, "medida"] <- mutinformation(mtg_discr$mpg_disc, mtg[[var]])
  }
}

relevancia <- relevancia %>% arrange(desc(medida))

# División y ajuste del modelo
set.seed(1234)
split_obj <- initial_split(mtg, prop = 0.7)
train_data <- training(split_obj)
test_data  <- testing(split_obj)

modelo_lm <- lm(mpg ~ ., data = train_data)

# Predicciones y métricas
predicciones <- predict(modelo_lm, newdata = test_data)
ecm <- mean((test_data$mpg - predicciones)^2)
sst <- sum((test_data$mpg - mean(test_data$mpg))^2)
sse <- sum((test_data$mpg - predicciones)^2)
r2_prueba <- 1 - sse / sst

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

# Ajuste con 3 variables más relevantes (wt, cyl, disp)
modelo_mrmr <- lm(mpg ~ wt + cyl + disp, data = train_data)

# Predicciones y métricas
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

# Comparación entre modelos
comparacion <- data.frame(
  Modelo = c("Completo (todas las variables)", "Reducido (3 variables mRMR)"),
  ECM    = c(round(ecm, 4), round(ecm_mrmr, 4)),
  `R^2`  = c(round(r2_prueba, 4), round(r2_mrmr, 4)),
  check.names = FALSE
)