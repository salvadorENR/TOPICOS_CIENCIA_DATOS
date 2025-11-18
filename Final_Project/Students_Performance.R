#################################################################################
## PROYECTO INTEGRADOR - CÓDIGO MAESTRO (FINAL + GRÁFICOS DE CORRELACIÓN)     ##
## -------------------------------------------------------------------------- ##
## - Incluye análisis de correlación (Point-Biserial) y gráficos automáticos  ##
#################################################################################

# ==============================================================================
# 0. CONFIGURACIÓN Y LIMPIEZA
# ==============================================================================
try(dev.off(), silent=TRUE)
rm(list = ls())
while(sink.number() > 0) { sink() }

# Instalar paquetes necesarios
if(!require(pacman)) install.packages("pacman")
pacman::p_load(readr, dplyr, ggplot2, rsample, recipes, themis, 
               caret, e1071, nnet, class, MASS, mRMRe, FNN, reshape2, gridExtra)

# ==============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ==============================================================================
file_path <- "Final_Project/student-mat.csv"

if(file.exists(file_path)) {
  datos <- read.csv(file_path, sep = ";", stringsAsFactors = TRUE)
  cat("¡Datos cargados correctamente desde:", file_path, "\n")
} else {
  stop("ERROR: No se encuentra 'Final_Project/student-mat.csv'.")
}

# Crear Target y limpiar
datos <- datos %>%
  mutate(Pass = ifelse(G3 >= 10, "yes", "no")) %>%
  mutate(Pass = factor(Pass, levels = c("yes", "no"))) %>%
  dplyr::select(-G3, -G1, -G2) 

cat("Variable 'Pass' creada. Distribución:\n")
print(table(datos$Pass))

# ==============================================================================
# 2. SELECCIÓN DE VARIABLES (ESTRATEGIA DOBLE)
# ==============================================================================

# INICIO DE GRABACIÓN EN ARCHIVO
sink("Final_Project/Resultados_Proyecto_Final.txt", split = TRUE)

cat("==========================================================\n")
cat(" REPORTE AVANZADO: ANÁLISIS COMPLETO \n")
cat("==========================================================\n\n")

cat("[1] Selección por AIC (Stepwise)...\n")
null_model <- glm(Pass ~ 1, data = datos, family = binomial)
full_model <- glm(Pass ~ ., data = datos, family = binomial)
step_model <- stepAIC(null_model, scope = list(lower = null_model, upper = full_model), 
                      direction = "both", trace = 0)
vars_aic <- attr(terms(step_model), "term.labels")
cat("Variables AIC:", paste(vars_aic, collapse = ", "), "\n\n")

cat("[2] Selección por mRMR / Experto...\n")
vars_mrmr <- tryCatch({
  datos_num <- datos
  datos_num[] <- lapply(datos_num, as.numeric)
  dd <- mRMR.data(data = data.frame(datos_num))
  res <- mRMR.classic(data = dd, target_indices = ncol(datos_num), feature_count = 10)
  v <- names(datos)[solutions(res)[[1]]]
  setdiff(v, c("Pass", "class"))
}, error = function(e) {
  return(c("failures", "absences", "Medu", "Fedu", "goout", 
           "Walc", "studytime", "internet", "freetime", "age"))
})
cat("Variables mRMR:", paste(vars_mrmr, collapse = ", "), "\n\n")

cat("[3] Intersección de variables...\n")
common_vars <- intersect(vars_aic, vars_mrmr)
if(length(common_vars) < 3) {
  final_vars <- unique(c(vars_aic, vars_mrmr))
} else {
  final_vars <- common_vars
}

cat("=== VARIABLES DEFINITIVAS ===\n")
print(final_vars)
cat("\n")

df_model <- datos[, c(final_vars, "Pass")]

# ==============================================================================
# NUEVO: GRÁFICOS DE CORRELACIÓN
# ==============================================================================
cat("Generando gráficos de correlación...\n")

# Preparar datos numéricos para correlación
df_cor <- df_model
df_cor$Pass_Num <- ifelse(df_cor$Pass == "yes", 1, 0) # Convertir Pass a numérico (1=Aprobado)
df_cor$Pass <- NULL

# Convertir factores a números para la matriz de correlación
df_cor[] <- lapply(df_cor, as.numeric)

# Calcular matriz de correlación
cor_matrix <- cor(df_cor, method = "pearson") # Pearson funciona bien para punto-biserial aquí

# Guardar correlaciones con la variable respuesta
target_cor <- data.frame(Variable = names(cor_matrix["Pass_Num",]), 
                         Correlacion = cor_matrix["Pass_Num",])
target_cor <- target_cor[target_cor$Variable != "Pass_Num",]

# 1. Gráfico de Barras: Correlación con 'Pass'
p_cor <- ggplot(target_cor, aes(x = reorder(Variable, Correlacion), y = Correlacion, fill = Correlacion > 0)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Correlación con el Rendimiento (Pass)", 
       x = "Variables Predictoras", y = "Coeficiente de Correlación") +
  scale_fill_manual(values = c("#F44336", "#4CAF50"), name = "Dirección", labels = c("Negativa (Riesgo)", "Positiva (Éxito)"))

ggsave("Final_Project/Grafico_Correlacion_Barras.png", p_cor, width = 8, height = 6)

# 2. Heatmap (Mapa de Calor) de todas las variables
melted_cormat <- melt(cor_matrix)
p_heat <- ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1,1)) +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 10, hjust = 1)) +
  coord_fixed() +
  labs(title = "Matriz de Correlación entre Variables")

ggsave("Final_Project/Grafico_Heatmap.png", p_heat, width = 8, height = 8)

cat("¡Gráficos guardados en Final_Project/!\n")

# ==============================================================================
# 3. CONFIGURACIÓN DE MODELADO (RECETAS Y MÉTRICAS)
# ==============================================================================

set.seed(2025)
split <- initial_split(df_model, prop = 0.75, strata = Pass)
train_data <- training(split)
test_data  <- testing(split)

# Función ENN Manual
apply_enn <- function(data, k = 3) {
  X <- data %>% dplyr::select(-Pass)
  y <- data$Pass
  majority_class <- "yes"
  minority_class <- "no"
  majority_indices <- which(y == majority_class)
  minority_indices <- which(y == minority_class)
  X_majority <- X[majority_indices, ]
  y_majority <- y[majority_indices]
  knn_indices <- knnx.index(data.matrix(X), data.matrix(X_majority), k = k + 1)
  knn_indices <- knn_indices[, -1]
  keep_indices <- logical(length(majority_indices))
  for (i in 1:length(majority_indices)) {
    neighbor_classes <- y[knn_indices[i, ]]
    same_class_neighbors <- sum(neighbor_classes == y_majority[i])
    if (same_class_neighbors >= ceiling(k/2)) keep_indices[i] <- TRUE
  }
  final_indices <- c(majority_indices[keep_indices], minority_indices)
  return(data[final_indices, ])
}

rec_base <- recipe(Pass ~ ., data = train_data) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

prep_base <- prep(rec_base)
train_base <- bake(prep_base, new_data = NULL)
test_proc <- bake(prep_base, new_data = test_data)

cat("Aplicando técnicas de balanceo...\n")
train_desbalanceado <- train_base
rec_smote <- rec_base %>% step_smote(Pass, seed = 2025)
train_smote <- prep(rec_smote) %>% bake(new_data = NULL)
cat("Aplicando ENN...\n")
train_enn <- apply_enn(train_base)
cat("Aplicando SMOTE + ENN...\n")
train_smote_temp <- prep(rec_smote) %>% bake(new_data = NULL)
train_smote_enn <- apply_enn(train_smote_temp)

variants <- list(
  "Desbalanceado"  = train_desbalanceado,
  "Solo SMOTE"     = train_smote,
  "Solo ENN"       = train_enn,
  "SMOTE + ENN"    = train_smote_enn
)

calc_metrics <- function(preds, actual) {
  preds <- factor(preds, levels = c("yes", "no"))
  actual <- factor(actual, levels = c("yes", "no"))
  cm <- table(Predicted = preds, Actual = actual)
  TP <- cm["yes","yes"]; TN <- cm["no","no"]
  FP <- cm["yes","no"];  FN <- cm["no","yes"]
  Sens <- TP/(TP+FN); Spec <- TN/(TN+FP)
  if(is.na(Sens)) Sens <- 0; if(is.na(Spec)) Spec <- 0
  PPV <- TP / (TP + FP); NPV <- TN / (TN + FN)
  if(is.na(PPV)) PPV <- 0; if(is.na(NPV)) NPV <- 0
  F1 <- 2 * ((PPV * Sens) / (PPV + Sens))
  if(is.na(F1)) F1 <- 0
  MCC_num <- (TP * TN) - (FP * FN)
  MCC_den <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  if(MCC_den == 0) MCC <- 0 else MCC <- MCC_num / MCC_den
  GMean <- sqrt(Sens * Spec)
  Acc <- (TP + TN) / sum(cm)
  return(round(c(Acc=Acc, Sens=Sens, Spec=Spec, GMean=GMean, PPV=PPV, NPV=NPV, F1=F1, MCC=MCC), 3))
}

run_model_variants <- function(model_name) {
  cat(paste0("\n", strrep("-", 60), "\n"))
  cat(paste0("=== ", model_name, " ===\n"))
  cat(paste0(strrep("-", 60), "\n"))
  for(var_name in names(variants)) {
    train_set <- variants[[var_name]]
    if(model_name == "KNN (k=5)") {
      fit <- knn3(Pass ~ ., data = train_set, k = 5)
      pred <- predict(fit, test_proc, type="class")
    } else if(model_name == "SVM (Radial)") {
      fit <- svm(Pass ~ ., data = train_set, kernel="radial")
      pred <- predict(fit, test_proc)
    } else if(model_name == "Regresión Logística") {
      fit <- glm(Pass ~ ., data = train_set, family = binomial)
      prob <- predict(fit, test_proc, type="response")
      pred <- ifelse(prob > 0.5, "no", "yes")
    } else if(model_name == "Red Neuronal") {
      set.seed(123)
      fit <- nnet(Pass ~ ., data = train_set, size = 5, trace = FALSE)
      pred <- predict(fit, test_proc, type="class")
    }
    cat(sprintf(">> %-15s: ", var_name))
    metrics <- calc_metrics(pred, test_proc$Pass)
    print(metrics)
  }
}

cat("INICIANDO EJECUCIÓN DE MODELOS...\n")
cat("Fecha y hora:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

run_model_variants("KNN (k=5)")
run_model_variants("SVM (Radial)")
run_model_variants("Regresión Logística")
run_model_variants("Red Neuronal")

cat("\n==========================================================\n")
cat(" FIN. Archivo guardado en: Final_Project/Resultados_Proyecto_Final.txt \n")
cat("==========================================================\n")
sink()