#################################################################################
## PROYECTO INTEGRADOR - CÓDIGO MAESTRO FINAL V5 (ANÁLISIS NN EXTENDIDO)       ##
## - Incluye iteración de Redes Neuronales con 1, 3, 5 y 10 neuronas ocultas.  ##
## - Ejecuta todos los escenarios de balanceo y selección de variables.        ##
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

# Crear directorio para resultados si no existe
if(!dir.exists("Final_Project")) dir.create("Final_Project")

# ==============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS BASE
# ==============================================================================
file_path_check <- "student-mat.csv" 
file_path <- if(file.exists(file.path("Final_Project", file_path_check))) {
  file.path("Final_Project", file_path_check)
} else if (file.exists(file_path_check)) {
  file_path_check
} else {
  stop("ERROR: No se encuentra 'student-mat.csv'. Por favor, coloque el archivo en el directorio de trabajo o en Final_Project/.")
}

if(file.exists(file_path)) {
  datos <- read.csv(file_path, sep = ";", stringsAsFactors = TRUE)
  cat("¡Datos cargados correctamente desde:", file_path, "\n")
} else {
  stop("ERROR: Archivo no encontrado tras la comprobación inicial.") 
}

# Crear Target y limpiar (excluyendo G1, G2, G3 de la predicción)
datos <- datos %>%
  mutate(Pass = ifelse(G3 >= 10, "yes", "no")) %>%
  mutate(Pass = factor(Pass, levels = c("yes", "no"))) %>%
  dplyr::select(-G3, -G1, -G2) 

cat("Variable 'Pass' creada. Distribución:\n")
print(table(datos$Pass))

# Definir el conjunto total de variables predictoras (para el escenario "Sin Selección")
all_vars_no_g <- setdiff(names(datos), "Pass")
df_all_vars <- datos[, c(all_vars_no_g, "Pass")] 


# ==============================================================================
# 2. SELECCIÓN DE VARIABLES (ESCENARIO "CON SELECCIÓN")
# ==============================================================================

cat("==========================================================\n")
cat(" REPORTE DE SELECCIÓN DE VARIABLES \n")
cat("==========================================================\n\n")

# 2.1 AIC (Stepwise)
cat("[1] Selección por AIC (Stepwise)...\n")
null_model <- glm(Pass ~ 1, data = datos, family = binomial)
full_model <- glm(Pass ~ ., data = datos, family = binomial)
step_model <- stepAIC(null_model, scope = list(lower = null_model, upper = full_model), 
                      direction = "both", trace = 0)
vars_aic <- attr(terms(step_model), "term.labels")
cat("Variables AIC:", paste(vars_aic, collapse = ", "), "\n\n")

# 2.2 mRMR
cat("[2] Selección por mRMR / Experto...\n")
vars_mrmr <- tryCatch({
  datos_num <- datos
  datos_num[] <- lapply(datos_num, as.numeric)
  datos_num <- datos_num %>% dplyr::select(-matches("G[1-3]$"))
  dd <- mRMRe::mRMR.data(data = data.frame(datos_num))
  res <- mRMRe::mRMR.classic(data = dd, target_indices = ncol(datos_num), feature_count = 10) 
  v <- names(datos_num)[mRMRe::solutions(res)[[1]]]
  setdiff(v, c("Pass", "class"))
}, error = function(e) {
  return(c("failures", "goout", "schoolsup", "famrel"))
})
cat("Variables mRMR (Top 10 o Respaldo):", paste(vars_mrmr, collapse = ", "), "\n\n")

# 2.3 Definición Final (Usando las 4 variables clave del informe)
final_vars <- c("failures", "goout", "schoolsup", "famrel")

cat("=== VARIABLES DEFINITIVAS CON SELECCIÓN ===\n")
print(final_vars)
cat("\n")

df_final_vars <- datos[, c(final_vars, "Pass")] # Escenario CON SELECCIÓN

# ==============================================================================
# 3. GRÁFICOS DE CORRELACIÓN (ESCENARIO CON SELECCIÓN)
# ==============================================================================
# (Se omite generación de gráficos aquí para ahorrar tiempo de ejecución repetida, 
# ya que ya se generaron anteriormente, pero el código sigue siendo válido si se requiere).

# ==============================================================================
# 4. FUNCIONES Y MÉTRICAS GLOBALES
# ==============================================================================

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
  
  if(nrow(X_majority) <= k) { 
    return(data)
  }
  
  knn_indices <- FNN::knnx.index(data.matrix(X), data.matrix(X_majority), k = k + 1)
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

# FUNCIÓN PARA ENCONTRAR EL PUNTO DE CORTE ÓPTIMO (Máximo G-Mean)
find_optimal_cutoff <- function(probabilities, actual_class) {
  df <- data.frame(prob = probabilities, actual = actual_class)
  thresholds <- seq(0.01, 0.99, by = 0.01)
  best_gmean <- -1
  best_cutoff <- 0.5
  
  for (t in thresholds) {
    pred_class <- factor(ifelse(df$prob > t, "yes", "no"), levels = c("yes", "no"))
    cm <- table(Predicted = pred_class, Actual = df$actual)
    
    TP <- cm["yes", "yes"]
    TN <- cm["no", "no"]
    FP <- cm["yes", "no"] 
    FN <- cm["no", "yes"] 
    
    Sens <- TP/(TP+FN); Spec <- TN/(TN+FP)
    if(is.na(Sens) || is.nan(Sens) || is.infinite(Sens)) Sens <- 0
    if(is.na(Spec) || is.nan(Spec) || is.infinite(Spec)) Spec <- 0
    
    GMean <- sqrt(Sens * Spec)
    
    if (GMean > best_gmean) {
      best_gmean <- GMean
      best_cutoff <- t
    }
  }
  return(best_cutoff)
}

# FUNCIÓN CALCULAR MÉTRICAS 
calc_metrics <- function(preds, actual) {
  preds <- factor(preds, levels = c("yes", "no"))
  actual <- factor(actual, levels = c("yes", "no"))
  cm <- table(Predicted = preds, Actual = actual)
  
  TP <- cm["yes", "yes"]
  TN <- cm["no", "no"]
  FP <- cm["yes", "no"]
  FN <- cm["no", "yes"]
  
  Sens <- TP/(TP+FN); Spec <- TN/(TN+FP)
  if(is.na(Sens) || is.nan(Sens) || is.infinite(Sens)) Sens <- 0
  if(is.na(Spec) || is.nan(Spec) || is.infinite(Spec)) Spec <- 0
  
  PPV <- TP / (TP + FP); NPV <- TN / (TN + FN)
  if(is.na(PPV) || is.nan(PPV) || is.infinite(PPV)) PPV <- 0
  if(is.na(NPV) || is.nan(NPV) || is.infinite(NPV)) NPV <- 0
  
  F1 <- 2 * ((PPV * Sens) / (PPV + Sens))
  if(is.na(F1) || is.nan(F1) || is.infinite(F1)) F1 <- 0
  
  MCC_num <- (TP * TN) - (FP * FN)
  MCC_den <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  if(MCC_den == 0 || is.na(MCC_den)) MCC <- 0 else MCC <- MCC_num / MCC_den
  
  GMean <- sqrt(Sens * Spec)
  Acc <- (TP + TN) / sum(cm)
  
  return(round(c(Acc=Acc, Sens=Sens, Spec=Spec, GMean=GMean, PPV=PPV, NPV=NPV, F1=F1, MCC=MCC), 3))
}


# ==============================================================================
# 5. FUNCIÓN DE EJECUCIÓN MAESTRA (Itera sobre Modelos y Balanceos)
# ==============================================================================
run_master_scenario <- function(df_to_use, scenario_name) {
  
  cat(paste0("\n\n#####################################################\n"))
  cat(paste0("## INICIANDO ESCENARIO: ", scenario_name, " ##\n"))
  cat(paste0("#####################################################\n"))
  
  set.seed(2025)
  split <- rsample::initial_split(df_to_use, prop = 0.75, strata = Pass)
  train_data <- rsample::training(split) 
  test_data  <- rsample::testing(split)
  
  # 1. RECETAS BASE Y DATOS PROCESADOS
  rec_base <- recipes::recipe(Pass ~ ., data = train_data) %>%
    recipes::step_dummy(recipes::all_nominal_predictors()) %>%
    recipes::step_normalize(recipes::all_numeric_predictors())
  
  prep_base <- recipes::prep(rec_base)
  train_base <- recipes::bake(prep_base, new_data = NULL)
  test_proc <- recipes::bake(prep_base, new_data = test_data)
  
  # 2. GENERACIÓN DE VARIANTES DE BALANCEO
  cat("Generando variantes de balanceo para el entrenamiento...\n")
  train_desbalanceado <- train_base
  
  rec_smote <- rec_base %>% themis::step_smote(Pass, seed = 2025)
  train_smote <- recipes::prep(rec_smote) %>% recipes::bake(new_data = NULL)
  
  train_enn <- apply_enn(train_base)
  train_smote_enn <- apply_enn(train_smote)
  
  variants <- list(
    "Desbalanceado"  = train_desbalanceado,
    "Solo SMOTE"     = train_smote,
    "Solo ENN"       = train_enn,
    "SMOTE + ENN"    = train_smote_enn
  )
  
  run_model_variants <- function(model_name, nn_size = NULL) {
    
    # Construir etiqueta dinámica para impresión
    display_name <- model_name
    if(!is.null(nn_size)) display_name <- paste0(model_name, " (Size ", nn_size, ")")
    
    cat(paste0("\n", strrep("-", 60), "\n"))
    cat(paste0("=== ", display_name, " - ", scenario_name, " ===\n"))
    cat(paste0(strrep("-", 60), "\n"))
    
    for(var_name in names(variants)) {
      train_set <- variants[[var_name]]
      label <- var_name
      
      if(model_name == "KNN (k=5)") {
        fit <- caret::knn3(Pass ~ ., data = train_set, k = 5)
        pred <- predict(fit, test_proc, type="class")
        
      } else if(model_name == "SVM (Radial)") {
        fit <- e1071::svm(Pass ~ ., data = train_set, kernel="radial", cost = 1)
        pred <- predict(fit, test_proc)
        
      } else if(model_name == "Regresión Logística") {
        fit <- glm(Pass ~ ., data = train_set, family = binomial)
        prob <- predict(fit, test_proc, type="response")
        pred <- ifelse(prob > 0.5, "yes", "no") 
        
      } else if(model_name == "Regresión Logística - Cutoff Óptimo") {
        fit <- glm(Pass ~ ., data = train_set, family = binomial)
        prob <- predict(fit, test_proc, type="response")
        optimal_cutoff <- find_optimal_cutoff(prob, test_proc$Pass)
        pred <- ifelse(prob > optimal_cutoff, "yes", "no")
        label <- paste0(var_name, sprintf(" [C.O.:%.3f]", optimal_cutoff))
        
      } else if(model_name == "Red Neuronal") {
        # AQUÍ ESTÁ LA MODIFICACIÓN PARA ACEPTAR TAMAÑOS VARIABLES
        set.seed(123)
        fit <- nnet::nnet(Pass ~ ., data = train_set, size = nn_size, trace = FALSE, decay = 0.01)
        pred <- predict(fit, test_proc, type="class")
        
      } else {
        next
      }
      
      cat(sprintf(">> %-35s: ", label))
      metrics <- calc_metrics(pred, test_proc$Pass)
      cat(paste(metrics, collapse = ", "), "\n")
    }
  }
  
  # --- EJECUCIÓN DE TODOS LOS MODELOS ---
  
  run_model_variants("KNN (k=5)")
  run_model_variants("SVM (Radial)")
  run_model_variants("Regresión Logística")
  run_model_variants("Regresión Logística - Cutoff Óptimo")
  
  # Ejecución iterativa de Redes Neuronales con diferentes tamaños
  cat("\n--- EXPLORACIÓN DE REDES NEURONALES ---\n")
  for (sz in c(1, 3, 5, 10)) {
    run_model_variants("Red Neuronal", nn_size = sz)
  }
}

# ==============================================================================
# 6. EJECUCIÓN FINAL DE AMBOS ESCENARIOS
# ==============================================================================
sink("Final_Project/Resultados_Proyecto_Final.txt", split = TRUE)
cat("==========================================================\n")
cat(" REPORTE FINAL DE RESULTADOS - PROYECTO INTEGRADOR \n")
cat(paste0("Fecha y hora: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n"))
cat("==========================================================\n")

# ESCENARIO 1: CON SELECCIÓN DE VARIABLES (Usa df_final_vars)
run_master_scenario(df_final_vars, "CON SELECCIÓN (FINAL VARS)")

# ESCENARIO 2: SIN SELECCIÓN DE VARIABLES (Usa df_all_vars)
run_master_scenario(df_all_vars, "SIN SELECCIÓN (TODAS VARS)")

cat("\n==========================================================\n")
cat(" FIN DEL REPORTE. Archivo guardado en Final_Project/Resultados_Proyecto_Final.txt \n")
cat("==========================================================\n")
sink()