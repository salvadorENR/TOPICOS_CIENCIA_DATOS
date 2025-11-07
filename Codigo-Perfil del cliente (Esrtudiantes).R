#################################################################################
## Códigos - Perfil del Cliente                                               ##
## Asignatura: Tópicos en Ciencia de Datos                                    ##
## Profesor: Ricardo Felipe Ferreira                                          ##
## Año: 2025/2                                                                ##
#################################################################################

set.seed(2025)

############################################################
## 1) Cargar los datos
############################################################

# ESCRIBA UN CÓDIGO PARA CARGAR LOS DATOS 

############################################################
## 2) Visualizar x1 vs x2 con colores por clase
############################################################

# define los colores de los puntos
cores <- ifelse(dados$y == 1, "blue", "red")

# COMPLETE LOS ESPACIOS FALTANTES EN LOS CÓDIGOS DE ABAJO 
# DONDE ESTÁ ESCRITO "VARIABLE DEL EJE X" Y "VARIABLE DEL EJE Y" USTED DEBE SUSTITUIR
# POR EL CÓDIGO CORRECTO

plot(VARIABLE DEL EJE X, VARIABLE DEL EJE Y,
     col = cores,  # define los colores a ser utilizados
     pch = ,       # define el tipo de punto
     xlab = ,      # rótulo del eje x
     ylab = )      # rótulo del eje y

# leyenda
legend("topleft",  # lugar del gráfico donde se ubicará la leyenda
       legend = c("Cliente premium (y = 1)",
                  "Cliente básico (y = -1)"),
       col    = c("blue", "red"),
       pch    = 19,
       bty    = "n")

# PREGUNTA: ¿QUÉ LE ESTÁN DICIENDO LOS DATOS? ¿ESTAMOS EN EL CASO LINEALMENTE SEPARABLE O 
# ¿NECESITAREMOS INTRODUCIR VARIABLES DE HOLGURA?

############################################################
## 3) División en entrenamiento y prueba (estratificada)
############################################################

# install.packages("rsample")   # si aún no lo tiene
library(rsample)

set.seed(2026)  # para reproducibilidad

# 70% entrenamiento, 30% prueba, estratificando por la variable de clase y
split <- initial_split(
  XXXXXXXXXXX,
  prop   = ,   # proporción en el conjunto de entrenamiento
  strata =     # estratificación por la clase
)

treino <- 
  teste  <- 
  
  # Verificar tamaños y proporciones en cada conjunto
  table()
table()
prop.table()
prop.table()

############################################################
## 4) Visualizar x1 vs x2 con colores por clase en el entrenamiento
############################################################

# ADAPTE EL CÓDIGO DEL ÍTEM 2) PARA EL CASO DEL CONJUNTO DE ENTRENAMIENTO

# PREGUNTA: ¿EL COMPORTAMIENTO ES PARECIDO AL GRÁFICO DE DISPERSIÓN DE LA BASE COMPLETA?
# ¿DEBERÍA SERLO?

############################################################
## 5) Ajustar SVM lineal con C grande (≈ hard margin)
############################################################

# install.packages("e1071")  # si aún no lo tiene
library(e1071)

svm_hard <- svm(
  y ~ x1 + x2,
  data   = treino,
  kernel = "linear",          # sin kernel no lineal
  type   = "C-classification",
  cost   = 1e6,               # C muy grande ≈ margen dura
  scale  = FALSE              # NO estandariza x1 y x2 (mantiene escala original)
)


############################################################
## 6) Extraer w y b e imprimir la ecuación del hiperplano
############################################################

w <- t(svm_hard$coefs) %*% svm_hard$SV   # vector (1 x 2): (w1, w2)
b <- -svm_hard$rho                       # intercepto

w1 <- as.numeric(w[1])
w2 <- as.numeric(w[2])
b0 <- as.numeric(b)

cat("El hiperplano óptimo tiene la siguiente ecuación:\n")
cat(sprintf("%.4f + %.4f * x1 + %.4f * x2 = 0\n", b0, w1, w2))

# margen geométrica
margem <- 1 / sqrt(w1^2 + w2^2)
cat(sprintf("La margen geométrica es aproximadamente: %.4f\n", margem))

############################################################
## 7) Gráfico del conjunto de entrenamiento con hiperplano y márgenes
############################################################

cores <- ifelse(treino$y == 1, "blue", "red")

plot(treino$x1, treino$x2,
     col = cores, pch = 19,
     xlab = "Horas/semana en la aplicación (x1)",
     ylab = "Gasto mensual en la aplicación (R$) (x2)")

legend("topleft",
       legend = c("Cliente premium (y = 1)",
                  "Cliente básico (y = -1)"),
       col    = c("blue", "red"),
       pch    = 19,
       bty    = "n")

# frontera: w1*x1 + w2*x2 + b0 = 0  => x2 = -(w1/w2)*x1 - b0/w2
abline(a = -b0 / w2, b = -w1 / w2, lwd = 2)

# márgenes: w1*x1 + w2*x2 + b0 = ±1  => x2 = -(w1/w2)*x1 - (b0 ± 1)/w2
abline(a = (1 - b0) / w2,  b = -w1 / w2, lty = 2)
abline(a = (-1 - b0) / w2, b = -w1 / w2, lty = 2)


############################################################
## 8) Evaluar el desempeño en el conjunto de prueba (SVM hard)
############################################################

# Predicciones en el conjunto de prueba con el modelo de margen "dura"
pred_hard <- predict() # COMPLETE

# Matriz de confusión (positivo = cliente premium, y = 1)
tab_hard <- table(
  Verdadeiro = teste$y,
  Predito    = pred_hard
)
tab_hard

# Extraer TP, TN, FP, FN
TP <- tab_hard["1",  "1"]
TN <- tab_hard["-1", "-1"]
FP <- tab_hard["-1", "1"]
FN <- tab_hard["1",  "-1"]

# Métricas para el SVM de margen dura
acc_hard   <- COMPLETE      # exactitud (acurácia)
sens_hard  <- COMPLETE      # sensibilidad (recall, TPR)
spec_hard  <- COMPLETE      # especificidad (TNR)
ppv_hard   <- COMPLETE      # valor predictivo positivo (precisión)
npv_hard   <- COMPLETE      # valor predictivo negativo
gmean_hard <- COMPLETE      # g-media
f1_hard    <- COMPLETE      # F1-score

# MCC
mcc_num <- COMPLETE
mcc_den <- COMPLETE
mcc_hard <- mcc_num / mcc_den

cat("Desempeño en el conjunto de prueba – SVM lineal (C grande, margen 'dura')\n")
cat(sprintf("Exactitud (acurácia) : %.4f\n", acc_hard))
cat(sprintf("Sensibilidad         : %.4f\n", sens_hard))
cat(sprintf("Especificidad        : %.4f\n", spec_hard))
cat(sprintf("VPP (precisión)      : %.4f\n", ppv_hard))
cat(sprintf("VPN                  : %.4f\n", npv_hard))
cat(sprintf("G-media              : %.4f\n", gmean_hard))
cat(sprintf("F1-score             : %.4f\n", f1_hard))
cat(sprintf("MCC                  : %.4f\n", mcc_hard))

############################################################
## 9) Ajustar SVM lineal con C moderado (≈ soft margin)
############################################################

# ADAPTE EL CÓDIGO DEL ÍTEM 5) PARA EL CASO EN QUE COST = 1 

############################################################
## 10) Extraer w y b e imprimir la ecuación del hiperplano
############################################################

# ADAPTE EL CÓDIGO DEL ÍTEM 6) PARA EL CASO EN QUE COST = 1 

############################################################
## 11) Gráfico del entrenamiento con hiperplano y márgenes (soft)
############################################################

# ADAPTE EL CÓDIGO DEL ÍTEM 7) PARA EL CASO EN QUE COST = 1 

############################################################
## 12) Evaluar el desempeño en el conjunto de prueba (SVM soft)
############################################################

# ADAPTE EL CÓDIGO DEL ÍTEM 8) PARA EL CASO EN QUE COST = 1 

############################################################
## 13) PARA REFLEXIONAR
############################################################

# A) COMPARE LOS RESULTADOS, ¿QUÉ SVM LINEAL SE DESEMPEÑÓ MEJOR, HARD O SOFT?

# B) ¿LA ELECCIÓN DEL HIPERPARÁMETRO C DE PENALIZACIÓN DE LAS VARIABLES DE HOLGURA 
# PUEDE HACERSE DE UNA MANERA MENOS SUBJETIVA? ¿CÓMO? ¿QUÉ SUGIERE USTED?










#Aquí va el siguiente código
#################################################################################
## Códigos - Perfil del Cliente                                               ##
## Asignatura: Tópicos en Ciencia de Datos                                    ##
## Profesor: Ricardo Felipe Ferreira                                          ##
## Año: 2025/2                                                                ##
#################################################################################

set.seed(2025)

############################################################
## 1) Cargar los datos
############################################################

# OPCIÓN A: CSV local (descomenta y ajusta la ruta/sep si hace falta)
# dados <- read.csv("perfil_clientes.csv", stringsAsFactors = TRUE)

# OPCIÓN B: ya tienes 'dados' en el entorno.
stopifnot(exists("dados"))

# Asegurar tipos
if (!is.factor(dados$y)) dados$y <- factor(dados$y)
# Reordenar niveles a c("-1","1") si vinieron distintos
if (!all(levels(dados$y) %in% c("-1","1"))) levels(dados$y) <- c("-1","1")

############################################################
## 2) Visualizar x1 vs x2 con colores por clase
############################################################

# Colores (azul: premium=1, rojo: básico=-1)
cores <- ifelse(dados$y == 1, "blue", "red")

plot(dados$x1, dados$x2,
     col  = cores,      # colores por clase
     pch  = 19,         # punto sólido
     xlab = "Horas/semana en la aplicación (x1)",
     ylab = "Gasto mensual en la aplicación (R$) (x2)")

legend("topleft",
       legend = c("Cliente premium (y = 1)",
                  "Cliente básico (y = -1)"),
       col    = c("blue", "red"),
       pch    = 19,
       bty    = "n")

# Pregunta guía:
# ¿Se ve una frontera aproximadamente lineal (linealmente separable) o habrá que usar holguras?

############################################################
## 3) División en entrenamiento y prueba (estratificada)
############################################################

library(rsample)
set.seed(2026)

split <- initial_split(
  dados,
  prop   = 0.7,  # 70% entrenamiento
  strata = y
)

treino <- training(split)
teste  <- testing(split)

# Verificar tamaños y proporciones
cat("\nTamaños: treino =", nrow(treino), " teste =", nrow(teste), "\n")
cat("\nFrecuencias absolutas:\n")
print(table(dados$y));  print(table(treino$y));  print(table(teste$y))
cat("\nFrecuencias relativas:\n")
print(prop.table(table(dados$y)))
print(prop.table(table(treino$y)))
print(prop.table(table(teste$y)))

############################################################
## 4) Visualizar x1 vs x2 (solo entrenamiento)
############################################################

cores_tr <- ifelse(treino$y == 1, "blue", "red")
plot(treino$x1, treino$x2,
     col = cores_tr, pch = 19,
     xlab = "Horas/semana en la aplicación (x1)",
     ylab = "Gasto mensual en la aplicación (R$) (x2)")
legend("topleft",
       legend = c("Cliente premium (y = 1)", "Cliente básico (y = -1)"),
       col    = c("blue","red"), pch = 19, bty = "n")

############################################################
## 5) Ajustar SVM lineal con C grande (≈ hard margin)
############################################################

library(e1071)

svm_hard <- svm(
  y ~ x1 + x2,
  data   = treino,
  kernel = "linear",
  type   = "C-classification",
  cost   = 1e6,     # C muy grande ~ margen dura
  scale  = FALSE
)

############################################################
## 6) Extraer w y b e imprimir la ecuación del hiperplano
############################################################

w <- t(svm_hard$coefs) %*% svm_hard$SV     # (1x2)
b <- -svm_hard$rho

w1 <- as.numeric(w[1]);  w2 <- as.numeric(w[2]);  b0 <- as.numeric(b)

cat("\n[HARD] Hiperplano: ", sprintf("%.4f + %.4f*x1 + %.4f*x2 = 0\n", b0, w1, w2))
margem <- 1 / sqrt(w1^2 + w2^2)
cat(sprintf("[HARD] Margen geométrica: %.4f\n", margem))

############################################################
## 7) Gráfico entrenamiento con hiperplano y márgenes (hard)
############################################################

plot(treino$x1, treino$x2,
     col = cores_tr, pch = 19,
     xlab = "Horas/semana en la aplicación (x1)",
     ylab = "Gasto mensual en la aplicación (R$) (x2)")
legend("topleft",
       legend = c("Cliente premium (y = 1)", "Cliente básico (y = -1)"),
       col    = c("blue","red"), pch = 19, bty = "n")

# frontera: w1*x1 + w2*x2 + b0 = 0  => x2 = -(w1/w2)*x1 - b0/w2
abline(a = -b0 / w2, b = -w1 / w2, lwd = 2)
# márgenes: = ±1
abline(a = (1 - b0) / w2,  b = -w1 / w2, lty = 2)
abline(a = (-1 - b0) / w2, b = -w1 / w2, lty = 2)

############################################################
## 8) Evaluación en prueba (SVM hard)
############################################################

pred_hard <- predict(svm_hard, newdata = teste)

tab_hard <- table(Verdadeiro = teste$y, Predito = pred_hard)
print(tab_hard)

TP <- tab_hard["1","1"];   TN <- tab_hard["-1","-1"]
FP <- tab_hard["-1","1"];  FN <- tab_hard["1","-1"]

acc_hard   <- (TP + TN) / sum(tab_hard)
sens_hard  <- TP / (TP + FN)
spec_hard  <- TN / (TN + FP)
ppv_hard   <- TP / (TP + FP)
npv_hard   <- TN / (TN + FN)
gmean_hard <- sqrt(sens_hard * spec_hard)
f1_hard    <- 2 * (ppv_hard * sens_hard) / (ppv_hard + sens_hard)

mcc_num <- (TP * TN) - (FP * FN)
mcc_den <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
mcc_hard <- ifelse(mcc_den > 0, mcc_num / mcc_den, NA_real_)

cat("\nDesempeño – SVM lineal (C grande, 'hard')\n")
cat(sprintf("Exactitud : %.4f\n", acc_hard))
cat(sprintf("Sensibilidad : %.4f\n", sens_hard))
cat(sprintf("Especificidad: %.4f\n", spec_hard))
cat(sprintf("PPV (Precisión): %.4f\n", ppv_hard))
cat(sprintf("VPN : %.4f\n", npv_hard))
cat(sprintf("G-media : %.4f\n", gmean_hard))
cat(sprintf("F1-score: %.4f\n", f1_hard))
cat(sprintf("MCC : %.4f\n", mcc_hard))

############################################################
## 9) Ajustar SVM lineal con C moderado (≈ soft margin)
############################################################

svm_soft <- svm(
  y ~ x1 + x2,
  data   = treino,
  kernel = "linear",
  type   = "C-classification",
  cost   = 1,       # C moderado ~ margen suave
  scale  = FALSE
)

############################################################
## 10) Extraer w y b (soft) e imprimir
############################################################

w_soft <- t(svm_soft$coefs) %*% svm_soft$SV
b_soft <- -svm_soft$rho

w1s <- as.numeric(w_soft[1]);  w2s <- as.numeric(w_soft[2]);  b0s <- as.numeric(b_soft)

cat("\n[SOFT] Hiperplano: ", sprintf("%.4f + %.4f*x1 + %.4f*x2 = 0\n", b0s, w1s, w2s))
margem_soft <- 1 / sqrt(w1s^2 + w2s^2)
cat(sprintf("[SOFT] Margen geométrica: %.4f\n", margem_soft))

############################################################
## 11) Gráfico entrenamiento con hiperplano y márgenes (soft)
############################################################

plot(treino$x1, treino$x2,
     col = cores_tr, pch = 19,
     xlab = "Horas/semana en la aplicación (x1)",
     ylab = "Gasto mensual en la aplicación (R$) (x2)")
legend("topleft",
       legend = c("Cliente premium (y = 1)", "Cliente básico (y = -1)"),
       col    = c("blue","red"), pch = 19, bty = "n")

abline(a = -b0s / w2s, b = -w1s / w2s, lwd = 2)
abline(a = (1 - b0s) / w2s,  b = -w1s / w2s, lty = 2)
abline(a = (-1 - b0s) / w2s, b = -w1s / w2s, lty = 2)

############################################################
## 12) Evaluación en prueba (SVM soft)
############################################################

pred_soft <- predict(svm_soft, newdata = teste)

tab_soft <- table(Verdadeiro = teste$y, Predito = pred_soft)
print(tab_soft)

TPs <- tab_soft["1","1"];   TNs <- tab_soft["-1","-1"]
FPs <- tab_soft["-1","1"];  FNs <- tab_soft["1","-1"]

acc_soft   <- (TPs + TNs) / sum(tab_soft)
sens_soft  <- TPs / (TPs + FNs)
spec_soft  <- TNs / (TNs + FPs)
ppv_soft   <- TPs / (TPs + FPs)
npv_soft   <- TNs / (TNs + FNs)
gmean_soft <- sqrt(sens_soft * spec_soft)
f1_soft    <- 2 * (ppv_soft * sens_soft) / (ppv_soft + sens_soft)

mcc_num_s <- (TPs * TNs) - (FPs * FNs)
mcc_den_s <- sqrt((TPs + FPs) * (TPs + FNs) * (TNs + FPs) * (TNs + FNs))
mcc_soft  <- ifelse(mcc_den_s > 0, mcc_num_s / mcc_den_s, NA_real_)

cat("\nDesempeño – SVM lineal (C = 1, 'soft')\n")
cat(sprintf("Exactitud : %.4f\n", acc_soft))
cat(sprintf("Sensibilidad : %.4f\n", sens_soft))
cat(sprintf("Especificidad: %.4f\n", spec_soft))
cat(sprintf("PPV (Precisión): %.4f\n", ppv_soft))
cat(sprintf("VPN : %.4f\n", npv_soft))
cat(sprintf("G-media : %.4f\n", gmean_soft))
cat(sprintf("F1-score: %.4f\n", f1_soft))
cat(sprintf("MCC : %.4f\n", mcc_soft))

############################################################
## 13) PARA REFLEXIONAR
############################################################
# A) Compare resultados: ¿hard o soft? (mira F1/G-mean/MCC si hay desbalance)
# B) C se puede elegir por validación cruzada (grid de C) maximizando F1/G-mean/MCC
#    y/o usando búsqueda con CV anidada para evitar sobreajuste en la selección.









