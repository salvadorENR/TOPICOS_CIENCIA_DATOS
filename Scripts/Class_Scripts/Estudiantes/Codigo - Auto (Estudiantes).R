#################################################################################
## Códigos - Auto                                                             ##
## Asignatura: Tópicos en Ciencia de Datos                                    ##
## Profesor: Ricardo Felipe Ferreira                                          ##
## Año: 2025/2                                                                ##
#################################################################################

############################################################
## 1) Cargar los datos
############################################################

# ESCRIBA UN CÓDIGO PARA CARGAR LOS DATOS 

############################################################
## 2) División en entrenamiento y prueba (estratificada)
############################################################

# install.packages("rsample")   # si aún no lo tiene
library(rsample)

set.seed(2026)  # para reproducibilidad

# 70% entrenamiento, 30% prueba, estratificando por la variable de clase y
split <- initial_split(
  #COMPLETE,
  prop   = #COMPLETE,   # proporción en el conjunto de entrenamiento
    strata = #COMPLETE    # estratificación por la clase
)

treino <- #COMPLETE
  teste  <- #COMPLETE
  
  # Verificar tamaños y proporciones en cada conjunto
  
  table(#COMPLETE) # tabla de frecuencia absoluta de la base completa  
    table(#COMPLETE) # tabla de frecuencia absoluta del entrenamiento
      table(#COMPLETE) # tabla de frecuencia absoluta de la prueba
        
        #COMPLETE # tabla de frecuencia relativa de la base completa
        #COMPLETE # tabla de frecuencia relativa del entrenamiento
        #COMPLETE # tabla de frequência relativa de la prueba
        
        ############################################################
        ## 3) Ajustar SVM lineal con C grande (≈ hard margin)
        ############################################################
        
        # install.packages("e1071")  # si aún no lo tiene
        library(e1071)
        
        svm_hard <- svm(
          #Y ~ #VAR 1 + VAR 2 + ... + VAR P, # complete con los nombres de las variables
          data   = dados,             # datos
          kernel = "linear",          # kernel lineal
          type   = "C-classification",
          cost   = 1e6,               # C muy grande ≈ margen dura
          scale  = FALSE              # NO estandariza x1 y x2 (mantiene la escala original)
        )
        
        w <- t(svm_hard$coefs) %*% svm_hard$SV   # vector (1 x 2): (w1, w2)
        b <- -svm_hard$rho                       # intercepto
        
        w
        b
        
        ############################################################
        ## 4) Evaluar el desempeño en el conjunto de prueba (SVM hard)
        ############################################################
        
        # Predicciones en el conjunto de prueba con el modelo de margen "dura"
        pred_hard <- predict(#COMPLETE, newdata = #COMPLETE)
          
          # Matriz de confusión (positivo = clase positiva, por ejemplo y = 1)
          tab_hard <- table(
            Verdadeiro = #COMPLETE,
              Predito    = #COMPLETE
          )
          tab_hard
          
          TP <- tab_hard["sim", "sim"]  # verdaderos positivos
          TN <- tab_hard["nao", "nao"]  # verdaderos negativos
          FP <- tab_hard["nao", "sim"]  # falsos positivos
          FN <- tab_hard["sim", "nao"]  # falsos negativos
          
          
          # Métricas para el SVM de margen dura (hard margin)
          acc_hard   <- #COMPLETE       # exactitud (acurácia)
            sens_hard  <- #COMPLETE       # sensibilidad (recall, TPR)
            spec_hard  <- #COMPLETE       # especificidad (TNR)
            ppv_hard   <- #COMPLETE       # valor predictivo positivo (precisión)
            npv_hard   <- #COMPLETE       # valor predictivo negativo
            gmean_hard <- #COMPLETE       # g-media
            f1_hard    <- #COMPLETE       # F1-score
            
            # MCC
            mcc_num <- (TP * TN - FP * FN)
          mcc_den <- sqrt( (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) )
          mcc_hard <- #COMPLETE
            
            cat("Desempenho no conjunto de teste – SVM linear (C grande, margem 'dura')\n")
          cat(sprintf("Acurácia          : %.4f\n", acc_hard))
          cat(sprintf("Sensibilidade     : %.4f\n", sens_hard))
          cat(sprintf("Especificidade    : %.4f\n", spec_hard))
          cat(sprintf("VPP (precisão)    : %.4f\n", ppv_hard))
          cat(sprintf("VPN               : %.4f\n", npv_hard))
          cat(sprintf("G-média           : %.4f\n", gmean_hard))
          cat(sprintf("F1-score          : %.4f\n", f1_hard))
          cat(sprintf("MCC               : %.4f\n", mcc_hard))
          
          ############################################################
          ## 5) Ajustar SVM lineal con C moderado (≈ soft margin)
          ############################################################
          
          svm_soft <- svm(
            #COMPLETE,
            data   = #COMPLETE,
              kernel = #COMPLETE,
              type   = #COMPLETE,
              cost   = 1,        # C moderado → permite holgura (margen "suave")
            scale  = #COMPLETE
          )
          
          w <- t(svm_soft$coefs) %*% svm_soft$SV   # vector (1 x 2): (w1, w2)
          b <- -svm_soft$rho                       # intercepto
          
          w
          b
          
          ############################################################
          ## 6) Evaluar el desempeño en el conjunto de prueba (SVM soft)
          ############################################################
          
          # Predicciones en el conjunto de prueba con el modelo de margen "suave"
          pred_soft <- #COMPLETE
            
            # Matriz de confusión (positivo = clase positiva, por ejemplo y = 1)
            tab_soft <- #COMPLETE
            tab_soft
          
          TP <- #COMPLETE  # verdaderos positivos
            TN <- #COMPLETE  # verdaderos negativos
            FP <- #COMPLETE  # falsos positivos
            FN <- #COMPLETE  # falsos negativos
            
            
            # Métricas para el SVM de margen suave (soft margin)
            acc_soft   <- #COMPLETE       # exactitud (acurácia)
            sens_soft  <- #COMPLETE       # sensibilidad (recall, TPR)
            spec_soft  <- #COMPLETE       # especificidad (TNR)
            ppv_soft   <- #COMPLETE       # valor predictivo positivo (precisión)
            npv_soft   <- #COMPLETE       # valor predictivo negativo
            gmean_soft <- #COMPLETE       # g-media
            f1_soft    <- #COMPLETE       # F1-score
            
            # MCC
            mcc_num <- #COMPLETE
            mcc_den <- #COMPLETE
            mcc_soft <- mcc_num / mcc_den
          
          cat("Desempenho no conjunto de teste – SVM linear (C pequeno, margem 'suave')\n")
          cat(sprintf("Acurácia          : %.4f\n", acc_soft))
          cat(sprintf("Sensibilidade     : %.4f\n", sens_soft))
          cat(sprintf("Especificidade    : %.4f\n", spec_soft))
          cat(sprintf("VPP (precisão)    : %.4f\n", ppv_soft))
          cat(sprintf("VPN               : %.4f\n", npv_soft))
          cat(sprintf("G-média           : %.4f\n", gmean_soft))
          cat(sprintf("F1-score          : %.4f\n", f1_soft))
          cat(sprintf("MCC               : %.4f\n", mcc_soft))
          
          #################################################################
          ## 7) Ajustar SVM lineal ajustado por CV
          ################################################################
          
          library(e1071)
          
          ## rejilla de valores candidatos para C (ajuste como desee)
          grid_C <- 10^seq(-3, 3, length.out = 5)
          grid_C
          
          set.seed(123)
          
          tune_lin <- tune.svm(
            #Y ~ #VAR 1 + VAR 2 + ... + VAR P, # complete con los nombres de las variables,
            data   = treino,
            kernel = "linear",
            type   = "C-classification",
            scale  = FALSE,
            cost   = grid_C,
            tunecontrol = tune.control(cross = 3)  # validación cruzada 3-fold
          )
          
          summary(tune_lin)          # muestra el C óptimo y los errores medios
          best_C     <- tune_lin$best.parameters$cost
          best_model <- tune_lin$best.model
          best_C
          
          #################################################################
          ## 8) Evaluar el desempeño en el conjunto de prueba (SVM lineal ajustado)
          ################################################################
          
          # Use el código de los ítems 4) y 6) para este ítem. Note que en el ítem 4) 
          # todas las variables creadas tenían el sufijo _hard, mientras que en el ítem 6) 
          # el sufijo era _soft. En este ítem utilice el sufijo _cv
          
          #################################################################
          ## 9) Ajustar SVM polinomial ajustado por CV
          ################################################################
          
          library(e1071)
          
          ## rejilla de valores candidatos para C, gamma y degree
          grid_C      <- 10^seq(-3, 3, length.out = 5)
          #grid_gamma  <- 10^seq(-3, 1, length.out = 4) # podríamos querer optimizar el hiperparámetro gamma, pero no lo haremos ahora
          #grid_degree <- c(2, 3) # podríamos querer optimizar el grado del polinomio, pero no lo haremos ahora
          
          set.seed(123)
          
          # Complete inspirándose en el ítem 7)
          
          tune_poly <- tune.svm(
            #COMPLETE,
            data   = #COMPLETE,
              kernel = "polynomial",
            type   = #COMPLETE,
              scale  = #COMPLETE,
              cost   = #COMPLETE,
              #gamma  = grid_gamma,  # si decide buscar el gamma óptimo
              #degree = grid_degree, # si decide buscar el grado óptimo del polinomio
              tunecontrol =  #COMPLETE  # validación cruzada 3-fold
          )
          
          summary(tune_poly)                 # muestra C, gamma y degree óptimos
          best_params_poly <-  #COMPLETE
            best_model_poly  <-  #COMPLETE
            best_params_poly
          
          #################################################################
          ## 10) Evaluar el desempeño en el conjunto de prueba (SVM polinomial ajustado)
          ################################################################
          
          # Use el código de los ítems 4), 6) y 8) para este ítem. Note que en el ítem 4) 
          # todas las variables creadas tenían el sufijo _hard, en el ítem 6) el sufijo era _soft 
          # y en el ítem 8) el sufijo era _cv. En este ítem utilice el sufijo _poly
          