# 1. Definir el eje X: Grado de Flexibilidad del Método
# La flexibilidad va de 1 (baja) a 100 (alta)
flexibilidad <- 1:100

# 2. Definir las curvas teóricas basándose en el trade-off (compromiso) entre sesgo y varianza:

# A. Error de Bayes (Error Irreducible)
# Es constante, representa la varianza del error irreducible (Var(epsilon) en Ec. 2.7 [3, 6]).
irreducible_error <- rep(5, 100) # Se establece en un valor constante bajo

# B. Sesgo Cuadrático (Squared Bias)
# Debe ser monótonamente decreciente a medida que aumenta la flexibilidad [2, 4, 7].
# Se simula con una función de decaimiento exponencial.
bias_sq <- 40 * exp(-0.04 * flexibilidad) + 5

# C. Varianza (Variance)
# Debe ser monótonamente creciente a medida que aumenta la flexibilidad [2, 4, 6].
# Se simula con una función que crece ligeramente más rápido que lineal.
variance <- 0.05 * flexibilidad + 0.005 * flexibilidad^1.8

# D. Error de Prueba (Test Error / Test MSE)
# Es la suma de los tres componentes: Varianza + Sesgo Cuadrático + Error Irreducible (Ec. 2.7 [3]).
test_error <- bias_sq + variance + irreducible_error

# E. Error de Entrenamiento (Training Error / Training MSE)
# Siempre disminuye a medida que la flexibilidad aumenta, y siempre es menor que el error de prueba [1, 8].
# Se simula como una función que disminuye rápidamente hacia el Error Irreducible.
training_error <- 30 * exp(-0.06 * flexibilidad) + 1.2 * irreducible_error

# 3. Crear el gráfico

# Iniciar el plot con el Error de Prueba (para asegurar que el rango Y cubra la forma de U)
plot(flexibilidad, test_error,
     type = "l", # Trazar como línea
     col = "red", # Rojo para el Error de Prueba [2]
     lwd = 2,
     xlab = "Grado de Flexibilidad del Método",
     ylab = "Error Cuadrático Medio (MSE)",
     main = "Trade-Off Sesgo-Varianza y Curvas de Error Típicas",
     ylim = range(c(irreducible_error, bias_sq, variance, test_error, training_error)))

# Agregar las curvas restantes usando lines()

# Varianza (Creciente)
lines(flexibilidad, variance, col = "orange", lwd = 2)

# Sesgo Cuadrático (Decreciente)
lines(flexibilidad, bias_sq, col = "blue", lwd = 2)

# Error de Entrenamiento (Siempre decreciente, debajo del Error de Prueba)
lines(flexibilidad, training_error, col = "darkgreen", lwd = 2)

# Error de Bayes / Irreducible (Constante)
# Se utiliza lty=2 para línea discontinua, como se ve en las figuras conceptuales [1, 2]
lines(flexibilidad, irreducible_error, col = "black", lwd = 2, lty = 2)

# 4. Añadir Leyenda para identificar cada curva
legend("topright", 
       legend = c("Error de Prueba (Test MSE)", "Sesgo Cuadrático", "Varianza", "Error de Entrenamiento", "Error de Bayes/Irreducible"),
       col = c("red", "blue", "orange", "darkgreen", "black"), 
       lwd = 2, 
       lty = c(1, 1, 1, 1, 2), 
       cex = 0.8)

# Fin del Código R









