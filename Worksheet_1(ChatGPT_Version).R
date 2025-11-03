# ----- Curvas sesgo–varianza y errores vs. flexibilidad (R + ggplot2) -----

# Paquetes (descomenta si necesitas instalarlos)
# install.packages(c("ggplot2", "tidyr", "dplyr"))

library(ggplot2)
library(tidyr)
library(dplyr)

# Eje x: grado de flexibilidad (0 = rígido, 1 = muy flexible)
x <- seq(0, 1, length.out = 400)

# Formas heurísticas (las magnitudes exactas no importan, solo la forma)
bias2        <- (1 - x)^2 + 0.02          # sesgo cuadrático ↓
variance     <- 0.60 * x^2                # varianza ↑
irreducible  <- rep(0.12, length(x))      # error de Bayes (irreducible) =
train_err    <- 0.65 - 0.60*x + 0.08*x^2  # error de entrenamiento ↓
test_err     <- bias2 + variance + irreducible  # error de prueba ≈ bias² + var + irreducible

df <- data.frame(
  flex = x,
  `Sesgo cuadrático (bias²)` = bias2,
  `Varianza`                 = variance,
  `Error de entrenamiento`   = train_err,
  `Error de prueba`          = test_err,
  `Error de Bayes (irreducible)` = irreducible
) |>
  pivot_longer(-flex, names_to = "curva", values_to = "valor")

ggplot(df, aes(flex, valor, color = curva, linetype = curva)) +
  geom_line(linewidth = 1) +
  labs(
    title = "Curvas típicas: sesgo–varianza y errores vs. flexibilidad",
    x = "Grado de flexibilidad del método",
    y = "Valor de la curva",
    color = NULL, linetype = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")




#**********************************************************************
#*************** (Opcional) versión base R, sin paquetes *****************
#***********************************************************************
#*
x <- seq(0, 1, length.out = 400)
bias2       <- (1 - x)^2 + 0.02
variance    <- 0.60 * x^2
irreducible <- rep(0.12, length(x))
train_err   <- 0.65 - 0.60*x + 0.08*x^2
test_err    <- bias2 + variance + irreducible

plot(x, test_err, type = "l", lwd = 2, xlab = "Flexibilidad",
     ylab = "Valor", main = "Sesgo–varianza y errores vs. flexibilidad")
lines(x, train_err, lwd = 2, col = "gray40")
lines(x, bias2, lwd = 2, col = "orange")
lines(x, variance, lwd = 2, col = "blue")
lines(x, irreducible, lwd = 2, lty = 2, col = "darkgreen")
legend("topright",
       legend = c("Error de prueba","Error de entrenamiento",
                  "Sesgo cuadrático (bias²)","Varianza",
                  "Error de Bayes (irreducible)"),
       lty = c(1,1,1,1,2), lwd = 2,
       col = c("black","gray40","orange","blue","darkgreen"), bty = "n")









