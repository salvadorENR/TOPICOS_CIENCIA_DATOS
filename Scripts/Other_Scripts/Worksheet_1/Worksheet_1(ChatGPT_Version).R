# ============================================================
# Problema 2 - Curvas Sesgo–Varianza Normalizadas (0–1)
# ============================================================

# install.packages(c("ggplot2","dplyr","tidyr","ggrepel","scales")) # if needed
library(ggplot2)
library(dplyr)
library(tidyr)
library(ggrepel)
library(scales)

# -------------------------------
# 1) Flexibilidad (0 a 1)
# -------------------------------
flex <- seq(0, 1, length.out = 600)

# -------------------------------
# 2) Curvas (en escala original)
# -------------------------------
coef_bias2 <- 0.45
coef_var   <- 0.55
err_irred  <- 0.06

bias2       <- coef_bias2 * (1 - flex)^2
variance    <- coef_var * (flex)^2
irreducible <- rep(err_irred, length(flex))
test_mse    <- bias2 + variance + irreducible
train_mse   <- 0.60 - 0.55*flex + 0.05*flex^2

# -------------------------------
# 3) Normalizar todas las curvas a [0,1]
# -------------------------------
normalize <- function(x) (x - min(x)) / (max(x) - min(x))
bias2_n       <- normalize(bias2)
variance_n    <- normalize(variance)
irreducible_n <- normalize(irreducible)  # será una línea constante
test_mse_n    <- normalize(test_mse)
train_mse_n   <- normalize(train_mse)

# -------------------------------
# 4) Punto de mínimo del test
# -------------------------------
i_min <- which.min(test_mse_n)
x_min <- flex[i_min]; y_min <- test_mse_n[i_min]

# -------------------------------
# 5) Data frame largo
# -------------------------------
df <- tibble(
  flex,
  `Sesgo cuadrático (bias²)`      = bias2_n,
  `Varianza`                      = variance_n,
  `Error de Bayes (irreducible)`  = irreducible_n,
  `Error de prueba (Test MSE)`    = test_mse_n,
  `Error de entrenamiento`        = train_mse_n
) |>
  pivot_longer(-flex, names_to = "curva", values_to = "valor")

# -------------------------------
# 6) Estilos
# -------------------------------
cols <- c(
  "Sesgo cuadrático (bias²)"     = "#1f78b4",
  "Varianza"                     = "#ff7f00",
  "Error de Bayes (irreducible)" = "#444444",
  "Error de prueba (Test MSE)"   = "#e31a1c",
  "Error de entrenamiento"       = "#33a02c"
)
lts <- c(
  "Sesgo cuadrático (bias²)"     = "solid",
  "Varianza"                     = "solid",
  "Error de Bayes (irreducible)" = "dashed",
  "Error de prueba (Test MSE)"   = "solid",
  "Error de entrenamiento"       = "dotdash"
)

# -------------------------------
# 7) Gráfico final
# -------------------------------
p <- ggplot(df, aes(x = flex, y = valor, color = curva, linetype = curva)) +
  geom_line(linewidth = 1.3) +
  geom_vline(xintercept = x_min, linewidth = 0.7, color = "#e31a1c", alpha = 0.6) +
  geom_point(
    data = data.frame(flex = x_min, valor = y_min),
    aes(x = flex, y = valor),
    color = "#e31a1c", size = 2.6, inherit.aes = FALSE
  ) +
  geom_label_repel(
    data = data.frame(flex = x_min, valor = y_min,
                      lab = sprintf("Mín. del Test\nflex = %.2f", x_min)),
    aes(x = flex, y = valor, label = lab),
    color = "#e31a1c", fill = "white", size = 3.4, label.size = 0.2,
    seed = 1, inherit.aes = FALSE, show.legend = FALSE
  ) +
  annotate("text", x = 0.05, y = 0.95,
           label = "Subajuste (alto sesgo)", hjust = 0, size = 3.6) +
  annotate("text", x = 0.95, y = 0.95,
           label = "Sobreajuste (alta varianza)", hjust = 1, size = 3.6) +
  scale_color_manual(values = cols) +
  scale_linetype_manual(values = lts) +
  scale_x_continuous(labels = percent_format(accuracy = 1), breaks = seq(0, 1, by = 0.1)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.1)) +
  labs(
    title = "Curvas típicas normalizadas: sesgo–varianza y errores vs. flexibilidad",
    x = "Grado de flexibilidad del método",
    y = "Error cuadrático medio (MSE normalizado)",
    color = NULL, linetype = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey80", linewidth = 0.4),
    plot.title = element_text(face = "bold")
  )

print(p)

# Optional: save
# ggsave("problema2_sesgo_varianza_normalizado.png", p, width = 10, height = 6, dpi = 300)
