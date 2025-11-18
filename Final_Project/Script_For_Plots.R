# ----------------------------------------------------------------------
# CÓDIGO R COMPLETO: GRÁFICOS DE DISTRIBUCIÓN PARA TODAS LAS VARIABLES
# RUTA DEL ARCHIVO: "Final_Project/student-mat.csv"
# ----------------------------------------------------------------------

# 0. CARGA DE PAQUETES
# Instala los paquetes si es necesario:
# install.packages(c("readr", "dplyr", "ggplot2"))
library(readr)
library(dplyr)
library(ggplot2)

# 1. CARGA Y PRE-PROCESAMIENTO DE DATOS
# ¡IMPORTANTE! Usando la ruta especificada por el usuario
file_path <- "Final_Project/student-mat.csv"
datos <- read_delim(file_path, delim = ";", show_col_types = FALSE)

# Convertir las variables de notas a numérico (necesario ya que read_delim las puede leer como texto)
datos <- datos %>%
  mutate(across(c("G1", "G2", "G3"), as.numeric))

# 2. CLASIFICACIÓN DE VARIABLES
# Separamos las variables en grupos para aplicar el tipo de gráfico adecuado.

# Variables Numéricas/Continuas o con amplio rango (para Histogramas/Densidad)
numeric_vars <- c("age", "absences", "G1", "G2", "G3")

# Variables Categóricas/Ordinales/Binarias (para Gráficos de Barras de Frecuencia)
all_vars <- names(datos)
categorical_vars <- all_vars[!all_vars %in% numeric_vars]

# Convertir explícitamente las variables categóricas a tipo factor
datos <- datos %>%
  mutate(across(all_of(categorical_vars), as.factor))


# ======================================================================
# PARTE A: GRÁFICOS PARA VARIABLES NUMÉRICAS (Histogramas y Densidad)
# ======================================================================

cat("Generando gráficos de Histograma/Densidad para variables NUMÉRICAS (revisa el panel de Plots de RStudio)...\n")

for (var in numeric_vars) {
  # Crear el gráfico
  p <- datos %>%
    ggplot(aes(x = .data[[var]])) +
    # Histograma
    geom_histogram(
      aes(y = after_stat(density)),
      bins = 20,
      fill = "#1f78b4",
      color = "white",
      alpha = 0.8
    ) +
    # Línea de densidad
    geom_density(color = "#e31a1c", linewidth = 1.2) +
    labs(
      title = paste("Distribución de:", var),
      x = var,
      y = "Densidad"
    ) +
    theme_minimal()
  
  # Mostrar el gráfico en el panel de RStudio
  print(p)
}

cat("...Generación de Histogramas completada.\n\n")


# ======================================================================
# PARTE B: GRÁFICOS PARA VARIABLES CATEGÓRICAS/ORDINALES (Barras de Frecuencia)
# ======================================================================

cat("Generando gráficos de Barras de Frecuencia para variables CATEGÓRICAS/ORDINALES (revisa el panel de Plots de RStudio)...\n")

for (var in categorical_vars) {
  # Crear el gráfico de barras
  p <- datos %>%
    ggplot(aes(x = .data[[var]])) +
    geom_bar(fill = "#33a02c", color = "black") +
    # Añadir el conteo de cada barra
    geom_text(aes(label = after_stat(count)), stat = "count", vjust = -0.5) +
    labs(
      title = paste("Frecuencia de:", var),
      x = var,
      y = "Conteo Absoluto"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
      legend.position = "none"
    )
  
  # Mostrar el gráfico en el panel de RStudio
  print(p)
}

cat("...Generación de Gráficos de Barras completada.\n")
cat("Proceso finalizado. Todos los gráficos se han mostrado secuencialmente en el panel de Plots.\n")