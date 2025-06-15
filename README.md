# Nonparametric-analysis-of-homicides-in-CDMX-by-borough

CONTENIDO

Introducción

Pruebas de Normalidad

Test de Shapiro-Wilk

Test de Jarque-Bera

Prueba de Homocedasticidad

Test de Levene

Pruebas de Comparación No Paramétricas

Test de Kruskal-Wallis

Prueba U de Mann-Whitney

Test de Friedman

Análisis de Correlaciones No Paramétricas

Correlación de Spearman

Correlación de Kendall

Análisis de Tendencias

LOWESS (Locally Weighted Scatterplot Smoothing)

Métodos de Remuestreo

Bootstrap para Intervalos de Confianza

Referencias



#Introducción#

El análisis de datos de homicidios presenta desafíos particulares: distribuciones asimétricas, heterogeneidad entre regiones y posibles valores atípicos. Los métodos paramétricos tradicionales (ANOVA, t-test) requieren supuestos de normalidad y homocedasticidad que pueden no cumplirse en estos datos. Por ello, adoptamos un enfoque no paramétrico que ofrece mayor robustez ante estas características.

El pipeline implementa una secuencia lógica: exploración, verificación de supuestos, pruebas estadísticas, y visualización e interpretación de resultados. Este documento explica los fundamentos matemáticos de cada método.

#Pruebas de Normalidad

Test de Shapiro-Wilk
El test de Shapiro-Wilk evalúa la hipótesis nula de que una muestra proviene de una población normalmente distribuida.

Fundamento Matemático
El estadístico W se calcula como:

$$W = \frac{(\sum_{i=1}^{n} a_i x_{(i)})^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$

Donde:

$x_{(i)}$ son los valores de la muestra ordenados
$a_i$ son constantes generadas a partir de los valores esperados, varianzas y covarianzas de las estadísticas de orden de una muestra de tamaño n de una distribución normal
$\bar{x}$ es la media muestral
Interpretación
W cercano a 1 indica normalidad
Valores p < 0.05 rechazan la hipótesis nula de normalidad
Potencia: Alta para n < 50, moderada para muestras mayores
Implementación en el Código
Test de Jarque-Bera
El test de Jarque-Bera evalúa la normalidad basándose en la asimetría y curtosis de la distribución.

Fundamento Matemático
El estadístico JB se calcula como:

$$JB = \frac{n}{6} \left( S^2 + \frac{(K-3)^2}{4} \right)$$

Donde:

n es el tamaño de la muestra
S es la asimetría muestral: $S = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{\sigma} \right)^3$
K es la curtosis muestral: $K = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{\sigma} \right)^4$
σ es la desviación estándar muestral
Bajo la hipótesis nula de normalidad, JB sigue asintóticamente una distribución chi-cuadrado con 2 grados de libertad.

Interpretación
Valores p < 0.05 rechazan la hipótesis nula de normalidad
Ventaja: Sensible tanto a la asimetría como a la curtosis no normal
Implementación en el Código
Prueba de Homocedasticidad
Test de Levene
El test de Levene evalúa la igualdad de varianzas entre dos o más grupos.

Fundamento Matemático
El estadístico W se calcula como:

$$W = \frac{(N-k)}{(k-1)} \frac{\sum_{i=1}^{k} n_i (Z_{i.} - Z_{..})^2}{\sum_{i=1}^{k} \sum_{j=1}^{n_i} (Z_{ij} - Z_{i.})^2}$$

Donde:

$N$ es el número total de observaciones
$k$ es el número de grupos
$n_i$ es el número de observaciones en el grupo i
$Z_{ij} = |x_{ij} - \bar{x}_i|$ (distancia absoluta desde la media del grupo)
$Z_{i.}$ es la media de $Z_{ij}$ para el grupo i
$Z_{..}$ es la media global de $Z_{ij}$
Interpretación
W se distribuye aproximadamente como F con parámetros k-1 y N-k
Valores p < 0.05 rechazan la hipótesis nula de homogeneidad de varianzas
Ventaja: Más robusto que el test de Bartlett ante desviaciones de normalidad
Implementación en el Código
Pruebas de Comparación No Paramétricas
Test de Kruskal-Wallis
El test de Kruskal-Wallis es la alternativa no paramétrica al ANOVA de una vía, evaluando si muestras independientes provienen de la misma distribución.

Fundamento Matemático
El estadístico H se calcula como:

$$H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)$$

Donde:

N es el número total de observaciones
k es el número de grupos
$n_i$ es el número de observaciones en el grupo i
$R_i$ es la suma de los rangos en el grupo i
Bajo la hipótesis nula, H se aproxima a una distribución chi-cuadrado con k-1 grados de libertad.

Interpretación
Valores p < 0.05 rechazan la hipótesis nula de igualdad de distribuciones
No requiere normalidad, pero asume que las distribuciones tienen formas similares
Implementación en el Código
Prueba U de Mann-Whitney
La prueba U de Mann-Whitney compara dos muestras independientes sin asumir normalidad, utilizada aquí para comparaciones post-hoc después de Kruskal-Wallis.

Fundamento Matemático
El estadístico U se calcula como:

$$U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1$$

O equivalentemente:

$$U' = n_1 n_2 + \frac{n_2(n_2+1)}{2} - R_2$$

Donde:

$n_1$ y $n_2$ son los tamaños de las dos muestras
$R_1$ es la suma de rangos para la primera muestra
$R_2$ es la suma de rangos para la segunda muestra
Interpretación
Para muestras grandes, U se aproxima a una distribución normal
Valores p < 0.05 rechazan la hipótesis nula de igualdad de distribuciones
Ventaja: Detecta diferencias en las ubicaciones (medianas) de las distribuciones
Implementación en el Código
Test de Friedman
El test de Friedman es una extensión no paramétrica del ANOVA de medidas repetidas para datos ordinales o que no cumplen normalidad.

Fundamento Matemático
El estadístico $\chi^2_r$ se calcula como:

$$\chi^2_r = \frac{12}{nk(k+1)} \sum_{j=1}^{k} R_j^2 - 3n(k+1)$$

Donde:

n es el número de bloques (sujetos o, en este caso, años)
k es el número de tratamientos (alcaldías)
$R_j$ es la suma de rangos para el tratamiento j
Interpretación
Bajo la hipótesis nula, $\chi^2_r$ se distribuye aproximadamente como chi-cuadrado con k-1 grados de libertad
Valores p < 0.05 rechazan la hipótesis nula de igualdad de distribuciones
Aplicación: Evalúa si hay diferencias en las tendencias temporales entre alcaldías
Implementación en el Código
Análisis de Correlaciones No Paramétricas
Correlación de Spearman
La correlación de Spearman (ρ) mide la fuerza y dirección de la asociación monotónica entre dos variables, sin asumir linealidad o normalidad.

Fundamento Matemático
El coeficiente ρ se calcula como:

$$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

Donde:

$d_i$ es la diferencia entre los rangos correspondientes de las observaciones
n es el número de pares de observaciones
Interpretación
ρ varía entre -1 (correlación negativa perfecta) y 1 (correlación positiva perfecta)
ρ = 0 indica ausencia de correlación monotónica
Ventaja: Detecta relaciones no lineales pero monotónicas
Implementación en el Código
Correlación de Kendall
La correlación de Kendall (τ) mide la asociación ordinal entre dos variables, basándose en la concordancia y discordancia de pares.

Fundamento Matemático
El coeficiente τ (tau-b) se calcula como:

$$\tau = \frac{n_c - n_d}{\sqrt{(n_0 - n_1)(n_0 - n_2)}}$$

Donde:

$n_c$ es el número de pares concordantes
$n_d$ es el número de pares discordantes
$n_0 = n(n-1)/2$
$n_1$ es el número de pares empatados en la primera variable
$n_2$ es el número de pares empatados en la segunda variable
Interpretación
τ varía entre -1 (inversamente ordenados) y 1 (igualmente ordenados)
τ = 0 indica independencia
Ventaja: Menos sensible a valores atípicos que Spearman
Implementación en el Código
Análisis de Tendencias
LOWESS (Locally Weighted Scatterplot Smoothing)
LOWESS es una técnica de regresión no paramétrica que ajusta modelos simples a subconjuntos locales de datos.

Fundamento Matemático
Para cada punto $(x_i, y_i)$, LOWESS:

Selecciona los k puntos más cercanos a $x_i$ (donde k = frac * n)
Asigna pesos $w_j$ a cada punto usando la función tricúbica: $$w_j = \left(1 - \left|\frac{x_j - x_i}{d_m}\right|^3\right)^3$$ donde $d_m$ es la distancia del punto más lejano en el vecindario
Ajusta una regresión ponderada usando los pesos $w_j$
El valor ajustado $\hat{y}_i$ es el valor predicho por la regresión en $x_i$
Interpretación
Captura patrones no lineales y tendencias locales
El parámetro frac controla el grado de suavizado (más alto = más suave)
Ventaja: Robustez ante valores atípicos y flexibilidad para modelar relaciones complejas
Implementación en el Código
Métodos de Remuestreo
Bootstrap para Intervalos de Confianza
El bootstrap es una técnica de remuestreo que permite estimar la distribución muestral de un estadístico sin asumir normalidad.

Fundamento Matemático
Para estimar el intervalo de confianza de un estadístico θ:

Generar B muestras bootstrap (remuestreo con reemplazo de la muestra original)
Calcular el estadístico θ* para cada muestra bootstrap
Construir el intervalo de confianza usando los percentiles de la distribución de θ*:
Límite inferior: percentil α/2 de θ*
Límite superior: percentil 1-α/2 de θ*
donde α = 1 - nivel de confianza (ej. α = 0.05 para IC 95%)

Interpretación
Proporciona estimaciones robustas de la incertidumbre
No requiere supuestos distribucionales
Ventaja: Aplicable a cualquier estadístico, incluso con muestras pequeñas
Implementación en el Código
Justificación del Enfoque No Paramétrico
La elección de métodos no paramétricos en este análisis se basa en una verificación rigurosa de supuestos:

Normalidad: Las pruebas de Shapiro-Wilk y Jarque-Bera evalúan si los datos de homicidios por alcaldía siguen una distribución normal.

Homocedasticidad: El test de Levene evalúa si las varianzas son homogéneas entre alcaldías.

Si estos supuestos no se cumplen (especialmente la homocedasticidad, que resultó significativa con p < 0.05), los métodos no paramétricos ofrecen ventajas importantes:

Mayor robustez ante valores atípicos
No requieren normalidad
Válidos para muestras pequeñas
Aplicables a datos ordinales o con asimetrías
El pipeline implementa un proceso completo: desde la verificación de supuestos hasta pruebas avanzadas de comparación, correlación y tendencias, proporcionando un análisis robusto y estadísticamente riguroso de los patrones de homicidios en la CDMX.

Referencias
Conover, W.J. (1999). Practical Nonparametric Statistics. Wiley, 3rd edition.
Hollander, M., Wolfe, D.A., & Chicken, E. (2013). Nonparametric Statistical Methods. Wiley, 3rd edition.
Cleveland, W.S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots". Journal of the American Statistical Association, 74(368), 829-836.
Efron, B., & Tibshirani, R.J. (1993). An Introduction to the Bootstrap. Chapman & Hall/CRC.
Shapiro, S.S., & Wilk, M.B. (1965). "An analysis of variance test for normality (complete samples)". Biometrika, 52(3/4), 591-611.
Jarque, C.M., & Bera, A.K. (1987). "A test for normality of observations and regression residuals". International Statistical Review, 55(2), 163-172.
Kruskal, W.H., & Wallis, W.A. (1952). "Use of ranks in one-criterion variance analysis". Journal of the American Statistical Association, 47(260), 583-621.
Friedman, M. (1937). "The use of ranks to avoid the assumption of normality implicit in the analysis of variance". Journal of the American Statistical Association, 32(200), 675-701.


Datos y Alcance

Se analizaron 144 observaciones (16 alcaldías × 9 años) con datos de homicidios en CDMX entre 2016-2024.
Se eliminaron 16 registros que estaban fuera del rango temporal establecido.
El rango de homicidios por alcaldía-año varía significativamente: desde 11 hasta 403 casos.

Distribución Geográfica
Alta concentración geográfica: Tres alcaldías acumulan casi el 45% del total de homicidios:
Iztapalapa (21.1%)
Gustavo A. Madero (14.1%)
Cuauhtémoc (9.5%)
Baja incidencia: Tres alcaldías representan menos del 5% del total:
Cuajimalpa (1.8%)
Magdalena Contreras (1.5%)
Milpa Alta (1.4%)

Análisis Estadístico
Normalidad: Sorprendentemente, todas las alcaldías muestran distribuciones normales (p>0.05 en pruebas Shapiro-Wilk y Jarque-Bera).
Homocedasticidad: El test de Levene indica que no hay homogeneidad de varianzas (p<0.001), justificando el uso de métodos no paramétricos.
Diferencias significativas: La prueba Kruskal-Wallis confirma diferencias estadísticamente significativas entre alcaldías (H=132.52, p<0.001).
Comparaciones múltiples: De 120 comparaciones posibles entre alcaldías, 98 (82%) mostraron diferencias significativas.
Tendencias Temporales
Correlación temporal: No existe correlación significativa entre año y número de homicidios (ρ=-0.012, p=0.886), indicando que no hay una tendencia lineal clara al alza o a la baja durante el periodo analizado.
Tendencia LOWESS: Muestra un cambio relativo positivo de +8.7% entre el inicio (2016) y final (2024) del periodo, sugiriendo un ligero aumento general a pesar de fluctuaciones.
Prueba de Friedman: Confirma diferencias significativas en tendencias temporales entre alcaldías (p<0.001), indicando que cada alcaldía sigue su propio patrón temporal.

Años críticos
Los años 2018 y 2019 registraron los mayores promedios de homicidios (111.94 y 112.81 respectivamente).
2022 registró el promedio más bajo (83.56), seguido de 2021 (85.06).

Conclusiones Generales
Heterogeneidad geográfica: Existen diferencias estadísticamente significativas entre alcaldías, con una concentración marcada en ciertas zonas.
Patrón temporal complejo: No hay una tendencia lineal simple, sino patrones diferenciados por alcaldía con fluctuaciones a lo largo del periodo.
Estabilidad reciente con repunte: Tras una disminución en 2021-2022, los datos de 2024 muestran un repunte (96.44 homicidios promedio).
Disparidad de riesgo: La mediana de homicidios en Iztapalapa (324) es casi 15 veces mayor que en Milpa Alta (23).
