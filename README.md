# Actividad-12-Implementaci-n-del-Mecanismo-de-Atenci-n-para-Series-de-Tiempo-y-un-Transformer-B-sico-

---
¿Qué hace este notebook?
Predice la concentración de CO2 atmosférico (ppm) a 6 meses de horizonte usando un Transformer Encoder-Decoder construido desde cero en PyTorch, comparado contra un LSTM de arquitectura equivalente.
Resultado	Transformer	LSTM
RMSE	0.777 ppm	0.80 ppm
R²	0.928	0.924
Dirección correcta	90.0 %	89.6 %
---
Dependencias
El notebook instala todo automáticamente. Si alguna librería falta, ejecuta:
```python
!pip install torch statsmodels scikit-learn scipy matplotlib pandas
```
> **Versiones probadas:** Python 3.10 · PyTorch 2.x · statsmodels 0.14 · scikit-learn 1.4
---
Estructura del notebook
#	Sección	Descripción
1	Importaciones	Librerías y semillas de reproducibilidad
2	Dataset y EDA	CO2 Mauna Loa — tendencia, estacionalidad, valores faltantes
3	Preprocesamiento	Imputación, 16 features, ventanas deslizantes, escalado
4	Mecanismo de atención	`AtenciónEscalada` + `MultiHeadAttention` desde cero
5	Transformer	Encoder-Decoder completo con máscara causal
6	LSTM	Encoder-Decoder recurrente para comparación
7	Entrenamiento	Teacher forcing, early stopping, gradient clipping
8	Métricas	MSE · RMSE · MAE · MAPE · R² · DA — con visualizaciones
9	Comparación	Transformer vs LSTM por horizonte y métricas globales
10	Pesos de atención	Mapas de calor por cabeza e interpretación
11	Corto/largo plazo	Experimento con ventanas L = 6 / 12 / 24 meses
12	Residuos	Normalidad, autocorrelación, heterocedasticidad
13	Conclusiones	Ventajas, limitaciones y casos de uso
---
Dataset
CO2 Mauna Loa — NOAA / Scripps Institution of Oceanography  
526 observaciones mensuales · 1958–2001 · unidad: ppm  
Cargado directamente desde `statsmodels.datasets.co2` (sin descarga manual).
---

Reproducibilidad
Todas las semillas están fijadas al inicio:
```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
```
Los resultados numéricos son idénticos en cada ejecución.
