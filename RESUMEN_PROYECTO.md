# Resumen de entrega: California Housing con K-Means

El proyecto agrupa registros de California Housing utilizando exclusivamente ubicación (`Latitude`, `Longitude`) e ingreso mediano (`MedInc`). Se entrenó K-Means con seis clusters y un clasificador de regresión logística para reproducir sus asignaciones en registros nuevos. El análisis completo, con explicaciones y salidas reales, está en [src/explore.ipynb](src/explore.ipynb).

## Recorrido del trabajo y decisiones

1. **Revisión del repositorio y los datos.** Se revisaron el template, los README y las dependencias. Se utilizó el `housing.csv` local, sin volver a descargarlo ni usar `solution.ipynb` como fuente. Sus columnas ya tenían los nombres requeridos.
2. **Auditoría y limpieza.** Se comprobaron tipos, nulos, valores no finitos y duplicados. Se conservaron las 20.640 filas: no había nulos, valores no finitos ni duplicados completos. Las cinco coincidencias en las tres características se conservaron porque no demuestran que los registros sean duplicados. Se mantuvo una referencia `row_id` a la posición original de cada fila.
3. **Exploración.** Se analizaron estadísticas y distribuciones, y se representó la ubicación coloreada por ingreso. Cada registro corresponde a un grupo de bloques censales de 1990, no a una vivienda individual. `MedInc` representa ingreso mediano en aproximadamente decenas de miles de USD de 1990; el notebook incluye las fuentes y documenta sus límites en los extremos.
4. **Preparación sin fuga de información.** Se dividieron las filas en entrenamiento (16.512) y test (4.128), con proporción 80/20 y semilla 42, antes de ajustar las transformaciones. StandardScaler se ajustó únicamente con entrenamiento y se conservó dentro de cada pipeline. No hizo falta imputar.
5. **Agrupación.** Se entrenó `KMeans(n_clusters=6, random_state=42, n_init=10)` solo con entrenamiento. Se calcularon tamaños de grupo, centroides en unidades originales, resúmenes, inercia y silhouette muestral. Test recibió etiquetas mediante `predict()` del pipeline fijo. Los gráficos comparan ambas particiones y muestran ingreso por cluster.
6. **Clasificación supervisada.** Se eligió regresión logística con parámetros fijados de antemano porque las fronteras entre centroides son lineales en el espacio escalado. Se usaron las mismas tres entradas y las etiquetas del K-Means como objetivo; se excluyeron precio, referencia de fila y columna `cluster`. No se ajustaron hiperparámetros usando test.
7. **Evaluación, persistencia y verificación.** Se evaluaron las seis categorías con accuracy, F1 macro, precision, recall y matriz de confusión. Se examinaron desacuerdos y se guardaron los pipelines, las predicciones, métricas, auditoría y configuración. El notebook se ejecutó desde un kernel limpio, con ocho celdas de código y cinco gráficos, sin errores.

## Resultados obtenidos

| Indicador | Resultado |
| --- | ---: |
| Registros conservados | 20.640 |
| Clusters | 6 |
| Accuracy del clasificador en test | 98,79 % |
| F1 macro en test | 0,9800 |
| Desacuerdos con K-Means en test | 50 de 4.128 |
| Inercia de entrenamiento | 8.069,1879 |
| Silhouette en muestra de 2.000 filas de train, semilla 42 | 0,4004 |

Los desacuerdos tienden a aparecer cerca de las fronteras de K-Means: la diferencia mediana entre las distancias a los dos centroides más próximos fue 0,0333 en los desacuerdos y 0,7885 en las coincidencias. Esa diferencia se mide en el espacio estandarizado y no constituye una probabilidad de error.

Las métricas supervisadas indican una alta fidelidad a las **pseudoetiquetas del modelo fijo**. No demuestran que existan seis categorías naturales ni evalúan precios. Los IDs de cluster son arbitrarios y no tienen orden. La distancia euclídea entre coordenadas escaladas no equivale a kilómetros; la partición aleatoria tampoco prueba generalización a otras regiones o periodos. El mapa muestra únicamente dos de las tres dimensiones usadas para agrupar.

## Entregables y comprobaciones

- [src/explore.ipynb](src/explore.ipynb): análisis, gráficos, evaluación y conclusiones.
- [src/housing_workflow.py](src/housing_workflow.py), [src/app.py](src/app.py) y [src/verify.py](src/verify.py): flujo reproducible, entrenamiento/predicción y comprobaciones.
- `models/`: ambos pipelines en joblib y archivos `metrics.json`, `config.json`, `audit.json` y `verification.json`.
- [data/processed/housing_clusters.csv](data/processed/housing_clusters.csv): tres características, referencia de fila, partición, cluster y predicción del clasificador.
- [data/processed/centroids.csv](data/processed/centroids.csv): centroides en unidades originales.
- README y `requirements.txt`: instrucciones y dependencias del experimento.

Pasaron las comprobaciones de particiones sin solapamiento, conservación de filas, alineación de la exportación, etiquetas enteras entre 0 y 5 y ajuste del escalador solo con entrenamiento. Durante el entrenamiento se verificó que las predicciones para todas las filas válidas fueran iguales antes y después de guardar y recargar los modelos. El verificador independiente también comprobó el hash de origen y la concordancia del CSV con los modelos guardados.

## Cómo reproducir y predecir

Desde la raíz del repositorio, usando Python 3.13 (ejecución original: 3.13.16):

```bash
python -m pip install -r requirements.txt
python src/app.py train
python src/verify.py
python -m nbconvert --to notebook --execute --inplace src/explore.ipynb --ExecutePreprocessor.timeout=180
```

Para nuevos registros, preparar un CSV con las columnas `Latitude,Longitude,MedInc` y ejecutar:

```bash
python src/app.py predict --input nuevos.csv --output predicciones.csv
```

La predicción carga los modelos existentes sin reajustarlos. El entrenamiento y la ejecución del notebook regeneran los entregables. Los cambios quedan listos para revisión, sin commit ni push.
