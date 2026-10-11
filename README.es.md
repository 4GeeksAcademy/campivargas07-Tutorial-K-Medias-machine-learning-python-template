# Plantilla de Proyecto de Ciencia de Datos

Esta plantilla está diseñada para impulsar proyectos de ciencia de datos proporcionando una configuración básica para conexiones de base de datos, procesamiento de datos, y desarrollo de modelos de aprendizaje automático. Incluye una organización estructurada de carpetas para tus conjuntos de datos y un conjunto de paquetes de Python predefinidos necesarios para la mayoría de las tareas de ciencia de datos.

## Estructura

El proyecto está organizado de la siguiente manera:

- **`src/app.py`** → Script principal de Python donde correrá tu proyecto.
- **`src/explore.ipynb`** → Notebook para exploración y pruebas. Una vez finalizada la exploración, migra el código limpio a `app.py`.
- **`src/utils.py`** → Funciones auxiliares, como conexión a bases de datos.
- **`requirements.txt`** → Lista de paquetes de Python necesarios.
- **`models/`** → Contendrá tus clases de modelos SQLAlchemy.
- **`data/`** → Almacena los datasets en diferentes etapas:
  - **`data/raw/`** → Datos sin procesar.
  - **`data/interim/`** → Datos transformados temporalmente.
  - **`data/processed/`** → Datos listos para análisis.


## ⚡ Configuración Inicial en Codespaces (Recomendado)

No es necesario realizar ninguna configuración manual, ya que **Codespaces se configura automáticamente** con los archivos predefinidos que ha creado la academia para ti. Simplemente sigue estos pasos:

1. **Espera a que el entorno se configure automáticamente**.
   - Todos los paquetes necesarios y la base de datos se instalarán por sí mismos.
   - El `username` y `db_name` creados automáticamente están en el archivo **`.env`** en la raíz del proyecto.
2. **Una vez que Codespaces esté listo, puedes comenzar a trabajar inmediatamente**.


## 💻 Configuración en Local (Solo si no puedes usar Codespaces)

**Prerrequisitos**

Asegúrate de tener Python 3.11+ instalado en tu máquina. También necesitarás pip para instalar los paquetes de Python.

**Instalación**

Clona el repositorio del proyecto en tu máquina local.

Navega hasta el directorio del proyecto e instala los paquetes de Python requeridos:

```bash
pip install -r requirements.txt
```

**Crear una base de datos (si es necesario)**

Crea una nueva base de datos dentro del motor Postgres personalizando y ejecutando el siguiente comando: 

```bash
$ psql -U postgres -c "DO \$\$ BEGIN 
    CREATE USER mi_usuario WITH PASSWORD 'mi_contraseña'; 
    CREATE DATABASE mi_base_de_datos OWNER mi_usuario; 
END \$\$;"
```
Conéctate al motor Postgres para usar tu base de datos, manipular tablas y datos: 

```bash
$ psql -U mi_usuario -d mi_base_de_datos
```

¡Una vez que estés dentro de PSQL podrás crear tablas, hacer consultas, insertar, actualizar o eliminar datos y mucho más!

**Variables de entorno**

Crea un archivo .env en el directorio raíz del proyecto para almacenar tus variables de entorno, como tu cadena de conexión a la base de datos:

```makefile
DATABASE_URL="postgresql://<USUARIO>:<CONTRASEÑA>@<HOST>:<PUERTO>/<NOMBRE_BD>"

#example
DATABASE_URL="postgresql://mi_usuario:mi_contraseña@localhost:5432/mi_base_de_datos"
```

## Ejecutando la Aplicación

Para ejecutar la aplicación, ejecuta el script app.py desde la raíz del directorio del proyecto:

```bash
python src/app.py
```

## Añadiendo Modelos

Para añadir clases de modelos SQLAlchemy, crea nuevos archivos de script de Python dentro del directorio models/. Estas clases deben ser definidas de acuerdo a tu esquema de base de datos.

Definición del modelo de ejemplo (`models/example_model.py`):

```py
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

Base = declarative_base()

class ExampleModel(Base):
    __tablename__ = 'example_table'
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True)
```

## Trabajando con Datos

Puedes colocar tus conjuntos de datos brutos en el directorio data/raw, conjuntos de datos intermedios en data/interim, y los conjuntos de datos procesados listos para el análisis en data/processed.

Para procesar datos, puedes modificar el script app.py para incluir tus pasos de procesamiento de datos, utilizando pandas para la manipulación y análisis de datos.

## Contribuyentes

Esta plantilla fue construida como parte del [Data Science and Machine Learning Bootcamp](https://4geeksacademy.com/us/coding-bootcamps/datascience-machine-learning) de 4Geeks Academy por [Alejandro Sanchez](https://twitter.com/alesanchezr) y muchos otros contribuyentes. Descubre más sobre [los programas BootCamp de 4Geeks Academy](https://4geeksacademy.com/us/programs) aquí.

Otras plantillas y recursos como este se pueden encontrar en la página de GitHub de la escuela.

## Experimento California Housing completado

Para revisar la entrega y el recorrido del trabajo, consultar [el resumen del proyecto](RESUMEN_PROYECTO.md).

El análisis en español y sus salidas están en `src/explore.ipynb`. Se utiliza el `housing.csv` local (20.640 grupos de bloques censales de 1990), exclusivamente con `Latitude`, `Longitude` y `MedInc`. Este último es ingreso mediano en aproximadamente 10.000 USD de 1990; véanse la [descripción de scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/descr/california_housing.rst) y la [aclaración publicada por O'Reilly](https://www.oreilly.com/catalog/errata.csp?isbn=0636920052289). No se requiere base de datos para este experimento.

Se conservan las 20.640 filas; no hay nulos, no finitos ni duplicados completos. Cinco coincidencias en las características se conservan. No se renombran columnas. `row_id` identifica la posición original, basada en cero; el SHA-256 del CSV queda registrado. Train/test son 16.512/4.128 filas (80/20, semilla 42). Ambos StandardScaler se ajustan solo con train. K-Means usa seis clusters, semilla 42 y `n_init=10`. La regresión logística multinomial usa C=1 sin ajustes sobre test.

Resultados: accuracy **0,987888**, F1 macro **0,979953**, **50** desacuerdos en test; inercia train **8069,187859** y silhouette **0,400446** sobre 2.000 filas de train (semilla 42). Las métricas supervisadas evalúan imitación de K-Means; no validan categorías reales ni predicen precios. La distancia sobre coordenadas escaladas no equivale a kilómetros y la partición aleatoria no evalúa extrapolación espacial o temporal.

Archivos reproducibles:

- `src/housing_workflow.py`: carga, auditoría, entrenamiento, exportación y predicción.
- `src/app.py`: CLI para entrenar o cargar y predecir.
- `src/verify.py`: comprobación independiente de artefactos existentes.
- `models/kmeans_pipeline.joblib` y `models/classifier_pipeline.joblib`: procesamiento y estimador juntos.
- `models/metrics.json`, `config.json`, `audit.json`, `verification.json`: resultados, parámetros, versiones y comprobaciones.
- `data/processed/housing_clusters.csv`: tres variables, `row_id`, partición, cluster y predicción del clasificador.
- `data/processed/centroids.csv`: centroides en unidades originales.

Ejecutar desde la raíz, con Python 3.13 (versión usada: 3.13.16; las versiones del intérprete y paquetes constan en `models/config.json`):

```bash
python -m pip install -r requirements.txt
python src/app.py train
python src/verify.py
python -m nbconvert --to notebook --execute --inplace src/explore.ipynb --ExecutePreprocessor.timeout=180
```

Para predecir nuevos registros, crear un CSV con encabezado `Latitude,Longitude,MedInc` y valores numéricos finitos, por ejemplo `34.05,-118.25,3.5`, y ejecutar:

```bash
python src/app.py predict --input nuevos.csv --output predicciones.csv
```

La predicción carga los modelos existentes sin entrenarlos. Los IDs de cluster son categorías arbitrarias asociadas al modelo guardado; no tienen orden y pueden cambiar al reentrenar. El nuevo CSV conserva el orden de entrada. El entrenamiento y la ejecución del notebook regeneran los artefactos del experimento. `verify.py` comprueba hash de origen, particiones sin solapamiento, conservación y alineación de filas, etiquetas enteras 0–5 y concordancia entre exportación y modelos recargados. Durante el entrenamiento también se comprueba igualdad de todas las predicciones antes y después de guardar. Cargar únicamente archivos joblib de confianza.
