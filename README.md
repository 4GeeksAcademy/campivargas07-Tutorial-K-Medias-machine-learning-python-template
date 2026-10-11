# Data Science Project Boilerplate

This boilerplate is designed to kickstart data science projects by providing a basic setup for database connections, data processing, and machine learning model development. It includes a structured folder organization for your datasets and a set of pre-defined Python packages necessary for most data science tasks.

## Structure

The project is organized as follows:

- **`src/app.py`** → Main Python script where your project will run.
- **`src/explore.ipynb`** → Notebook for exploration and testing. Once exploration is complete, migrate the clean code to `app.py`.
- **`src/utils.py`** → Auxiliary functions, such as database connection.
- **`requirements.txt`** → List of required Python packages.
- **`models/`** → Will contain your SQLAlchemy model classes.
- **`data/`** → Stores datasets at different stages:
  - **`data/raw/`** → Raw data.
  - **`data/interim/`** → Temporarily transformed data.
  - **`data/processed/`** → Data ready for analysis.


## ⚡ Initial Setup in Codespaces (Recommended)

No manual setup is required, as **Codespaces is automatically configured** with the predefined files created by the academy for you. Just follow these steps:

1. **Wait for the environment to configure automatically**.
   - All necessary packages and the database will install themselves.
   - The automatically created `username` and `db_name` are in the **`.env`** file at the root of the project.
2. **Once Codespaces is ready, you can start working immediately**.


## 💻 Local Setup (Only if you can't use Codespaces)

**Prerequisites**

Make sure you have Python 3.11+ installed on your machine. You will also need pip to install the Python packages.

**Installation**

Clone the project repository to your local machine.

Navigate to the project directory and install the required Python packages:

```bash
pip install -r requirements.txt
```

**Create a database (if necessary)**

Create a new database within the Postgres engine by customizing and executing the following command:

```bash
$ psql -U postgres -c "DO \$\$ BEGIN 
    CREATE USER my_user WITH PASSWORD 'my_password'; 
    CREATE DATABASE my_database OWNER my_user; 
END \$\$;"
```
Connect to the Postgres engine to use your database, manipulate tables, and data:

```bash
$ psql -U my_user -d my_database
```

Once inside PSQL, you can create tables, run queries, insert, update, or delete data, and much more!

**Environment Variables**

Create a .env file in the root directory of the project to store your environment variables, such as your database connection string:

```makefile
DATABASE_URL="postgresql://<USER>:<PASSWORD>@<HOST>:<PORT>/<DB_NAME>"

#example
DATABASE_URL="postgresql://my_user:my_password@localhost:5432/my_database"
```

## Running the Application

To run the application, execute the app.py script from the root directory of the project:

```bash
python src/app.py
```

## Adding Models

To add SQLAlchemy model classes, create new Python script files within the models/ directory. These classes should be defined according to your database schema.

Example model definition (`models/example_model.py`):

```py
from sqlalchemy.orm import declarative_base
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

Base = declarative_base()

class ExampleModel(Base):
    __tablename__ = 'example_table'
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True)
```

## Working with Data

You can place your raw datasets in the data/raw directory, intermediate datasets in data/interim, and processed datasets ready for analysis in data/processed.

To process data, you can modify the app.py script to include your data processing steps, using pandas for data manipulation and analysis.

## Contributors

This template was built as part of the [Data Science and Machine Learning Bootcamp](https://4geeksacademy.com/us/coding-bootcamps/datascience-machine-learning) by 4Geeks Academy by [Alejandro Sanchez](https://twitter.com/alesanchezr) and many other contributors. Learn more about [4Geeks Academy BootCamp programs](https://4geeksacademy.com/us/programs) here.

Other templates and resources like this can be found on the school's GitHub page.

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
