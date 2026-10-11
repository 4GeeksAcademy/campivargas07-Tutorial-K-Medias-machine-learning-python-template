"""Experimento reproducible: tres variables, K-Means fijo y pseudoetiquetas."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['Latitude', 'Longitude', 'MedInc']
LABELS = list(range(6))
SEED = 42


def load_data(path=None):
    path = Path(path) if path else ROOT / 'housing.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Falta el CSV local: {path}. No se descargan datos.')
    raw = pd.read_csv(path)
    missing = set(FEATURES) - set(raw.columns)
    if missing:
        raise ValueError(f'Faltan columnas indispensables: {sorted(missing)}')
    if not all(pd.api.types.is_numeric_dtype(raw[c]) for c in FEATURES):
        raise ValueError('Las tres características deben ser numéricas.')
    numeric = raw.select_dtypes(include='number')
    audit = {
        'rows_before': len(raw), 'columns': list(raw.columns),
        'dtypes': raw.dtypes.astype(str).to_dict(),
        'nulls': raw.isna().sum().astype(int).to_dict(),
        'nonfinite_numeric': {c: int((~np.isfinite(numeric[c])).sum()) for c in numeric},
        'full_duplicates': int(raw.duplicated().sum()),
        'feature_duplicates': int(raw.duplicated(FEATURES).sum()),
        'renamed_columns': {},
    }
    valid = np.isfinite(raw[FEATURES]).all(axis=1)
    clean = raw.loc[valid, FEATURES].copy()
    clean.index.name = 'row_id'  # posición original del CSV, basada en cero
    audit.update(rows_after=len(clean), removed_rows=int((~valid).sum()),
                 cleaning='Excluir solo filas no finitas en las tres variables; conservar coincidencias de características.')
    if len(clean) < 10:
        raise ValueError('No hay suficientes filas válidas para el experimento.')
    return raw, clean, audit


def train_experiment(path=None):
    path = Path(path) if path else ROOT / 'housing.csv'
    raw, clean, audit = load_data(path)
    train, test = train_test_split(clean, test_size=0.2, random_state=SEED)
    kmeans = Pipeline([('scaler', StandardScaler()),
                       ('kmeans', KMeans(n_clusters=6, random_state=SEED, n_init=10))])
    classifier = Pipeline([('scaler', StandardScaler()),
                           ('classifier', LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=SEED))])
    with threadpool_limits(limits=1):
        kmeans.fit(train[FEATURES])
        y_train = kmeans.predict(train[FEATURES])
        y_test = kmeans.predict(test[FEATURES])
        classifier.fit(train[FEATURES], y_train)
        pred_test = classifier.predict(test[FEATURES])
        silhouette = silhouette_score(kmeans['scaler'].transform(train[FEATURES]), y_train,
                                      sample_size=min(2000, len(train)), random_state=SEED)
    train = train.assign(cluster=y_train)
    test = test.assign(cluster=y_test)
    exported = pd.concat([
        train.assign(partition='train', classifier_prediction=classifier.predict(train[FEATURES])),
        test.assign(partition='test', classifier_prediction=pred_test)
    ]).sort_index()
    exported = exported[FEATURES + ['partition', 'cluster', 'classifier_prediction']].reset_index()
    centers = pd.DataFrame(kmeans['scaler'].inverse_transform(kmeans['kmeans'].cluster_centers_),
                           columns=FEATURES).rename_axis('cluster')
    metrics = {
        'accuracy': float(accuracy_score(y_test, pred_test)),
        'macro_f1': float(f1_score(y_test, pred_test, labels=LABELS, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, pred_test, labels=LABELS).tolist(),
        'classification_report': classification_report(y_test, pred_test, labels=LABELS, output_dict=True, zero_division=0),
        'train_inertia': float(kmeans['kmeans'].inertia_),
        'train_silhouette_sample': float(silhouette),
        'test_disagreements': int((y_test != pred_test).sum()),
        'train_rows': len(train), 'test_rows': len(test),
        'cluster_counts': {part: frame.cluster.value_counts().reindex(LABELS, fill_value=0).to_dict()
                           for part, frame in [('train', train), ('test', test)]},
    }
    config = {
        'features': FEATURES, 'random_state': SEED, 'test_size': 0.2,
        'kmeans': {'n_clusters': 6, 'random_state': SEED, 'n_init': 10},
        'classifier': {'type': 'LogisticRegression', 'C': 1.0, 'max_iter': 2000, 'solver': 'lbfgs', 'random_state': SEED},
        'scaler': 'StandardScaler ajustado exclusivamente con train en cada pipeline',
        'silhouette': {'sample_size': 2000, 'random_state': SEED, 'partition': 'train'},
        'source': str(path.resolve()), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'row_id': 'posición original del CSV, basada en cero; estable mientras no cambie el archivo',
        'python': platform.python_version(), 'threads': 1,
        'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'joblib', 'matplotlib', 'threadpoolctl']},
    }
    return dict(raw=raw, clean=clean, audit=audit, train=train, test=test, kmeans=kmeans,
                classifier=classifier, exported=exported, centers=centers, metrics=metrics, config=config)


def verify_artifacts(result, output_dir=ROOT):
    output_dir = Path(output_dir)
    train, test, clean = (result[k] for k in ['train', 'test', 'clean'])
    assert set(train.index).isdisjoint(test.index), 'Particiones solapadas'
    assert set(train.index) | set(test.index) == set(clean.index), 'Filas perdidas'
    saved = pd.read_csv(output_dir / 'data/processed/housing_clusters.csv')
    pd.testing.assert_frame_equal(saved, result['exported'], check_dtype=False)
    assert saved.row_id.is_unique and len(saved) == len(clean)
    assert all(pd.api.types.is_integer_dtype(saved[c]) and saved[c].isin(LABELS).all()
               for c in ['cluster', 'classifier_prediction'])
    for name in ['kmeans', 'classifier']:
        loaded = joblib.load(output_dir / f'models/{name}_pipeline.joblib')
        assert list(loaded.feature_names_in_) == FEATURES
        expected = result[name].predict(clean[FEATURES])
        np.testing.assert_array_equal(expected, loaded.predict(clean[FEATURES]))
    np.testing.assert_allclose(result['kmeans']['scaler'].mean_, train[FEATURES].mean())
    return {'partitions_disjoint': True, 'rows_conserved': True, 'labels_valid': True,
            'csv_aligned': True, 'reload_predictions_equal': True, 'scaler_train_only': True}


def save_artifacts(result, output_dir=ROOT):
    output_dir = Path(output_dir)
    models = output_dir / 'models'
    processed = output_dir / 'data/processed'
    models.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    for name in ['kmeans', 'classifier']:
        joblib.dump(result[name], models / f'{name}_pipeline.joblib')
    result['exported'].to_csv(processed / 'housing_clusters.csv', index=False)
    result['centers'].to_csv(processed / 'centroids.csv')
    for name in ['metrics', 'config', 'audit']:
        (models / f'{name}.json').write_text(json.dumps(result[name], indent=2, ensure_ascii=False) + '\n')
    checks = verify_artifacts(result, output_dir)
    (models / 'verification.json').write_text(json.dumps(checks, indent=2) + '\n')
    return checks


def predict_records(records, model_dir=ROOT / 'models'):
    frame = pd.DataFrame(records)
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f'Faltan características: {sorted(missing)}')
    x = frame[FEATURES]
    if x.empty or not all(pd.api.types.is_numeric_dtype(x[c]) for c in FEATURES) or not np.isfinite(x).all().all():
        raise ValueError('Se necesitan registros numéricos, finitos y no vacíos.')
    out = x.copy()
    for name, column in [('kmeans', 'cluster'), ('classifier', 'classifier_prediction')]:
        out[column] = joblib.load(Path(model_dir) / f'{name}_pipeline.joblib').predict(x)
    return out
