"""Verifica entregables existentes sin volver a entrenar."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from housing_workflow import ROOT, FEATURES, load_data, verify_artifacts


def main():
    config = json.loads((ROOT / 'models/config.json').read_text())
    source = ROOT / 'housing.csv'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == config['source_sha256'], 'CSV de origen modificado'
    _, clean, _ = load_data(source)
    train, test = train_test_split(clean, test_size=config['test_size'], random_state=config['random_state'])
    result = {'clean': clean, 'train': train, 'test': test,
              'exported': pd.read_csv(ROOT / 'data/processed/housing_clusters.csv')}
    for name in ['kmeans', 'classifier']:
        result[name] = joblib.load(ROOT / f'models/{name}_pipeline.joblib')
    exported = result['exported'].set_index('row_id')
    pd.testing.assert_frame_equal(exported[FEATURES], clean, check_dtype=False)
    assert set(exported.index[exported.partition == 'train']) == set(train.index)
    assert set(exported.index[exported.partition == 'test']) == set(test.index)
    for name, column in [('kmeans', 'cluster'), ('classifier', 'classifier_prediction')]:
        np.testing.assert_array_equal(result[name].predict(clean[FEATURES]), exported[column])
    print(json.dumps(verify_artifacts(result), indent=2))


if __name__ == '__main__':
    main()
