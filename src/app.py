"""CLI de entrenamiento y predicción; no requiere conexión a una base de datos."""
import argparse
import json
from pathlib import Path
import pandas as pd
from housing_workflow import ROOT, predict_records, save_artifacts, train_experiment


def main():
    parser = argparse.ArgumentParser(description='California Housing: K-Means y clasificador')
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train', help='Entrenar con el CSV local y guardar entregables')
    train.add_argument('--data', type=Path, default=ROOT / 'housing.csv')
    predict = sub.add_parser('predict', help='Cargar modelos y predecir un CSV nuevo')
    predict.add_argument('--input', type=Path, required=True)
    predict.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'train':
        result = train_experiment(args.data)
        checks = save_artifacts(result)
        print(json.dumps({'metrics': result['metrics'], 'verification': checks}, indent=2))
    else:
        output = predict_records(pd.read_csv(args.input))
        if args.output:
            output.to_csv(args.output, index=False)
        else:
            print(output.to_csv(index=False), end='')


if __name__ == '__main__':
    main()
