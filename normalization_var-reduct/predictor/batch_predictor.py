#!/usr/bin/env python
# coding: utf-8
"""
Batch runner for predictor.py: executes chosen combinations of scaler, reduction, and model.
"""
import subprocess
import os
import itertools
import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Batch execute predictor.py for multiple combinations.")
    parser.add_argument('--training', required=True, help='Path to training Excel file')
    parser.add_argument('--validation', required=True, help='Path to validation Excel file')
    parser.add_argument('--scalers', default='zscore,minmax,decimal,robust,unit,none',
                        help='Comma-separated list of scalers to test')
    parser.add_argument('--reductions', default='pca,pca_2,efa,plsr,lasso,rfe,rfe_2,sparsepca,sparsepca_2,mihr,pfi,pfi_2,none',
                        help='Comma-separated list of reductions to test')
    parser.add_argument('--models', default='mlr,svm,dt,rf,xgb,knn,ann,gam,gpr,lgbm,catb,gbr,adaboost',
                        help='Comma-separated list of models to test')
    parser.add_argument('--trials', type=int, default=500, help='Number of Optuna trials per run')
    parser.add_argument('--script', default='predictor.py', help='Predictor script name')
    parser.add_argument('--logs-dir', default='logs', help='Directory for successful logs')
    parser.add_argument('--error-dir', default='logs/error', help='Directory for error logs')
    parser.add_argument("--shap", default='off', choices=['on', 'off'], help="Activa la generación de gráficos SHAP ('on' para activar). Por defecto: 'off'.")
    parser.add_argument('--no_intro', action='store_true', help='Pass --no_intro flag to predictor')
    return parser.parse_args()


def ensure_dirs(base_dir, error_dir):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)


def build_combinations(scalers, reductions, models):
    return list(itertools.product(scalers, reductions, models))


def main():
    args = parse_args()
    scalers = args.scalers.split(',')
    reductions = args.reductions.split(',')
    models = args.models.split(',')

    ensure_dirs(args.logs_dir, args.error_dir)
    combos = build_combinations(scalers, reductions, models)
    total = len(combos)
    print(f"Total combinations: {total}")

    start_all = datetime.now()
    for idx, (scaler, reduction, model) in enumerate(combos, 1):
        name = f"{scaler}_{reduction}_{model}"
        log_base = f"{name}.txt"
        cmd = [
            'python', args.script,
            '--training', args.training,
            '--validation', args.validation,
            '--scaler', scaler,
            '--reduction', reduction,
            '--model', model,
            '--trials', str(args.trials)
        ]
        if args.no_intro:
            cmd.append('--no_intro')
        if args.shap == 'on':
            cmd.append('--shap')

        print(f"[{idx}/{total}] Running: {name}")
        start = datetime.now()
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        duration = datetime.now() - start

        # choose log directory
        if result.returncode == 0:
            log_path = os.path.join(args.logs_dir, log_base)
            print(f"Success: {name} in {duration}")
        else:
            log_path = os.path.join(args.error_dir, log_base)
            print(f"Error: {name} in {duration}")

        # write log
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Started: {start.isoformat()}\n")
            f.write("--- stdout ---\n")
            f.write(result.stdout or 'None')
            f.write("\n--- stderr ---\n")
            f.write(result.stderr or 'None')
            if result.returncode != 0:
                f.write(f"\nExit code: {result.returncode}\n")

        # ETA
        elapsed = datetime.now() - start_all
        avg = elapsed / idx
        remaining = avg * (total - idx)
        print(f"ETA remaining: {remaining}\n")

    print(f"Done all {total}. Total time: {datetime.now() - start_all}")

if __name__ == '__main__':
    main()
