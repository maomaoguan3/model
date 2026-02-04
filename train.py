import os
os.makedirs("out/lightgbm/plots", exist_ok=True)
os.makedirs("out/lightgbm", exist_ok=True)
import json
import csv
import functools
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.interpolate import make_interp_spline
import optuna.visualization as vis
from optuna.integration import LightGBMPruningCallback

SELECTED_FEATURES = [
    'inst_inst_code','eb_iban_num','inst_fund_channel_code','eb_card_brand',
    'eb_merchant_create_source','eb_cnt_by_card_amt','eb_biz_product_code',
    'eb_card_bank_code','eb_kyc_username_keyword_2','eb_first_bind',
    'eb_merchant_name_keyword_1','pmt_pay_channel','eb_pay_amount',
    'eb_pay_channel','eb_customer_event_type','eb_member_country',
    'eb_kyc_username_keyword_1','eb_bind_card_days','eb_kyc_finish_days',
    'eb_card_country','eb_register_days','eb_merchant_id','inst_amount',
    'eb_anonymous_pay','cmf_amount','eb_transaction_amount','eb_cnt_by_member_amt',
    'eb_three_ds_success_count','inst_communicate_status','eb_online',
    'eb_billing_agreement','eb_card_num','inst_product_code','eb_app_name',
    'eb_card_ocr','eb_green_point_used','eb_card_is_verified','eb_account_balance',
    'eb_event_time_minute','eb_merchant_name_keyword_2','inst_pay_mode',
    'eb_kyc_username_keyword_3','eb_bank_cardholder_name_keyword_1',
    'eb_data_eden_aml_switch','eb_on_card','eb_kyc_type','eb_payment_type',
    'eb_content_language','eb_kyc_source',
]

def _smooth_roc(fpr, tpr, n=200):
    uniq, idx = np.unique(fpr, return_index=True)
    fpr_u = fpr[np.sort(idx)]
    tpr_u = tpr[np.sort(idx)]
    if len(fpr_u) >= 4:
        xs = np.linspace(fpr_u.min(), fpr_u.max(), n)
        ys = make_interp_spline(fpr_u, tpr_u, k=3)(xs)
        return xs, ys
    return fpr, tpr

def make_pruning_cb(trial, metric="auc", valid_name="valid", step_offset=0):
    def _cb(env):
        for data_name, eval_name, value, _ in env.evaluation_result_list:
            if data_name == valid_name and eval_name == metric:
                step = step_offset + env.iteration
                trial.report(float(value), step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"Pruned at step {step}")
    return _cb

def objective(trial, train_X, train_y, test_X, test_y, kf):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),          # alias: bagging_fraction
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),  # alias: feature_fraction
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "seed": 42,
        "bagging_seed": 42,
        "feature_fraction_seed": 42,
    }

    oof_pred = np.zeros(len(train_X), dtype=float)
    test_preds_fold = []
    fold_best_iters = []
    auc_scores = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_X, train_y)):
        dtrain = lgb.Dataset(train_X.iloc[tr_idx], label=train_y.iloc[tr_idx])
        dvalid = lgb.Dataset(train_X.iloc[va_idx], label=train_y.iloc[va_idx])

        step_offset = fold * 10000

        gbm = lgb.train(
            params=params,
            train_set=dtrain,
            valid_sets=[dvalid],
            valid_names=["valid"],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                make_pruning_cb(trial, metric="auc", valid_name="valid", step_offset=step_offset),
                lgb.log_evaluation(0),
            ],
        )

        oof_pred[va_idx] = gbm.predict(train_X.iloc[va_idx], num_iteration=gbm.best_iteration)
        auc = roc_auc_score(train_y.iloc[va_idx], oof_pred[va_idx])
        auc_scores.append(auc)

        test_preds_fold.append(gbm.predict(test_X, num_iteration=gbm.best_iteration))
        fold_best_iters.append(gbm.best_iteration or gbm.current_iteration())

    cv_auc = float(np.mean(auc_scores))
    test_pred = np.mean(np.vstack(test_preds_fold), axis=0)
    test_auc = roc_auc_score(test_y, test_pred)

    fpr, tpr, _ = roc_curve(test_y, test_pred)
    xs, ys = _smooth_roc(fpr, tpr)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    try:
        ax.plot(xs, ys, label=f"Test AUC: {test_auc:.5f}")
    except Exception as e:
        print(f"[Trial {trial.number}] ROC smooth failed: {e}, fallback to raw ROC")
        ax.plot(fpr, tpr, label=f"Test AUC: {test_auc:.5f}")
    ax.legend()
    ax.set_title("ROC Curve (Test)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    short_param = ", ".join(f"{k}={v:.3g}" for k, v in trial.params.items())
    ax.text(0.05, 0.05, short_param, fontsize=8, color='black',
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.5),
            transform=ax.transAxes)
    plt.savefig(f"out/lightgbm/plots/light_gbm_roc_curve_{trial.number}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[Trial {trial.number}] CV AUC: {cv_auc:.5f} | Test AUC: {test_auc:.5f}")

    os.makedirs("out/lightgbm", exist_ok=True)
    with open("out/lightgbm/optuna_trial_results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if trial.number == 0 and f.tell() == 0:
            writer.writerow(["trial_number", "cv_auc", "test_auc", "best_iter_mean", "params"])
        writer.writerow([trial.number, cv_auc, test_auc, int(np.mean(fold_best_iters)), json.dumps(trial.params)])

    trial.set_user_attr("cv_best_iter", int(np.mean(fold_best_iters)))
    return cv_auc

if __name__ == "__main__":
    train_df = pd.read_parquet("out/train_df.parquet")
    test_df  = pd.read_parquet("out/test_df.parquet")
    train_X, train_y = train_df[SELECTED_FEATURES], train_df["is_fraud"]
    test_X,  test_y  = test_df[SELECTED_FEATURES],  test_df["is_fraud"]

    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Study
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    obj = functools.partial(objective, train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y, kf=kf)

    study.optimize(obj, n_trials=30)

    vis.plot_optimization_history(study).write_html("out/lightgbm/lightgbm_opt_history.html")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) >= 2:
        vis.plot_param_importances(study).write_html("out/lightgbm/lightgbm_param_importances.html")
    else:
        print(f"[Skip] Need >=2 completed trials for param importances, got {len(completed)}")

    best = study.best_trial
    print("Best trial:")
    print(f"  CV AUC: {best.value:.5f}")
    print("  Best parameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    best_iter = best.user_attrs.get("cv_best_iter", 1000)
    final_params = {
        **best.params,
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "seed": 42,
        "bagging_seed": 42,
        "feature_fraction_seed": 42,
    }
    final_gbm = lgb.train(
        params=final_params,
        train_set=lgb.Dataset(train_X, label=train_y),
        num_boost_round=int(best_iter),
        valid_sets=[],
        callbacks=[lgb.log_evaluation(0)],
    )
    final_gbm.save_model("out/lightgbm/best_model.txt")