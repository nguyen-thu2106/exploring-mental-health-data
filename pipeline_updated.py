# ═══════════════════════════════════════════════════════════════════════
# PIPELINE CẬP NHẬT: Optuna Tuning + OOF Threshold Search
# ═══════════════════════════════════════════════════════════════════════
# Hướng dẫn: Copy từng section (đánh dấu CELL) vào notebook cell tương ứng.
# Thời gian chạy ước tính: ~30-60 phút (do Optuna tuning)
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# CELL 1: IMPORTS (thay thế cell imports cũ)
# ═══════════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pprint
import optuna

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from IPython.display import display

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
import joblib

# Tắt log verbose của Optuna (chỉ hiện summary)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════
# CELL 2: LOAD DATA (giữ nguyên)
# ═══════════════════════════════════════════════════════════════════════
train = pd.read_csv('data/processed/train_preprocessed.csv')
test = pd.read_csv('data/processed/test_preprocessed.csv')


# ═══════════════════════════════════════════════════════════════════════
# CELL 3: FEATURE SELECTION (giữ nguyên toàn bộ code cũ)
# ═══════════════════════════════════════════════════════════════════════
# ... (giữ nguyên cell Feature Selection từ notebook hiện tại)
# Kết quả: X_train_final, y_train_final, X_test_final


# ═══════════════════════════════════════════════════════════════════════
# CELL 4: HÀM OPTUNA TUNING (MỚI HOÀN TOÀN)
# ═══════════════════════════════════════════════════════════════════════

def optuna_tune_all_models(X, y, n_splits=5, n_trials=20, random_state=42):
    """
    Tự động tuning hyperparameters cho 4 base models bằng Optuna.

    Cơ chế:
    - Mỗi trial, Optuna đề xuất 1 bộ tham số
    - Đánh giá bằng Stratified K-Fold OOF AUC (mean)
    - Sau n_trials, trả về bộ tham số tốt nhất cho từng model

    Lưu ý: Ensemble (Soft/Weighted Voting, Stacking) KHÔNG cần Optuna vì:
    - Soft Voting: không có tham số
    - Weighted Voting: trọng số tính tự động từ AUC
    - Stacking (LR): LR đủ đơn giản, tune thêm dễ overfit 4 cột meta-features
    """

    pos_count = (y == 1).sum()
    neg_count = (y == 0).sum()
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    best_params_all = {}
    model_names = ["XGBoost", "Random Forest", "CatBoost", "LightGBM"]

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f" OPTUNA TUNING: {model_name} ({n_trials} trials × {n_splits} folds)")
        print(f"{'='*60}")

        start_time = time.time()

        def objective(trial):
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            fold_aucs = []

            # ── Đề xuất tham số tùy theo model ──
            if model_name == "XGBoost":
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'max_depth': trial.suggest_int('max_depth', 4, 10),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.5),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 1.0, log=True),
                }
                clf_template = XGBClassifier(
                    objective='binary:logistic', eval_metric='auc',
                    n_estimators=1000, scale_pos_weight=scale_pos_weight,
                    tree_method='hist', random_state=random_state,
                    n_jobs=-1, verbosity=0, early_stopping_rounds=100,
                    **params
                )

            elif model_name == "Random Forest":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                }
                clf_template = RandomForestClassifier(
                    class_weight='balanced', random_state=random_state,
                    **params
                )

            elif model_name == "CatBoost":
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                }
                clf_template = CatBoostClassifier(
                    iterations=2000, auto_class_weights='Balanced',
                    eval_metric='AUC', early_stopping_rounds=100,
                    verbose=0, random_state=random_state,
                    **params
                )

            elif model_name == "LightGBM":
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 1.0, log=True),
                }
                clf_template = LGBMClassifier(
                    objective='binary', n_estimators=2000,
                    class_weight='balanced', random_state=random_state,
                    verbosity=-1,
                    **params
                )

            # ── Chạy K-Fold CV ──
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

                clf = clone(clf_template)

                if isinstance(clf, LGBMClassifier):
                    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc',
                            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
                elif isinstance(clf, XGBClassifier):
                    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                elif isinstance(clf, CatBoostClassifier):
                    clf.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, verbose=False)
                else:
                    clf.fit(X_tr, y_tr)

                val_prob = clf.predict_proba(X_va)[:, 1]
                fold_aucs.append(roc_auc_score(y_va, val_prob))

            return np.mean(fold_aucs)

        # ── Chạy Optuna study ──
        study = optuna.create_study(direction='maximize',
                                     study_name=f'{model_name}_tuning')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        elapsed = time.time() - start_time
        best_params_all[model_name] = study.best_params

        print(f"\n Best AUC (CV mean): {study.best_value:.4f}")
        print(f" Thời gian: {elapsed:.1f}s")
        print(f" Best params:")
        for k, v in study.best_params.items():
            print(f"   {k}: {v}")

    return best_params_all


def build_models_from_tuned_params(best_params_all, scale_pos_weight, random_state=42):
    """
    Tạo dict models từ best params của Optuna.
    """
    models = {}

    if "XGBoost" in best_params_all:
        models["XGBoost"] = XGBClassifier(
            objective='binary:logistic', eval_metric='auc',
            n_estimators=1000, scale_pos_weight=scale_pos_weight,
            tree_method='hist', random_state=random_state,
            n_jobs=-1, verbosity=0, early_stopping_rounds=100,
            **best_params_all["XGBoost"]
        )

    if "Random Forest" in best_params_all:
        models["Random Forest"] = RandomForestClassifier(
            class_weight='balanced', random_state=random_state,
            **best_params_all["Random Forest"]
        )

    if "CatBoost" in best_params_all:
        models["CatBoost"] = CatBoostClassifier(
            iterations=2000, auto_class_weights='Balanced',
            eval_metric='AUC', early_stopping_rounds=100,
            verbose=100, random_state=random_state,
            **best_params_all["CatBoost"]
        )

    if "LightGBM" in best_params_all:
        models["LightGBM"] = LGBMClassifier(
            objective='binary', n_estimators=2000,
            class_weight='balanced', random_state=random_state,
            verbosity=-1,
            **best_params_all["LightGBM"]
        )

    return models


# ═══════════════════════════════════════════════════════════════════════
# CELL 5: CHẠY OPTUNA TUNING (MỚI)
# ═══════════════════════════════════════════════════════════════════════

# Tính scale_pos_weight
pos_count = (y_train_final == 1).sum()
neg_count = (y_train_final == 0).sum()
scale_pos_weight = float(neg_count / max(pos_count, 1))

# Chạy Optuna tuning cho tất cả models
best_params_all = optuna_tune_all_models(
    X_train_final, y_train_final,
    n_splits=5,
    n_trials=20,   # Có thể tăng lên 50 nếu có thời gian
    random_state=42
)

# Tạo models dict từ best params
models = build_models_from_tuned_params(best_params_all, scale_pos_weight)


# ═══════════════════════════════════════════════════════════════════════
# CELL 6: HÀM EVALUATE (giữ nguyên hàm evaluate_team_models cũ)
# ═══════════════════════════════════════════════════════════════════════

# Giữ nguyên hàm evaluate_team_models từ notebook hiện tại
# Hàm này đã return: results_df, oof_predictions
# oof_predictions chứa OOF probs cho từng model + ensemble


# ═══════════════════════════════════════════════════════════════════════
# CELL 7: CHẠY EVALUATION VỚI TUNED MODELS
# ═══════════════════════════════════════════════════════════════════════

results_df, oof_preds = evaluate_team_models(
    models,
    X_train_final,
    y_train_final,
    add_ensembles=True
)


# ═══════════════════════════════════════════════════════════════════════
# CELL 8: OOF THRESHOLD SEARCH (THAY THẾ cells retrain cũ)
# ═══════════════════════════════════════════════════════════════════════

# ── Bước 1: Chọn Best Model dựa trên bảng xếp hạng ──
# results_df đã sort theo Raw_AUC_Mean giảm dần (index 0 = tốt nhất)
best_model_name = results_df.iloc[0]['Model']
print(f"Best Model/Ensemble: {best_model_name}")

# ── Bước 2: Threshold Search trên OOF predictions ──
#
# Tại sao dùng OOF thay vì 15% val?
#   - OOF dùng TOÀN BỘ tập train (~140K mẫu), mỗi mẫu được dự đoán
#     bởi model CHƯA TỪNG thấy nó → không bị data leakage
#   - 15% val chỉ có ~21K mẫu → kém ổn định hơn
#   - OOF KHÔNG bị bias cho threshold search
#
# Tại sao KHÔNG average threshold từng fold?
#   - Tìm trên full OOF vector là cách chuẩn, ổn định hơn
#   - Average từng fold thêm phức tạp mà không cải thiện

best_oof_probs = oof_preds[best_model_name]
y_arr = y_train_final.to_numpy()

thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = []

for thr in thresholds:
    preds = (best_oof_probs >= thr).astype(int)
    f1_scores.append(f1_score(y_arr, preds))

best_idx = int(np.argmax(f1_scores))
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"\n{'='*60}")
print(f" OOF THRESHOLD SEARCH")
print(f"{'='*60}")
print(f" Threshold tối ưu (F1 cao nhất): {best_threshold:.2f}")
print(f" F1-Score tại ngưỡng tối ưu    : {best_f1:.4f}")

# ── Vẽ đồ thị F1 theo Threshold ──
plt.figure(figsize=(10, 4))
plt.plot(thresholds, f1_scores, color='steelblue', linewidth=2, label='F1-Score')
plt.axvline(best_threshold, color='crimson', linestyle='--', linewidth=1.5,
            label=f'Optimal threshold = {best_threshold:.2f}')
plt.scatter([best_threshold], [best_f1], color='crimson', zorder=5, s=80)
plt.xlabel('Threshold')
plt.ylabel('F1-Score')
plt.title(f'OOF Threshold Search (trên toàn bộ {len(y_arr)} mẫu train)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ── Báo cáo metrics tại ngưỡng tối ưu trên OOF ──
oof_preds_optimal = (best_oof_probs >= best_threshold).astype(int)
print(f"\n── Báo cáo trên OOF (threshold tối ưu = {best_threshold:.2f}) ──")
print(f"  F1-Score : {f1_score(y_arr, oof_preds_optimal):.4f}")
print(f"  Precision: {precision_score(y_arr, oof_preds_optimal):.4f}")
print(f"  Recall   : {recall_score(y_arr, oof_preds_optimal):.4f}")
print(f"  Accuracy : {accuracy_score(y_arr, oof_preds_optimal):.4f}")
print(f"  AUC      : {roc_auc_score(y_arr, best_oof_probs):.4f}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 9: RETRAIN BEST MODEL + DỰ ĐOÁN TEST
# ═══════════════════════════════════════════════════════════════════════

# Chỉ retrain nếu best model là single model (không phải ensemble)
# Nếu best là ensemble → cần retrain tất cả base models
is_ensemble = "Ensemble" in best_model_name

if not is_ensemble:
    print(f"\n{'='*60}")
    print(f" RETRAIN: {best_model_name}")
    print(f"{'='*60}")

    # Tạo lại model với best params từ Optuna
    final_model_params = best_params_all[best_model_name]

    if best_model_name == "LightGBM":
        best_model = LGBMClassifier(
            objective='binary', n_estimators=2000,
            class_weight='balanced', random_state=42, verbosity=-1,
            **final_model_params
        )
    elif best_model_name == "XGBoost":
        best_model = XGBClassifier(
            objective='binary:logistic', eval_metric='auc',
            n_estimators=1000, scale_pos_weight=scale_pos_weight,
            tree_method='hist', random_state=42, n_jobs=-1,
            verbosity=0, early_stopping_rounds=100,
            **final_model_params
        )
    elif best_model_name == "CatBoost":
        best_model = CatBoostClassifier(
            iterations=2000, auto_class_weights='Balanced',
            eval_metric='AUC', early_stopping_rounds=100,
            verbose=100, random_state=42,
            **final_model_params
        )
    elif best_model_name == "Random Forest":
        best_model = RandomForestClassifier(
            class_weight='balanced', random_state=42,
            **final_model_params
        )

    # Retrain trên 85% + 15% validation cho early stopping
    print("Đang retrain trên 85% tập Train (15% cho early stopping)...")
    X_tr_full, X_val_full, y_tr_full, y_val_full = train_test_split(
        X_train_final, y_train_final,
        test_size=0.15, stratify=y_train_final, random_state=42
    )

    if isinstance(best_model, LGBMClassifier):
        best_model.fit(
            X_tr_full, y_tr_full,
            eval_set=[(X_val_full, y_val_full)],
            eval_metric='auc',
            callbacks=[early_stopping(50, verbose=False), log_evaluation(100)]
        )
        print(f"\nDừng ở iteration: {best_model.best_iteration_}")
        print(f"AUC tốt nhất trên Val: {best_model.best_score_['valid_0']['auc']:.4f}")

    elif isinstance(best_model, XGBClassifier):
        best_model.fit(
            X_tr_full, y_tr_full,
            eval_set=[(X_val_full, y_val_full)],
            verbose=False
        )
        print(f"\nDừng ở iteration: {best_model.best_iteration}")

    elif isinstance(best_model, CatBoostClassifier):
        best_model.fit(
            X_tr_full, y_tr_full,
            eval_set=(X_val_full, y_val_full),
            use_best_model=True, verbose=100
        )
        print(f"\nDừng ở iteration: {best_model.best_iteration_}")

    else:
        # Random Forest — không có early stopping
        best_model.fit(X_tr_full, y_tr_full)
        print("Hoàn tất training (RF không dùng early stopping)")

    # ── Dự đoán test ──
    test_probs = best_model.predict_proba(X_test_final)[:, 1]
    test_predictions = (test_probs >= best_threshold).astype(int)

    print(f"\nThreshold áp dụng: {best_threshold:.2f}")
    print(f"Phân phối nhãn test: 0 → {(test_predictions==0).sum()} | "
          f"1 → {(test_predictions==1).sum()}")

else:
    # ── Xử lý trường hợp Best là Ensemble ──
    print(f"\n{'='*60}")
    print(f" RETRAIN ENSEMBLE: {best_model_name}")
    print(f"{'='*60}")

    # Split cho early stopping
    X_tr_full, X_val_full, y_tr_full, y_val_full = train_test_split(
        X_train_final, y_train_final,
        test_size=0.15, stratify=y_train_final, random_state=42
    )

    # Retrain tất cả base models
    base_model_names = list(best_params_all.keys())
    retrained_models = {}

    for bm_name in base_model_names:
        print(f"  Retrain: {bm_name}...")
        bm_params = best_params_all[bm_name]

        if bm_name == "LightGBM":
            bm = LGBMClassifier(objective='binary', n_estimators=2000,
                                class_weight='balanced', random_state=42, verbosity=-1,
                                **bm_params)
            bm.fit(X_tr_full, y_tr_full, eval_set=[(X_val_full, y_val_full)],
                    eval_metric='auc',
                    callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])

        elif bm_name == "XGBoost":
            bm = XGBClassifier(objective='binary:logistic', eval_metric='auc',
                               n_estimators=1000, scale_pos_weight=scale_pos_weight,
                               tree_method='hist', random_state=42, n_jobs=-1,
                               verbosity=0, early_stopping_rounds=100, **bm_params)
            bm.fit(X_tr_full, y_tr_full, eval_set=[(X_val_full, y_val_full)], verbose=False)

        elif bm_name == "CatBoost":
            bm = CatBoostClassifier(iterations=2000, auto_class_weights='Balanced',
                                     eval_metric='AUC', early_stopping_rounds=100,
                                     verbose=0, random_state=42, **bm_params)
            bm.fit(X_tr_full, y_tr_full, eval_set=(X_val_full, y_val_full),
                    use_best_model=True, verbose=False)

        elif bm_name == "Random Forest":
            bm = RandomForestClassifier(class_weight='balanced', random_state=42, **bm_params)
            bm.fit(X_tr_full, y_tr_full)

        retrained_models[bm_name] = bm

    # Tạo test predictions từ ensemble
    test_probs_matrix = np.column_stack([
        retrained_models[name].predict_proba(X_test_final)[:, 1]
        for name in base_model_names
    ])

    if best_model_name == "Ensemble - Soft Voting":
        test_probs = test_probs_matrix.mean(axis=1)

    elif best_model_name == "Ensemble - Weighted Voting":
        # Lấy weights từ OOF AUC
        base_aucs = np.array([
            roc_auc_score(y_arr, oof_preds[name]) for name in base_model_names
        ])
        weights = base_aucs / base_aucs.sum()
        test_probs = np.average(test_probs_matrix, axis=1, weights=weights)

    elif best_model_name == "Ensemble - Stacking (LR)":
        # Retrain meta-learner trên OOF
        oof_matrix = np.column_stack([oof_preds[name] for name in base_model_names])
        meta_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        meta_model.fit(oof_matrix, y_arr)
        test_probs = meta_model.predict_proba(test_probs_matrix)[:, 1]

    test_predictions = (test_probs >= best_threshold).astype(int)

    print(f"\nThreshold áp dụng: {best_threshold:.2f}")
    print(f"Phân phối nhãn test: 0 → {(test_predictions==0).sum()} | "
          f"1 → {(test_predictions==1).sum()}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 10: LƯU SUBMISSION + MODEL
# ═══════════════════════════════════════════════════════════════════════

# Tạo file submission
submission = pd.DataFrame({'Depression': test_predictions})
submission.to_csv('submission_lgbm_final.csv', index=False)
print("Đã lưu file submission_lgbm_final.csv")
print(submission['Depression'].value_counts())

# Lưu model + threshold + Optuna best params
artifact = {
    'best_model_name': best_model_name,
    'threshold': best_threshold,
    'optuna_params': best_params_all,
}
if not is_ensemble:
    artifact['model'] = best_model
else:
    artifact['base_models'] = retrained_models
    if best_model_name == "Ensemble - Stacking (LR)":
        artifact['meta_model'] = meta_model

joblib.dump(artifact, 'best_model_optuna.pkl')
print(f"\nĐã lưu model + threshold ({best_threshold:.2f}) + Optuna params "
      f"vào 'best_model_optuna.pkl'")
