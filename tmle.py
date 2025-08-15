import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.pipeline import make_pipeline  
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import StackingClassifier 
from scipy.stats import norm
from sklearn.metrics import precision_score, recall_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.calibration import CalibratedClassifierCV  # 🆕 新增校準功能
from sklearn.isotonic import IsotonicRegression  # 🆕 等溫回歸校準
import warnings
from tqdm import tqdm
import time
from sklearn.utils import resample
import matplotlib.pyplot as plt  # 🆕 用於校準圖表
from sklearn.calibration import calibration_curve
############################################################################
# Base Models
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifier, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
import lightgbm as lgb
import warnings

def tmle_project(file_path, test_size=0.3, random_state=42, calibration_method='platt'):
    
    def data_loading(file_path, test_size=0.3, random_state=42):
        df = pd.read_csv(file_path)
        Y = df["Y"].values
        A = df["A"].values
        W = df.drop(columns=["Y", "A"])
        W_A = df.drop(columns=["Y"])
        
        # 🆕 添加train/test分割
        indices = np.arange(len(Y))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, 
            stratify=A  # 確保treatment分佈一致
        )
        
        print(f"📊 Data Split Summary:")
        print(f"   Total samples: {len(Y)}")
        print(f"   Train samples: {len(train_idx)} ({len(train_idx)/len(Y)*100:.1f}%)")
        print(f"   Test samples: {len(test_idx)} ({len(test_idx)/len(Y)*100:.1f}%)")
        print(f"   Train treatment prop: {A[train_idx].mean():.3f}")
        print(f"   Test treatment prop: {A[test_idx].mean():.3f}")
        
        return {
            'train': {
                'Y': Y[train_idx], 'A': A[train_idx], 
                'W': W.iloc[train_idx], 'W_A': W_A.iloc[train_idx],
                'indices': train_idx
            },
            'test': {
                'Y': Y[test_idx], 'A': A[test_idx],
                'W': W.iloc[test_idx], 'W_A': W_A.iloc[test_idx], 
                'indices': test_idx
            }
        }

    def data_preprocessing(W_train, W_test=None):
        '''data preprocessing function with proper train/test handling'''
        print(f"<< shape of raw train data>>: {W_train.shape}")
        print(f"<< missing values in train>>: {W_train.isnull().sum().sum()}")
        
        # 處理訓練數據
        W_train_clean = W_train.fillna(W_train.median())
        scaler = StandardScaler()
        W_train_standardized = scaler.fit_transform(W_train_clean)
        W_train_df = pd.DataFrame(W_train_standardized, columns=W_train.columns)
        
        if W_test is not None:
            # 處理測試數據 - 使用訓練集的參數
            W_test_clean = W_test.fillna(W_train.median())
            W_test_standardized = scaler.transform(W_test_clean)
            W_test_df = pd.DataFrame(W_test_standardized, columns=W_test.columns)
            return W_train_df, W_test_df, scaler
        
        return W_train_df, scaler

    def get_base_learners(n_features):
        '''base learners with improved parameters'''
        base_learners = [
            # ========== LINEAR MODELS ==========
            ('logistic_cv', LogisticRegressionCV(
                cv=5, 
                max_iter=10000, 
                random_state=42,
                solver='lbfgs',
                scoring='roc_auc',
                class_weight='balanced'
            )),
            
            ('logistic_l1', LogisticRegressionCV(
                cv=5, 
                max_iter=10000, 
                penalty='l1',
                solver='liblinear', 
                random_state=42,
                tol=1e-4,
                scoring='roc_auc',
                class_weight='balanced'
            )),
            
            ('logistic_elastic', make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(
                    cv=5,
                    penalty='elasticnet',
                    solver='saga',
                    l1_ratios=[0.1, 0.5, 0.7, 0.9],
                    max_iter=5000,
                    random_state=42,
                    class_weight='balanced'
                )
            )),
            
            # ========== TREE-BASED MODELS ==========
            ('rf', RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )),
            
            ('extra_trees', ExtraTreesClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )),
            
            ('gbm', GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                max_features='sqrt',
                random_state=42,
                validation_fraction=0.1,
                n_iter_no_change=10
            )),
            
            ('xgb', XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                scale_pos_weight=1,
                random_state=42,
                eval_metric='logloss',
                verbosity=0,
                n_jobs=-1
            )),
            
            ('lgbm', lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=-1,
                n_jobs=-1,
                class_weight='balanced'
            )),
            
            # ========== NON-LINEAR MODELS ==========
            ('svm_rbf', make_pipeline(
                StandardScaler(),
                SVC(
                    probability=True,
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    class_weight='balanced',
                    random_state=42
                )
            )),
            
            ('svm_linear', make_pipeline(
                StandardScaler(),
                SVC(
                    probability=True,
                    kernel='linear',
                    C=0.1,
                    class_weight='balanced',
                    random_state=42
                )
            )),
            
            ('mlp', make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(50, 30, 15),
                    max_iter=3000,
                    alpha=0.01,
                    learning_rate='adaptive',
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                    random_state=42
                )
            )),
            
            # ========== SIMPLE MODELS ==========
            ('nb', make_pipeline(
                StandardScaler(),
                GaussianNB()
            )),
            
            ('knn', make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(
                    n_neighbors=15,
                    weights='distance',
                    n_jobs=-1
                )
            )),
            
            ('dt', DecisionTreeClassifier(
                max_depth=6,
                min_samples_split=30,
                min_samples_leaf=15,
                class_weight='balanced',
                random_state=42
            ))
        ]
        return base_learners

    def fit_superlearner(X, Y, base_learners, model_name="SuperLearner"):
        print(f"\n << Fitting {model_name} >>")
        
        pbar = tqdm(total=len(base_learners) + 2, desc=f"<< Training {model_name} >>", leave=False)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            meta_learner = LogisticRegressionCV(cv=3, max_iter=5000, 
                                              random_state=42, solver='lbfgs')
            
            pbar.set_description(f"Building {model_name}")
            pbar.update(1)
            
            sl = StackingClassifier(
                estimators=base_learners,
                cv=3,
                stack_method='predict_proba',
                final_estimator=meta_learner,
                n_jobs=-1
            )
            
            try:
                pbar.set_description(f"<< Fitting {model_name} >>   ")
                sl.fit(X, Y)
                pbar.update(1)

                pbar.set_description(f"<< Evaluating {model_name} >>")
                cv_scores = cross_val_score(sl, X, Y, cv=3, scoring='roc_auc')
                print(f"{model_name} Train CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"Warning: {model_name} fitting failed: {str(e)}")
                print("Using simplified model...")
                from sklearn.linear_model import LogisticRegression
                sl = LogisticRegression(max_iter=5000, random_state=42)
                sl.fit(X, Y)
            
            finally:
                pbar.close()
        
        return sl

    # 🆕 新增校準相關函數
    def evaluate_calibration(y_true, y_prob, n_bins=10):
        """評估校準品質"""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )
        
        # 計算校準誤差 (Calibration Error)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_error = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Brier Score
        brier_score = brier_score_loss(y_true, y_prob)
        
        return {
            'calibration_error': calibration_error,
            'brier_score': brier_score,
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value
        }

    def calibrate_classifier(base_model, X_train, y_train, method='platt'):
        """
        對分類器進行校準
        method: 'platt' (sigmoid), 'isotonic' (isotonic regression)
        """
        print(f"   🎯 Calibrating classifier using {method} method...")
        
        if method == 'platt':
            calibrated_model = CalibratedClassifierCV(
                base_model, method='sigmoid', cv=3
            )
        elif method == 'isotonic':
            calibrated_model = CalibratedClassifierCV(
                base_model, method='isotonic', cv=3
            )
        else:
            print(f"   ⚠️  Unknown calibration method: {method}. Using Platt scaling.")
            calibrated_model = CalibratedClassifierCV(
                base_model, method='sigmoid', cv=3
            )
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            calibrated_model.fit(X_train, y_train)
        
        return calibrated_model

    def predict_Q_models(sl, W_A_data, A_data, is_test=False):
        """Predict Q models - 可在訓練集或測試集上預測"""
        data_type = "test" if is_test else "train"
        print(f"   Predicting Q models on {data_type} set...")
        
        try:
            Q_A = sl.predict_proba(W_A_data)[:, 1]
        except:
            Q_A = sl.predict_proba(W_A_data)[:, 1]
        
        # Predict Q1 and Q0
        W_A1 = W_A_data.copy()
        W_A1.iloc[:, -1] = 1  # 假設A是最後一列
        Q_1 = sl.predict_proba(W_A1)[:, 1]
        
        W_A0 = W_A_data.copy()
        W_A0.iloc[:, -1] = 0
        Q_0 = sl.predict_proba(W_A0)[:, 1]
        
        return Q_A, Q_1, Q_0

    # 🆕 修改 estimate_g 函數，加入校準和訓練集性能評估
    def estimate_g_with_calibration(A_train, W_train_standardized, A_test, W_test_standardized, base_learners, calibration_method='platt'):
        """
        帶校準的propensity score estimation，並返回訓練和測試集性能
        """
        print(f"\n << Estimating propensity scores with {calibration_method} calibration >>")
        
        # 1. 在訓練集上進行下採樣
        treated_idx = A_train == 1
        control_idx = A_train == 0
        W_treated = W_train_standardized[treated_idx]
        W_control = W_train_standardized[control_idx]
        A_treated = A_train[treated_idx]
        A_control = A_train[control_idx]

        if len(W_treated) > len(W_control):
            W_treated_down = resample(W_treated, replace=False, n_samples=len(W_control), random_state=42)
            A_treated_down = resample(A_treated, replace=False, n_samples=len(W_control), random_state=42)
            W_down = np.vstack([W_treated_down, W_control])
            A_down = np.concatenate([A_treated_down, A_control])
        else:
            W_control_down = resample(W_control, replace=False, n_samples=len(W_treated), random_state=42)
            A_control_down = resample(A_control, replace=False, n_samples=len(W_treated), random_state=42)
            W_down = np.vstack([W_treated, W_control_down])
            A_down = np.concatenate([A_treated, A_control_down])

        print(f"g-model training samples (downsampled): {W_down.shape}, A=1 proportion: {np.mean(A_down):.2f}")

        # 2. 訓練基礎g-model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            base_g_model = fit_superlearner(W_down, A_down, base_learners, "Base Propensity Score Model")

        # 🆕 3. 校準g-model
        print("   📊 Evaluating calibration before calibration...")
        base_probs_train = base_g_model.predict_proba(W_down)[:, 1]
        pre_cal_metrics = evaluate_calibration(A_down, base_probs_train)
        print(f"   Pre-calibration: CE={pre_cal_metrics['calibration_error']:.4f}, Brier={pre_cal_metrics['brier_score']:.4f}")
        
        calibrated_g_model = calibrate_classifier(base_g_model, W_down, A_down, calibration_method)
        
        # 評估校準後效果
        cal_probs_train = calibrated_g_model.predict_proba(W_down)[:, 1]
        post_cal_metrics = evaluate_calibration(A_down, cal_probs_train)
        print(f"   Post-calibration: CE={post_cal_metrics['calibration_error']:.4f}, Brier={post_cal_metrics['brier_score']:.4f}")
        
        improvement_ce = pre_cal_metrics['calibration_error'] - post_cal_metrics['calibration_error']
        improvement_brier = pre_cal_metrics['brier_score'] - post_cal_metrics['brier_score']
        print(f"   📈 Improvement: CE Δ={improvement_ce:+.4f}, Brier Δ={improvement_brier:+.4f}")

        # 🆕 4. 在完整訓練集上評估校準後模型的性能
        print("   📊 Evaluating calibrated model on full training set...")
        g_w_train_full = calibrated_g_model.predict_proba(W_train_standardized)[:, 1]
        train_cal_metrics = evaluate_calibration(A_train, g_w_train_full)
        
        # 🆕 5. 在測試集上預測propensity scores (使用校準後的模型)
        print("   Predicting calibrated propensity scores on test set...")
        g_w_test = calibrated_g_model.predict_proba(W_test_standardized)[:, 1]

        # 評估測試集校準品質
        test_cal_metrics = evaluate_calibration(A_test, g_w_test)
        print(f"   Test set calibration: CE={test_cal_metrics['calibration_error']:.4f}, Brier={test_cal_metrics['brier_score']:.4f}")

        # 🆕 6. OVERLAP WEIGHTING計算 - 同時為訓練集和測試集
        # 訓練集overlap weights
        g_w_train_trimmed = np.clip(g_w_train_full, 0.05, 0.95)
        overlap_weights_train = g_w_train_trimmed * (1 - g_w_train_trimmed)
        H_overlap_train = A_train * (1 - g_w_train_trimmed) - (1 - A_train) * g_w_train_trimmed
        H_1_overlap_train = (1 - g_w_train_trimmed)
        H_0_overlap_train = g_w_train_trimmed
        
        # 測試集overlap weights
        g_w_test_trimmed = np.clip(g_w_test, 0.05, 0.95)
        overlap_weights_test = g_w_test_trimmed * (1 - g_w_test_trimmed)
        H_overlap_test = A_test * (1 - g_w_test_trimmed) - (1 - A_test) * g_w_test_trimmed
        H_1_overlap_test = (1 - g_w_test_trimmed)
        H_0_overlap_test = g_w_test_trimmed

        # 7. DIAGNOSTIC INFORMATION
        print(f"\n📊 Training Set PS Performance:")
        print(f"   Propensity Score: min={g_w_train_full.min():.4f}, max={g_w_train_full.max():.4f}, mean={g_w_train_full.mean():.4f}")
        print(f"   Overlap Weights: min={overlap_weights_train.min():.4f}, max={overlap_weights_train.max():.4f}, mean={overlap_weights_train.mean():.4f}")
        train_good_overlap = np.sum((g_w_train_trimmed >= 0.1) & (g_w_train_trimmed <= 0.9))
        print(f"   Good overlap samples (0.1 ≤ PS ≤ 0.9): {train_good_overlap} ({train_good_overlap/len(g_w_train_trimmed)*100:.1f}%)")

        # 🆕 印出downsample+overlap weight+calibration後的平均
        print(f"   [Average after downsample+overlap+calibration] g_w_train_trimmed mean: {g_w_train_trimmed.mean():.4f}")
        print(f"   [Average after downsample+overlap+calibration] overlap_weights_train mean: {overlap_weights_train.mean():.4f}")

        # 🆕 更詳細的訓練集 g-model 分布圖
        import seaborn as sns
        plt.figure(figsize=(8,5))
        # 分組 treated/control
        treated_scores = g_w_train_trimmed[A_train == 1]
        control_scores = g_w_train_trimmed[A_train == 0]
        sns.histplot(treated_scores, bins=30, color='royalblue', label='Treated', kde=True, stat='density', alpha=0.6)
        sns.histplot(control_scores, bins=30, color='orange', label='Control', kde=True, stat='density', alpha=0.6)
        # 標註 mean/median
        plt.axvline(g_w_train_trimmed.mean(), color='green', linestyle='--', label=f'Mean: {g_w_train_trimmed.mean():.2f}')
        plt.axvline(np.median(g_w_train_trimmed), color='red', linestyle=':', label=f'Median: {np.median(g_w_train_trimmed):.2f}')
        plt.title('Training Set Propensity Score Distribution (g_w_train_trimmed)')
        plt.xlabel('Propensity Score')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        print(f"\n📊 Test Set PS Performance:")
        print(f"   Propensity Score: min={g_w_test.min():.4f}, max={g_w_test.max():.4f}, mean={g_w_test.mean():.4f}")
        print(f"   Overlap Weights: min={overlap_weights_test.min():.4f}, max={overlap_weights_test.max():.4f}, mean={overlap_weights_test.mean():.4f}")
        test_good_overlap = np.sum((g_w_test_trimmed >= 0.1) & (g_w_test_trimmed <= 0.9))
        print(f"   Good overlap samples (0.1 ≤ PS ≤ 0.9): {test_good_overlap} ({test_good_overlap/len(g_w_test_trimmed)*100:.1f}%)")
        print(f"   [Average after overlap+calibration] g_w_test_trimmed mean: {g_w_test_trimmed.mean():.4f}")
        print(f"   [Average after overlap+calibration] overlap_weights_test mean: {overlap_weights_test.mean():.4f}")

        # 🆕 更詳細的測試集 g-model 分布圖
        plt.figure(figsize=(8,5))
        treated_scores_test = g_w_test_trimmed[A_test == 1]
        control_scores_test = g_w_test_trimmed[A_test == 0]
        sns.histplot(treated_scores_test, bins=30, color='royalblue', label='Treated', kde=True, stat='density', alpha=0.6)
        sns.histplot(control_scores_test, bins=30, color='orange', label='Control', kde=True, stat='density', alpha=0.6)
        plt.axvline(g_w_test_trimmed.mean(), color='green', linestyle='--', label=f'Mean: {g_w_test_trimmed.mean():.2f}')
        plt.axvline(np.median(g_w_test_trimmed), color='red', linestyle=':', label=f'Median: {np.median(g_w_test_trimmed):.2f}')
        plt.title('Test Set Propensity Score Distribution (g_w_test_trimmed)')
        plt.xlabel('Propensity Score')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        # 🆕 為了與原始模型的測試性能比較，也在測試集上評估基礎模型
        base_probs_test = base_g_model.predict_proba(W_test_standardized)[:, 1]
        base_test_metrics = evaluate_calibration(A_test, base_probs_test)

        # 🆕 返回完整的校準相關信息，包含訓練集性能與平均值
        calibration_info = {
            'pre_calibration_metrics': pre_cal_metrics,
            'post_calibration_metrics': post_cal_metrics,
            'train_calibration_metrics': train_cal_metrics,  # 🆕 訓練集校準性能
            'test_calibration_metrics': test_cal_metrics,
            'base_test_metrics': base_test_metrics,
            'calibration_method': calibration_method,
            # 🆕 訓練集相關信息
            'train_ps_scores': g_w_train_full,
            'train_overlap_weights': overlap_weights_train,
            # 🆕 平均值
            'g_w_train_trimmed_mean': g_w_train_trimmed.mean(),
            'overlap_weights_train_mean': overlap_weights_train.mean(),
            'g_w_test_trimmed_mean': g_w_test_trimmed.mean(),
            'overlap_weights_test_mean': overlap_weights_test.mean()
        }
        from figure2 import plot_g_model_distribution

        # 訓練集分布
        plot_g_model_distribution(
            g_w_train_trimmed, A_train,
            title="Propensity Score Distribution (Train, Downsampled + Overlap + Calibration)",
            extreme_focus=True, show_kde=True, show_mean_median=True
        )

        # 測試集分布
        plot_g_model_distribution(
            g_w_test_trimmed, A_test,
            title="Propensity Score Distribution (Test, Overlap + Calibration)",
            extreme_focus=True, show_kde=True, show_mean_median=True
        )
        # 🆕 返回額外的訓練集信息
        return {
            'model': calibrated_g_model,
            'test': {
                'g_w': g_w_test_trimmed,
                'H_1': H_1_overlap_test,
                'H_0': H_0_overlap_test,
                'H_overlap': H_overlap_test,
                'overlap_weights': overlap_weights_test
            },
            'train': {  # 🆕 訓練集信息
                'g_w': g_w_train_trimmed,
                'H_1': H_1_overlap_train,
                'H_0': H_0_overlap_train,
                'H_overlap': H_overlap_train,
                'overlap_weights': overlap_weights_train
            },
            'calibration_info': calibration_info
        }

    def estimate_fluctuation_param(Y, Q_A, H_1, H_0, A, H_overlap=None):
        """
        Estimation of fluctuation parameters for Overlap Weighting
        """
        print("📈 Estimating fluctuation parameter (Overlap Weighting)...")
        
        Q_A_clipped = np.clip(Q_A, 1e-6, 1 - 1e-6)
        logit_QA = np.log(Q_A_clipped / (1 - Q_A_clipped))

        if H_overlap is not None:
            H_A = H_overlap
        else:
            H_A = A * H_1 - (1 - A) * H_0
            
        H_A = H_A.reshape(-1, 1) if H_A.ndim == 1 else H_A

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = sm.GLM(Y, H_A, offset=logit_QA, family=sm.families.Binomial()).fit()
                eps = model.params[0]
                print(f"   Fluctuation parameter (epsilon): {eps:.6f}")
        except Exception as e:
            print(f"   GLM fitting failed, using fallback method: {str(e)}")
            eps = 0.0
            
        return eps

    def update_Q(Q_base, H, eps):
        """Update Q values using the fluctuation parameter"""
        Q_clipped = np.clip(Q_base, 1e-6, 1 - 1e-6)
        logit_Q = np.log(Q_clipped / (1 - Q_clipped))
        updated_Q = 1 / (1 + np.exp(-(logit_Q + eps * H)))
        return np.clip(updated_Q, 1e-6, 1 - 1e-6)

    def compute_tmle(Y, A, Q_A_update, Q_1_update, Q_0_update, H_1, H_0, overlap_weights=None, H_overlap=None):
        """
        簡化版本: 只計算 ATE 和 ATT
        """
        print("🎯 Computing TMLE estimates (ATE, ATT)...")
        
        if overlap_weights is not None:
            # ATE with Overlap Weighting
            ate_numerator = np.mean((Q_1_update - Q_0_update) * overlap_weights)
            ate_denominator = np.mean(overlap_weights)
            ate = ate_numerator / ate_denominator
            
            # ATT
            treated_idx = (A == 1)
            if np.any(treated_idx):
                att_numerator = np.mean((Q_1_update[treated_idx] - Q_0_update[treated_idx]) * overlap_weights[treated_idx])
                att_denominator = np.mean(overlap_weights[treated_idx])
                att = att_numerator / att_denominator if att_denominator > 0 else np.nan
            else:
                att = np.nan

            # Influence function
            if H_overlap is not None:
                term1 = H_overlap * (Y - Q_A_update)
            else:
                term1 = (A * H_1 - (1 - A) * H_0) * (Y - Q_A_update)
                
            term2 = ((Q_1_update - Q_0_update) * overlap_weights - ate * overlap_weights) / ate_denominator
            infl_fn = term1 + term2 - ate
            
        else:
            # Fallback: 傳統計算（無overlap weighting）
            ate = np.mean(Q_1_update - Q_0_update)
            att = np.mean((Q_1_update - Q_0_update)[A == 1]) if np.any(A == 1) else np.nan
            
            H_A = A * H_1 - (1 - A) * H_0
            infl_fn = H_A * (Y - Q_A_update) + (Q_1_update - Q_0_update) - ate
        
        # 計算標準誤和置信區間
        se = np.sqrt(np.var(infl_fn) / len(Y))
        ci_low = ate - 1.96 * se
        ci_high = ate + 1.96 * se
        p_value = 2 * (1 - norm.cdf(abs(ate / se))) if se > 0 else 1.0
        
        return ate, se, ci_low, ci_high, p_value, infl_fn, att

    # 🆕 增強diagnostic_checks函數，加入更多測試集性能指標
    def diagnostic_checks(Y, A, W, Q_A, Q_1, Q_0, g_w, stage="First", is_test=False, model_performance=None):
        """Enhanced diagnostic checks with comprehensive test performance metrics"""
        data_type = "Test" if is_test else "Train"
        print(f"\n=== {stage} {data_type} Diagnostic Checks ===")
        
        treatment_prop = A.mean()
        print(f"{data_type} Treatment group proportion: {treatment_prop:.4f}")

        g_treated = g_w[A==1].mean()
        g_control = g_w[A==0].mean()
        g_overlap = np.minimum(g_w, 1-g_w).mean()

        print(f"{data_type} Propensity Score - Treated Mean: {g_treated:.4f}")
        print(f"{data_type} Propensity Score - Control Mean: {g_control:.4f}")
        print(f"{data_type} Overlap Measure: {g_overlap:.4f} (Higher is better)")

        # 🆕 Q-model performance metrics
        y_pred = (Q_A >= 0.5).astype(int)
        auc = roc_auc_score(Y, Q_A)
        precision = precision_score(Y, y_pred, zero_division=0)
        recall = recall_score(Y, y_pred, zero_division=0)
        
        print(f"{data_type} Q model AUC: {auc:.4f}")
        print(f"{data_type} Q model Precision: {precision:.4f}")
        print(f"{data_type} Q model Recall: {recall:.4f}")
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            print(f"{data_type} Q model F1 Score: {f1:.4f}")
        else:
            f1 = 0.0

        # 🆕 G-model (propensity score) performance metrics - 支援訓練集和測試集
        print(f"\n--- {data_type} G-model Performance ---")
        g_pred = (g_w >= 0.5).astype(int)
        g_auc = roc_auc_score(A, g_w)
        g_precision = precision_score(A, g_pred, zero_division=0)
        g_recall = recall_score(A, g_pred, zero_division=0)
        g_brier = brier_score_loss(A, g_w)
        
        print(f"{data_type} G model AUC: {g_auc:.4f}")
        print(f"{data_type} G model Precision: {g_precision:.4f}")
        print(f"{data_type} G model Recall: {g_recall:.4f}")
        if g_precision + g_recall > 0:
            g_f1 = 2 * (g_precision * g_recall) / (g_precision + g_recall)
            print(f"{data_type} G model F1 Score: {g_f1:.4f}")
        else:
            g_f1 = 0.0
        print(f"{data_type} G model Brier Score: {g_brier:.4f}")
        
        # 🆕 校準相關指標 - 支援訓練集和測試集
        if model_performance is not None:
            if is_test and 'test_calibration_metrics' in model_performance:
                cal_metrics = model_performance['test_calibration_metrics']
                print(f"{data_type} G model Calibration Error: {cal_metrics['calibration_error']:.4f}")
                print(f"{data_type} G model Calibration Brier: {cal_metrics['brier_score']:.4f}")
            elif not is_test and 'train_calibration_metrics' in model_performance:
                cal_metrics = model_performance['train_calibration_metrics']
                print(f"{data_type} G model Calibration Error: {cal_metrics['calibration_error']:.4f}")
                print(f"{data_type} G model Calibration Brier: {cal_metrics['brier_score']:.4f}")

        print(f"{data_type} Q_A distribution: min={Q_A.min():.4f}, max={Q_A.max():.4f}, mean={Q_A.mean():.4f}")
        print(f"{data_type} Q_1 distribution: min={Q_1.min():.4f}, max={Q_1.max():.4f}, mean={Q_1.mean():.4f}")
        print(f"{data_type} Q_0 distribution: min={Q_0.min():.4f}, max={Q_0.max():.4f}, mean={Q_0.mean():.4f}")

        raw_ate = np.mean(Q_1 - Q_0)
        print(f"{data_type} {stage} ATE estimate: {raw_ate:.6f}")

        extreme_ps = np.sum((g_w < 0.05) | (g_w > 0.95))
        print(f"{data_type} Extreme propensity score samples: {extreme_ps} ({extreme_ps/len(g_w)*100:.2f}%)")

        # 🆕 返回更完整的診斷信息
        diagnostic_results = {
            'treatment_prop': treatment_prop,
            'g_treated': g_treated,
            'g_control': g_control,
            'g_overlap': g_overlap,
            'q_auc': auc,
            'q_precision': precision,
            'q_recall': recall,
            'q_f1': f1,
            'raw_ate': raw_ate,
            'extreme_ps_count': extreme_ps,
            # 🆕 G-model性能指標
            'g_auc': g_auc,
            'g_precision': g_precision,
            'g_recall': g_recall,
            'g_f1': g_f1,
            'g_brier': g_brier
        }

        return diagnostic_results

    # 🆕 新增訓練/測試性能比較函數
    def print_train_test_comparison(train_metrics, test_metrics, title="Model Performance Comparison"):
        """打印訓練集和測試集的性能比較"""
        print(f"\n" + "="*80)
        print(f"                    {title}")
        print("="*80)
        
        print("Q-MODEL (Outcome Model) Performance:")
        print(f"{'Metric':<20} {'Train':<12} {'Test':<12} {'Difference':<12} {'Overfitting?':<12}")
        print("-" * 80)
        
        q_metrics = ['q_auc', 'q_precision', 'q_recall', 'q_f1']
        q_names = ['AUC', 'Precision', 'Recall', 'F1 Score']
        
        for metric, name in zip(q_metrics, q_names):
            train_val = train_metrics.get(metric, 0)
            test_val = test_metrics.get(metric, 0)
            diff = train_val - test_val
            overfitting = "Yes" if diff > 0.05 else "No"
            print(f"{name:<20} {train_val:<12.4f} {test_val:<12.4f} {diff:+12.4f} {overfitting:<12}")
        
        print("\nG-MODEL (Propensity Score Model) Performance:")
        print(f"{'Metric':<20} {'Train':<12} {'Test':<12} {'Difference':<12} {'Overfitting?':<12}")
        print("-" * 80)
        
        g_metrics = ['g_auc', 'g_precision', 'g_recall', 'g_f1', 'g_brier']
        g_names = ['AUC', 'Precision', 'Recall', 'F1 Score', 'Brier Score']
        
        for metric, name in zip(g_metrics, g_names):
            train_val = train_metrics.get(metric, 0)
            test_val = test_metrics.get(metric, 0)
            if metric == 'g_brier':  # Brier score越低越好
                diff = test_val - train_val  # 測試集比訓練集高表示overfitting
                overfitting = "Yes" if diff > 0.02 else "No"
            else:
                diff = train_val - test_val
                overfitting = "Yes" if diff > 0.05 else "No"
            print(f"{name:<20} {train_val:<12.4f} {test_val:<12.4f} {diff:+12.4f} {overfitting:<12}")
        
        print("\nOverlap & PS Quality:")
        print(f"{'Metric':<20} {'Train':<12} {'Test':<12} {'Difference':<12}")
        print("-" * 65)
        
        overlap_metrics = ['g_overlap', 'extreme_ps_count']
        overlap_names = ['Overlap Quality', 'Extreme PS Count']
        
        for metric, name in zip(overlap_metrics, overlap_names):
            train_val = train_metrics.get(metric, 0)
            test_val = test_metrics.get(metric, 0)
            diff = train_val - test_val
            if metric == 'extreme_ps_count':
                print(f"{name:<20} {train_val:<12.0f} {test_val:<12.0f} {diff:+12.0f}")
            else:
                print(f"{name:<20} {train_val:<12.4f} {test_val:<12.4f} {diff:+12.4f}")
        
        # 🆕 總體評估
        print("\n" + "="*80)
        print("                         OVERFITTING ASSESSMENT")
        print("="*80)
        
        # Q-model overfitting check
        q_auc_diff = train_metrics.get('q_auc', 0) - test_metrics.get('q_auc', 0)
        q_f1_diff = train_metrics.get('q_f1', 0) - test_metrics.get('q_f1', 0)
        
        if q_auc_diff > 0.1 or q_f1_diff > 0.1:
            q_assessment = "🔴 Significant Q-model overfitting detected"
        elif q_auc_diff > 0.05 or q_f1_diff > 0.05:
            q_assessment = "🟡 Moderate Q-model overfitting detected"
        else:
            q_assessment = "🟢 Q-model shows good generalization"
        
        # G-model overfitting check
        g_auc_diff = train_metrics.get('g_auc', 0) - test_metrics.get('g_auc', 0)
        g_f1_diff = train_metrics.get('g_f1', 0) - test_metrics.get('g_f1', 0)
        g_brier_diff = test_metrics.get('g_brier', 0) - train_metrics.get('g_brier', 0)
        
        if g_auc_diff > 0.1 or g_f1_diff > 0.1 or g_brier_diff > 0.05:
            g_assessment = "🔴 Significant G-model overfitting detected"
        elif g_auc_diff > 0.05 or g_f1_diff > 0.05 or g_brier_diff > 0.02:
            g_assessment = "🟡 Moderate G-model overfitting detected"
        else:
            g_assessment = "🟢 G-model shows good generalization"
        
        print(q_assessment)
        print(g_assessment)
        
        # 校準比較
        if 'train_calibration_metrics' in train_metrics and 'test_calibration_metrics' in test_metrics:
            train_ce = train_metrics['train_calibration_metrics']['calibration_error']
            test_ce = test_metrics['test_calibration_metrics']['calibration_error']
            ce_diff = test_ce - train_ce
            
            if ce_diff > 0.05:
                cal_assessment = "🔴 Significant calibration degradation on test set"
            elif ce_diff > 0.02:
                cal_assessment = "🟡 Moderate calibration degradation on test set"
            else:
                cal_assessment = "🟢 Calibration maintains well on test set"
            
            print(cal_assessment)
            print(f"   Train Calibration Error: {train_ce:.4f}")
            print(f"   Test Calibration Error: {test_ce:.4f}")
            print(f"   Degradation: {ce_diff:+.4f}")

    # 🆕 修改 print_results 函數，加入完整的train/test性能比較
    def print_results(ate, se, ci_low, ci_high, p_value, raw_ate, train_diagnostics, test_pre_diagnostics, test_post_diagnostics, att, calibration_info=None):
        """Enhanced results printing with comprehensive train/test performance comparison"""
        print("\n" + "="*80)
        print("    TMLE Results (Overlap Weighting + Calibrated G-Model + Train/Test Split)")
        print("="*80)
        
        # 主要結果表格
        print(f"{'Estimand':<12} {'Estimate':<12} {'Std.Err':<10} {'95% CI':<25} {'P-value':<10} {'Significant':<12}")
        print("-" * 80)
        print(f"{'ATE':<12} {ate:<12.6f} {se:<10.6f} [{ci_low:.6f}, {ci_high:.6f}] {p_value:<10.6f} {'Yes' if p_value < 0.05 else 'No':<12}")
        
        if not np.isnan(att):
            print(f"{'ATT':<12} {att:<12.6f} {'---':<10} {'---':<25} {'---':<10} {'---':<12}")
        else:
            print(f"{'ATT':<12} {'N/A':<12} {'---':<10} {'---':<25} {'---':<10} {'---':<12}")

        # 🆕 校準信息
        if calibration_info is not None:
            print("\n" + "="*80)
            print("                      CALIBRATION ASSESSMENT")
            print("="*80)
            print(f"📊 Calibration Method: {calibration_info['calibration_method'].title()}")
            print("\n" + "-"*60)
            print("            Calibration Metrics Comparison (Downsampled Training)")
            print("-"*60)
            print(f"{'Metric':<20} {'Before':<12} {'After':<12} {'Improvement':<12}")
            print("-"*60)
            
            pre_ce = calibration_info['pre_calibration_metrics']['calibration_error']
            post_ce = calibration_info['post_calibration_metrics']['calibration_error']
            ce_improvement = pre_ce - post_ce
            
            pre_brier = calibration_info['pre_calibration_metrics']['brier_score']
            post_brier = calibration_info['post_calibration_metrics']['brier_score']
            brier_improvement = pre_brier - post_brier
            
            print(f"{'Calibration Error':<20} {pre_ce:<12.4f} {post_ce:<12.4f} {ce_improvement:+.4f}")
            print(f"{'Brier Score':<20} {pre_brier:<12.4f} {post_brier:<12.4f} {brier_improvement:+.4f}")
            
            # 🆕 完整訓練集和測試集校準性能比較
            print("\n" + "-"*60)
            print("           Full Train vs Test Set Calibration Performance")
            print("-"*60)
            train_ce = calibration_info['train_calibration_metrics']['calibration_error']
            train_brier = calibration_info['train_calibration_metrics']['brier_score']
            test_ce = calibration_info['test_calibration_metrics']['calibration_error']
            test_brier = calibration_info['test_calibration_metrics']['brier_score']
            
            print(f"{'Train Set CE':<20} {train_ce:<12.4f}")
            print(f"{'Test Set CE':<20} {test_ce:<12.4f}")
            print(f"{'CE Degradation':<20} {test_ce - train_ce:+12.4f}")
            print(f"{'Train Set Brier':<20} {train_brier:<12.4f}")
            print(f"{'Test Set Brier':<20} {test_brier:<12.4f}")
            print(f"{'Brier Degradation':<20} {test_brier - train_brier:+12.4f}")
            
            # 🆕 加入訓練集診斷信息到校準信息中
            if 'train_calibration_metrics' in calibration_info:
                train_diagnostics.update({
                    'train_calibration_metrics': calibration_info['train_calibration_metrics']
                })
            if 'test_calibration_metrics' in calibration_info:
                test_post_diagnostics.update({
                    'test_calibration_metrics': calibration_info['test_calibration_metrics']
                })
            
            print("\n📈 Calibration Assessment:")
            if ce_improvement > 0.01:
                print("   ✅ Significant improvement in training calibration error")
            elif ce_improvement > 0:
                print("   ✓ Slight improvement in training calibration error")
            else:
                print("   ⚠️  No improvement in training calibration error")
                
            if test_ce < 0.05:
                print("   ✅ Excellent test set calibration (CE < 0.05)")
            elif test_ce < 0.10:
                print("   ✓ Good test set calibration (CE < 0.10)")
            else:
                print("   ⚠️  Poor test set calibration (CE ≥ 0.10)")

        # 🆕 訓練/測試性能比較
        print_train_test_comparison(train_diagnostics, test_post_diagnostics, "COMPREHENSIVE TRAIN/TEST PERFORMANCE COMPARISON")

        print("\n" + "="*80)
        print("                          INTERPRETATIONS")
        print("="*80)
        print("🎯 ATE (Average Treatment Effect):")
        print("   - Population-wide average causal effect")
        print("   - Uses overlap weighting and calibrated propensity scores")
        print(f"   - Estimate: {ate:.6f}")
        
        print("\n🎪 ATT (Average Treatment Effect on the Treated):")
        print("   - Average causal effect among those who received treatment")
        print("   - Policy-relevant for understanding treatment effectiveness")
        if not np.isnan(att):
            print(f"   - Estimate: {att:.6f}")
        else:
            print("   - Not available (no treated units)")

        print("\n" + "-"*60)
        print("            Raw vs TMLE Comparison (Test Set)")
        print("-"*60)
        print(f"Raw ATE (Pre-update):     {test_pre_diagnostics['raw_ate']:.6f}")
        print(f"TMLE ATE (Post-update):   {ate:.6f}")
        print(f"Adjustment Magnitude:     {abs(ate - test_pre_diagnostics['raw_ate']):.6f}")
        if test_pre_diagnostics['raw_ate'] != 0:
            relative_change = abs(ate - test_pre_diagnostics['raw_ate'])/abs(test_pre_diagnostics['raw_ate'])*100
            print(f"Relative Change:          {relative_change:.2f}%")

        print("\n" + "="*80)
        
        # 效應大小解釋
        if abs(ate) < 0.01:
            effect_size = "Negligible"
        elif abs(ate) < 0.05:
            effect_size = "Small"
        elif abs(ate) < 0.1:
            effect_size = "Medium"
        else:
            effect_size = "Large"

        direction = "Positive" if ate > 0 else "Negative"
        significance = "Statistically Significant" if p_value < 0.05 else "Not Statistically Significant"

        print(f"📊 Effect Summary: {effect_size} {direction} Treatment Effect, {significance}")
        print("   🎯 Enhanced with calibrated propensity scores for improved reliability")
        print("   📈 Comprehensive train/test performance comparison provided")
        print("   🔍 Overfitting assessment included for model validation")
        print("="*80)

    #############################################################################
    # Main Process
    print("***** Start TMLE Analysis with Calibrated G-Model & Train/Test Comparison *****")
    print("="*80)
    print(f"🎯 Using {calibration_method} calibration method")

    with tqdm(total=12, desc="TMLE Progress", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as main_pbar:

        try:
            # 1. Data Loading and Splitting
            main_pbar.set_description("***** Loading and Splitting Data *****")
            data_splits = data_loading(file_path, test_size, random_state)
            main_pbar.update(1)
            time.sleep(0.1)

            # 2. Data Preprocessing
            main_pbar.set_description("***** Preprocessing Data *****")
            W_train_std, W_test_std, scaler_W = data_preprocessing(
                data_splits['train']['W'], data_splits['test']['W']
            )
            
            W_A_train_std = pd.concat([W_train_std, pd.DataFrame(data_splits['train']['A'], columns=['A'])], axis=1)
            W_A_test_std = pd.concat([W_test_std, pd.DataFrame(data_splits['test']['A'], columns=['A'])], axis=1)
            main_pbar.update(1)
            time.sleep(0.1)

            # 3. Set up base learners
            main_pbar.set_description("***** Setting Up Base Learners *****")
            n_features = data_splits['train']['W'].shape[1]
            base_learners = get_base_learners(n_features)
            main_pbar.update(1)
            time.sleep(0.1)

            # 4. Fit Q models (只在訓練集上訓練)
            main_pbar.set_description("> Step 1: Fit Outcome Models (Q) on Train Set")
            print("\nStep 1: Fit Outcome Models (Q) on Training Set")
            sl = fit_superlearner(W_A_train_std, data_splits['train']['Y'], base_learners, "Outcome Model")
            
            # 🆕 在訓練集和測試集上都預測Q models
            Q_A_train, Q_1_train, Q_0_train = predict_Q_models(sl, W_A_train_std, data_splits['train']['A'], is_test=False)
            Q_A_test, Q_1_test, Q_0_test = predict_Q_models(sl, W_A_test_std, data_splits['test']['A'], is_test=True)
            main_pbar.update(1)

            # 5. 🆕 Estimate calibrated propensity scores (包含訓練集性能)
            main_pbar.set_description("> Step 2: Estimate Calibrated Propensity Scores (g)")
            print("\nStep 2: Estimate Calibrated Propensity Scores (g)")
            g_results = estimate_g_with_calibration(
                data_splits['train']['A'], W_train_std, 
                data_splits['test']['A'], W_test_std, 
                base_learners, calibration_method
            )
            main_pbar.update(1)

            # 🆕 6. Training set diagnostic checks
            main_pbar.set_description("***** Diagnostic Checks - Training Set *****")
            train_diagnostics = diagnostic_checks(
                data_splits['train']['Y'], data_splits['train']['A'], data_splits['train']['W'], 
                Q_A_train, Q_1_train, Q_0_train, g_results['train']['g_w'], "Training", is_test=False, 
                model_performance=g_results['calibration_info']
            )
            main_pbar.update(1)

            # 7. 🆕 Pre-update diagnostic checks (測試集)
            main_pbar.set_description("***** Diagnostic Checks - Pre-update (Test Set) *****")
            test_pre_diagnostics = diagnostic_checks(
                data_splits['test']['Y'], data_splits['test']['A'], data_splits['test']['W'], 
                Q_A_test, Q_1_test, Q_0_test, g_results['test']['g_w'], "Pre-update", is_test=True, 
                model_performance=g_results['calibration_info']
            )
            main_pbar.update(1)

            # 8. Estimate and update fluctuation parameters (測試集)
            main_pbar.set_description("> Step 3: TMLE Update (Test Set)")
            print("\nStep 3: TMLE Update on Test Set")
            eps = estimate_fluctuation_param(
                data_splits['test']['Y'], Q_A_test, g_results['test']['H_1'], g_results['test']['H_0'], 
                data_splits['test']['A'], g_results['test']['H_overlap']
            )

            # Q function updates (測試集)
            Q_A_update_test = update_Q(Q_A_test, g_results['test']['H_overlap'], eps)
            Q_1_update_test = update_Q(Q_1_test, (1 - g_results['test']['g_w']), eps)
            Q_0_update_test = update_Q(Q_0_test, (-g_results['test']['g_w']), eps)
            main_pbar.update(1)

            # 9. 🆕 Post-update diagnostic checks (測試集)
            main_pbar.set_description("***** Diagnostic Checks - Post-update (Test Set) *****")
            test_post_diagnostics = diagnostic_checks(
                data_splits['test']['Y'], data_splits['test']['A'], data_splits['test']['W'], 
                Q_A_update_test, Q_1_update_test, Q_0_update_test, g_results['test']['g_w'], "Post-update", is_test=True,
                model_performance=g_results['calibration_info']
            )
            main_pbar.update(1)

            # 10. Compute final results (測試集)
            main_pbar.set_description("***** Compute Final Results (Test Set) *****")
            ate, se, ci_low, ci_high, p_value, infl_fn, att = compute_tmle(
                data_splits['test']['Y'], data_splits['test']['A'], 
                Q_A_update_test, Q_1_update_test, Q_0_update_test, 
                g_results['test']['H_1'], g_results['test']['H_0'], 
                g_results['test']['overlap_weights'], g_results['test']['H_overlap']
            )
            
            raw_ate = test_pre_diagnostics['raw_ate']
            main_pbar.update(1)
            
            # 11. 🆕 打印結果（包含完整train/test性能比較）
            main_pbar.set_description("***** Printing Results *****")
            print_results(ate, se, ci_low, ci_high, p_value, raw_ate, train_diagnostics, test_pre_diagnostics, test_post_diagnostics, att, g_results['calibration_info'])
            main_pbar.update(1)
            
            # 12. 額外信息
            main_pbar.set_description("***** Final Summary *****")
            print(f"\n📊 Dataset Information:")
            print(f"   Training set size: {len(data_splits['train']['Y'])}")
            print(f"   Test set size: {len(data_splits['test']['Y'])}")
            print(f"   Test size ratio: {test_size*100:.1f}%")
            print(f"   Calibration method: {calibration_method}")
            main_pbar.update(1)
            
            return {
                'ate': ate, 
                'att': att, 
                'se': se, 
                'ci_low': ci_low, 
                'ci_high': ci_high, 
                'p_value': p_value, 
                'raw_ate': raw_ate, 
                'influence_function': infl_fn,
                'train_diagnostics': train_diagnostics,  # 🆕 訓練集診斷
                'test_pre_diagnostics': test_pre_diagnostics, 
                'test_post_diagnostics': test_post_diagnostics,
                'overlap_weights': g_results['test']['overlap_weights'], 
                'g_scores': g_results['test']['g_w'],
                'calibration_info': g_results['calibration_info'],
                'train_size': len(data_splits['train']['Y']), 
                'test_size': len(data_splits['test']['Y']),
                'test_ratio': test_size,
                'calibration_method': calibration_method
            }
        
        except Exception as e:
            main_pbar.set_description("<< Analysis Failed >>")
            print(f"An error occurred during the analysis: {str(e)}")
            print("Please check the data format and path")
            return None


# 🆕 執行分析 - 現在包含完整的train/test性能比較
if __name__ == "__main__":
    # 可選的校準方法: 'platt' (Platt scaling) 或 'isotonic' (Isotonic regression)
    calibration_methods = ['platt', 'isotonic']
    
    print("🎯 Available calibration methods:")
    print("   - 'platt': Platt scaling (sigmoid function)")
    print("   - 'isotonic': Isotonic regression (monotonic function)")
    print("\n" + "="*60)
    
    # 使用 Platt scaling 作為預設
    results = tmle_project(
        '/Users/chendawei/Desktop/MIT TMLE ICU project/original /yasmeen tmle/tmle_data.csv', 
        test_size=0.3, 
        random_state=42,
        calibration_method='platt'  # 可選 'isotonic' 或 'platt'
    )
    
    # 🆕 如果想要比較不同校準方法，可以取消下面的註解
    """
    print("\n" + "="*80)
    print("          COMPARING DIFFERENT CALIBRATION METHODS")
    print("="*80)
    
    calibration_results = {}
    
    for method in ['platt', 'isotonic']:
        print(f"\n🔄 Running analysis with {method} calibration...")
        result = tmle_project(
            '/Users/chendawei/Desktop/Task 2 /yasmeen tmle/tmle_data.csv', 
            test_size=0.3, 
            random_state=42,  # 保持相同的隨機種子以便比較
            calibration_method=method
        )
        calibration_results[method] = result
    
    # 比較結果
    print("\n" + "="*80)
    print("              CALIBRATION METHOD COMPARISON")
    print("="*80)
    print(f"{'Method':<12} {'ATE':<12} {'SE':<10} {'P-value':<10} {'Test CE':<12} {'Test Brier':<12} {'Train-Test CE Diff':<18}")
    print("-" * 100)
    
    for method in calibration_methods:
        if calibration_results[method] is not None:
            result = calibration_results[method]
            cal_info = result['calibration_info']
            train_ce = cal_info['train_calibration_metrics']['calibration_error']
            test_ce = cal_info['test_calibration_metrics']['calibration_error']
            test_brier = cal_info['test_calibration_metrics']['brier_score']
            ce_diff = test_ce - train_ce
            
            print(f"{method.title():<12} {result['ate']:<12.6f} {result['se']:<10.6f} {result['p_value']:<10.6f} {test_ce:<12.4f} {test_brier:<12.4f} {ce_diff:+18.4f}")
    
    # 推薦最佳方法
    best_method = min(calibration_results.keys(), 
                     key=lambda x: calibration_results[x]['calibration_info']['test_calibration_metrics']['calibration_error'] 
                     if calibration_results[x] is not None else float('inf'))
    print(f"\n🏆 Recommended method based on test calibration error: {best_method}")
    
    # 🆕 額外的train/test generalization比較
    print(f"\n🔍 Generalization Assessment:")
    for method in calibration_methods:
        if calibration_results[method] is not None:
            result = calibration_results[method]
            train_diag = result['train_diagnostics']
            test_diag = result['test_post_diagnostics']
            
            # Q-model generalization
            q_auc_diff = train_diag['q_auc'] - test_diag['q_auc']
            g_auc_diff = train_diag['g_auc'] - test_diag['g_auc']
            
            if q_auc_diff > 0.05 or g_auc_diff > 0.05:
                generalization = "Poor"
            elif q_auc_diff > 0.02 or g_auc_diff > 0.02:
                generalization = "Fair"
            else:
                generalization = "Good"
            
            print(f"   {method.title()}: {generalization} generalization (Q-AUC diff: {q_auc_diff:+.3f}, G-AUC diff: {g_auc_diff:+.3f})")
    """