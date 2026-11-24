from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import f1_score, make_scorer
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from catboost import CatBoostClassifier, CatBoostRegressor
from pathlib import Path
import optuna
import json
from sklearn.preprocessing import StandardScaler

class DataCleaner(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.noa_median_ = X['Number of Open Accounts'].median()
        self.moc_median_ = X['Maximum Open Credit'].median()
        self.cla_median_ = X['Current Loan Amount'].median()
        self.noa_q99_ = X['Number of Open Accounts'].quantile(0.99)

        return self

    def transform(self, X):
        X = X.copy()

        home_ownership_mask = X['Home Ownership'] == 'Have Mortgage'
        X.loc[home_ownership_mask, 'Home Ownership'] = 'Home Mortgage'

        yicj_less_1_year_mask = (X['Years in current job'] == '< 1 year')
        yicj_more_10_years_mask = (X['Years in current job'] == '10+ years')
        X['less_1_year_in_current_job'] = yicj_less_1_year_mask * 1
        X['more_10_years_in_current_job'] = yicj_more_10_years_mask * 1
        X.loc[yicj_less_1_year_mask, 'Years in current job'] = '1 year'
        X.loc[yicj_more_10_years_mask, 'Years in current job'] = '10 years'
        X['Years in current job'] = X['Years in current job'].apply(lambda g: int(g.split(' ')[0]) if isinstance(g, str) else g)

        noa_greater_q99 = (X['Number of Open Accounts'] > self.noa_q99_)
        X['number_of_open_accounts_was_scaled'] = noa_greater_q99 * 1
        X.loc[noa_greater_q99, 'Number of Open Accounts'] = self.noa_median_

        moc_big_value_mask = (X['Maximum Open Credit'] > 1e7) & (X['Maximum Open Credit'].notna())
        X.loc[moc_big_value_mask, 'Maximum Open Credit'] = self.moc_median_
        moc_0_mask = (X['Maximum Open Credit'] == 0)
        X.loc[moc_0_mask, 'Maximum Open Credit'] = self.moc_median_

        cla_scaled_mask = X['Current Loan Amount'] == 99999999.0
        X['current_loan_amount_scaled'] = cla_scaled_mask * 1
        X.loc[cla_scaled_mask, 'Current Loan Amount'] = self.cla_median_

        monthly_debt_gt_cla_mask = (X['Monthly Debt'] > X['Current Loan Amount']) | (X['Monthly Debt'] == 0)
        X.loc[monthly_debt_gt_cla_mask, 'Monthly Debt'] = X.loc[monthly_debt_gt_cla_mask, 'Current Loan Amount']

        credit_score_scaled_mask = (X['Credit Score'] > 751)
        X['credit_score_scaled'] = credit_score_scaled_mask * 1
        X.loc[credit_score_scaled_mask, 'Credit Score'] = X.loc[credit_score_scaled_mask, 'Credit Score'] // 10

        msld_isna_nocp_1 = X['Months since last delinquent'].isna() & (X['Number of Credit Problems'] >= 1)
        msld_isna_nocp_0 = X['Months since last delinquent'].isna() & (X['Number of Credit Problems'] == 0)
        X['not_debtor'] = msld_isna_nocp_0 * 1
        X.loc[msld_isna_nocp_1, 'Months since last delinquent'] = 0
        X.loc[msld_isna_nocp_0, 'Months since last delinquent'] = -1

        annual_income_credit_score_missed_mask = X['Annual Income'].isna() & X['Credit Score'].isna()
        X['annual_income_credit_score_missed'] = annual_income_credit_score_missed_mask * 1

        return X

class NaNCatboostImputer(TransformerMixin, BaseEstimator):
    param_grid = {
        "depth": optuna.distributions.IntDistribution(2, 10),
        "l2_leaf_reg": optuna.distributions.FloatDistribution(1e-2, 15, log=True),
        "iterations": optuna.distributions.IntDistribution(20, 1500, log=True),
        "learning_rate": optuna.distributions.FloatDistribution(1e-3, 0.5, log=True),
        "random_strength": optuna.distributions.FloatDistribution(1e-2, 1)     
    }
    def __init__(self, target_col_name: str, model_type: str):
        self.target_col_name = target_col_name
        self.model_type = model_type

    def fit(self, X, y=None):
        if (self.model_type == 'regression'):
            self.frozen_params = {
                'silent': True,
                'random_state': 42,
                'eval_metric' : 'RMSE'
            }
            self.imputer_ = CatBoostRegressor(**self.frozen_params)
            self.optuna_metric = 'neg_root_mean_squared_error'
        if (self.model_type == 'classification'):
            self.frozen_params = {
                'silent': True,
                'random_state': 42
            }
            self.imputer_ = CatBoostClassifier(**self.frozen_params)
            self.optuna_metric = make_scorer(f1_score, average='weighted')

        if (self.target_col_name == 'Years in current job'):
            self.json_prefix = 'yicj'
        elif (self.target_col_name == 'Bankruptcies'):
            self.json_prefix = 'bankruptcies'
        elif (self.target_col_name == 'Credit Score'):
            self.json_prefix = 'credit_score'
        elif (self.target_col_name == 'Annual Income'):
            self.json_prefix = 'annual_income'

        fit_mask = X[self.target_col_name].notna()
        y_train = X.loc[fit_mask, self.target_col_name]
        X_train = X[fit_mask].drop(self.target_col_name, axis=1)

        project_root = next(
            parent for parent in Path(__file__).resolve().parents
            if (parent / 'configs').is_dir()
        )

        config_path = project_root / 'configs' / 'model_config.json'

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as config:
                configs = json.load(config)
        else:
            configs = []

        model_info = next((p for p in configs if p["model_name"] == f'{self.json_prefix}_imputer'), None)
        self.best_params_ = model_info["params"] if model_info else None

        if self.best_params_ is None:

            cv = KFold(n_splits=3, random_state=42, shuffle=True)

            optuna_search = optuna.integration.OptunaSearchCV(
                self.imputer_,
                self.param_grid,
                cv=cv,
                n_trials=30,
                scoring=self.optuna_metric,
                random_state=42
            )

            optuna_search.fit(X_train, y_train)
            self.best_params_ = optuna_search.best_params_

            model_info = {
                "model_name": f'{self.json_prefix}_imputer',
                "params": self.best_params_
            }

            configs.append(model_info)

            with open(config_path, "w", encoding="utf-8") as config:
                json.dump(configs, config, indent=4, ensure_ascii=False)

        self.imputer_ = self.imputer_.set_params(**self.frozen_params, **self.best_params_)
        self.imputer_.fit(X_train, y_train)

        return self

    def transform(self, X):
        X = X.copy()
        miss_mask = X[self.target_col_name].isna()
        X[f'{self.json_prefix}_filled'] = miss_mask * 1
        X_to_model = X[miss_mask]
        X_to_model = X_to_model.drop(columns=[self.target_col_name, f'{self.json_prefix}_filled'], axis=1)
        X.loc[miss_mask, self.target_col_name] = self.imputer_.predict(X_to_model)

        return X

class FeatureCreator(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.fitted_flag_ = True

        return self

    def transform(self, X):
        X = X.copy()

        X['income_to_payment_ratio'] = X['Annual Income'] / (X['Monthly Debt'] * 12)
        X['income_to_loan_ratio'] = X['Annual Income'] / X['Current Loan Amount']
        X['income_to_max_credit_ratio'] = X['Annual Income'] / X['Maximum Open Credit']
        X['debt_to_income_ratio'] = X['Monthly Debt'] * 12 / X['Annual Income']
        X['loan_to_income_ratio'] = X['Current Loan Amount'] / X['Annual Income']
        X['income_per_account'] = X['Annual Income'] / X['Number of Open Accounts']

        return X

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class MetaDataframeCreator(BaseEstimator, TransformerMixin):
    def __init__(self, lr_model, lgbm_model, n_splits=5, random_state=42):
        self.lr_model = lr_model
        self.lgbm_model = lgbm_model
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y=None):

        self.lr_model_ = clone(self.lr_model)
        self.lgbm_model_ = clone(self.lgbm_model)
        self.scaler_ = StandardScaler()

        X = X.copy()
        y = np.asarray(y)

        X_scaled = self.scaler_.fit_transform(X)

        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        # === OOF  ===
        oof_lr = cross_val_predict(
            self.lr_model_,
            X_scaled,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=1
        )[:, 1]

        oof_lgbm = cross_val_predict(
            self.lgbm_model_,
            X,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=1
        )[:, 1]

        self.oof_ = pd.DataFrame({
            "lr_proba": oof_lr,
            "lgbm_proba": oof_lgbm
        }, index=X.index)

        self.lr_model_.fit(X_scaled, y)
        self.lgbm_model_.fit(X, y)

        return self

    def transform(self, X):
        X = X.copy()
        X_scaled = self.scaler_.transform(X)

        return pd.DataFrame({
            "lr_proba": self.lr_model_.predict_proba(X_scaled)[:, 1],
            "lgbm_proba": self.lgbm_model_.predict_proba(X)[:, 1]
        }, index=X.index)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).oof_


class SimpleErrorDeleter(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.fitted_ = True

        return self

    def transform(self, X):
        X = X.copy()
        X.drop(columns=['Home Ownership', 'Years in current job', 'Purpose', 'Term'], inplace=True)
        return X
        
