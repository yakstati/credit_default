from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, make_scorer
import pandas as pd
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier, CatBoostRegressor
from pathlib import Path
import optuna
import json

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

        monthly_debt_gt_cla_mask = X['Monthly Debt'] > X['Current Loan Amount']
        X.loc[monthly_debt_gt_cla_mask, 'Monthly Debt'] = X.loc[monthly_debt_gt_cla_mask, 'Current Loan Amount']

        credit_score_scaled_mask = (X['Credit Score'] > 751)
        X['credit_score_scaled'] = credit_score_scaled_mask * 1
        X.loc[credit_score_scaled_mask, 'Credit Score'] = X.loc[credit_score_scaled_mask, 'Credit Score'] // 10

        msld_isna_nocp_1 = X['Months since last delinquent'].isna() & (X['Number of Credit Problems'] >= 1)
        msld_isna_nocp_0 = X['Months since last delinquent'].isna() & (X['Number of Credit Problems'] == 0)
        X['not_debtor'] = msld_isna_nocp_0 * 1
        X.loc[msld_isna_nocp_1, 'Months since last delinquent'] = 0
        X.loc[msld_isna_nocp_0, 'Months since last delinquent'] = -1

        X = pd.get_dummies(data=X, prefix=None, prefix_sep='_', columns=['Home Ownership', 'Purpose', 'Term'], drop_first=True, dtype=int)

        return X

class YicjImputer(TransformerMixin, BaseEstimator):
    frozen_params = {
            'silent':True,
            #'cat_features':CAT_FEATURE_NAMES,
            #'eval_metric':'RMSE',
            'early_stopping_rounds':20,
            'random_state':42
    }

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame=None):
        X = X.copy()

        fit_mask = X['Years in current job'].notna()
        target = X.loc[fit_mask, 'Years in current job']
        X_train = X[fit_mask].drop('Years in current job', axis=1)

        imputer = CatBoostClassifier(**self.frozen_params)

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

        model_info = next((p for p in configs if p["model_name"] == "yicj_imputer"), None)
        self.best_params_ = model_info["params"] if model_info else None

        if self.best_params_ is None:
            f1_weight = make_scorer(f1_score, average='weighted')
            param_grid = {
                    "depth": optuna.distributions.IntDistribution(2, 10),
                    "l2_leaf_reg": optuna.distributions.FloatDistribution(1e-2, 15, log=True),
                    "iterations": optuna.distributions.IntDistribution(20, 1500, log=True),
                    "learning_rate": optuna.distributions.FloatDistribution(1e-3, 0.5, log=True),
                    "random_strength": optuna.distributions.FloatDistribution(1e-2, 1)     
                }

            cv = KFold(n_splits=3, random_state=42, shuffle=True)

            optuna_search = optuna.integration.OptunaSearchCV(
                imputer,
                param_grid,
                cv=cv,
                n_trials=30,
                scoring=f1_weight,
                random_state=42
            )

            optuna_search.fit(X_train, target)
            self.best_params_ = optuna_search.best_params_

            model_info = {
                "model_name": "yicj_imputer",
                "params": self.best_params_
            }

            configs.append(model_info)

            with open(config_path, "w", encoding="utf-8") as config:
                json.dump(configs, config, indent=4, ensure_ascii=False)

        self.yicj_catboost_imputer_ = CatBoostClassifier(**self.frozen_params,**self.best_params_)
        self.yicj_catboost_imputer_.fit(X_train, target)

        self.feature_names_in_ = X.columns.tolist()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy() 
        miss_mask = X['Years in current job'].isna()
        X['years_in_current_job_filled'] = miss_mask
        X_to_model = X[miss_mask].drop('Years in current job', axis=1)
        X.loc[miss_mask, 'Years in current job'] = self.yicj_catboost_imputer_.predict(X_to_model)

        return X

class Bankruptciesmputer(TransformerMixin, BaseEstimator):
    frozen_params = {
            'silent':True,
            #'cat_features':CAT_FEATURE_NAMES,
            #'eval_metric':'RMSE',
            'early_stopping_rounds':20,
            'random_state':42
    }

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame=None):
        X = X.copy()

        fit_mask = X['Bankruptcies'].notna()
        target = X.loc[fit_mask, 'Bankruptcies']
        X_train = X[fit_mask].drop('Bankruptcies', axis=1)

        imputer = CatBoostClassifier(**self.frozen_params)

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

        model_info = next((p for p in configs if p["model_name"] == "bankruptcies_imputer"), None)
        self.best_params_ = model_info["params"] if model_info else None

        if self.best_params_ is None:
            f1_weight = make_scorer(f1_score, average='weighted')
            param_grid = {
                    "depth": optuna.distributions.IntDistribution(2, 10),
                    "l2_leaf_reg": optuna.distributions.FloatDistribution(1e-2, 15, log=True),
                    "iterations": optuna.distributions.IntDistribution(20, 1500, log=True),
                    "learning_rate": optuna.distributions.FloatDistribution(1e-3, 0.5, log=True),
                    "random_strength": optuna.distributions.FloatDistribution(1e-2, 1)     
                }

            cv = KFold(n_splits=3, random_state=42, shuffle=True)

            optuna_search = optuna.integration.OptunaSearchCV(
                imputer,
                param_grid,
                cv=cv,
                n_trials=30,
                scoring=f1_weight,
                random_state=42
            )

            optuna_search.fit(X_train, target)
            self.best_params_ = optuna_search.best_params_

            model_info = {
                "model_name": "bankruptcies_imputer",
                "params": self.best_params_
            }

            configs.append(model_info)

            with open(config_path, "w", encoding="utf-8") as config:
                json.dump(configs, config, indent=4, ensure_ascii=False)

        self.bankruptcies_catboost_imputer_ = CatBoostClassifier(**self.frozen_params,**self.best_params_)
        self.bankruptcies_catboost_imputer_.fit(X_train, target)

        self.feature_names_in_ = X.columns.tolist()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy() 
        miss_mask = X['Bankruptcies'].isna()
        X['bankruptcies_filled'] = miss_mask
        X_to_model = X[miss_mask].drop('Bankruptcies', axis=1)
        X.loc[miss_mask, 'Bankruptcies'] = self.bankruptcies_catboost_imputer_.predict(X_to_model)

        return X

class CreditScoreImputer(TransformerMixin, BaseEstimator):
    frozen_params = {
            'silent':True,
            #'cat_features':CAT_FEATURE_NAMES,
            'eval_metric':'RMSE',
            'early_stopping_rounds':20,
            'random_state':42
    }

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame=None):
        X = X.copy()

        fit_mask = X['Credit Score'].notna()
        target = X.loc[fit_mask, 'Credit Score']
        X_train = X[fit_mask].drop('Credit Score', axis=1)

        imputer = CatBoostRegressor(**self.frozen_params)

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

        model_info = next((p for p in configs if p["model_name"] == "credit_score_imputer"), None)
        self.best_params_ = model_info["params"] if model_info else None

        if self.best_params_ is None:
            param_grid = {
                    "depth": optuna.distributions.IntDistribution(2, 10),
                    "l2_leaf_reg": optuna.distributions.FloatDistribution(1e-2, 15, log=True),
                    "iterations": optuna.distributions.IntDistribution(20, 1500, log=True),
                    "learning_rate": optuna.distributions.FloatDistribution(1e-3, 0.5, log=True),
                    "random_strength": optuna.distributions.FloatDistribution(1e-2, 1)     
                }

            cv = KFold(n_splits=3, random_state=42, shuffle=True)

            optuna_search = optuna.integration.OptunaSearchCV(
                imputer,
                param_grid,
                cv=cv,
                n_trials=30,
                scoring='neg_root_mean_squared_error',
                random_state=42
            )

            optuna_search.fit(X_train, target)
            self.best_params_ = optuna_search.best_params_

            model_info = {
                "model_name": "credit_score_imputer",
                "params": self.best_params_
            }

            configs.append(model_info)

            with open(config_path, "w", encoding="utf-8") as config:
                json.dump(configs, config, indent=4, ensure_ascii=False)

        self.credit_score_catboost_imputer_ = CatBoostRegressor(**self.frozen_params,**self.best_params_)
        self.credit_score_catboost_imputer_.fit(X_train, target)

        self.feature_names_in_ = X.columns.tolist()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy() 
        miss_mask = X['Credit Score'].isna()
        X['credit_score_filled'] = miss_mask
        X_to_model = X[miss_mask].drop('Credit Score', axis=1)
        X.loc[miss_mask, 'Credit Score'] = self.credit_score_catboost_imputer_.predict(X_to_model)

        return X

class AnnualIncomeImputer(TransformerMixin, BaseEstimator):
    frozen_params = {
            'silent':True,
            #'cat_features':CAT_FEATURE_NAMES,
            'eval_metric':'RMSE',
            'early_stopping_rounds':20,
            'random_state':42
    }

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame=None):
        X = X.copy()

        fit_mask = X['Annual Income'].notna()
        target = X.loc[fit_mask, 'Annual Income']
        X_train = X[fit_mask].drop('Annual Income', axis=1)

        imputer = CatBoostRegressor(**self.frozen_params)

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

        model_info = next((p for p in configs if p["model_name"] == "annual_income_imputer"), None)
        self.best_params_ = model_info["params"] if model_info else None

        if self.best_params_ is None:
            param_grid = {
                    "depth": optuna.distributions.IntDistribution(2, 10),
                    "l2_leaf_reg": optuna.distributions.FloatDistribution(1e-2, 15, log=True),
                    "iterations": optuna.distributions.IntDistribution(20, 1500, log=True),
                    "learning_rate": optuna.distributions.FloatDistribution(1e-3, 0.5, log=True),
                    "random_strength": optuna.distributions.FloatDistribution(1e-2, 1)     
                }

            cv = KFold(n_splits=3, random_state=42, shuffle=True)

            optuna_search = optuna.integration.OptunaSearchCV(
                imputer,
                param_grid,
                cv=cv,
                n_trials=30,
                scoring='neg_root_mean_squared_error',
                random_state=42
            )

            optuna_search.fit(X_train, target)
            self.best_params_ = optuna_search.best_params_

            model_info = {
                "model_name": "annual_income_imputer",
                "params": self.best_params_
            }

            configs.append(model_info)

            with open(config_path, "w", encoding="utf-8") as config:
                json.dump(configs, config, indent=4, ensure_ascii=False)

        self.annual_income_catboost_imputer_ = CatBoostRegressor(**self.frozen_params,**self.best_params_)
        self.annual_income_catboost_imputer_.fit(X_train, target)

        self.feature_names_in_ = X.columns.tolist()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy() 
        miss_mask = X['Annual Income'].isna()
        X['annual_income_filled'] = miss_mask
        X_to_model = X[miss_mask].drop('Annual Income', axis=1)
        X.loc[miss_mask, 'Annual Income'] = self.annual_income_catboost_imputer_.predict(X_to_model)

        return X


