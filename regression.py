from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from utils import load_boston, eval_metrics

def basic_models(X_train, X_test, y_train, y_test):
    """Train and evaluate basic regressors."""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "SVR": SVR()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse, r2 = eval_metrics(y_test, preds)
        print(f"{name} — MSE: {mse:.2f}, R²: {r2:.2f}")

def tuned_models(X_train, X_test, y_train, y_test):
    """Grid-search, then train and evaluate regressors."""
    tuned = {
        "Ridge": (Ridge(), {
            'alpha': [0.1, 1.0, 10.0],
            'fit_intercept': [True, False],
            'solver': ['auto', 'svd', 'cholesky']
        }),
        "Random Forest": (RandomForestRegressor(), {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }),
        "SVR": (SVR(), {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        })
    }
    for name, (model, params) in tuned.items():
        grid = GridSearchCV(model, params, cv=3, scoring='r2')
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        preds = best.predict(X_test)
        mse, r2 = eval_metrics(y_test, preds)
        print(f"{name} (tuned) — MSE: {mse:.2f}, R²: {r2:.2f}, params: {grid.best_params_}")

if __name__ == "__main__":
    # load and split data
    df = load_boston()
    X, y = df.drop("MEDV", axis=1), df["MEDV"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("----- Basic Models -----")
    basic_models(X_tr, X_te, y_tr, y_te)

    print("\n----- Tuned Models -----")
    tuned_models(X_tr, X_te, y_tr, y_te)
