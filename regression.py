from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from utils import load_boston, eval_metrics

def basic_models(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "SVR": SVR()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse, r2 = eval_metrics(y_test, preds)
        print(f"{name} - MSE: {mse:.2f}, R2: {r2:.2f}")

if __name__ == "__main__":
    df = load_boston()
    X, y = df.drop("MEDV", axis=1), df["MEDV"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("----- Model Comparision -----")
    basic_models(X_train, X_test, y_train, y_test)
