import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pmdarima import auto_arima
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, GridSearchCV
import tensorflow as tf
import streamlit as st

# Funkcje pomocnicze

def evaluate_model(y_test, y_pred, label="Model"):
    """Oblicza metryki błędów i zwraca je w formie słownika."""
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    # Skalowanie błędów
    scale_factor = np.mean(y_test)
    rmse_scaled = rmse / scale_factor
    mae_scaled = mae / scale_factor

    print(f"{label} - RMSE (scaled): {rmse_scaled:.4f}, MAE (scaled): {mae_scaled:.4f}, MAPE: {mape:.2f}%")
    return {
        "RMSE": rmse_scaled,
        "MAE": mae_scaled,
        "MAPE (%)": mape
    }


def visualize_results(y_test, y_pred, title, years=None, ylabel="Liczba ludności (miliony)"):
    """Tworzy wykres prognozy vs rzeczywiste dane."""
    plt.figure(figsize=(10, 6))
    if years is not None:
        plt.plot(years, y_test / 1e6, label="Rzeczywista (miliony)", marker='o', color="blue")
        plt.plot(years, y_pred / 1e6, label="Prognozowana (miliony)", marker='x', color="red")
    plt.title(title)
    plt.xlabel('Rok')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_combined_results(y_train, y_pred, y_test, years_train, years_test, model_name, country, in_streamlit=False):
    """Tworzy wykres z danymi treningowymi, testowymi i predykcjami."""
    plt.figure(figsize=(10, 6))

    # Wizualizacja danych treningowych
    plt.plot(years_train, np.array(y_train) / 1e6, label="Dane treningowe (w milionach)", color="blue")

    # Wizualizacja rzeczywistych danych testowych
    plt.plot(years_test, np.array(y_test) / 1e6, label="Dane testowe (w milionach)", color="green")

    # Wizualizacja przewidywanych wartości testowych
    plt.plot(years_test, np.array(y_pred) / 1e6, label="Prognozowane (w milionach)", color="red")

    plt.title(f"Model: {model_name} - {country}")
    plt.xlabel("Rok")
    plt.ylabel("Liczba ludności (miliony)")
    plt.legend()
    plt.grid(True)

    plt.show()  # Wyświetlenie lokalnie


def sliding_window(data, target_column, lags=3):
    """Tworzy przesuwane okno czasowe (sliding window) dla danych."""
    df = data.copy()
    for lag in range(1, lags + 1):
        df[f"{target_column}_lag{lag}"] = df[target_column].shift(lag)
    return df.dropna()



# Funkcja dynamicznego podziału danych

def split_data_dynamic(data, target_column, forecast_horizon):
    """Dzieli dane na zbiór treningowy i testowy."""
    test_start_year = data['year'].max() - forecast_horizon + 1
    train = data[data['year'] < test_start_year]
    test = data[data['year'] >= test_start_year]
    return train, test



# Funkcja modelowania ARIMA

def run_arima_dynamic(data, country, forecast_horizon):
    """Trenuje model ARIMA i generuje prognozę."""
    print(f"\n=== Processing ARIMA for {country} - {forecast_horizon}-year horizon ===")

    # Filtrowanie danych dla wybranego kraju
    country_data = data[data['country'] == country][['year', 'population']]

    # Dynamiczne dopasowanie zbioru treningowego i testowego
    train, test = split_data_dynamic(country_data, 'population', forecast_horizon)

    # Skalowanie danych
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train['population'].values.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test['population'].values.reshape(-1, 1)).flatten()

    # Dopasowanie modelu ARIMA
    model = auto_arima(train_scaled, seasonal=False, stepwise=True, max_p=5, max_d=2, max_q=5)
    print(f"Selected ARIMA order for {country}: {model.order}")

    # Prognozowanie
    forecast_scaled = model.predict(n_periods=forecast_horizon)
    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

    # Ocena wyników
    y_test = test['population'].values
    metrics = evaluate_model(y_test, forecast, label=f"ARIMA ({forecast_horizon}-year horizon)")

    # Wizualizacja wyników
    visualize_results(y_test, forecast, f"ARIMA Forecast vs Actual for {country} ({forecast_horizon} years)", test['year'])

    visualize_combined_results(
        y_train=train['population'].values,
        y_pred=forecast,
        y_test=test['population'].values,
        years_train=train['year'].values,
        years_test=test['year'].values,
        model_name="ARIMA",
        country=country
    )

    # Przygotowanie szczegółów prognozy
    details = pd.DataFrame({
        "Year": test['year'].values,
        "Predicted Population": forecast,
        "Actual Population": y_test
    })

    # Przygotowanie danych do zwrotu
    return {
        "metrics": metrics,
        "y_test": test['population'].values,
        "y_pred": forecast,
        "years_test": test['year'].values,
        "y_train": train['population'].values,
        "years_train": train['year'].values,
        "details": details
    }


def run_xgboost(data, country, forecast_horizon, lags=10):
    """Trenuje model XGBoost z tuningiem i generuje prognozę."""
    print(f"\n=== Processing XGBoost for {country} - {forecast_horizon}-year horizon ===")

    # Filtrowanie danych dla wybranego kraju
    country_data = data[data['country'] == country][['year', 'population']].dropna()

    # Podział danych na treningowe i testowe
    train, test = split_data_dynamic(country_data, 'population', forecast_horizon)

    # Tworzenie danych z przesuwanym oknem
    lagged_train = sliding_window(train, target_column='population', lags=lags).dropna()

    # Skalowanie danych
    scaler = StandardScaler()
    X_train = lagged_train.drop(columns=['population', 'year'])
    X_train_scaled = scaler.fit_transform(X_train)
    y_train = lagged_train['population']

    # Strojenie modelu XGBoost
    param_grid = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 0.5, 1],  # L1 regularization
        'reg_lambda': [1, 1.5, 2, 5]    # L2 regularization
    }

    model = XGBRegressor(objective='reg:squarederror', random_state=42)

    # RandomizedSearchCV z TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=10)
    search = RandomizedSearchCV(
        model, param_distributions=param_grid, n_iter=50,
        scoring='neg_mean_squared_error', cv=tscv,
        verbose=1, random_state=42, n_jobs=-1
    )
    search.fit(X_train_scaled, y_train)
    best_model = search.best_estimator_

    print(f"Best parameters: {search.best_params_}")

    # Prognozowanie iteracyjne
    forecast = []
    last_known_row = lagged_train.iloc[-1, :].copy()

    for _ in range(forecast_horizon):
        input_features = last_known_row[X_train.columns].values.reshape(1, -1)
        input_features_scaled = scaler.transform(input_features)
        predicted_value = best_model.predict(input_features_scaled)[0]
        forecast.append(predicted_value)

        # Aktualizacja lagów
        last_known_row[f"population_lag{lags}"] = predicted_value
        last_known_row = last_known_row.shift(-1)

    # Wyniki i metryki
    y_test = test['population'].values[:forecast_horizon]
    forecast = np.array(forecast)

    metrics = evaluate_model(y_test, forecast, label=f"XGBoost ({forecast_horizon}-year horizon)")

    # Wizualizacje wyników
    visualize_results(
        y_test=y_test,
        y_pred=forecast,
        title=f"XGBoost Forecast vs Actual for {country} ({forecast_horizon} years)",
        years=test['year'].values[:forecast_horizon]
    )
    visualize_combined_results(
        y_train=train['population'].values,
        y_pred=forecast,
        y_test=test['population'].values[:forecast_horizon],
        years_train=train['year'].values,
        years_test=test['year'].values[:forecast_horizon],
        model_name="XGBoost",
        country=country
    )
    # Przygotowanie szczegółów prognozy
    details = pd.DataFrame({
        "Year": test['year'].values[:forecast_horizon],
        "Predicted Population": forecast,
        "Actual Population": y_test
    })

    # Przygotowanie danych do zwrotu
    return {
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": forecast,
        "years_test": test['year'].values[:forecast_horizon],
        "y_train": train['population'].values,
        "years_train": train['year'].values,
        "details": details
    }

def run_random_forest(data, country, forecast_horizon, lags=3):
    """Trenuje model Random Forest i generuje prognozę."""

    print(f"\n=== Processing Random Forest for {country} - {forecast_horizon}-year horizon ===")

    # Filtrowanie danych dla wybranego kraju
    country_data = data[data['country'] == country][['year', 'population']].dropna()

    # Dynamiczny podział danych na treningowe i testowe
    train, test = split_data_dynamic(country_data, 'population', forecast_horizon)

    # Tworzenie danych z przesuwanym oknem
    lagged_train = sliding_window(train, target_column='population', lags=lags).dropna()

    # Skalowanie danych
    scaler = StandardScaler()
    X_train = lagged_train.drop(columns=['population', 'year'])
    X_train_scaled = scaler.fit_transform(X_train)
    y_train = lagged_train['population']

    # Dopasowanie modelu Random Forest z Grid Search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_

    print(f"Best parameters: {grid_search.best_params_}")

    # Prognozowanie iteracyjne na zbiorze testowym
    forecast = []
    last_known_row = lagged_train.iloc[-1, :].copy()  # Ostatni wiersz zbioru treningowego

    for _ in range(forecast_horizon):
        input_features = last_known_row[X_train.columns].values.reshape(1, -1)
        input_features_scaled = scaler.transform(input_features)
        predicted_value = best_model.predict(input_features_scaled)[0]
        forecast.append(predicted_value)

        # Aktualizacja lagów
        last_known_row = last_known_row.shift(-1)
        last_known_row[f"population_lag{lags}"] = predicted_value

    y_test = test['population'].values[:forecast_horizon]
    forecast = np.array(forecast)
    metrics = evaluate_model(y_test, forecast, label=f"Random Forest ({forecast_horizon}-year horizon)")

    # Wizualizacja wyników
    visualize_results(
        y_test=y_test,
        y_pred=forecast,
        title=f"Random Forest Forecast vs Actual for {country} ({forecast_horizon} years)",
        years=test['year'].values[:forecast_horizon]
    )

    visualize_combined_results(
        y_train=train['population'].values,
        y_pred=forecast,
        y_test=test['population'].values[:forecast_horizon],
        years_train=train['year'].values,
        years_test=test['year'].values[:forecast_horizon],
        model_name="Random Forest",
        country=country
    )

    # Przygotowanie szczegółów prognozy
    details = pd.DataFrame({
        "Year": test['year'].values[:forecast_horizon],
        "Predicted Population": forecast,
        "Actual Population": y_test
    })

    # Przygotowanie danych do zwrotu
    return {
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": forecast,
        "years_test": test['year'].values[:forecast_horizon],
        "y_train": train['population'].values,
        "years_train": train['year'].values,
        "details": details
    }

def run_lstm(data, country, forecast_horizon, lags=3, units=64, dropout_rate=0.2, epochs=50, batch_size=16, l2_reg=0.01):
    """Trenuje model LSTM z określonymi hiperparametrami i generuje prognozę."""

    # Filtrowanie danych dla wybranego kraju
    country_data = data[data['country'] == country][['year', 'population', 'birth_rate', 'death_rate', 'net_migration_rate']].dropna().reset_index(drop=True)

    # Skalowanie danych populacji i dodatkowych cech
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(country_data[['population', 'birth_rate', 'death_rate', 'net_migration_rate']])

    # Tworzenie lagów (sekwencji czasowych)
    X, y = [], []
    for i in range(len(scaled_data) - lags):
        X.append(scaled_data[i:i + lags])
        y.append(scaled_data[i + lags, 0])  # Tylko populacja jako zmienna docelowa
    X, y = np.array(X), np.array(y)

    # Podział na dane treningowe i testowe
    train_size = len(X) - forecast_horizon
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    years_train = country_data['year'][lags:train_size + lags].values
    years_test = country_data['year'][train_size + lags:].values

    # Budowanie modelu LSTM z regularizacją
    model = Sequential([
        LSTM(units, activation='tanh', kernel_regularizer=tf.keras.regularizers.L2(l2_reg), input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Trenowanie modelu
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Prognozowanie iteracyjne
    forecast = []
    last_known_sequence = X_train[-1]

    for _ in range(forecast_horizon):
        predicted_scaled = model.predict(last_known_sequence.reshape(1, lags, -1))[0]
        forecast.append(predicted_scaled)
        last_known_sequence = np.roll(last_known_sequence, -1, axis=0)
        last_known_sequence[-1, 0] = predicted_scaled
        last_known_sequence[-1, 1:] = X_test[0, -1, 1:]  # Zachowanie cech dodatkowych

    # Reskalowanie prognoz i danych rzeczywistych
    forecast_rescaled = scaler.inverse_transform(np.hstack((forecast, np.zeros((len(forecast), scaled_data.shape[1] - 1)))))[:, 0]
    y_test_rescaled = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), scaled_data.shape[1] - 1)))))[:, 0]
    y_train_rescaled = scaler.inverse_transform(np.hstack((y_train.reshape(-1, 1), np.zeros((len(y_train), scaled_data.shape[1] - 1)))))[:, 0]

    # Ocena modelu
    metrics = evaluate_model(y_test_rescaled, forecast_rescaled, label=f"LSTM ({forecast_horizon}-year horizon)")

    # Wizualizacja wyników
    visualize_results(
        y_test=y_test_rescaled.flatten(),
        y_pred=forecast_rescaled.flatten(),
        title=f"LSTM Forecast vs Actual for {country} ({forecast_horizon} years)",
        years=years_test
    )

    visualize_combined_results(
        y_train=y_train_rescaled.flatten(),
        y_pred=forecast_rescaled.flatten(),
        y_test=y_test_rescaled.flatten(),
        years_train=years_train,
        years_test=years_test,
        model_name="LSTM",
        country=country
    )

    # Przygotowanie szczegółów prognozy
    details = pd.DataFrame({
        "Year": years_test,
        "Predicted Population": forecast_rescaled.flatten(),
        "Actual Population": y_test_rescaled.flatten()
    })

    # Przygotowanie danych do zwrotu
    return {
        "metrics": metrics,
        "y_test": y_test_rescaled.flatten(),
        "y_pred": forecast_rescaled.flatten(),
        "years_test": years_test,
        "y_train": y_train_rescaled.flatten(),
        "years_train": years_train,
        "details": details
    }


def set_random_seed(seed=42):
    """Ustawienie losowego ziarna dla stabilnych wyników."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

def plot_tuning_results(param_values, mape_values, param_name):
    """Wizualizuje wyniki tuningu jako wykres liniowy MAPE dla danego parametru."""
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, mape_values, marker='o', label="MAPE")
    plt.xlabel(f"{param_name}")
    plt.ylabel("MAPE (%)")
    plt.title(f"Tuning {param_name}: MAPE vs {param_name}")
    plt.grid(True)
    plt.legend()
    plt.show()

def tune_lstm_iteratively(data, country, forecast_horizon, lags=3):
    """Strojenie modelu LSTM krok po kroku dla poszczególnych parametrów z lepszym zakresem i wizualizacją."""
    print(f"\n=== Tuning LSTM Iteratively for {country} - {forecast_horizon}-year horizon ===")

    # Przygotowanie danych
    country_data = data[data['country'] == country][
        ['year', 'population', 'birth_rate', 'death_rate', 'net_migration_rate']].dropna().reset_index(drop=True)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(country_data[['population', 'birth_rate', 'death_rate', 'net_migration_rate']])

    X, y = [], []
    for i in range(len(scaled_data) - lags):
        X.append(scaled_data[i:i + lags])
        y.append(scaled_data[i + lags, 0])
    X, y = np.array(X), np.array(y)

    train_size = len(X) - forecast_horizon
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Tuned parameters
    best_params = {}
    mape_results = []

    def plot_tuning_results(param_values, mape_values, param_name):
        plt.figure(figsize=(10, 6))
        plt.plot(param_values, mape_values, marker='o', linestyle='-', label=f'MAPE vs {param_name}')
        plt.title(f'Tuning {param_name} for LSTM')
        plt.xlabel(param_name)
        plt.ylabel('MAPE (%)')
        plt.grid(True)
        plt.legend()
        plt.show()

    # Testowanie liczby epok
    epoch_values = [20, 50, 100, 200]
    mape_epochs = []
    for epochs in epoch_values:
        set_random_seed()
        model = Sequential([
            LSTM(64, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0)

        y_pred = model.predict(X_test)
        y_pred_rescaled = scaler.inverse_transform(
            np.hstack((y_pred, np.zeros((len(y_pred), scaled_data.shape[1] - 1)))))[:, 0]
        y_test_rescaled = scaler.inverse_transform(
            np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), scaled_data.shape[1] - 1)))))[:, 0]

        metrics = evaluate_model(y_test_rescaled, y_pred_rescaled, label=f"LSTM (Epochs={epochs})")
        mape_epochs.append(metrics['MAPE (%)'])
    plot_tuning_results(epoch_values, mape_epochs, "Epochs")
    best_params['epochs'] = epoch_values[np.argmin(mape_epochs)]

    # Testowanie dropout rate
    dropout_values = [0.1, 0.2, 0.3, 0.5]
    mape_dropouts = []
    for dropout_rate in dropout_values:
        set_random_seed()
        model = Sequential([
            LSTM(64, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(dropout_rate),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=16, verbose=0)

        y_pred = model.predict(X_test)
        y_pred_rescaled = scaler.inverse_transform(
            np.hstack((y_pred, np.zeros((len(y_pred), scaled_data.shape[1] - 1)))))[:, 0]
        y_test_rescaled = scaler.inverse_transform(
            np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), scaled_data.shape[1] - 1)))))[:, 0]

        metrics = evaluate_model(y_test_rescaled, y_pred_rescaled, label=f"LSTM (Dropout={dropout_rate})")
        mape_dropouts.append(metrics['MAPE (%)'])
    plot_tuning_results(dropout_values, mape_dropouts, "Dropout Rate")
    best_params['dropout_rate'] = dropout_values[np.argmin(mape_dropouts)]

    # Testowanie liczby neuronów
    unit_values = [32, 64, 128, 256]
    mape_units = []
    for units in unit_values:
        set_random_seed()
        model = Sequential([
            LSTM(units, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(best_params['dropout_rate']),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=16, verbose=0)

        y_pred = model.predict(X_test)
        y_pred_rescaled = scaler.inverse_transform(
            np.hstack((y_pred, np.zeros((len(y_pred), scaled_data.shape[1] - 1)))))[:, 0]
        y_test_rescaled = scaler.inverse_transform(
            np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), scaled_data.shape[1] - 1)))))[:, 0]

        metrics = evaluate_model(y_test_rescaled, y_pred_rescaled, label=f"LSTM (Units={units})")
        mape_units.append(metrics['MAPE (%)'])
    plot_tuning_results(unit_values, mape_units, "Units")
    best_params['units'] = unit_values[np.argmin(mape_units)]

    # Testowanie L2 regularizacji wag
    l2_values = [0.01, 0.1, 0.5]
    mape_l2 = []
    for l2_reg in l2_values:
        set_random_seed()
        model = Sequential([
            LSTM(best_params['units'], activation='tanh', kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
                 input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(best_params['dropout_rate']),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=16, verbose=0)

        y_pred = model.predict(X_test)
        y_pred_rescaled = scaler.inverse_transform(
            np.hstack((y_pred, np.zeros((len(y_pred), scaled_data.shape[1] - 1)))))[:, 0]
        y_test_rescaled = scaler.inverse_transform(
            np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), scaled_data.shape[1] - 1)))))[:, 0]

        metrics = evaluate_model(y_test_rescaled, y_pred_rescaled, label=f"LSTM (L2 Reg={l2_reg})")
        mape_l2.append(metrics['MAPE (%)'])
    plot_tuning_results(l2_values, mape_l2, "L2 Regularization")
    best_params['l2_reg'] = l2_values[np.argmin(mape_l2)]

    print(f"\nFinal Best Parameters: {best_params}")
    return best_params





# Streamlit App

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.title("System Predykcji Liczby Ludności")

# Wczytaj dane
data = pd.read_csv("prepared_data.csv")

# Statyczna lista krajów i algorytmów
allowed_countries = ["Polska", "Dania"]
allowed_forecast_years = [5, 10, 20]
algorithms = ["ARIMA", "Random Forest", "XGBoost", "LSTM"]

# Mapa krajów z polskiego na angielski
country_translation_reverse = {"Polska": "Poland", "Dania": "Denmark"}

# UI
country_polish = st.selectbox("Wybierz kraj:", allowed_countries)
algorithm = st.selectbox("Wybierz algorytm:", algorithms)
forecast_years = st.selectbox("Wybierz horyzont czasowy (w latach):", allowed_forecast_years)

# Mapowanie wybranego kraju z polskiego na angielski
country = country_translation_reverse[country_polish]

if st.button("Uruchom"):
    with st.spinner("Trwa przetwarzanie danych, proszę czekać..."):
    # Uruchamianie wybranego algorytmu
        results = None  # Zmienna na przechowanie wyników
        if algorithm == "ARIMA":
            results = run_arima_dynamic(data, country, forecast_years)
        elif algorithm == "Random Forest":
            results = run_random_forest(data, country, forecast_years, lags=3)
        elif algorithm == "XGBoost":
            results = run_xgboost(data, country, forecast_years, lags=5)
        elif algorithm == "LSTM":
            st.write("#### Optymalizacja hiperparametrów modelu LSTM...")
            best_params = tune_lstm_iteratively(data, country, forecast_years)
            results = run_lstm(data, country, forecast_years, **best_params)

        if results:
            # Szczegóły prognozy dla ostatniego roku
            st.write("### Szczegóły prognozy")
            last_year = results["years_test"][-1]
            last_predicted = results["y_pred"][-1]
            last_actual = results["y_test"][-1]

            st.write(f"Rok prognozy: **{last_year}**")
            st.markdown(f"- Prognozowana liczba ludności w {last_year} dla kraju {country_polish}: **{last_predicted:,.0f}**")
            st.markdown(f"- Rzeczywista liczba ludności w {last_year}: **{last_actual:,.0f}**")

            # Wyświetlanie metryk w tabelce
            st.write("### Ocena jakości modelu")
            metrics_df = pd.DataFrame(results["metrics"].items(), columns=["Metryka", "Wartość"])
            metrics_df["Wartość"] = metrics_df["Wartość"].apply(lambda x: f"{x:.4f}")  # Formatowanie wartości
            st.table(metrics_df)

            st.write("### Wizualizacja wyników")

            # Układ kolumn dla wizualizacji
            col1, col2 = st.columns(2)

            with col1:
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                ax1.plot(results["years_test"], results["y_test"] / 1e6, label="Wartości rzeczywiste", marker='o', color="green")
                ax1.plot(results["years_test"], results["y_pred"] / 1e6, label="Wartości predykcji", marker='x', color="red")
                ax1.set_title(f"Model: {algorithm} - {country_polish} ")
                ax1.set_xlabel("Rok")
                ax1.set_ylabel("Liczba ludności (miliony)")
                ax1.legend()
                ax1.grid(True)
                st.pyplot(fig1)

            with col2:
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                ax2.plot(results["years_train"], results["y_train"] / 1e6, label="Dane treningowe", color="blue")
                ax2.plot(results["years_test"], results["y_test"] / 1e6, label="Dane testowe", color="green")
                ax2.plot(results["years_test"], results["y_pred"] / 1e6, label="Predykcja", color="red")
                ax2.set_title(f"Model: {algorithm} - {country_polish}")
                ax2.set_xlabel("Rok")
                ax2.set_ylabel("Liczba ludności (miliony)")
                ax2.legend()
                ax2.grid(True)
                st.pyplot(fig2)
        else:
            st.error("Nie udało się uruchomić modelu. Sprawdź dane wejściowe.")
