from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUTA_DATOS = Path(__file__).resolve().parent.parent / "trafico_limpio.csv"

def cargar_dataset_limpio(ruta):
    datos = pd.read_csv(ruta, float_precision="round_trip")
    datos["date_time"] = pd.to_datetime(datos["date_time"], errors="raise")
    datos = datos.set_index("date_time").sort_index()
    if not datos.index.is_unique or datos.isna().any().any():
        raise ValueError("El dataset tiene fechas repetidas o valores faltantes.")
    y = datos["traffic_volume"].copy()
    X = datos.drop(columns=["traffic_volume"])
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Las entradas contienen valores no finitos.")
    print("Dimensiones de X:", X.shape)
    return X, y

def separar_datos(X, y):
    corte_train = int(len(X) * 0.70)
    corte_val = int(len(X) * 0.85)

    X_train = X.iloc[:corte_train].copy()
    X_val = X.iloc[corte_train:corte_val].copy()
    X_test = X.iloc[corte_val:].copy()

    y_train = y.iloc[:corte_train].copy()
    y_val = y.iloc[corte_train:corte_val].copy()
    y_test = y.iloc[corte_val:].copy()

    for nombre, conjunto in [
        ("Entrenamiento", X_train),
        ("Validación", X_val),
        ("Prueba", X_test)
    ]:
        print(
            nombre,
            "| Registros:", len(conjunto),
            "| Desde:", conjunto.index.min(),
            "| Hasta:", conjunto.index.max()
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


def normalizar_datos(X_train, X_val, X_test):
    columnas = ["temp", "rain_1h", "clouds_all"]

    
    minimos = X_train[columnas].min()
    rangos = X_train[columnas].max() - minimos

    if (rangos == 0).any():
        raise ValueError("Una columna numérica tiene rango cero.")

    train_norm = X_train.copy()
    val_norm = X_val.copy()
    test_norm = X_test.copy()

    for conjunto in [train_norm, val_norm, test_norm]:
        conjunto[columnas] = (
            conjunto[columnas] - minimos
        ) / rangos

    return train_norm, val_norm, test_norm


def predecir(datos, pesos, b):
    return datos @ pesos + b


def calcular_mse(reales, predicciones):
    return np.mean((predicciones - reales) ** 2)


def calcular_rmse(reales, predicciones):
    return np.sqrt(calcular_mse(reales, predicciones))


def calcular_mae(reales, predicciones):
    return np.mean(np.abs(predicciones - reales))


def calcular_r2(reales, predicciones):
    suma_errores = np.sum((reales - predicciones) ** 2)
    suma_total = np.sum((reales - np.mean(reales)) ** 2)

    if suma_total == 0:
        return np.nan

    return 1 - suma_errores / suma_total


def entrenar_regresion(
    datos_train,
    objetivos_train,
    datos_val,
    objetivos_val,
    learning_rate,
    epocas
):
    # Cada entrenamiento comienza desde cero.
    pesos = np.zeros(datos_train.shape[1])
    b = 0.0
    n = len(objetivos_train)

    historial_train = [
        calcular_mse(
            objetivos_train,
            predecir(datos_train, pesos, b)
        )
    ]

    historial_val = [
        calcular_mse(
            objetivos_val,
            predecir(datos_val, pesos, b)
        )
    ]

    for epoca in range(epocas):
        predicciones = predecir(datos_train, pesos, b)
        errores = predicciones - objetivos_train

        # Solo entrenamiento actualiza los parámetros.
        gradiente_pesos = (2 / n) * (datos_train.T @ errores)
        gradiente_b = 2 * np.mean(errores)

        pesos = pesos - learning_rate * gradiente_pesos
        b = b - learning_rate * gradiente_b

        mse_train = calcular_mse(
            objetivos_train,
            predecir(datos_train, pesos, b)
        )

        mse_val = calcular_mse(
            objetivos_val,
            predecir(datos_val, pesos, b)
        )

        historial_train.append(mse_train)
        historial_val.append(mse_val)

        if (epoca + 1) % 500 == 0:
            print(
                f"Época {epoca + 1}: "
                f"Train = {mse_train:.2f} | Val = {mse_val:.2f}"
            )

    return pesos, b, historial_train, historial_val


def main():
    X, y = cargar_dataset_limpio(RUTA_DATOS)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    ) = separar_datos(X, y)

    X_train_norm, X_val_norm, X_test_norm = normalizar_datos(
        X_train, X_val, X_test
    )

    datos_train = X_train_norm.to_numpy(dtype=float)
    objetivos_train = y_train.to_numpy(dtype=float)

    datos_val = X_val_norm.to_numpy(dtype=float)
    objetivos_val = y_val.to_numpy(dtype=float)

    print("\nMODELO BASE: learning_rate = 0.01")

    pesos_base, b_base, train_base, val_base = entrenar_regresion(
        datos_train,
        objetivos_train,
        datos_val,
        objetivos_val,
        learning_rate=0.01,
        epocas=5000
    )

    print("\nMODELO AJUSTADO: learning_rate = 0.05")

    pesos, b, historial_train, historial_val = entrenar_regresion(
        datos_train,
        objetivos_train,
        datos_val,
        objetivos_val,
        learning_rate=0.05,
        epocas=5000
    )

    resultados = []

    for nombre, pesos_modelo, intercepto in [
        ("Base", pesos_base, b_base),
        ("Ajustado", pesos, b)
    ]:
        pred_train = predecir(
            datos_train, pesos_modelo, intercepto
        )
        pred_val = predecir(
            datos_val, pesos_modelo, intercepto
        )

        resultados.append({
            "Modelo": nombre,
            "MSE train": calcular_mse(objetivos_train, pred_train),
            "MSE val": calcular_mse(objetivos_val, pred_val),
            "R2 train": calcular_r2(objetivos_train, pred_train),
            "R2 val": calcular_r2(objetivos_val, pred_val)
        })

    print("\nCOMPARACIÓN DE MODELOS")
    print(
        pd.DataFrame(resultados)
        .round(4)
        .to_string(index=False)
    )

    predicciones_train = predecir(datos_train, pesos, b)
    predicciones_val = predecir(datos_val, pesos, b)

    comparacion = pd.DataFrame({
        "real": objetivos_val,
        "prediccion": predicciones_val
    }, index=X_val.index)

    comparacion["error"] = (
        comparacion["prediccion"] - comparacion["real"]
    )

    print("\nPREDICCIONES DEL MODELO AJUSTADO EN VALIDACIÓN")
    print(comparacion.head(24).round(1).to_string())

    errores_por_hora = comparacion.groupby(
        comparacion.index.hour
    )["error"].agg(
        error_promedio="mean",
        error_absoluto_promedio=lambda errores: errores.abs().mean(),
        observaciones="count"
    )

    print("\nERRORES DE VALIDACIÓN POR HORA")
    print(errores_por_hora.round(1))

    mejor_epoca = int(np.argmin(historial_val))

    print("\nÉpoca con menor MSE de validación:", mejor_epoca)
    print("Menor MSE de validación:", historial_val[mejor_epoca])

    # Evaluar prueba con la configuración previamente fijada.
    # No se ajustan pesos ni parámetros usando prueba.
    datos_test = X_test_norm.to_numpy(dtype=float)
    objetivos_test = y_test.to_numpy(dtype=float)

    predicciones_test = predecir(datos_test, pesos, b)

    resumen_metricas = []

    for nombre, reales, estimaciones in [
        ("Entrenamiento", objetivos_train, predicciones_train),
        ("Validación", objetivos_val, predicciones_val),
        ("Prueba", objetivos_test, predicciones_test)
    ]:
        resumen_metricas.append({
            "Conjunto": nombre,
            "MSE": calcular_mse(reales, estimaciones),
            "RMSE": calcular_rmse(reales, estimaciones),
            "MAE": calcular_mae(reales, estimaciones),
            "R2": calcular_r2(reales, estimaciones)
        })

    print("\nMÉTRICAS DEL MODELO AJUSTADO")
    print(
        pd.DataFrame(resumen_metricas)
        .round(4)
        .to_string(index=False)
    )

    fig, ejes = plt.subplots(1, 2, figsize=(13, 5))

    ejes[0].plot(historial_train, label="Entrenamiento")
    ejes[0].plot(historial_val, label="Validación")
    ejes[0].set_title("Modelo ajustado: evolución del error")

    ejes[1].plot(
        val_base,
        label="Base: learning rate 0.01"
    )
    ejes[1].plot(
        historial_val,
        label="Ajustado: learning rate 0.05"
    )
    ejes[1].set_title("Comparación del error de validación")

    for eje in ejes:
        eje.set_xlabel("Época")
        eje.set_ylabel("MSE")
        eje.legend()
        eje.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()

