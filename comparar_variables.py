from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modelo_manual import (
    cargar_dataset_limpio, separar_datos, normalizar_datos,
    entrenar_regresion, predecir, calcular_mse,
    calcular_rmse, calcular_mae, calcular_r2
)

def comparar_variables(
    X_train_norm,
    y_train,
    X_val_norm,
    y_val
):
    columnas_temporales = [
        columna
        for columna in X_train_norm.columns
        if columna.startswith(("hora_", "dia_"))
    ]

    columnas_clima = ["temp", "rain_1h", "clouds_all"] + [
        columna
        for columna in X_train_norm.columns
        if columna.startswith("clima_")
    ]

    versiones = {
        "Temporal": columnas_temporales,
        "Temporal + festivos": columnas_temporales + ["holiday"],
        "Temporal + clima": columnas_temporales + columnas_clima,
        "Completa": list(X_train_norm.columns)
    }

    objetivos_train = y_train.to_numpy(dtype=float)
    objetivos_val = y_val.to_numpy(dtype=float)

    resultados = []
    curvas = {}

    for nombre, columnas in versiones.items():
        print(f"\nENTRENANDO: {nombre}")

        datos_train = X_train_norm[columnas].to_numpy(dtype=float)
        datos_val = X_val_norm[columnas].to_numpy(dtype=float)

        pesos, b, historial_train, historial_val = entrenar_regresion(
            datos_train,
            objetivos_train,
            datos_val,
            objetivos_val,
            learning_rate=0.05,
            epocas=5000
        )

        pred_train = predecir(datos_train, pesos, b)
        pred_val = predecir(datos_val, pesos, b)

        mejora_final = (
            (historial_val[4500] - historial_val[5000])
            / historial_val[4500]
        ) * 100

        resultados.append({
            "Versión": nombre,
            "Variables": len(columnas),
            "MSE train": calcular_mse(objetivos_train, pred_train),
            "MSE val": calcular_mse(objetivos_val, pred_val),
            "RMSE val": calcular_rmse(objetivos_val, pred_val),
            "MAE val": calcular_mae(objetivos_val, pred_val),
            "R2 train": calcular_r2(objetivos_train, pred_train),
            "R2 val": calcular_r2(objetivos_val, pred_val),
            "Mejora val últimas 500 (%)": mejora_final
        })

        curvas[nombre] = historial_val

    tabla = pd.DataFrame(resultados)

    print("\nCOMPARACIÓN DE GRUPOS DE VARIABLES")
    print(tabla.round(4).to_string(index=False))
    print("\nEsta comparación no evalúa el conjunto de prueba.")

    fig, ejes = plt.subplots(1, 2, figsize=(13, 5))

    for nombre, historial in curvas.items():
        ejes[0].plot(historial, label=nombre)
        ejes[1].plot(
            range(4000, len(historial)),
            historial[4000:],
            label=nombre
        )

    ejes[0].set_title("Error de validación: entrenamiento completo")
    ejes[1].set_title("Detalle de las últimas 1,000 épocas")

    for eje in ejes:
        eje.set_xlabel("Época")
        eje.set_ylabel("MSE de validación")
        eje.legend()
        eje.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()

    return tabla



def main():
    ruta = Path(__file__).resolve().parent / "trafico_limpio.csv"
    X, y = cargar_dataset_limpio(ruta)
    xt, xv, xp, yt, yv, yp = separar_datos(X, y)
    nt, nv, _ = normalizar_datos(xt, xv, xp)
    comparar_variables(nt, yt, nv, yv)


if __name__ == "__main__":
    main()

