"""Regresión del tráfico con Random Forest y scikit-learn.

Utiliza el mismo trafico_limpio.csv y la misma división cronológica que
la regresión manual. La selección se realiza únicamente con validación.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from modelo_manual import cargar_dataset_limpio, separar_datos


CARPETA = Path(__file__).resolve().parent
RUTA_DATOS = CARPETA / "trafico_limpio.csv"
CARPETA_RESULTADOS = CARPETA / "resultados_random_forest"


def calcular_metricas(valores_reales, predicciones):
    """Calcula métricas de regresión con las unidades del proyecto."""
    mse = mean_squared_error(valores_reales, predicciones)
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(valores_reales, predicciones),
        "R2": r2_score(valores_reales, predicciones),
    }


def seleccionar_mejor_configuracion(resultados):
    """Devuelve la primera configuración con menor MAE de validación."""
    indice = resultados["MAE validación"].idxmin()
    return resultados.loc[indice]


def agrupar_importancias(columnas, importancias):
    """Agrupa las importancias internas según el tipo de entrada."""
    totales = {"Temporales": 0.0, "Clima": 0.0, "Festivos": 0.0}

    for columna, importancia in zip(columnas, importancias):
        if columna.startswith("hora_") or columna.startswith("dia_"):
            grupo = "Temporales"
        elif columna == "holiday":
            grupo = "Festivos"
        else:
            grupo = "Clima"
        totales[grupo] += importancia

    resumen = pd.DataFrame(
        {"Grupo": list(totales.keys()), "Importancia": list(totales.values())}
    )
    return resumen.sort_values("Importancia", ascending=False).reset_index(drop=True)


def guardar_figura(figura, nombre):
    figura.tight_layout()
    figura.savefig(CARPETA_RESULTADOS / nombre, dpi=220, bbox_inches="tight")
    plt.close(figura)


def crear_modelo(configuracion):
    return RandomForestRegressor(
        n_estimators=configuracion["n_estimators"],
        max_depth=configuracion["max_depth"],
        min_samples_leaf=configuracion["min_samples_leaf"],
        max_features=configuracion["max_features"],
        random_state=42,
        n_jobs=-1,
    )


def configuraciones_evaluadas():
    """Conjunto breve de configuraciones definidas antes de evaluar prueba."""
    return [
        {"nombre": "Base", "n_estimators": 100, "max_depth": None,
         "min_samples_leaf": 1, "max_features": 1.0},
        {"nombre": "50 árboles", "n_estimators": 50, "max_depth": None,
         "min_samples_leaf": 1, "max_features": 1.0},
        {"nombre": "200 árboles", "n_estimators": 200, "max_depth": None,
         "min_samples_leaf": 1, "max_features": 1.0},
        {"nombre": "Profundidad 10", "n_estimators": 100, "max_depth": 10,
         "min_samples_leaf": 1, "max_features": 1.0},
        {"nombre": "Profundidad 20", "n_estimators": 100, "max_depth": 20,
         "min_samples_leaf": 1, "max_features": 1.0},
        {"nombre": "Profundidad 30", "n_estimators": 100, "max_depth": 30,
         "min_samples_leaf": 1, "max_features": 1.0},
        {"nombre": "Mínimo hoja 2", "n_estimators": 100, "max_depth": None,
         "min_samples_leaf": 2, "max_features": 1.0},
        {"nombre": "Mínimo hoja 5", "n_estimators": 100, "max_depth": None,
         "min_samples_leaf": 5, "max_features": 1.0},
        {"nombre": "Mínimo hoja 10", "n_estimators": 100, "max_depth": None,
         "min_samples_leaf": 10, "max_features": 1.0},
        {"nombre": "Variables sqrt", "n_estimators": 100, "max_depth": None,
         "min_samples_leaf": 1, "max_features": "sqrt"},
        {"nombre": "50 % variables", "n_estimators": 100, "max_depth": None,
         "min_samples_leaf": 1, "max_features": 0.5},
        {"nombre": "Combinada", "n_estimators": 200, "max_depth": 20,
         "min_samples_leaf": 2, "max_features": 0.5},
    ]


def ejecutar_experimento():
    CARPETA_RESULTADOS.mkdir(exist_ok=True)

    X, y = cargar_dataset_limpio(RUTA_DATOS)
    X_train, X_val, X_test, y_train, y_val, y_test = separar_datos(X, y)

    configuraciones = configuraciones_evaluadas()
    resultados = []
    modelos = {}

    print("\nCOMPARACIÓN DE CONFIGURACIONES EN VALIDACIÓN")
    for configuracion in configuraciones:
        modelo = crear_modelo(configuracion)
        modelo.fit(X_train, y_train)
        modelos[configuracion["nombre"]] = modelo

        metricas_train = calcular_metricas(y_train, modelo.predict(X_train))
        metricas_val = calcular_metricas(y_val, modelo.predict(X_val))
        fila = {
            "Configuración": configuracion["nombre"],
            "Árboles": configuracion["n_estimators"],
            "Profundidad máxima": str(configuracion["max_depth"]),
            "Mínimo por hoja": configuracion["min_samples_leaf"],
            "Variables por división": str(configuracion["max_features"]),
            "MAE entrenamiento": metricas_train["MAE"],
            "MAE validación": metricas_val["MAE"],
            "RMSE validación": metricas_val["RMSE"],
            "R2 entrenamiento": metricas_train["R2"],
            "R2 validación": metricas_val["R2"],
        }
        resultados.append(fila)
        print(
            f"{configuracion['nombre']}: "
            f"MAE val = {metricas_val['MAE']:.2f} | "
            f"R² val = {metricas_val['R2']:.4f}"
        )

    tabla_resultados = pd.DataFrame(resultados)
    tabla_resultados.to_csv(
        CARPETA_RESULTADOS / "comparacion_configuraciones.csv", index=False
    )
    mejor = seleccionar_mejor_configuracion(tabla_resultados)
    nombre_mejor = mejor["Configuración"]
    modelo_final = modelos[nombre_mejor]

    print("\nCONFIGURACIÓN SELECCIONADA:", nombre_mejor)
    print("Criterio: menor MAE de validación")

    predicciones_train = modelo_final.predict(X_train)
    predicciones_val = modelo_final.predict(X_val)
    predicciones_test = modelo_final.predict(X_test)

    resumen_metricas = []
    for nombre, reales, predicciones in [
        ("Entrenamiento", y_train, predicciones_train),
        ("Validación", y_val, predicciones_val),
        ("Prueba", y_test, predicciones_test),
    ]:
        fila = {"Conjunto": nombre}
        fila.update(calcular_metricas(reales, predicciones))
        resumen_metricas.append(fila)

    metricas = pd.DataFrame(resumen_metricas)
    metricas.to_csv(CARPETA_RESULTADOS / "metricas_finales.csv", index=False)
    print("\nMÉTRICAS DEL RANDOM FOREST SELECCIONADO")
    print(metricas.round(4).to_string(index=False))

    predicciones_prueba = pd.DataFrame(
        {"real": y_test, "prediccion": predicciones_test}, index=X_test.index
    )
    predicciones_prueba["error"] = (
        predicciones_prueba["prediccion"] - predicciones_prueba["real"]
    )
    predicciones_prueba.to_csv(CARPETA_RESULTADOS / "predicciones_prueba.csv")

    importancias = pd.DataFrame(
        {"Variable": X_train.columns, "Importancia": modelo_final.feature_importances_}
    ).sort_values("Importancia", ascending=False)
    importancias.to_csv(CARPETA_RESULTADOS / "importancias_variables.csv", index=False)
    grupos = agrupar_importancias(
        X_train.columns, modelo_final.feature_importances_
    )
    grupos.to_csv(CARPETA_RESULTADOS / "importancias_grupos.csv", index=False)

    errores_val = pd.DataFrame(
        {"real": y_val, "prediccion": predicciones_val}, index=X_val.index
    )
    errores_val["error"] = errores_val["prediccion"] - errores_val["real"]
    errores_val["error_absoluto"] = errores_val["error"].abs()
    errores_hora = errores_val.groupby(errores_val.index.hour).agg(
        error_promedio=("error", "mean"),
        MAE=("error_absoluto", "mean"),
        observaciones=("error", "size"),
    )
    errores_hora.index.name = "hora"
    errores_hora.to_csv(CARPETA_RESULTADOS / "errores_validacion_hora.csv")

    siete = errores_val[errores_val.index.hour == 7].copy()
    errores_siete = siete.groupby(siete.index.dayofweek).agg(
        error_promedio=("error", "mean"),
        MAE=("error_absoluto", "mean"),
        observaciones=("error", "size"),
    )
    errores_siete.index.name = "weekday"
    errores_siete.to_csv(CARPETA_RESULTADOS / "errores_0700_dia.csv")

    comparacion_modelos = pd.DataFrame(
        [
            {
                "Modelo": "Regresión lineal manual",
                "MSE prueba": 626723.0078,
                "RMSE prueba": 791.6584,
                "MAE prueba": 583.5983,
                "R2 prueba": 0.8400247,
            },
            {
                "Modelo": "Random Forest",
                "MSE prueba": metricas.loc[2, "MSE"],
                "RMSE prueba": metricas.loc[2, "RMSE"],
                "MAE prueba": metricas.loc[2, "MAE"],
                "R2 prueba": metricas.loc[2, "R2"],
            },
        ]
    )
    comparacion_modelos.to_csv(
        CARPETA_RESULTADOS / "comparacion_modelos.csv", index=False
    )

    figura, eje = plt.subplots(figsize=(9, 5))
    orden = tabla_resultados.sort_values("MAE validación")
    eje.barh(orden["Configuración"], orden["MAE validación"])
    eje.set_title("MAE de validación por configuración")
    eje.set_xlabel("MAE (vehículos por hora)")
    guardar_figura(figura, "comparacion_configuraciones.png")

    figura, ejes = plt.subplots(1, 2, figsize=(11, 4.5))
    ejes[0].plot(errores_hora.index, errores_hora["MAE"], marker="o")
    ejes[0].set(
        title="MAE de validación por hora", xlabel="Hora",
        ylabel="Vehículos por hora", xticks=range(0, 24, 2),
    )
    dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    ejes[1].bar(dias, errores_siete["error_promedio"])
    ejes[1].axhline(0, color="black", linewidth=0.8)
    ejes[1].set(
        title="Error promedio a las 07:00", xlabel="Día",
        ylabel="Predicción - valor real",
    )
    guardar_figura(figura, "errores_temporales.png")

    figura, eje = plt.subplots(figsize=(7, 4.5))
    eje.bar(grupos["Grupo"], grupos["Importancia"])
    eje.set_title("Importancia interna acumulada por grupo")
    eje.set_ylabel("Proporción de importancia")
    guardar_figura(figura, "importancia_grupos.png")

    muestra = errores_val.iloc[:168]
    figura, eje = plt.subplots(figsize=(10, 4.5))
    eje.plot(muestra.index, muestra["real"], label="Real")
    eje.plot(muestra.index, muestra["prediccion"], label="Predicción")
    eje.set_title("Primera semana de validación")
    eje.set_ylabel("Vehículos por hora")
    eje.legend()
    figura.autofmt_xdate()
    guardar_figura(figura, "predicciones_semana.png")

    print("\nResultados guardados en:", CARPETA_RESULTADOS)
    return tabla_resultados, metricas, modelo_final


if __name__ == "__main__":
    ejecutar_experimento()
