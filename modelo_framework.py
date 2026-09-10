from pathlib import Path

import pandas as pd
from scipy.io import arff
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def calcular_metricas(valores_reales, predicciones):
    """La clase positiva es phishing: es_phishing = 1."""
    resultados = {
        "Accuracy": accuracy_score(valores_reales, predicciones),
        "Precision phishing": precision_score(
            valores_reales, predicciones, pos_label=1, zero_division=0
        ),
        "Recall phishing": recall_score(
            valores_reales, predicciones, pos_label=1, zero_division=0
        ),
        "F1 phishing": f1_score(
            valores_reales, predicciones, pos_label=1, zero_division=0
        ),
    }
    return resultados


def guardar_figura(figura, ruta):
    figura.tight_layout()
    figura.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(figura)


def graficar_comparacion(tabla, etiqueta, ruta):
    """Cada punto corresponde a un árbol diferente, no a una época."""
    figura, eje = plt.subplots(figsize=(8, 5))
    eje.plot(
        tabla["Valor"], tabla["Accuracy train"], "o-", label="Entrenamiento"
    )
    eje.plot(
        tabla["Valor"], tabla["Accuracy val"], "o-", label="Validación"
    )
    eje.set_xlabel(etiqueta)
    eje.set_ylabel("Accuracy")
    eje.set_title("Comparación de entrenamiento y validación")
    eje.set_xticks(tabla["Valor"])
    eje.set_ylim(0.85, 1.0)
    eje.grid(alpha=0.3)
    eje.legend()
    guardar_figura(figura, ruta)


def graficar_matriz(valores_reales, predicciones, nombre, ruta):
    matriz = confusion_matrix(valores_reales, predicciones, labels=[0, 1])
    figura, eje = plt.subplots(figsize=(6, 5))
    visualizacion = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=["Legítimo", "Phishing"],
    )
    visualizacion.plot(
        ax=eje, cmap="Blues", values_format="d", colorbar=False
    )
    eje.set_title("Árbol base: " + nombre)
    eje.set_xlabel("Clase predicha")
    eje.set_ylabel("Clase real")
    guardar_figura(figura, ruta)
    return matriz


# 2. Cargar el dataset y separar las entradas de la respuesta

carpeta = Path(__file__).resolve().parent
ruta_dataset = carpeta / "Training Dataset.arff"
carpeta_resultados = carpeta / "resultados"

datos, metadata = arff.loadarff(ruta_dataset)
df = pd.DataFrame(datos).astype(int)

assert set(df["Result"].unique()) == {-1, 1}, "Revisar las etiquetas originales."

# Original: -1 phishing, 1 legítimo. Nueva etiqueta: 1 phishing, 0 legítimo.
df["es_phishing"] = df["Result"].map({-1: 1, 1: 0})
X = df.drop(columns=["Result", "es_phishing"]).copy()
y = df["es_phishing"].copy()

assert X.shape[1] == 30, "Se esperaban 30 características."
assert not X.isna().any().any(), "Hay valores faltantes en las entradas."
print("Dimensiones de X:", X.shape)


# Agrupar perfiles idénticos sin borrar registros
# Un mismo grupo contiene las mismas 30 características, aunque cambie la etiqueta.

grupos = X.groupby(list(X.columns), sort=False, dropna=False).ngroup()
etiquetas_por_grupo = y.groupby(grupos).nunique()
grupos_contradiccion = (etiquetas_por_grupo > 1).sum()

print("Grupos distintos:", grupos.nunique())
print("Grupos con etiquetas contradictorias:", grupos_contradiccion)


#entrenamiento, validación y prueba por grupos

division_train = GroupShuffleSplit(
    n_splits=1, train_size=0.70, random_state=42
)
indices_train, indices_restantes = next(
    division_train.split(X, y, groups=grupos)
)

division_restante = GroupShuffleSplit(
    n_splits=1, test_size=0.50, random_state=42
)
indices_val_local, indices_test_local = next(
    division_restante.split(
        X.iloc[indices_restantes],
        y.iloc[indices_restantes],
        groups=grupos.iloc[indices_restantes],
    )
)

# Convertir posiciones dentro del conjunto restante a posiciones originales.
indices_val = indices_restantes[indices_val_local]
indices_test = indices_restantes[indices_test_local]

X_train = X.iloc[indices_train].copy()
y_train = y.iloc[indices_train].copy()
X_val = X.iloc[indices_val].copy()
y_val = y.iloc[indices_val].copy()
X_test = X.iloc[indices_test].copy()
y_test = y.iloc[indices_test].copy()

# isdisjoint comprueba que dos conjuntos no tengan elementos en común.
grupos_train = set(grupos.iloc[indices_train])
grupos_val = set(grupos.iloc[indices_val])
grupos_test = set(grupos.iloc[indices_test])

assert grupos_train.isdisjoint(grupos_val)
assert grupos_train.isdisjoint(grupos_test)
assert grupos_val.isdisjoint(grupos_test)
assert len(X_train) + len(X_val) + len(X_test) == len(X)
indices_totales = set(indices_train).union(indices_val, indices_test)
assert len(indices_totales) == len(X)

conjuntos = {
    "Entrenamiento": indices_train,
    "Validacion": indices_val,
    "Prueba": indices_test,
}

for nombre, indices in conjuntos.items():
    porcentaje = len(indices) / len(X)
    print(f"\n{nombre}: {len(indices)} registros ({porcentaje:.2%})")
    print(y.iloc[indices].value_counts().sort_index())

print("\nVerificación correcta: no hay grupos compartidos.")

# Guardar la división para poder comprobarla después.
carpeta_resultados.mkdir(exist_ok=True)
particiones = pd.DataFrame({"fila_original": X.index, "grupo": grupos})
for nombre, indices in conjuntos.items():
    particiones.loc[indices, "conjunto"] = nombre
particiones.to_csv(carpeta_resultados / "particiones.csv", index=False)


#Entrenar el árbol base

modelo_base = DecisionTreeClassifier(
    criterion="gini",
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
)

# Solo entrenamiento participa en el aprendizaje.
modelo_base.fit(X_train, y_train)
predicciones_train = modelo_base.predict(X_train)
predicciones_val = modelo_base.predict(X_val)

metricas_train = calcular_metricas(y_train, predicciones_train)
metricas_val = calcular_metricas(y_val, predicciones_val)

print("\nBASE: ENTRENAMIENTO")
print(classification_report(
    y_train, predicciones_train, labels=[0, 1],
    target_names=["Legítimo", "Phishing"], digits=4, zero_division=0
))
print("\nBASE: VALIDACIÓN")
print(classification_report(
    y_val, predicciones_val, labels=[0, 1],
    target_names=["Legítimo", "Phishing"], digits=4, zero_division=0
))
print("Profundidad base:", modelo_base.get_depth())
print("Hojas del base:", modelo_base.get_n_leaves())

mejor_nombre = "Base"
mejor_f1 = metricas_val["F1 phishing"]


#Comparar profundidades usando solamente validación

resultados_profundidad = []

for profundidad in [3, 5, 8, 12, 20]:
    modelo = DecisionTreeClassifier(
        criterion="gini",
        max_depth=profundidad,
        min_samples_leaf=1,
        random_state=42,
    )
    modelo.fit(X_train, y_train)

    predicciones_train_actual = modelo.predict(X_train)
    predicciones_val_actual = modelo.predict(X_val)
    metricas_train_actual = calcular_metricas(y_train, predicciones_train_actual)
    metricas_val_actual = calcular_metricas(y_val, predicciones_val_actual)

    resultados_profundidad.append({
        "Valor": profundidad,
        "Profundidad": modelo.get_depth(),
        "Hojas": modelo.get_n_leaves(),
        "Accuracy train": metricas_train_actual["Accuracy"],
        "Accuracy val": metricas_val_actual["Accuracy"],
        "Precision phishing val": metricas_val_actual["Precision phishing"],
        "Recall phishing val": metricas_val_actual["Recall phishing"],
        "F1 phishing val": metricas_val_actual["F1 phishing"],
    })

    if metricas_val_actual["F1 phishing"] > mejor_f1:
        mejor_f1 = metricas_val_actual["F1 phishing"]
        mejor_nombre = f"Profundidad máxima: {profundidad}"

comparacion_profundidad = pd.DataFrame(resultados_profundidad)
print("\nCOMPARACIÓN DE PROFUNDIDADES")
print(comparacion_profundidad.round(4).to_string(index=False))
comparacion_profundidad.to_csv(
    carpeta_resultados / "comparacion_profundidad.csv", index=False
)
graficar_comparacion(
    comparacion_profundidad,
    "Profundidad máxima",
    carpeta_resultados / "efecto_profundidad.png",
)


#Comparar el mínimo de registros por hoja

resultados_hojas = []

for minimo_hoja in [1, 5, 10, 20]:
    modelo = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_leaf=minimo_hoja,
        random_state=42,
    )
    modelo.fit(X_train, y_train)

    predicciones_train_actual = modelo.predict(X_train)
    predicciones_val_actual = modelo.predict(X_val)
    metricas_train_actual = calcular_metricas(y_train, predicciones_train_actual)
    metricas_val_actual = calcular_metricas(y_val, predicciones_val_actual)

    resultados_hojas.append({
        "Valor": minimo_hoja,
        "Profundidad": modelo.get_depth(),
        "Hojas": modelo.get_n_leaves(),
        "Accuracy train": metricas_train_actual["Accuracy"],
        "Accuracy val": metricas_val_actual["Accuracy"],
        "Precision phishing val": metricas_val_actual["Precision phishing"],
        "Recall phishing val": metricas_val_actual["Recall phishing"],
        "F1 phishing val": metricas_val_actual["F1 phishing"],
    })

    if metricas_val_actual["F1 phishing"] > mejor_f1:
        mejor_f1 = metricas_val_actual["F1 phishing"]
        mejor_nombre = f"Mínimo por hoja: {minimo_hoja}"

comparacion_hojas = pd.DataFrame(resultados_hojas)
print("\nCOMPARACIÓN DEL MÍNIMO POR HOJA")
print(comparacion_hojas.round(4).to_string(index=False))
comparacion_hojas.to_csv(
    carpeta_resultados / "comparacion_minimo_hoja.csv", index=False
)
graficar_comparacion(
    comparacion_hojas,
    "Mínimo de registros por hoja",
    carpeta_resultados / "efecto_minimo_hoja.png",
)

print(f"\nMejor F1 observado en validación: {mejor_nombre} ({mejor_f1:.4f})")


# Evalución final del modelo 

modelo_final = modelo_base
print("Configuración final fijada: árbol base. Sin reajuste con prueba.")
predicciones_test = modelo_final.predict(X_test)
metricas_test = calcular_metricas(y_test, predicciones_test)

print("\nEVALUACIÓN FINAL EN PRUEBA")
print(classification_report(
    y_test, predicciones_test, labels=[0, 1],
    target_names=["Legítimo", "Phishing"], digits=4, zero_division=0
))

# Guardar las métricas de los tres conjuntos en una sola tabla.
resumen = pd.DataFrame(
    [metricas_train, metricas_val, metricas_test]
)
resumen.insert(0, "Conjunto", ["Entrenamiento", "Validacion", "Prueba"])
resumen.to_csv(carpeta_resultados / "metricas.csv", index=False)

tabla_predicciones = pd.DataFrame({
    "fila_original": indices_test,
    "real_es_phishing": y_test.to_numpy(),
    "prediccion_es_phishing": predicciones_test,
})
tabla_predicciones.to_csv(
    carpeta_resultados / "predicciones_prueba.csv", index=False
)
print("\nEJEMPLOS DE PREDICCIONES (1 = phishing, 0 = legítimo)")
print(tabla_predicciones.head(10).to_string(index=False))


# Matrices de confusioón y primeras decisiones del árbol base

matriz_val = graficar_matriz(
    y_val, predicciones_val, "Validación",
    carpeta_resultados / "matriz_validacion.png",
)
matriz_test = graficar_matriz(
    y_test, predicciones_test, "Prueba",
    carpeta_resultados / "matriz_prueba.png",
)

print("\nLegítimos clasificados correctamente:", matriz_test[0, 0])
print("Falsas alarmas:", matriz_test[0, 1])
print("Casos de phishing no detectados:", matriz_test[1, 0])
print("Casos de phishing detectados:", matriz_test[1, 1])

figura, eje = plt.subplots(figsize=(18, 9))
plot_tree(
    modelo_base,
    feature_names=X.columns.tolist(),
    class_names=["Legítimo", "Phishing"],
    max_depth=2,  # Solo limita el dibujo, no el modelo entrenado.
    filled=True,
    rounded=True,
    impurity=False,
    fontsize=9,
    ax=eje,
)
eje.set_title("Primeras decisiones del árbol base")
guardar_figura(figura, carpeta_resultados / "arbol_base.png")

print("\nResultados y gráficas guardados en:", carpeta_resultados)
