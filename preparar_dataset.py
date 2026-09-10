from pathlib import Path
import pandas as pd

def cargar_datos(ruta):
    datos = pd.read_csv(ruta, keep_default_na=False)

    datos["date_time"] = pd.to_datetime(
        datos["date_time"],
        errors="raise"
    )

    print("Dimensiones originales:", datos.shape)
    return datos


def preparar_datos(datos):
    datos = datos.copy()

    # Identificar festivos antes de eliminar registros.
    fechas_festivas = (
        datos.loc[datos["holiday"] != "None", "date_time"]
        .dt.normalize()
        .unique()
    )

    # Extender la etiqueta a todas las horas de esas fechas.
    datos["holiday"] = (
        datos["date_time"]
        .dt.normalize()
        .isin(fechas_festivas)
        .astype(int)
    )

    print("Fechas festivas identificadas:", len(fechas_festivas))

    # Eliminar temperaturas de cero Kelvin.
    cantidad_inicial = len(datos)
    datos = datos.loc[datos["temp"] != 0].copy()

    print(
        "Temperaturas cero eliminadas:",
        cantidad_inicial - len(datos)
    )

    # Eliminar el valor anómalo revisado en el análisis.
    cantidad_inicial = len(datos)
    datos = datos.loc[datos["rain_1h"] != 9831.3].copy()

    print(
        "Registros de lluvia eliminados:",
        cantidad_inicial - len(datos)
    )

    # Excluir la medición numérica de nieve.
    # Las descripciones de nieve se conservan.
    datos = datos.drop(columns=["snow_1h"])

    datos["weather_description"] = (
        datos["weather_description"]
        .str.strip()
        .str.lower()
    )

    # Comprobar consistencia antes de usar el primer valor.
    variaciones = datos.groupby("date_time")[
        ["holiday", "traffic_volume"]
    ].nunique()

    if (variaciones > 1).any().any():
        raise ValueError(
            "Hay horas con valores distintos de holiday o traffic_volume."
        )

    datos_por_hora = datos.groupby("date_time").agg(
        holiday=("holiday", "first"),
        traffic_volume=("traffic_volume", "first")
    )

    # Promediar valores distintos para evitar ponderar mediciones
    # por el número de descripciones climáticas asociadas.
    for columna in ["temp", "rain_1h", "clouds_all"]:
        datos_por_hora[columna] = (
            datos.groupby("date_time")[columna]
            .agg(lambda valores: valores.drop_duplicates().mean())
        )

    # Representar las descripciones presentes en cada hora.
    # Se conserva el procedimiento del notebook: las categorías
    # se identifican antes de la separación temporal.
    indicadores_clima = pd.get_dummies(
        datos["weather_description"],
        prefix="clima",
        dtype=int
    )

    clima_por_hora = indicadores_clima.groupby(
        datos["date_time"]
    ).max()

    datos_por_hora["hour"] = datos_por_hora.index.hour
    datos_por_hora["weekday"] = datos_por_hora.index.dayofweek

    # Referencia: hora 0.
    horas = pd.get_dummies(
        datos_por_hora["hour"],
        prefix="hora",
        dtype=int
    ).drop(columns=["hora_0"])

    # Referencia: lunes, día 0.
    dias = pd.get_dummies(
        datos_por_hora["weekday"],
        prefix="dia",
        dtype=int
    ).drop(columns=["dia_0"])

    X = datos_por_hora[
        ["holiday", "temp", "rain_1h", "clouds_all"]
    ].copy()

    X = X.join(clima_por_hora, validate="one_to_one")
    X = X.join(horas, validate="one_to_one")
    X = X.join(dias, validate="one_to_one")
    X = X.sort_index()

    y = datos_por_hora.loc[X.index, "traffic_volume"].copy()

    if X.isna().any().any() or y.isna().any():
        raise ValueError(
            "Hay valores faltantes después de preparar los datos."
        )

    print("Dimensiones de X:", X.shape)
    print("Dimensiones de y:", y.shape)

    return X, y



def main():
    carpeta = Path(__file__).resolve().parent
    datos = cargar_datos(carpeta / "Metro_Interstate_Traffic_Volume.csv")
    X, y = preparar_datos(datos)
    salida = X.copy()
    salida["traffic_volume"] = y
    ruta = carpeta / "trafico_limpio.csv"
    salida.to_csv(ruta, index_label="date_time")
    print("Dataset preparado, sin normalizar:", ruta)
    print("Filas:", len(salida), "| Columnas incluyendo fecha:", len(salida.columns) + 1)


if __name__ == "__main__":
    main()

