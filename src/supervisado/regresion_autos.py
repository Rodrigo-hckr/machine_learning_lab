"""
script: regresion_autos.py
categoria: Aprendizaje supervisado -> Regresion lineal

Por que hacemos este ejercicio?
En el aprendizaje supervisado entrenamos un modelo usando ejemplos con entradas (x) y respuestas conocidas o etiquetadas (y) para aprender la relacion entre ellas y predecir salidas en datos nuevos. Aqui usamos el ejemplo clasico de autos usados: estimar el precio (y) a partir del kilometraje y el año de fabricacion (x).
"""

#paso 1 -> importar las librerias
#usamos pandas para organizar los datos en tablas y scikit-learn
# para entender el modelo de regresion lineal y medir su error con MAE.
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def main():
    # Paso 2 -> preparar el dataset supervisado (entradas conocidas y salida y)
    #cada fila representa un vehiculo con sus caracteristicas y su precio real
    # contar etiquetas 'y' es lo que define el aprendizaje supervisado.
    datos = pd.DataFrame({
        "kilometraje_km": [51715, 7980, 82497, 85199, 62948, 35000, 91000, 22000],
        "anio": [2012, 2013, 2012, 2011, 2008, 2014, 2010, 2015],
        "precio_usd": [15985, 19600, 14095, 12490, 13985, 21200, 11500, 23500],
    })

    # Paso 3 -> separar variables independientes 'x' y dependiente 'y'
    # X contiene los atributos que influyen en el resultado. Y es el valor
    # continuo que queremos predecir, osea el precio en usd.

    X = datos[["kilometraje_km", "anio"]]
    y = datos["precio_usd"]

    # paso -> 4 crear y entrenar el modelo de regresion lineal
    # fit(x, y) ajusta los parametros del modelo para encontrar la relacion
    # lineal que mejor aproxima 'y' y apartir de 'x' minimiza el error.

    modelo = LinearRegression()
    modelo.fit(X, y)

    # paso -> 5 Evaluar el desempeño del modelo con mean absolute Error
    # comprobamos los valores reales 'y' contra las predicciones del modelo para saber
    # en promedio en cuantos usd se equivoca.

    predicciones = modelo.predict(X)
    mae = mean_absolute_error(y, predicciones)

    #paso ->6 hacer una prediccion sobre un auto nuevo *sin precio conocido*
    # alimentamos nuevas caracteristicas X al modelo ya entrenado para estimar 'y'
    auto_nuevo = pd.DataFrame({"kilometraje_km": [40000], "anio": [2013]})
    precio_estimado = modelo.predict(auto_nuevo)[0]

    # paso 7 -> Mostramos los resultados claros en la consola para registrarlos o compartirlos
    print("=== Aprendizaje Supervisado: Regresion Lineal de Autos ===")
    print(f"Error absoluto medio (MAE): ${mae:,.2f} USD")
    print(f"Prediccion para auto (40,000 km, año 2013)): ${precio_estimado:,.2f} USD")

if __name__ == "__main__":
    main()

