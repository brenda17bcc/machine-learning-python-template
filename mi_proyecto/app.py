# app.py - Aplicación web con Flask para usar el modelo Iris
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Cargamos el modelo entrenado 
modelo = joblib.load('modelo_iris.pkl')

# Nombres de las especies 
especies = ['Setosa', 'Versicolor', 'Virginica']

# Nombres de columnas EXACTOS con los que se entrenó el modelo
columnas = ['sepal length (cm)', 'sepal width (cm)',
            'petal length (cm)', 'petal width (cm)']

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    if request.method == 'POST':
        largo_sepalo = float(request.form['largo_sepalo'])
        ancho_sepalo = float(request.form['ancho_sepalo'])
        largo_petalo = float(request.form['largo_petalo'])
        ancho_petalo = float(request.form['ancho_petalo'])

        datos = pd.DataFrame(
            [[largo_sepalo, ancho_sepalo, largo_petalo, ancho_petalo]],
            columns=columnas
        )
        prediccion = modelo.predict(datos)[0]
        resultado = especies[prediccion]

    return render_template('index.html', resultado=resultado)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)