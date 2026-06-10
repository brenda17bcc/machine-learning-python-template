from utils import db_connect
engine = db_connect()

# Paso 1 - Cargar y explorar el dataset Iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Cargamos el dataset como un DataFrame de pandas
iris = load_iris(as_frame=True)
df = iris.frame  

df['especie'] = df['target'].map(dict(enumerate(iris.target_names)))

# --- Primer vistazo a los datos ---
print("Forma del dataset (filas, columnas):", df.shape)
print("\nPrimeras filas:")
print(df.head())
print("\nInformación general:")
df.info()
print("\nEstadísticas básicas:")
print(df.describe())
print("\nCantidad de ejemplos por especie:")
print(df['especie'].value_counts())

colores = {'setosa': '#FF1E56', 'versicolor': '#00E5FF', 'virginica': '#7CFC00'}

plt.figure(figsize=(8, 6))
for especie, color in colores.items():
    subset = df[df['especie'] == especie]
    plt.scatter(subset['petal length (cm)'], subset['petal width (cm)'],
                c=color, label=especie, s=70, edgecolors='black', alpha=0.85)

plt.title('Iris: largo vs ancho del pétalo por especie', fontsize=14, fontweight='bold')
plt.xlabel('Largo del pétalo (cm)')
plt.ylabel('Ancho del pétalo (cm)')
plt.legend(title='Especie')
plt.grid(True, alpha=0.3)
plt.show()

pip install pandas matplotlib scikit-learn

# Paso 2 - Entrenar y optimizar un modelo de clasificación
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1) Cargamos los datos otra vez 
iris = load_iris(as_frame=True)
X = iris.data      
y = iris.target     

# 2) Separamos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Tamaño entrenamiento:", X_train.shape[0], "| Tamaño prueba:", X_test.shape[0])

# 3) Modelo base
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)
pred_base = modelo.predict(X_test)
print("\nAccuracy del modelo base:", round(accuracy_score(y_test, pred_base), 3))

# 4) Optimización: probamos varias combinaciones de hiperparámetros
parametros = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 3, 5]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), parametros, cv=5)
grid.fit(X_train, y_train)
print("\nMejores parámetros encontrados:", grid.best_params_)
mejor_modelo = grid.best_estimator_

# 5) Evaluamos el mejor modelo
pred = mejor_modelo.predict(X_test)
print("\nAccuracy del modelo optimizado:", round(accuracy_score(y_test, pred), 3))
print("\nReporte de clasificación:")
print(classification_report(y_test, pred, target_names=iris.target_names))

matriz = confusion_matrix(y_test, pred)
plt.figure(figsize=(6, 5))
sns.heatmap(matriz, annot=True, fmt='d', cmap='plasma',
            xticklabels=iris.target_names, yticklabels=iris.target_names,
            linewidths=1, linecolor='white')
plt.title('Matriz de confusión', fontsize=14, fontweight='bold')
plt.xlabel('Predicción')
plt.ylabel('Valor real')
plt.show()

importancias = pd.Series(mejor_modelo.feature_importances_, index=X.columns).sort_values()
colores_barras = ['#FF1E56', '#FFAC41', '#00E5FF', '#7CFC00']
plt.figure(figsize=(8, 5))
importancias.plot(kind='barh', color=colores_barras, edgecolor='black')
plt.title('Importancia de cada característica', fontsize=14, fontweight='bold')
plt.xlabel('Importancia')
plt.tight_layout()
plt.show()

# Guardamos el modelo para usarlo en Flask (Paso 3)
joblib.dump(mejor_modelo, 'modelo_iris.pkl')
print("\n✅ Modelo guardado como 'modelo_iris.pkl'")


