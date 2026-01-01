import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn import tree





 ############# GARGAR DATASET Y DEFINIICION DE VARIABLES

# Cargar el archivo CSV

df = "Pacientes.csv"

df = pd.read_csv(df)



# Verificar las primeras filas del Dataset

print(df.head(10))



# 3. Graficar el pairplot

sns.pairplot(df, hue="problema_cardiaco")



#Establecer variables dependientes y independientes

variablex = df[["edad", "colesterol"]].values  # Variables independientes

print(variablex[:10])  

#. la variable a predeir como variable dependiente

variabley = df["problema_cardiaco"].values.reshape(-1, 1) # Variables dependientes

print(variabley[:10])







 ############  ENTRENAMIENTO DEL MODELO ####################

# 4 y 5:  Crear los datasets de entrenamiento y prueba

x_train, x_test, y_train, y_test = train_test_split(variablex, variabley, test_size=0.2, random_state=42)



# Crear y entrenar el modelo de arbol de decisión

modelTree = DecisionTreeClassifier(criterion="gini", max_depth=3)

modelTree.fit(x_train, y_train)



# Hacer predicciones

y_pred = modelTree.predict(x_test)



#  6 y 7   Calcular la precisión y F1-score

f1 = f1_score(y_test, y_pred)  

print(f"El modelo obtuvo un indice F1 de: {f1}")



percent = modelTree.score(x_test, y_test)

print(f"El modelo obtuvo un {percent*100:.2f}% de precisiion para clasificar")



#  8. Mostrar el arbol de decision

plt.figure(figsize=(12, 6))

tree.plot_tree(modelTree, feature_names=["Edad", "Colesterol"])

plt.show()



# 9.  Matriz de confusion

conf_matrix = confusion_matrix(y_test, y_pred)

cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])

cm_display.plot(cmap=plt.cm.Blues)

plt.show()