# Laboratorio Machine Learning
# Capitulo (1 al 3)

# Conceptos importantes.
- **comando directo vs Machine Learning** Wn peogrmamcion tradicional se envia un ocmando diercto con una respuesta predefinida '2+2' -> '4'. en  machine learning se entregan datos de entrada 'x' para construir un modelo mediante patrones, razonamiento porbabilisticos y prueba y error. 
- **Disciplina y mineria de datos**  Machine learning enfatiza autoaprendizaje a partir de laexposicion a datos para mejorar predicciones, mientras que la mineria de datos suele extraer patrones o relaciones ocultas en grandes conjuntos de datos.
- **Entrenamiento y prueba** Lo s datos se dividen entre datos de entrenamiento *para desarrollar le modelo* y datos de prueba (para evaluar su desempeño de datos no vistos) 

## Cuadro comparativo de categorias y ramas

| Categoría / Rama | Respuestas Previas (`y`) | Objetivo Principal | Ejemplo Clásico |
| :--- | :--- | :--- | :--- |
| **Aprendizaje supervisado** | Sí (etiquetadas) | Aprender la relación entre entradas (`X`) y salidas (`y`) para predecir `y` en datos nuevos[cite: 2, 3] | Estimar el precio de un auto usado a partir de atributos como marca, año y kilometraje[cite: 2, 3] |
| **Aprendizaje no supervisado** | No (sin etiquetas) | Descubrir patrones, estructuras o grupos ocultos en `X` sin etiquetas de salida[cite: 2, 3] | Agrupar observaciones similares en clústeres[cite: 2, 3] |
| **Aprendizaje semisupervisado** | Parcial (mezcla) | Aprovechar datos no etiquetados junto con ejemplos etiquetados para reforzar el modelo[cite: 2, 3] | Entrenar con casos etiquetados y extender el etiquetado o reentrenamiento a datos sin etiqueta[cite: 2, 3] |
| **Aprendizaje por refuerzo** | No fijas (recompensa / retroalimentación) | Aprender mediante prueba y error e iteraciones para maximizar el desempeño o recompensa hacia un objetivo[cite: 2, 3] | Un jugador o agente que prueba acciones en un entorno y ajusta su comportamiento según la retroalimentación[cite: 2, 3] |
| **Minería de datos (afín)** | Variable / enfoque exploratorio | Extraer patrones o relaciones ocultos a partir de datos[cite: 2, 3] | Explorar patrones o relaciones en conjuntos de datos históricos[cite: 2, 3] | 
## Ejecucuion 
```bash
python3 src/categorias_ml.py
python3 src/refuerzo_q.py 

