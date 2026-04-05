# Clasificador Inteligente de Imágenes de Ropa
**Empresa:** StyleNet — Área de Ciencia de Datos

---

## Descripción del proyecto

StyleNet necesita automatizar la clasificación de imágenes de productos de ropa subidas por usuarios y vendedores. Este proyecto implementa y compara dos arquitecturas de redes neuronales para clasificar prendas de vestir usando el dataset Fashion-MNIST.

---

## Estructura del repositorio

```
clasificador-ropa-stylenet/
├── clasificador_ropa_modulo8.ipynb   # Notebook principal
├── README.md                          # Esta documentación
├── modelo_cnn_stylenet.keras          # Modelo CNN exportado
├── modelo_ann_stylenet.keras          # Modelo ANN exportado
└── graficos/
    ├── fashion_mnist_clases.png
    ├── distribucion_clases.png
    ├── curvas_entrenamiento.png
    ├── confusion_ann.png
    └── confusion_cnn.png
```

---

## Parte 1 — Justificación de Arquitecturas

### Red Neuronal Densa (ANN) vs Red Neuronal Convolutiva (CNN)

#### ¿Por qué una ANN densa no es óptima para imágenes?

Cuando una imagen de 28×28 píxeles se aplana en un vector de 784 valores, se **pierde toda la información espacial**. Cada píxel se procesa de forma independiente y la red no puede saber que dos píxeles adyacentes forman un borde, una costura o un patrón de textura.

Además, una capa `Dense(256)` conectada a una entrada de 784 neuronas genera 784 × 256 = **200.704 parámetros solo en la primera capa**, lo que aumenta el riesgo de overfitting y el costo computacional.

#### ¿Por qué la CNN es superior para clasificación de imágenes?

| Aspecto | ANN Densa | CNN |
|---|---|---|
| Estructura de entrada | Vector plano (784D) | Matriz 2D (28×28×1) |
| Información espacial | Perdida al aplanar | Preservada por los filtros |
| Patrones detectados | Globales, no locales | Locales: bordes, texturas, formas |
| Parámetros | Muchos (conexiones densas) | Pocos (filtros compartidos) |
| Invariancia a traslación | No | Sí (vía MaxPooling) |
| Accuracy típica en Fashion-MNIST | ~88% | ~92%+ |

La CNN aplica **filtros convolucionales** que recorren la imagen detectando características locales en una jerarquía:

```
Capa Conv 1 (32 filtros)  → bordes, líneas, contornos
Capa Conv 2 (64 filtros)  → texturas, patrones repetitivos
Capa Conv 3 (128 filtros) → formas complejas, partes de prendas
Capas densas finales       → clasificación en 10 categorías
```

El **parámetro de filtro se comparte** a lo largo de toda la imagen, lo que reduce drásticamente la cantidad de parámetros respecto a una capa densa equivalente.

---

## Parte 2 — Explicación Técnica del Modelo

### Dataset: Fashion-MNIST

| Característica | Valor |
|---|---|
| Imágenes totales | 70.000 (60K train / 10K test) |
| Resolución | 28 × 28 píxeles |
| Canales | 1 (escala de grises) |
| Clases | 10 categorías de ropa |
| Preprocesamiento | Normalización a [0, 1] |

**Las 10 categorías:**
`Remera/Top · Pantalón · Suéter · Vestido · Abrigo · Sandalia · Camisa · Zapatilla · Cartera · Botita`

---

### Arquitectura ANN Densa

```
Entrada: imagen 28×28
    ↓
Flatten → vector de 784 valores
    ↓
Dense(256, ReLU)    — 784×256+256 = 200.960 parámetros
    ↓
Dropout(0.3)        — apaga el 30% de neuronas aleatoriamente
    ↓
Dense(128, ReLU)    — 256×128+128 = 32.896 parámetros
    ↓
Dropout(0.3)
    ↓
Dense(10, Softmax)  — 128×10+10 = 1.290 parámetros
```

**Total aproximado: ~235.000 parámetros**

---

### Arquitectura CNN

```
Entrada: imagen 28×28×1
    ↓
Conv2D(32, 3×3, ReLU, padding='same')  → 28×28×32
MaxPooling2D(2×2)                       → 14×14×32
    ↓
Conv2D(64, 3×3, ReLU, padding='same')  → 14×14×64
MaxPooling2D(2×2)                       → 7×7×64
    ↓
Conv2D(128, 3×3, ReLU, padding='same') → 7×7×128
    ↓
Flatten → vector de 7×7×128 = 6.272 valores
    ↓
Dense(128, ReLU)
Dropout(0.4)
Dense(10, Softmax)
```

**Total aproximado: ~900.000 parámetros** (la mayor parte en las capas densas finales)

---

### Funciones de activación

| Función | Fórmula | Dónde se usa | Por qué |
|---|---|---|---|
| **ReLU** | `f(x) = max(0, x)` | Capas ocultas | Simple, evita el problema de gradientes desvanecientes, acelera el entrenamiento |
| **Softmax** | `f(x_i) = e^x_i / Σe^x_j` | Capa de salida | Convierte logits en probabilidades que suman 1, ideal para multiclase |

---

### Función de pérdida

Se usa **Sparse Categorical Crossentropy** porque:
- Las etiquetas son enteros directos (0–9), no vectores one-hot
- Mide la diferencia entre la distribución de probabilidad predicha y la real
- Equivale a minimizar `-log(p_clase_correcta)`: penaliza más cuando el modelo está seguro pero equivocado

---

### Optimizador: Adam

Adam (Adaptive Moment Estimation) combina dos técnicas:
- **Momentum:** acumula gradientes del pasado para navegar mejor los valles del paisaje de pérdida
- **RMSprop:** adapta la tasa de aprendizaje por parámetro según la magnitud de sus gradientes

Parámetros utilizados: `lr=0.001` (valor por defecto), `beta_1=0.9`, `beta_2=0.999`

---

### Regularización: Dropout

El Dropout desactiva aleatoriamente una fracción de neuronas durante cada paso de entrenamiento. Esto:

1. Impide que la red dependa de neuronas específicas (co-adaptación)
2. Fuerza la redundancia: múltiples caminos aprenden la misma representación
3. Equivale a entrenar un ensamble implícito de redes más pequeñas

**Tasas usadas:**
- ANN: Dropout(0.3) — 30% de neuronas apagadas
- CNN: Dropout(0.4) — 40%, ya que las CNNs tienden más al overfitting

**Importante:** el Dropout solo se activa durante el entrenamiento. En inferencia, todas las neuronas están activas y los pesos se escalan proporcionalmente.

---

### Early Stopping

Callback que monitorea `val_loss` y detiene el entrenamiento si no mejora durante `patience=5` epochs consecutivos. Recupera automáticamente los pesos del mejor epoch (`restore_best_weights=True`), evitando entrenar de más.

---

## Parte 3 — Análisis de Métricas

### Resultados esperados

| Métrica | ANN Densa | CNN |
|---|---|---|
| Accuracy en prueba | ~88% | ~92% |
| Loss en prueba | ~0.34 | ~0.24 |
| Epochs hasta convergencia | 15–20 | 15–25 |

> Los valores exactos dependerán de la semilla aleatoria y el entorno de ejecución. Los rangos indicados son representativos del comportamiento de estos modelos con Fashion-MNIST.

### Interpretación de la curva de entrenamiento

Una curva saludable muestra:
- `train_accuracy` y `val_accuracy` crecen juntas sin una brecha grande → el Dropout está funcionando
- `val_loss` disminuye hasta estabilizarse → el Early Stopping detiene en el momento correcto
- Si `train_accuracy >> val_accuracy` → overfitting → aumentar Dropout o reducir capacidad del modelo

### Interpretación de la matriz de confusión

Las categorías con mayor confusión en Fashion-MNIST son:
- **Camisa** vs **Remera/Top**: formas similares, colores variables
- **Suéter** vs **Abrigo**: siluetas parecidas
- **Zapatilla** vs **Sandalia**: estructuras de calzado similares

La CNN reduce estas confusiones porque sus filtros aprenden texturas y detalles estructurales que la ANN densa no puede capturar al procesar cada píxel de forma independiente.

### Métricas por clase

El reporte de clasificación muestra `precision`, `recall` y `f1-score` por categoría:
- **Precision:** de todas las predicciones de clase X, ¿cuántas eran correctas?
- **Recall:** de todos los ejemplos reales de clase X, ¿cuántos detectó el modelo?
- **F1-score:** media armónica entre precision y recall

Las categorías de calzado y accesorios (cartera, botita) suelen tener el F1 más alto por su forma más distintiva. La camisa y la remera suelen tener el F1 más bajo.

---

## Uso del modelo en producción

```python
import tensorflow as tf
import numpy as np

# Cargar modelo entrenado
modelo = tf.keras.models.load_model('modelo_cnn_stylenet.keras')

# Preprocesar imagen externa
img = tf.keras.utils.load_img('prenda.jpg', color_mode='grayscale', target_size=(28, 28))
img_array = tf.keras.utils.img_to_array(img) / 255.0
img_batch = np.expand_dims(img_array, axis=0)  # (1, 28, 28, 1)

# Predicción
probs    = modelo.predict(img_batch)[0]
categoria = np.argmax(probs)
print(f'Categoría predicha: {categoria} | Confianza: {probs[categoria]*100:.1f}%')
```

---

## Posibles mejoras

| Mejora | Descripción | Impacto esperado |
|---|---|---|
| Data Augmentation | Rotaciones, zoom, flips horizontales | +1–2% accuracy, mayor robustez |
| BatchNormalization | Normaliza activaciones entre capas | Entrenamiento más estable y rápido |
| Transfer Learning | Usar pesos de ResNet o EfficientNet preentrenados | +3–5% accuracy |
| Learning Rate Scheduler | Reducir lr cuando val_loss se estanca | Convergencia más fina |

---

## Referencias

- Keras Documentation: https://keras.io/
- TensorFlow API: https://www.tensorflow.org/api_docs
- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
- Machine Learning Mastery: https://machinelearningmastery.com/start-here/
- DotCSV — Redes Neuronales: https://www.youtube.com/@DotCSV

---
