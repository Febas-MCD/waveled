# Clasificación de Audio: Música vs No Música

## Objetivo General del Proyecto

El proyecto tiene como objetivo desarrollar un sistema capaz de clasificar segmentos de audio en dos categorías: **música** y **no música**. Esta clasificación es un paso preliminar esencial para tareas más complejas, como la **clasificación por género musical**.

---

## Etapa 1: Modelos Clásicos de Clasificación Supervisada

### Objetivo de la Etapa

En la primera etapa del proyecto, se utilizaron modelos clásicos de clasificación supervisada para abordar el problema. El objetivo fue evaluar la viabilidad de técnicas tradicionales antes de incorporar arquitecturas más avanzadas.

### Modelos Evaluados

Se probaron distintos modelos supervisados, cuyos resultados fueron registrados usando **DagsHub**:

| Modelo              | Accuracy | F1-score | Precision | Recall |
|---------------------|----------|----------|-----------|--------|
| Regresión Logística | 0.8623   | 0.8268   | 0.8003    | 0.8550 |
| LSVM                | 0.8515   | 0.8123   | 0.7904    | 0.8354 |
| Random Forest       | 0.8554   | 0.8047   | 0.8368    | 0.7749 |
| XGBoost             | 0.8601   | 0.8201   | 0.8110    | 0.8295 |

### Modelo Seleccionado

Aunque todos los modelos mostraron un rendimiento competitivo, **Regresión Logística** fue el que obtuvo el **mejor F1-score (0.8268)** y el **mayor Recall (0.8550)**. Esto indica que el modelo fue particularmente eficaz al detectar segmentos musicales, sin perder demasiada precisión. Además, se observó que su curva de aprendizaje es estable y sin señales de sobreajuste, lo que refuerza su idoneidad para una primera versión funcional.

Elegimos **retener el modelo de Regresión Logística**, por su simplicidad, buena capacidad generalizadora y facilidad de interpretación.

### Comparación con la Literatura

Modelos similares en tareas de detección de música suelen reportar **F1-scores entre 0.80 y 0.90**, dependiendo del dataset y del preprocesamiento aplicado. Por ejemplo:

- En el proyecto de clasificación de géneros musicales utilizando SVM y CNNs, se reportaron F1-scores de hasta 0.85 empleando MFCCs como características de entrada. [Fuente: GitHub - leemgs/music-genre-class-f1-score](https://github.com/leemgs/music-genre-class-f1-score)

- En un estudio que utilizó el conjunto de datos GTZAN y clasificadores como Random Forest y Regresión Logística, se obtuvieron F1-scores de 0.912 y 0.798 respectivamente. [Fuente: Large-Scale Music Genre Analysis and Classification Using Machine Learning with Apache Spark](https://www.mdpi.com/2079-9292/11/16/2567)

En este contexto, nuestro resultado de **0.8268** con Regresión Logística es **competitivo y dentro del rango esperado** para una etapa inicial.

---

## Etapa 2: Modelo Neuronal Personalizado

### Introducción

A modo de prueba y para verificar el comportamiento de arquitecturas más complejas, se construyó un modelo neuronal con mecanismos de **atención multi-cabeza** y **redes convolucionales**, entrenado sobre secuencias de embeddings de audio. Si bien la **regresión logística fue el modelo seleccionado**, este modelo neuronal se diseñó como experimento adicional. Aún no se ha implementado transferencia de aprendizaje, pero el modelo sirve como línea base para explorar mejoras futuras.

### Arquitectura del Modelo

```python
def build_model(input_shape=(10, 128)):
    inputs = layers.Input(shape=input_shape)
    x = layers.MultiHeadAttention(num_heads=2, key_dim=64, dropout=0.2)(inputs, inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model
```

Este modelo logró métricas satisfactorias, justificando su uso como base para futuras mejoras.

### ¿Es necesaria la transferencia de aprendizaje?

Aunque el modelo actual logra buenos resultados, existen escenarios donde la transferencia de aprendizaje puede aportar beneficios adicionales:

- Capturar características más abstractas aprendidas de grandes corpus de audio etiquetado.
- Acelerar el entrenamiento en conjuntos de datos limitados.
- Mejorar la generalización hacia nuevos dominios o estilos musicales.

Por ello, sí existe justificación para explorar modelos con transferencia de aprendizaje como siguiente paso del proyecto.

### Modelos con Transferencia de Aprendizaje Recomendados

Se consideran las siguientes opciones, todas entrenadas sobre conjuntos de datos amplios como AudioSet:

1. VGGish
- Fuente: Google Research
- Pre-entrenamiento: AudioSet
- Uso: Como extractor de características.
- Ventajas: Arquitectura robusta; integración sencilla con TensorFlow/Keras.

2. YAMNet
- Basado en: MobileNetV1
- Pre-entrenamiento: AudioSet
- Ventajas: Ligero, ideal para dispositivos móviles o aplicaciones en tiempo real.

3. OpenL3
- Basado en: arquitectura L3-Net.
- Ventajas: Produce embeddings semánticos útiles para tareas generales de audio, incluyendo clasificación de música.

### Procedimiento Propuesto

1. Usar el modelo preentrenado como extractor de características (por ejemplo, generar embeddings con VGGish o YAMNet).

2. Entrenar una red neuronal ligera (como una capa densa o RNN) sobre estas características.

3. Comparar resultados con los modelos entrenados desde cero para evaluar si la transferencia de aprendizaje aporta mejoras sustanciales.

## Producción y Mejoras Futuras

Para llevar el sistema a producción, se plantean:
- Implementación de un pipeline de inferencia robusto con preprocesamiento adecuado.
- Incorporación de métricas adicionales como el tiempo de inferencia y uso de memoria.
- Posible calibración del modelo si el sistema requiere priorizar la precisión sobre el recall (o viceversa).

## Próximos Pasos

- Evaluar modelos con transferencia de aprendizaje.
- Desarrollar sistema de clasificación por género musical.
- Posiblemente incorporar arquitecturas como CNNs profundas o transformers específicos para audio.

## Conclusión
El proyecto avanza en fases bien estructuradas. Tras validar modelos clásicos y construir una red neuronal funcional, los siguientes pasos apuntan hacia el uso de transferencia de aprendizaje para mejorar la precisión, robustez y escalabilidad del sistema.






