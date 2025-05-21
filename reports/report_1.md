# Audio Classification: Music vs Non-Music

## General Objective of the Project

The goal of this project is to develop a system capable of classifying audio segments into two categories: **music** and **non-music**. This classification serves as a crucial preliminary step for more complex tasks such as **music genre classification**.

---

## Stage 1: Classical Supervised Classification Models

### Stage Objective

In the first stage of the project, classical supervised classification models were used to approach the problem. The objective was to assess the feasibility of traditional techniques before incorporating more advanced architectures.

### Evaluated Models

Different supervised models were tested, and their results were recorded using **DagsHub**:

| Model               | Accuracy | F1-score | Precision | Recall |
|---------------------|----------|----------|-----------|--------|
| Logistic Regression | 0.8623   | 0.8268   | 0.8003    | 0.8550 |
| LSVM                | 0.8515   | 0.8123   | 0.7904    | 0.8354 |
| Random Forest       | 0.8554   | 0.8047   | 0.8368    | 0.7749 |
| XGBoost             | 0.8601   | 0.8201   | 0.8110    | 0.8295 |

### Selected Model

Although all models showed competitive performance, **Logistic Regression** achieved the **best F1-score (0.8268)** and the **highest Recall (0.8550)**. This indicates that the model was particularly effective at detecting musical segments without losing too much precision. Additionally, its learning curve was stable and showed no signs of overfitting, reinforcing its suitability for a first functional version.

We chose to **retain the Logistic Regression model** due to its simplicity, good generalization capabilities, and ease of interpretation.

### Comparison with Literature

Similar models used in music detection tasks often report **F1-scores between 0.80 and 0.90**, depending on the dataset and preprocessing applied. For example:

- In a music genre classification project using SVM and CNNs, F1-scores of up to 0.85 were reported using MFCCs as input features. [Source: GitHub - leemgs/music-genre-class-f1-score](https://github.com/leemgs/music-genre-class-f1-score)

- In a study using the GTZAN dataset and classifiers like Random Forest and Logistic Regression, F1-scores of 0.912 and 0.798 were obtained, respectively. [Source: Large-Scale Music Genre Analysis and Classification Using Machine Learning with Apache Spark](https://www.mdpi.com/2079-9292/11/16/2567)

In this context, our result of **0.8268** with Logistic Regression is **competitive and within the expected range** for an initial stage.

---

## Stage 2: Custom Neural Model

### Introduction

As a test and to explore the behavior of more complex architectures, a neural model was built using **multi-head attention** mechanisms and **convolutional networks**, trained on sequences of audio embeddings. Although **Logistic Regression was the selected model**, this neural model was designed as an additional experiment. Transfer learning has not yet been implemented, but the model serves as a baseline for exploring future improvements.

### Model Architecture

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

This model achieved satisfactory metrics, justifying its use as a basis for future improvements.

### Is Transfer Learning Necessary?

Although the current model performs well, there are scenarios where transfer learning could provide additional benefits:

- Capture more abstract features learned from large labeled audio corpora.
- Speed up training on limited datasets.
- Improve generalization to new domains or musical styles.

Therefore, there is justification to explore transfer learning models as the next step in the project.

### Recommended Transfer Learning Models

The following options are considered, all trained on large datasets such as AudioSet:

1. VGGish
- Source: Google Research
- Pre-training: AudioSet
- Use: As a feature extractor
- Advantages: Robust architecture; easy integration with TensorFlow/Keras

2. YAMNet
- Based on: MobileNetV1
- Pre-training: AudioSet
- Advantages: Lightweight, ideal for mobile or real-time applications

3. OpenL3
- Based on: L3-Net architecture
- Advantages: Produces semantic embeddings useful for general audio tasks, including music classification

### Proposed Procedure

1. Use the pre-trained model as a feature extractor (e.g., generate embeddings with VGGish or YAMNet).

2. Train a lightweight neural network (such as a dense layer or RNN) on these features.

3. Compare results with models trained from scratch to evaluate whether transfer learning provides substantial improvements.

## Production and Future Improvements

To take the system to production, the following are proposed:
- Implementation of a robust inference pipeline with proper preprocessing.
- Incorporation of additional metrics such as inference time and memory usage.
- Possible model calibration if the system needs to prioritize precision over recall (or vice versa).

## Next Steps

- Evaluate models with transfer learning.
- Develop a music genre classification system.
- Possibly incorporate architectures such as deep CNNs or audio-specific transformers.

## Conclusion
The project is progressing through well-structured phases. After validating classical models and building a functional neural network, the next steps point toward the use of transfer learning to enhance the system's precision, robustness, and scalability.






