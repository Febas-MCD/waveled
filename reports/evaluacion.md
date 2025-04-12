# Evaluación Comparativa de Modelos para Clasificación de Audio

## 📊 Resultados en Conjunto de Test

### Modelo de Regresión Logística con TensorFlow
| Métrica           | Valor (Umbral=0.5) | Valor (Umbral Óptimo=0.4848) | Benchmark |
|-------------------|--------------------|------------------------------|-----------|
| Accuracy          | 0.8517             | 0.8500 (-0.2%)               | >0.80    | 
| Loss              | 0.4724             | -                            | <0.50    | 
| PR-AUC            | 0.8677             | -                            | >0.85    | 
| Precision         | 0.7726             | 0.7700 (-0.3%)               | >0.70    | 
| Recall            | 0.8509             | 0.8600 (+1.1%)               | >0.80    | 
| F1-Score          | 0.8095             | 0.8120 (+0.3%)               | >0.75    | 

### Modelo CNN+Attention
| Métrica           | Valor (Umbral=0.5) | Valor (Umbral Óptimo) | Cambio |
|-------------------|--------------------|-----------------------|--------|
| Accuracy          | 0.8531             | 0.8500 (-0.4%)        | +0.2% vs RL | 
| Loss              | 0.4731             | -                     | +0.0007 vs RL | 
| PR-AUC            | 0.8666             | -                     | -0.1% vs RL | 
| Precision         | 0.7772             | 0.7600 (-2.2%)        | +0.5% vs RL | 
| Recall            | 0.8469             | 0.8700 (+2.3%)        | -0.4% vs RL | 
| F1-Score          | 0.8100             | 0.8110 (+0.1%)        | +0.2% vs RL | 




## 🏭 Viabilidad para Producción

### Modelo de Regresión Logística
**✅ Ventajas:**
- Eficiencia computacional 
- Bajo consumo de recursos 
- Fácil mantenimiento e interpretabilidad

### Modelo CNN+Attention
**✅ Ventajas:**
- Mejora en precisión (+0.5% en métricas clave)
- Arquitectura escalable para mejoras futuras
- Mecanismo de atención para patrones complejos

**⚠️ Consideraciones:**
- Requiere infraestructura GPU
- Mayor costo operacional
- Complejidad de mantenimiento

## 📌 Recomendación Final

**Se recomienda implementar el modelo de Regresión Logística porque:**
1. Las diferencias en rendimiento son mínimas (+0.2% F1, -0.1% PR-AUC)
2. Supera consistentemente los KPIs mínimos requeridos:
   - Recall: 85.09% (vs 85% requerido)
   - F1-Score: 0.8095 (vs 0.80 objetivo)
3. Ofrece ventajas operacionales decisivas

**Considerar el modelo CNN+Attention solo si:**
- Aparecen requisitos de >90% recall
- Se dispone de presupuesto para infraestructura GPU
- Se necesita detectar subpatrones complejos en audio
