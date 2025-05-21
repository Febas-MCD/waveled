# Preliminary Comparative Evaluation of Models for Audio Classification

## 📊 Test Set Results

### Logistic Regression Model with TensorFlow
| Metric            | Value (Threshold=0.5) | Value (Optimal Threshold=0.4848) | Benchmark |
|-------------------|--------------------|-------------------------------------|-----------|
| Accuracy          | 0.8517             | 0.8500 (-0.2%)                      | >0.80     | 
| Loss              | 0.4724             | -                                   | <0.50     | 
| PR-AUC            | 0.8677             | -                                   | >0.85     | 
| Precision         | 0.7726             | 0.7700 (-0.3%)                      | >0.70     | 
| Recall            | 0.8509             | 0.8600 (+1.1%)                      | >0.80     | 
| F1-Score          | 0.8095             | 0.8120 (+0.3%)                      | >0.75     | 

**✅ Advantages:**
- Computational efficiency  
- Low resource consumption  
- Easy maintenance and interpretability

### CNN+Attention Model
| Metric            | Value (Threshold=0.5) | Value (Optimal Threshold) | Change         |
|-------------------|--------------------|------------------------------|----------------|
| Accuracy          | 0.8531             | 0.8500 (-0.4%)               | +0.2% vs RL    | 
| Loss              | 0.4731             | -                            | +0.0007 vs RL  | 
| PR-AUC            | 0.8666             | -                            | -0.1% vs RL    | 
| Precision         | 0.7772             | 0.7600 (-2.2%)               | +0.5% vs RL    | 
| Recall            | 0.8469             | 0.8700 (+2.3%)               | -0.4% vs RL    | 
| F1-Score          | 0.8100             | 0.8110 (+0.1%)               | +0.2% vs RL    | 

**✅ Advantages:**
- Improved precision (+0.5% in key metrics)  
- Scalable architecture for future enhancements  
- Attention mechanism for complex pattern recognition


**⚠️ Considerations:**
- Requires GPU infrastructure  
- Higher operational cost  
- Increased maintenance complexity

