import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

def visualize_results(predictions, probabilities, duration, model_name):
    """Visualize the results of audio analysis"""
    num_segments = len(predictions)
    time_axis = np.linspace(0, duration, num_segments, endpoint=False)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    plt.suptitle(f'Análisis de Música: {model_name}\nDuración: {duration:.2f}s')
    
    # Gráfico de clasificación
    colors = ['red' if p == 0 else 'green' for p in predictions]
    ax1.bar(time_axis, [1]*num_segments, width=SEGMENT_DURATION, 
            color=colors, alpha=0.6, align='edge')
    ax1.set_ylabel('Clasificación')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['No música', 'Música'])
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Gráfico de probabilidades
    ax2.bar(time_axis, probabilities, width=SEGMENT_DURATION, 
            color='blue', alpha=0.6, align='edge')
    ax2.set_xlabel('Tiempo (segundos)')
    ax2.set_ylabel('Probabilidad')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Leyenda y estadísticas
    legend_elements = [
        Patch(facecolor='green', label='Música', alpha=0.6),
        Patch(facecolor='red', label='No música', alpha=0.6)
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    music_segments = sum(predictions)
    music_percentage = music_segments / num_segments * 100
    avg_prob = np.mean(probabilities) * 100
    
    plt.tight_layout()
    plt.show()
    
    print("\n📊 Resumen:")
    print(f"- Segmentos de música: {music_segments}/{num_segments} ({music_percentage:.1f}%)")
    print(f"- Probabilidad promedio: {avg_prob:.1f}%")
    print("🎼 CONCLUSIÓN: El audio es PREDOMINANTEMENTE MÚSICA" if music_percentage > 50 
          else "🔇 CONCLUSIÓN: El audio NO ES PREDOMINANTEMENTE MÚSICA")