import streamlit as st
import numpy as np
import plotly.graph_objects as go

# 1. Configuración de la interfaz (Estética Welch Labs)
st.set_page_config(page_title="Perceptrón Manual", layout="wide")
st.title("📟 Laboratorio de Perceptrón: Clasificación Manual")
st.write("Ajusta los pesos (perillas) para separar los patrones positivos de los negativos.")

# 2. Sidebar: Configuración de los datos (Etiquetas deseadas)
st.sidebar.header("🎯 Configuración de Patrones")
patterns = np.array([[0,0], [0,1], [1,0], [1,1]])
targets = []

for i, p in enumerate(patterns):
    label = st.sidebar.radio(f"Patrón {p}:", ["Negativo (-1)", "Positivo (+1)"], index=0, key=f"target_{i}")
    targets.append(1 if "Positivo" in label else -1)

# 3. Controles: Las 3 Perillas (Sliders)
st.header("🎛️ Perillas de Ajuste")
col1, col2, col3 = st.columns(3)
with col1:
    w1 = st.slider("Peso W1", -2.0, 2.0, 0.5, 0.1)
with col2:
    w2 = st.slider("Peso W2", -2.0, 2.0, 0.5, 0.1)
with col3:
    b = st.slider("Bias (Sesgo)", -2.0, 2.0, -0.5, 0.1)

# 4. Lógica del Perceptrón
def predict(x1, x2):
    suma_ponderada = (x1 * w1) + (x2 * w2) + b
    activacion = 1 if suma_ponderada >= 0 else -1
    return suma_ponderada, activacion

predicciones = [predict(p[0], p[1]) for p in patterns]
aciertos = sum(1 for i in range(4) if predicciones[i][1] == targets[i])

# 5. Visualización con Plotly (Frontera de decisión)
fig = go.Figure()

# Dibujar la frontera de decisión: w1*x + w2*y + b = 0  => y = (-w1*x - b) / w2
x_range = np.linspace(-0.5, 1.5, 10)
if w2 != 0:
    y_range = (-w1 * x_range - b) / w2
    fig.add_trace(go.Scatter(x=x_range, y=y_range, name="Frontera de Decisión", line=dict(color='blue', width=3, dash='dash')))

# Dibujar los puntos
for i, p in enumerate(patterns):
    es_correcto = predicciones[i][1] == targets[i]
    color = "green" if es_correcto else "red"
    simbolo = "circle" if targets[i] == 1 else "x"
    
    fig.add_trace(go.Scatter(
        x=[p[0]], y=[p[1]],
        marker=dict(size=20, color=color, symbol=simbolo, line=dict(width=2, color="black")),
        name=f"Punto {p} ({'Correcto' if es_correcto else 'Error'})"
    ))

fig.update_layout(xaxis_range=[-0.5, 1.5], yaxis_range=[-0.5, 1.5], template="simple_white")
st.plotly_chart(fig, use_container_width=True)

# 6. Contador de Aciertos
st.subheader(f"📊 Puntuación: {aciertos} / 4 patrones clasificados")
if aciertos == 4:
    st.success("¡Logrado! Has encontrado una solución lineal.")
