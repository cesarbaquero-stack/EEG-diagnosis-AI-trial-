import streamlit as st
import numpy as np
import tensorflow as tf
import mne
import os
import matplotlib.pyplot as plt
import tempfile

# ==========================================
# 1. CONFIGURACIÓN E INICIALIZACIÓN
# ==========================================
st.set_page_config(
    page_title="NeuroAI: Detector de Epilepsia", 
    page_icon="🧠", 
    layout="wide"
)

st.title("🧠 NeuroAI: Detección de Crisis en EEG")
st.markdown("""
Esta aplicación utiliza una **Red Neuronal Convolucional (EEGNet)** para analizar archivos de electroencefalografía (EEG) en formato **.EDF**.
El sistema escanea la señal buscando patrones convulsivos y genera una línea de tiempo de probabilidades.
""")

# --- PARÁMETROS DEL MODELO (Actualizados) ---
FS = 256            
DURATION = 5.0      
N_CHANNELS = 28     # Tu modelo pide 28
POINTS = 1281       # Tu modelo pide 1281

# ==========================================
# 2. CARGA DEL MODELO (Optimizado)
# ==========================================
@st.cache_resource
def load_model():
    model_path = 'modelo_ligero.h5'  
    
    if not os.path.exists(model_path):
        st.error(f"⚠️ Error Crítico: No encuentro el archivo '{model_path}'.")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None

model = load_model()

# ==========================================
# 3. PRE-PROCESAMIENTO
# ==========================================
def preprocess_window(data_window):
    """
    Prepara una ventana: Padding a (28, 1281), Clip y Escalado Fijo (/100).
    """
    # A. Ajuste de Canales (Padding hasta 28)
    current_channels = data_window.shape[0]
    if current_channels < N_CHANNELS:
        pad = np.zeros((N_CHANNELS - current_channels, data_window.shape[1]))
        data_window = np.concatenate([data_window, pad], axis=0)
    elif current_channels > N_CHANNELS:
        data_window = data_window[:N_CHANNELS, :]
    
    # B. Ajuste de Tiempo (Padding hasta 1281)
    current_points = data_window.shape[1]
    if current_points < POINTS:
        pad_time = np.zeros((data_window.shape[0], POINTS - current_points))
        data_window = np.concatenate([data_window, pad_time], axis=1)
    elif current_points > POINTS:
        data_window = data_window[:, :POINTS]

    # C. Limpieza y Escalado (Igual al entrenamiento)
    data_window = np.clip(data_window, -500, 500)
    data_window = data_window / 100.0
    
    # D. Formato de Imagen 4D
    data_window = data_window[np.newaxis, ..., np.newaxis]
    
    return data_window

# ==========================================
# 4. LÓGICA DE LA APLICACIÓN
# ==========================================
uploaded_file = st.file_uploader("Sube tu archivo .EDF aquí", type=["edf"])

if uploaded_file is not None and model is not None:
    st.success("Archivo recibido. Iniciando procesamiento...")
    
    # Archivo temporal para MNE
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
    tfile.write(uploaded_file.read())
    tfile.close() 
    
    try:
        # --- PASO 1: CARGA Y LIMPIEZA ---
        with st.spinner('Filtrando ruido y estandarizando señal...'):
            raw = mne.io.read_raw_edf(tfile.name, preload=True, verbose=False)
            
            # 1. CONVERTIR A MICROVOLTIOS (CRÍTICO)
            raw.apply_function(lambda x: x * 1e6)
            
            # 2. Filtros Estándar
            raw.notch_filter(60, verbose=False)
            raw.filter(1.0, 40.0, verbose=False)
            
            # 3. Resampleo
            if raw.info['sfreq'] != FS:
                raw.resample(FS, verbose=False)
            
            # 4. Referencia Promedio
            raw.set_eeg_reference('average', projection=True, verbose=False)
            raw.apply_proj()
        
        # --- PASO 2: SEGMENTACIÓN Y PREDICCIÓN ---
        data = raw.get_data() 
        total_seconds = data.shape[1] / FS
        num_windows = int(total_seconds / DURATION)
        
        st.write(f"📊 **Análisis Técnico:** Duración: `{total_seconds:.2f}s` | Ventanas: `{num_windows}`")
        
        progress_bar = st.progress(0)
        predictions = []
        times_sec = []
        
        status_text = st.empty()
        status_text.text("🧠 Escaneando cerebro...")
        
        for i in range(num_windows):
            start_idx = int(i * DURATION * FS)
            end_idx = int((i + 1) * DURATION * FS)
            
            window = data[:, start_idx:end_idx]
            
            # Verificar que la ventana tenga datos suficientes para procesar
            if window.shape[1] > 0:
                processed_window = preprocess_window(window)
                
                # --- DEBUG VISUAL (Solo primera ventana) ---
                if i == 0:
                    st.write("### 🕵️‍♂️ Diagnóstico de Rayos X")
                    debug_signal = processed_window[0, :, :, 0]
                    st.write(f"**Rango de Valores (IA):** Min {debug_signal.min():.2f} | Max {debug_signal.max():.2f}")
                    
                    fig_debug, ax_debug = plt.subplots(figsize=(10, 2))
                    ax_debug.plot(debug_signal[0, :])
                    ax_debug.set_title("Señal vista por la IA (Canal 1)")
                    st.pyplot(fig_debug)
