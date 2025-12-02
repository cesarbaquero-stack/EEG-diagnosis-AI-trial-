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

# --- PARÁMETROS DEL MODELO (Deben coincidir con tu entrenamiento) ---
FS = 256            # Frecuencia de muestreo
DURATION = 5.0      # Segundos por ventana
N_CHANNELS = 23     # Canales que espera la IA
POINTS = int(FS * DURATION) # 1280 puntos

# ==========================================
# 2. CARGA DEL MODELO (Optimizado para Nube)
# ==========================================
@st.cache_resource
def load_model():
    # Buscamos el modelo en la misma carpeta del script
    model_path = 'modelo_ligero.h5'
    
    if not os.path.exists(model_path):
        st.error(f"⚠️ Error Crítico: No encuentro el archivo '{model_path}'. Asegúrate de subirlo a GitHub junto con este código.")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None

model = load_model()

# ==========================================
# 3. PRE-PROCESAMIENTO (Espejo del Entrenamiento)
# ==========================================
def preprocess_window(data_window):
    """
    Prepara una ventana de 5 segundos para que la IA la entienda.
    Aplica: Padding de canales, Clip de artefactos y Escalado Fijo.
    """
    # A. Ajuste de Canales (Debe tener 23)
    current_channels = data_window.shape[0]
    
    if current_channels > N_CHANNELS:
        data_window = data_window[:N_CHANNELS, :]
    elif current_channels < N_CHANNELS:
        # Rellenar con ceros si faltan canales
        pad = np.zeros((N_CHANNELS - current_channels, data_window.shape[1]))
        data_window = np.concatenate([data_window, pad], axis=0)
    
    # B. Limpieza de Artefactos (Clip)
    # Elimina picos locos de movimiento mayores a 500 uV
    data_window = np.clip(data_window, -500, 500)
    
    # C. Escalado Fijo (LA CLAVE DEL ÉXITO)
    # Dividimos por 100 uV para mantener la relación física de la señal
    data_window = data_window / 100.0
    
    # D. Formato de Imagen para Keras (Batch, Alto, Ancho, Color)
    # Entrada esperada: (1, 23, 1280, 1)
    data_window = data_window[np.newaxis, ..., np.newaxis]
    
    return data_window

# ==========================================
# 4. LÓGICA DE LA APLICACIÓN
# ==========================================
uploaded_file = st.file_uploader("Sube tu archivo .EDF aquí", type=["edf"])

if uploaded_file is not None and model is not None:
    st.success("Archivo recibido. Iniciando procesamiento...")
    
    # MNE necesita leer desde un archivo físico en disco, no desde memoria RAM.
    # Creamos un archivo temporal para engañarlo.
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
    tfile.write(uploaded_file.read())
    tfile.close() 
    
    try:
        # --- PASO 1: CARGA Y LIMPIEZA ---
        with st.spinner('Filtrando ruido y estandarizando señal...'):
            raw = mne.io.read_raw_edf(tfile.name, preload=True, verbose=False)
            
            # 1. CONVERTIR A MICROVOLTIOS (CRÍTICO)
            # MNE carga en Volts. Multiplicamos por 1 millón.
            raw.apply_function(lambda x: x * 1e6)
            
            # 2. Filtros Estándar
            raw.notch_filter(60, verbose=False)        # Ruido eléctrico
            raw.filter(1.0, 40.0, verbose=False)       # Frecuencias cerebrales
            
            # 3. Resampleo (si el archivo no es de 256Hz)
            if raw.info['sfreq'] != FS:
                raw.resample(FS, verbose=False)
            
            # 4. Referencia Promedio (CAR)
            raw.set_eeg_reference('average', projection=True, verbose=False)
            raw.apply_proj()
        
        # --- PASO 2: SEGMENTACIÓN Y PREDICCIÓN ---
        data = raw.get_data() # (Canales, Tiempo Total)
        total_seconds = data.shape[1] / FS
        num_windows = int(total_seconds / DURATION)
        
        st.write(f"📊 **Análisis Técnico:**")
        st.write(f"- Duración Total: `{total_seconds:.2f} segundos`")
        st.write(f"- Ventanas a analizar: `{num_windows}`")
        
        progress_bar = st.progress(0)
        predictions = []
        times_sec = []
        
        status_text = st.empty()
        status_text.text("🧠 La IA está escaneando el cerebro...")
        
        for i in range(num_windows):
            start_idx = int(i * DURATION * FS)
            end_idx = int((i + 1) * DURATION * FS)
            
            # Extraer ventana
            window = data[:, start_idx:end_idx]
            
            # Verificar tamaño exacto (a veces la última ventana queda corta)
            if window.shape[1] == POINTS:
                # Pre-procesar (Padding + Clip + Scale)
                processed_window = preprocess_window(window)
                
                # Predecir
                pred_prob = model.predict(processed_window, verbose=0)[0][0]
                predictions.append(pred_prob)
                times_sec.append(i * DURATION)
            
            # Actualizar barra de progreso
            if i % 5 == 0:
                progress_bar.progress(min(i / num_windows, 1.0))
        
        progress_bar.progress(1.0)
        status_text.text("✅ Análisis finalizado.")
        
        # --- PASO 3: VISUALIZACIÓN ---
        predictions = np.array(predictions)
        
        # Slider para que el médico ajuste la sensibilidad
        threshold = st.slider("Ajustar Umbral de Sensibilidad", 0.0, 1.0, 0.5, 0.05, 
                              help="Si bajas el umbral, detectará más crisis pero podría haber falsas alarmas.")
        
        is_seizure = predictions > threshold
        
        # Gráfica
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(times_sec, predictions, color='#2c3e50', label='Probabilidad IA', linewidth=1)
        
        # Rellenar zonas de crisis
        ax.fill_between(times_sec, 0, 1, where=is_seizure, color='#e74c3c', alpha=0.3, label='Crisis Detectada')
        
        ax.axhline(threshold, color='#3498db', linestyle='--', label=f'Umbral ({threshold})')
        ax.set_title("Línea de Tiempo de Probabilidad de Crisis", fontsize=14)
        ax.set_ylabel("Probabilidad (0-1)")
        ax.set_xlabel("Tiempo (segundos)")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right")
        
        st.pyplot(fig)
        
        # Resumen numérico
        total_time_seizure = np.sum(is_seizure) * DURATION
        
        col1, col2 = st.columns(2)
        with col1:
            if total_time_seizure > 0:
                st.error(f"⚠️ **ALERTA:** Se detectaron patrones compatibles con epilepsia.")
            else:
                st.success(f"✅ **NORMAL:** No se detectaron eventos de crisis.")
                
        with col2:
            st.metric("Tiempo Total de Crisis", f"{total_time_seizure:.1f} s")

    except Exception as e:
        st.error(f"Ocurrió un error procesando el archivo: {e}")
    finally:
        # Limpieza del archivo temporal
        os.unlink(tfile.name)
