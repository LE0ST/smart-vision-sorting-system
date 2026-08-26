# Sistema Inteligente de Clasificación Industrial mediante PDI e Interfaz Gestual sin Contacto

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Landmarks%20Inference-00A67E)](https://mediapipe.dev)
[![Arduino](https://img.shields.io/badge/Arduino-ATmega328P-00979D?logo=arduino&logoColor=white)](https://arduino.cc)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sistema automatizado de clasificación mecatrónica distribuida basado en **Procesamiento Digital de Imágenes (PDI)**, inferencia biométrica en tiempo real y co-diseño Hardware-Software. El sistema opera mediante una estación central en Python (nodo maestro) y un controlador embebido Arduino UNO (nodo esclavo) sincronizados por protocolo UART no bloqueante.

---

## 📁 Estructura del Repositorio

```text
smart-vision-sorting-system/
├── arduino/
│   └── prototipo_arduino_v2/
│       └── prototipo_arduino_v2.ino     # Firmware Arduino con FSM no bloqueante
├── src/
│   ├── __init__.py                      # Inicializador del paquete Python
│   ├── config.py                        # Calibración, puertos serial y constantes
│   ├── main.py                          # Lógica principal, visión y GUI CustomTkinter
│   └── operadores/                      # Dataset de rostros para reconocimiento facial
│       └── .gitkeep
├── docs/                                # Documentación técnica y diagramas
│   └── .gitkeep
├── .gitignore                           # Exclusiones de control de versiones
├── requirements.txt                     # Dependencias del entorno Python
└── README.md                            # Documentación técnica del proyecto
```

---

## 📐 Arquitectura General del Sistema

El flujo operacional se fundamenta en una **máquina de estados finitos (FSM)** gobernada por eventos visuales, eliminando la necesidad de pulsadores físicos y minimizando perturbaciones lumínicas en planta.

```text
[ Cámara Laptop (Op.) ] ──> [ MediaPipe (Hands/Pose/Face) ] ──> [ HMI CustomTkinter ]
                                                                       │
                                                            (Gesto OK / Auth LBPH)
                                                                       ▼
[ Cámara Rampa (Prod.)] ──> [ Segmentación HSV + Morfología ] ──> [ Decisión de Ruta ]
                                                                       │
                                                               (UART 9600 Baud)
                                                                       ▼
[ Arduino UNO (FSM)   ] ──> [ Servos SG90 (Rampa + Compuerta)] ──> [ Canalización A/B/C ]
```

---

## 🛠️ Módulos Técnicos y Algorítmicos

### 1. Segmentación Cromática y Morfología Espacial (Visión de Inspección)
* **Espacio de Color HSV:** Desacopla luminancia ($V$) de la información cromática ($H$, $S$) para garantizar invarianza ante sombras y gradientes de luz de laboratorio.
* **Filtrado Morfológico No Lineal:** Aplicación secuencial de Apertura (`cv2.MORPH_OPEN`) y Clausura (`cv2.MORPH_CLOSE`) con elemento estructurante de $5\times5$ para suprimir ruido espurio y sellar discontinuidades.
* **Análisis de Conectividad:** Extracción de contornos (`cv2.findContours`) y discriminación de masa activa por umbralización de área (`cv2.countNonZero`).

### 2. HMI Manos Libres y Control Gestual (MediaPipe Hands)
* **Filtrado Temporal Anti-Parpadeo:** Algoritmo de votación por mayoría simple basado en colas circulares (`collections.deque(maxlen=5)`) para estabilizar la detección y prevenir reinicios de temporizadores por ruido.
* **Navegación Dinámica:** Evaluación de flexión articular normalizada ($\gamma = 0.02$) y control de ganancia de brillo mediante distancia euclidiana entre landmarks:
  $$d = \sqrt{(x_8 - x_4)^2 + (y_8 - y_4)^2}$$

### 3. Seguridad Operacional y Biometría
* **Reconocimiento Facial (LBPH):** Extracción de regiones de interés facial con Haar Cascade y validación mediante Local Binary Patterns Histograms (`cv2.face.LBPHFaceRecognizer`) calibrado a un umbral estricto ($\text{distancia} < 115.0$).
* **Monitoreo de Fatiga (EAR):** Cálculo del *Eye Aspect Ratio* sobre 6 puntos interpalpebrales de MediaPipe FaceMesh para advertencia inmediata ante somnolencia ($\text{EAR} < 0.22$):
  $$\text{EAR} = \frac{\|P_2 - P_6\| + \|P_3 - P_5\|}{2 \|P_1 - P_4\|}$$
* **Ergonomía Postural:** Inferencia de vectores de hombros y cuello mediante MediaPipe Pose para alertar desviaciones articulares prolongadas ($> 30\text{ px}$).

### 4. Control Embebido y Actuación (Arduino UNO)
* **FSM Asíncrona (No Bloqueante):** Control temporal por `millis()` de actuadores servomotores SG90 para direccionamiento angular ($60^\circ, 90^\circ, 120^\circ$) y apertura de compuerta gravitacional.
* **Protocolo Serial:** Comunicación bidireccional a 9600 baud con confirmación por paquetes `ACK` y parada de emergencia inmediata (`'X'`).

---

## 🚀 Instalación y Despliegue

### Requisitos Previos
* Python 3.10 o superior
* Arduino IDE 2.0+
* 2 Cámaras web (integrada + auxiliar / IP Webcam)

### Configuración del Entorno
```bash
# Clonar repositorio
git clone https://github.com/LE0ST/smart-vision-sorting-system.git
cd smart-vision-sorting-system

# Crear entorno virtual (opcional pero recomendado)
python -m venv .venv
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecución
1. Cargar `arduino/prototipo_arduino_v2/prototipo_arduino_v2.ino` en la placa Arduino UNO.
2. Registrar fotos de los operarios en la carpeta `src/operadores/` con el formato `nombre_apellido_1.jpg`.
3. Iniciar la interfaz central:
```bash
python src/main.py
```

---

## 👥 Autores
* **Leonardo Yactayo Tolentino** — Estudiante de Ingeniería Electrónica, UNMSM.
* **Max Gil Machaca** — Estudiante de Ingeniería Electrónica, UNMSM.
