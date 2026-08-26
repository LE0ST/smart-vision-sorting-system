"""Configuracion y parametros del Sistema de Clasificacion y Vision Artificial."""

import os
import numpy as np

# Rutas del Sistema
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPERATORS_DIR = os.path.join(BASE_DIR, "operadores")

# ─────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────
FRAME_W       = 640
FRAME_H       = 480
MAX_FPS       = 15
SERIAL_PORT   = "COM9"
SERIAL_BAUD   = 9600
ACK_TIMEOUT   = 2.0
OK_HOLD_TIME  = 0.8
CANCEL_HOLD   = 0.5
PALM_HOLD     = 1.0
FACE_INTERVAL = 2.0
EAR_THRESH    = 0.22
LBPH_THRESH   = 115.0   # calibrado para Max Gil

CAP_HSV    = [(np.array([90,60,30]), np.array([130,255,180]))]
CAP_MIN_AREA = 3000

COLOR_RANGES = {
    "AZUL":     [(np.array([100, 80, 40]), np.array([130,255,255]))],
    "AMARILLO": [(np.array([20,  80, 80]), np.array([35, 255,255]))],
}
COLOR_TO_DEST = {"AZUL":"A","AMARILLO":"B"}   # cualquier otro color -> "C"
DEST_BGR      = {"A":(255,100,0),"B":(0,220,220),"C":(0,200,0)}
GENERIC_MIN_AREA = 2500   # área mínima para detectar "otro objeto" -> C

# Pantallas
class Screen:
    MENU     = "MENU"
    TRABAJO  = "AREA_TRABAJO"
    BRILLO   = "BRILLO"
    SEÑAS    = "SEÑAS"
    CORPORAL = "CORPORAL"
    FILTROS  = "FILTROS"

# Estados clasificación
class St:
    BLOQUEADO="BLOQUEADO"; DETECCION="DETECCIÓN"
    ESPERA="ESPERA CONF."; CLASIF="CLASIFICANDO"; CANCEL="CANCELADO"


