"""Sistema de Clasificacion con Gestos y Vision Artificial - Modulo Principal."""

import os
import sys

# Asegurar que el modulo config sea accesible al ejecutar directamente
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import mediapipe as mp
import numpy as np
import serial
import serial.tools.list_ports
import threading
import time
import queue
import glob
import math
from datetime import datetime
from collections import deque
import customtkinter as ctk
from PIL import Image

from config import (
    FRAME_W, FRAME_H, MAX_FPS, SERIAL_PORT, SERIAL_BAUD, ACK_TIMEOUT,
    OK_HOLD_TIME, CANCEL_HOLD, PALM_HOLD, FACE_INTERVAL, EAR_THRESH, LBPH_THRESH,
    OPERATORS_DIR, CAP_HSV, CAP_MIN_AREA, COLOR_RANGES, COLOR_TO_DEST, DEST_BGR,
    GENERIC_MIN_AREA, Screen, St
)

class FaceRecognizer:
    def __init__(self, folder=OPERATORS_DIR):
        self._det = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._rec      = cv2.face.LBPHFaceRecognizer_create()
        self._id_map   = {}   # label_id → nombre limpio
        self._trained  = False
        self._train(folder)

    def _clean_name(self, filename):
        """juan_perez_1.jpg → Juan Perez"""
        base = os.path.splitext(filename)[0]
        # quitar sufijo numérico final (_1, _2, etc.)
        parts = base.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            base = parts[0]
        return base.replace("_", " ").title()

    def _train(self, folder):
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
            print(f"[Face] Carpeta '{folder}' creada.")
            return
        name_to_id = {}
        faces, labels = [], []
        next_id = 0
        for ext in ("*.jpg","*.jpeg","*.png"):
            for path in sorted(glob.glob(os.path.join(folder, ext))):
                name = self._clean_name(os.path.basename(path))
                if name not in name_to_id:
                    name_to_id[name] = next_id
                    self._id_map[next_id] = name
                    next_id += 1
                lid  = name_to_id[name]
                img  = cv2.imread(path)
                if img is None: continue
                gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                gray = cv2.resize(gray,(200,200))
                faces.append(gray); labels.append(lid)
                print(f"[Face] {name} ← {os.path.basename(path)}")
        if faces:
            self._rec.train(faces, np.array(labels))
            self._trained = True
            print(f"[Face] LBPH entrenado: {len(self._id_map)} persona(s), {len(faces)} foto(s).")
        else:
            print("[Face] Sin fotos — reconocimiento desactivado.")

    def recognize(self, frame_bgr):
        if not self._trained: return "Desconocido", 999.0, None
        gray  = cv2.equalizeHist(cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2GRAY))
        rects = self._det.detectMultiScale(gray,1.1,5,minSize=(60,60))
        best_name,best_conf,best_box = "Desconocido",999.0,None
        for (x,y,w,h) in rects:
            roi = cv2.resize(gray[y:y+h,x:x+w],(200,200))
            lid,conf = self._rec.predict(roi)
            if conf < LBPH_THRESH and conf < best_conf:
                best_conf=conf; best_name=self._id_map.get(lid,"Desconocido")
                best_box=(x,y,w,h)
        return best_name,best_conf,best_box


# ═══════════════════════════════════════════════════════════════════════
# GestureEngine — todos los gestos
# ═══════════════════════════════════════════════════════════════════════
class GestureEngine:
    """
    Retorna dict con:
      fingers     : int 0-5 (dedos extendidos)
      palm        : bool (palma extendida, todos los dedos + pulgar)
      ok_conf     : bool (pulgar arriba confirmado 0.8s)
      cancel_conf : bool (puño confirmado 0.5s)
      peace_conf  : bool (✌ confirmado 0.5s) — para activar cara en trabajo
      ok_instant  : bool
      cancel_instant: bool
      pinch_dist  : float
      asl_letter  : str|None
      palm_progress: float 0-1
      func_progress: float 0-1
    """
    def __init__(self):
        _h = mp.solutions.hands
        self._hands = _h.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.6)
        self._draw = mp.solutions.drawing_utils
        self._CONN = mp.solutions.hands.HAND_CONNECTIONS

        self._last_func   = None; self._t0_func   = None
        self._t0_palm     = None; self._t0_peace  = None
        self._t0_fist_nav = None
        # Temporización para navegar menú con dedos
        self._last_fingers = 0;   self._t0_fingers = None
        FINGER_HOLD = 1.5   # segundos manteniendo N dedos para entrar

        # ── Suavizado anti-parpadeo ──────────────────────────────
        # Ventanas cortas de votación: en vez de confiar en el frame
        # instantáneo (que puede "parpadear" por ruido de MediaPipe),
        # se decide por mayoría sobre los últimos SMOOTH_N frames.
        # Esto evita que pequeños temblores reinicien los temporizadores
        # de sostenimiento (gestos, navegación, letras).
        SMOOTH_N = 5
        self._buf_fingers = deque(maxlen=SMOOTH_N)
        self._buf_func    = deque(maxlen=SMOOTH_N)
        self._buf_asl     = deque(maxlen=SMOOTH_N)
        self._buf_fist    = deque(maxlen=SMOOTH_N)
        self._buf_palm_g  = deque(maxlen=SMOOTH_N)
        self._buf_peace   = deque(maxlen=SMOOTH_N)

    @staticmethod
    def _vote(buf, value):
        """Agrega `value` al buffer y retorna el valor más frecuente
        en la ventana (mayoría simple). None cuenta como un valor más."""
        buf.append(value)
        counts = {}
        for v in buf:
            counts[v] = counts.get(v,0)+1
        return max(counts.items(), key=lambda kv: kv[1])[0]

    # ── Helpers ───────────────────────────────────────────────────
    # MARGIN: separación mínima (en coordenadas normalizadas 0-1) que debe
    # haber entre la punta y la base del dedo para contarlo como "arriba".
    # Sin este margen, cuando el dedo está casi recto el punto puede
    # oscilar de un lado a otro del umbral con el más mínimo temblor.
    _MARGIN = 0.02

    def _up(self, lm, tip, pip):
        return lm[tip].y < lm[pip].y - self._MARGIN
    def _thumb_extended(self, lm):
        """Pulgar extendido hacia el lado (no arriba como gesto OK)."""
        # El pulgar está extendido si su punta está lejos de la base de la palma
        # y los otros dedos también están arriba (para diferenciar del gesto OK)
        tip  = lm[4]; base = lm[2]; mcp = lm[9]
        # Distancia horizontal del pulgar respecto al centro de la palma
        dist = abs(tip.x - mcp.x)
        return dist > 0.12   # umbral empírico
    def _count_fingers(self, lm):
        """Cuenta 1-5: 4 dedos normales + pulgar si está extendido."""
        n = sum(self._up(lm,t,t-2) for t in [8,12,16,20])
        if self._thumb_extended(lm):
            n += 1
        return min(n, 5)
    def _thumb_up(self, lm):
        return (lm[4].y < lm[3].y - self._MARGIN < lm[2].y - self._MARGIN and
                all(lm[t].y > lm[t-2].y - self._MARGIN for t in [8,12,16,20]))
    def _fist(self, lm):
        closed = all(lm[t].y > lm[t-2].y - self._MARGIN for t in [8,12,16,20])
        bent   = lm[4].x > lm[3].x if lm[0].x < lm[9].x else lm[4].x < lm[3].x
        return closed and bent
    def _palm(self, lm):
        fingers = all(self._up(lm,t,t-2) for t in [8,12,16,20])
        thumb   = lm[4].y < lm[2].y - self._MARGIN
        return fingers and thumb
    def _peace(self, lm):
        return (self._up(lm,8,6) and self._up(lm,12,10) and
                not self._up(lm,16,14) and not self._up(lm,20,18))
    def _pinch(self, lm, w, h):
        x1,y1 = lm[4].x*w, lm[4].y*h
        x2,y2 = lm[8].x*w, lm[8].y*h
        return math.hypot(x2-x1,y2-y1)
    def _dist(self, lm, i, j):
        return math.hypot(lm[i].x-lm[j].x, lm[i].y-lm[j].y)

    def _asl(self, lm):
        f8  = self._up(lm,8,6);  f12 = self._up(lm,12,10)
        f16 = self._up(lm,16,14); f20 = self._up(lm,20,18)
        f4  = lm[4].y < lm[3].y - self._MARGIN

        # Distancias usadas por las letras nuevas (D, I, O, X)
        thumb_mid_dist  = self._dist(lm, 4, 12)   # pulgar ↔ punta medio
        thumb_index_dist= self._dist(lm, 4, 8)    # pulgar ↔ punta índice
        index_curl      = lm[8].y - lm[6].y       # >0 si la punta del índice
                                                   # quedó más abajo que su
                                                   # articulación (dedo en gancho)

        # ── D: índice arriba + pulgar tocando el dedo medio ─────────
        if (f8 and not f12 and not f16 and not f20
                and thumb_mid_dist < 0.08):
            return "D"

        # ── I: solo meñique arriba, resto cerrado y pulgar cerrado ──
        if (f20 and not f8 and not f12 and not f16 and not f4):
            return "I"

        # ── O: dedos curvados formando círculo con el pulgar ────────
        if (not f8 and not f12 and not f16 and not f20
                and thumb_index_dist < 0.07
                and self._dist(lm,4,16) < 0.12):
            return "O"

        # ── X: índice en gancho, resto cerrado ───────────────────────
        if (not f12 and not f16 and not f20 and not f4
                and index_curl > 0.015 and index_curl < 0.09):
            return "X"

        if f8 and not f12 and not f16 and not f20:
            return "L" if f4 else "1"
        if f8 and f12 and not f16 and not f20: return "2"
        if f8 and f12 and f16 and not f20:     return "3"
        if f4 and not f8 and not f12 and not f16 and f20: return "Y"
        if f4 and f8 and f12 and f16 and f20:  return "B"
        if not any([f4,f8,f12,f16,f20]):       return "A"
        return None

    # ── Proceso ───────────────────────────────────────────────────
    def process(self, frame_bgr):
        out = dict(fingers=0, palm=False, palm_instant=False,
                   ok_conf=False, cancel_conf=False,
                   peace_conf=False, ok_instant=False, cancel_instant=False,
                   pinch_dist=0.0, asl_letter=None,
                   palm_progress=0.0, func_progress=0.0)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)
        ann = frame_bgr.copy()
        h,w = frame_bgr.shape[:2]
        now = time.time()

        if not res.multi_hand_landmarks:
            self._last_func=None; self._t0_func=None
            self._t0_palm=None;   self._t0_peace=None
            self._buf_fingers.clear(); self._buf_func.clear()
            self._buf_asl.clear();     self._buf_fist.clear()
            self._buf_palm_g.clear();  self._buf_peace.clear()
            return ann, out

        hl = res.multi_hand_landmarks[0]
        self._draw.draw_landmarks(ann, hl, self._CONN,
            self._draw.DrawingSpec(color=(0,255,128),thickness=2),
            self._draw.DrawingSpec(color=(255,255,0), thickness=1))
        lm = hl.landmark

        # Valores crudos del frame actual
        raw_fingers = self._count_fingers(lm)
        raw_fist    = self._fist(lm)
        raw_palm_g  = self._palm(lm)
        raw_asl     = self._asl(lm)
        raw_func    = "OK" if self._thumb_up(lm) else ("CANCEL" if raw_fist else None)
        raw_peace   = self._peace(lm)

        # Suavizado por mayoría de votos sobre los últimos frames: evita que
        # un parpadeo de 1 frame (ruido de MediaPipe) reinicie los
        # temporizadores de sostenimiento y haga sentir los gestos "locos".
        fingers = self._vote(self._buf_fingers, raw_fingers)
        fist    = self._vote(self._buf_fist,    raw_fist)
        palm_g  = self._vote(self._buf_palm_g,  raw_palm_g)
        asl     = self._vote(self._buf_asl,     raw_asl)
        func    = self._vote(self._buf_func,    raw_func)
        peace   = self._vote(self._buf_peace,   raw_peace)

        out["fingers"]      = fingers
        out["pinch_dist"]   = self._pinch(lm,w,h)
        out["asl_letter"]   = asl
        out["palm_instant"] = palm_g

        # ── 5 dedos mantenidos 1.5s → volver al menú ─────────────
        MENU_HOLD = 1.5
        if fingers == 5:
            if self._t0_palm is None: self._t0_palm = now
            prog = min((now-self._t0_palm)/MENU_HOLD, 1.0)
            out["palm_progress"] = prog
            bw = int(150*prog)
            cv2.rectangle(ann,(w-160,h-30),(w-10,h-10),(40,40,40),-1)
            cv2.rectangle(ann,(w-160,h-30),(w-160+bw,h-10),(0,200,255),-1)
            cv2.putText(ann,"[MENU]",(w-155,h-35),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,255),2)
            if prog >= 1.0:
                out["palm"]=True; self._t0_palm=None
        else:
            self._t0_palm = None

        # ── Puño mantenido 1.5s → opción 5 (Filtros) desde menú ──
        FIST_NAV_HOLD = 1.5
        if fist and fingers != 5:
            if self._t0_fist_nav is None: self._t0_fist_nav = now
            hold_fist = now - self._t0_fist_nav
            pct_fist  = min(hold_fist / FIST_NAV_HOLD, 1.0)
            bw_fist   = int(150 * pct_fist)
            cv2.rectangle(ann,(10,h-60),(165,h-44),(40,40,40),-1)
            cv2.rectangle(ann,(10,h-60),(10+bw_fist,h-44),(255,100,200),-1)
            cv2.putText(ann,f"Puño {int(pct_fist*100)}%",
                (10,h-65),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,100,200),1)
            if hold_fist >= FIST_NAV_HOLD:
                out["finger_nav"] = 5
                self._t0_fist_nav = now
        else:
            self._t0_fist_nav = None

        # ── N dedos mantenidos 1.5s → navegar menú (1-4) ─────────
        out["finger_nav"] = out.get("finger_nav", 0)
        FINGER_HOLD = 1.5
        n_f = fingers
        if 1 <= n_f <= 4 and not fist and func != "OK" and fingers != 5:
            if n_f != self._last_fingers:
                self._last_fingers = n_f
                self._t0_fingers   = now
            if self._t0_fingers:
                hold_f = now - self._t0_fingers
                pct_f  = min(hold_f / FINGER_HOLD, 1.0)
                bw_f   = int(150 * pct_f)
                cols   = [(0,100,255),(0,200,100),(255,180,0),(200,0,255),(255,100,200)]
                col_n  = cols[n_f-1]
                cv2.rectangle(ann,(10,h-60),(165,h-44),(40,40,40),-1)
                cv2.rectangle(ann,(10,h-60),(10+bw_f,h-44),col_n,-1)
                cv2.putText(ann,f"{n_f} dedos {int(pct_f*100)}%",
                    (10,h-65),cv2.FONT_HERSHEY_SIMPLEX,0.45,col_n,1)
                if hold_f >= FINGER_HOLD:
                    out["finger_nav"] = n_f
                    self._t0_fingers  = now
        else:
            if not (1 <= n_f <= 5):
                self._last_fingers = 0
                self._t0_fingers   = None

        # ── Gestos funcionales (OK / CANCEL) ─────────────────────
        out["ok_instant"]     = func == "OK"
        out["cancel_instant"] = func == "CANCEL"

        if func != self._last_func:
            self._last_func = func
            self._t0_func   = now if func else None

        if func and self._t0_func:
            needed = OK_HOLD_TIME if func=="OK" else CANCEL_HOLD
            hold   = now - self._t0_func
            pct    = min(hold/needed, 1.0)
            out["func_progress"] = pct
            col = (0,220,60) if func=="OK" else (0,60,220)
            cv2.rectangle(ann,(10,h-30),(210,h-10),(40,40,40),-1)
            cv2.rectangle(ann,(10,h-30),(10+int(200*pct),h-10),col,-1)
            if hold >= needed:
                if func=="OK":     out["ok_conf"]     = True
                else:              out["cancel_conf"]  = True
                self._t0_func = now

        # ── Paz ✌ ─────────────────────────────────────────────────
        if peace:
            if self._t0_peace is None: self._t0_peace = now
            if now - self._t0_peace >= 0.6:
                out["peace_conf"]=True; self._t0_peace=None
        else:
            self._t0_peace = None

        # ── Overlay fingers ───────────────────────────────────────
        n = out["fingers"]
        cv2.putText(ann, f"Dedos: {n}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,200,0),2)

        return ann, out

    def close(self): self._hands.close()


# ═══════════════════════════════════════════════════════════════════════
# BodyAnalyzer — MediaPipe Pose + análisis de apariencia
# ═══════════════════════════════════════════════════════════════════════
class BodyAnalyzer:
    def __init__(self):
        _p = mp.solutions.pose
        self._pose = _p.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self._draw = mp.solutions.drawing_utils
        self._CONN = mp.solutions.pose.POSE_CONNECTIONS

    def _dominant_color_name(self, roi_bgr):
        """Devuelve nombre de color dominante en una región."""
        if roi_bgr.size == 0: return "—"
        hsv  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        h    = hsv[:,:,0].flatten()
        s    = hsv[:,:,1].flatten()
        v    = hsv[:,:,2].flatten()
        # Filtrar píxeles válidos (no muy oscuros ni muy desaturados)
        mask = (s > 40) & (v > 40)
        if mask.sum() < 20: return "Oscuro/Negro"
        h_val = np.median(h[mask])
        s_val = np.median(s[mask])
        v_val = np.median(v[mask])
        if v_val < 50:   return "Negro"
        if s_val < 30:
            if v_val > 180: return "Blanco"
            return "Gris"
        if h_val < 10 or h_val > 170:  return "Rojo"
        if h_val < 25:  return "Naranja"
        if h_val < 35:  return "Amarillo"
        if h_val < 85:  return "Verde"
        if h_val < 130: return "Azul"
        if h_val < 150: return "Morado"
        return "Rosa"

    def _hair_color(self, roi_bgr):
        """
        Detecta color de cabello usando solo colores naturales.
        Clasifica en: Negro, Castaño Oscuro, Castaño, Rubio, Canoso, Blanco.
        Ignora fondos y colores no naturales.
        """
        if roi_bgr.size == 0: return "—"
        # Convertir a múltiples espacios para mejor clasificación
        hsv  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        h_ch = hsv[:,:,0].flatten()
        s_ch = hsv[:,:,1].flatten()
        v_ch = hsv[:,:,2].flatten()
        g_ch = gray.flatten()

        # Solo píxeles oscuros/marrones (excluye fondo claro y colores saturados)
        # Cabello natural: saturación media-baja, matiz en rango cálido o neutro
        nat_mask = (
            (s_ch < 120) &          # no muy saturado (excluye fondos coloridos)
            (v_ch > 20)  &          # no completamente negro/sombra
            (v_ch < 230) &          # no fondo blanco brillante
            ~((h_ch > 85) & (h_ch < 150) & (s_ch > 40))  # excluye verde/azul/cyan
        )

        if nat_mask.sum() < 30: return "—"

        v_med = np.median(v_ch[nat_mask])
        s_med = np.median(s_ch[nat_mask])
        h_med = np.median(h_ch[nat_mask])

        # Clasificación por luminosidad y saturación
        if v_med < 45:
            return "Negro"
        if v_med < 80:
            if s_med > 30 and h_med < 25: return "Castaño Oscuro"
            return "Negro"
        if v_med < 120:
            if s_med > 25: return "Castaño"
            return "Castaño Oscuro"
        if v_med < 170:
            if s_med > 30: return "Castaño Claro"
            return "Castaño"
        if v_med < 210:
            if s_med > 20: return "Rubio"
            return "Canoso"
        return "Blanco/Rubio"

    def _hair_analysis(self, frame_bgr, landmarks, w, h):
        """Analiza zona sobre la cabeza para estimar orden y color del cabello."""
        nose = landmarks[0]
        nx, ny = int(nose.x*w), int(nose.y*h)
        # ROI: franja sobre la nariz (zona del cabello)
        y1 = max(0, ny-130); y2 = max(0, ny-30)
        x1 = max(0, nx-80);  x2 = min(w, nx+80)
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0: return "—", False
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Gradiente: alto = borde/cabello despeinado
        gx   = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
        gy   = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
        grad = np.sqrt(gx**2+gy**2).mean()
        ordered = grad < 18
        color = self._hair_color(roi)
        return color, ordered

    def _skin_tone(self, frame_bgr, landmarks, w, h):
        """Tono de piel estimado desde la zona de la frente."""
        nose = landmarks[0]
        nx, ny = int(nose.x*w), int(nose.y*h)
        y1=max(0,ny-60); y2=max(0,ny-10)
        x1=max(0,nx-40); x2=min(w,nx+40)
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0: return "—"
        # Máscara de piel YCrCb
        ycr = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        sk  = cv2.inRange(ycr,
              np.array([0,133,77]), np.array([255,173,127]))
        if sk.sum() == 0: return "—"
        mean_v = roi[sk>0].mean(axis=0)   # BGR medio
        b,g,r  = mean_v
        brightness = (r+g+b)/3
        if   brightness > 180: return "Claro"
        elif brightness > 130: return "Medio"
        elif brightness > 80:  return "Moreno"
        else:                  return "Oscuro"

    def _posture(self, landmarks, w, h):
        """Analiza si los hombros están nivelados."""
        ls = landmarks[11]; rs = landmarks[12]
        lx,ly = ls.x*w, ls.y*h
        rx,ry = rs.x*w, rs.y*h
        diff_y = abs(ly-ry)
        # Inclinación lateral
        if diff_y > 30:
            return "Inclinado", False
        # Hombros caídos: si y del hombro es muy baja vs nariz
        nose = landmarks[0]
        ny   = nose.y*h
        avg_sh_y = (ly+ry)/2
        drop = avg_sh_y - ny
        if drop > h*0.35:
            return "Caídos", False
        return "Rectos ✓", True

    def process(self, frame_bgr):
        ann  = frame_bgr.copy()
        h, w = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res  = self._pose.process(rgb)
        info = {"ropa":"—","cabello_color":"—","cabello_orden":"—",
                "piel":"—","postura":"—","postura_ok":False}

        if not res.pose_landmarks:
            cv2.putText(ann,"Sin cuerpo detectado",(10,60),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,60,200),2)
            return ann, info

        self._draw.draw_landmarks(ann, res.pose_landmarks, self._CONN,
            self._draw.DrawingSpec(color=(0,255,200),thickness=2,circle_radius=3),
            self._draw.DrawingSpec(color=(200,200,0),thickness=2))

        lm = res.pose_landmarks.landmark

        # ── Color de ropa: ROI entre hombros y cadera ─────────────
        ls = lm[11]; rs = lm[12]
        lh = lm[23]; rh = lm[24]
        lsx,lsy = int(ls.x*w), int(ls.y*h)
        rsx,rsy = int(rs.x*w), int(rs.y*h)
        lhx,lhy = int(lh.x*w), int(lh.y*h)
        rhx,rhy = int(rh.x*w), int(rh.y*h)
        x1r = max(0,min(lsx,rsx)); x2r = min(w,max(lsx,rsx))
        y1r = max(0,min(lsy,rsy)); y2r = min(h,max(lhy,rhy))
        if x2r>x1r and y2r>y1r:
            roi_ropa = frame_bgr[y1r:y2r, x1r:x2r]
            info["ropa"] = self._dominant_color_name(roi_ropa)
            cv2.rectangle(ann,(x1r,y1r),(x2r,y2r),(255,200,0),2)
            cv2.putText(ann,"ROPA",(x1r,y1r-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,200,0),1)

        # ── Cabello ───────────────────────────────────────────────
        hair_col, ordered = self._hair_analysis(frame_bgr, lm, w, h)
        info["cabello_color"] = hair_col
        info["cabello_orden"] = "Ordenado ✓" if ordered else "Despeinado"

        # ── Piel ──────────────────────────────────────────────────
        info["piel"] = self._skin_tone(frame_bgr, lm, w, h)

        # ── Postura ───────────────────────────────────────────────
        post, post_ok = self._posture(lm, w, h)
        info["postura"]    = post
        info["postura_ok"] = post_ok

        return ann, info

    def close(self): self._pose.close()


# ═══════════════════════════════════════════════════════════════════════
# EarDetector — fatiga EAR con FaceMesh
# ═══════════════════════════════════════════════════════════════════════
class EarDetector:
    _L = [362,385,387,263,373,380]
    _R = [33, 160,158,133,153,144]
    def __init__(self):
        _fm = mp.solutions.face_mesh
        self._fm = _fm.FaceMesh(max_num_faces=1,refine_landmarks=True,
            min_detection_confidence=0.5,min_tracking_confidence=0.5)
    def _ear(self,lm,idx,w,h):
        pts=[(int(lm[i].x*w),int(lm[i].y*h)) for i in idx]
        A=np.linalg.norm(np.array(pts[1])-np.array(pts[5]))
        B=np.linalg.norm(np.array(pts[2])-np.array(pts[4]))
        C=np.linalg.norm(np.array(pts[0])-np.array(pts[3]))
        return (A+B)/(2*C+1e-6)
    def process(self,frame_bgr):
        h,w=frame_bgr.shape[:2]
        rgb=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
        res=self._fm.process(rgb)
        if not res.multi_face_landmarks: return 0.0
        lm=res.multi_face_landmarks[0].landmark
        return round(((self._ear(lm,self._L,w,h)+self._ear(lm,self._R,w,h))/2),2)
    def close(self): self._fm.close()


# ═══════════════════════════════════════════════════════════════════════
# FaceFilter — filtros tipo Snapchat dibujados con OpenCV + FaceMesh
# ═══════════════════════════════════════════════════════════════════════
class FaceFilter:
    """
    Dibuja filtros AR sobre la cara usando landmarks de MediaPipe FaceMesh.
    Filtros disponibles:
      0 = ninguno
      1 = gorra
      2 = lentes
      3 = bigote
    Todo dibujado con OpenCV puro, sin archivos externos.
    """
    # Índices FaceMesh relevantes
    # Frente: 10 (top), 338 (right temple), 109 (left temple)
    # Ojos: ojo izq 33,133 / ojo der 362,263
    # Nariz: 1 (punta), 129 (ala izq), 358 (ala der)
    # Boca: 61 (comisura izq), 291 (comisura der), 0 (labio sup centro)

    def __init__(self):
        _fm = mp.solutions.face_mesh
        self._fm = _fm.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.active_filter = 0   # 0=ninguno,1=gorra,2=lentes,3=bigote

    def _pt(self, lm, idx, w, h):
        return (int(lm[idx].x * w), int(lm[idx].y * h))

    def _draw_necklace(self, frame, lm, w, h):
        """Collar dibujado entre los hombros con cadena y dije."""
        # Puntos de referencia: base del cuello y hombros
        neck_l = self._pt(lm, 234, w, h)   # sien/mandíbula izq (aprox hombro)
        neck_r = self._pt(lm, 454, w, h)   # sien/mandíbula der
        chin   = self._pt(lm, 152, w, h)   # mentón (base cara)

        cx     = (neck_l[0] + neck_r[0]) // 2
        neck_w = int(abs(neck_r[0] - neck_l[0]) * 0.75)
        base_y = chin[1] + int(abs(chin[1] - neck_l[1]) * 0.6)

        # ── Cadena: arco de puntos ──────────────────────────────
        chain_pts = []
        steps = 30
        for i in range(steps + 1):
            t   = i / steps
            x_c = int(cx - neck_w + 2 * neck_w * t)
            # Curva catenaria (parábola simple)
            sag = int(neck_w * 0.35 * (4 * t * (1 - t)))
            y_c = base_y + sag
            chain_pts.append((x_c, y_c))

        # Dibujar cadena dorada con grosor variable
        for i in range(len(chain_pts) - 1):
            cv2.line(frame, chain_pts[i], chain_pts[i+1], (0, 180, 255), 2)
        # Eslabones decorativos
        for i in range(0, len(chain_pts), 4):
            cv2.circle(frame, chain_pts[i], 3, (0, 210, 255), -1)

        # ── Dije central: estrella de 6 puntas ─────────────────
        mid_pt = chain_pts[steps // 2]
        dije_x, dije_y = mid_pt[0], mid_pt[1] + 10
        r_out, r_in = 12, 6
        star_pts = []
        for i in range(12):
            angle = math.radians(i * 30 - 90)
            r     = r_out if i % 2 == 0 else r_in
            star_pts.append((
                int(dije_x + r * math.cos(angle)),
                int(dije_y + r * math.sin(angle))
            ))
        star_arr = np.array(star_pts, dtype=np.int32)
        cv2.fillPoly(frame, [star_arr], (0, 200, 255))
        cv2.polylines(frame, [star_arr], True, (0, 255, 220), 1)
        # Brillito central
        cv2.circle(frame, (dije_x, dije_y), 3, (255, 255, 255), -1)

        # ── Cierre del collar en los extremos ──────────────────
        cv2.circle(frame, chain_pts[0],     5, (0, 180, 255), -1)
        cv2.circle(frame, chain_pts[steps], 5, (0, 180, 255), -1)

    def _draw_glasses(self, frame, lm, w, h):
        """Lentes de sol dibujados con elipses."""
        # Centros de los ojos
        le_l = self._pt(lm, 33,  w, h)
        le_r = self._pt(lm, 133, w, h)
        re_l = self._pt(lm, 362, w, h)
        re_r = self._pt(lm, 263, w, h)

        lc = ((le_l[0]+le_r[0])//2, (le_l[1]+le_r[1])//2)
        rc = ((re_l[0]+re_r[0])//2, (re_l[1]+re_r[1])//2)

        r_x = max(int(abs(le_r[0]-le_l[0])*0.75), 18)
        r_y = max(int(r_x * 0.55), 12)

        # Lentes con transparencia simulada (overlay oscuro)
        overlay = frame.copy()
        cv2.ellipse(overlay, lc, (r_x, r_y), 0, 0, 360, (10,10,10), -1)
        cv2.ellipse(overlay, rc, (r_x, r_y), 0, 0, 360, (10,10,10), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Borde dorado
        cv2.ellipse(frame, lc, (r_x, r_y), 0, 0, 360, (0, 180, 255), 2)
        cv2.ellipse(frame, rc, (r_x, r_y), 0, 0, 360, (0, 180, 255), 2)

        # Puente entre lentes
        bridge_lx = lc[0] + r_x
        bridge_rx = rc[0] - r_x
        bridge_y  = (lc[1] + rc[1]) // 2
        cv2.line(frame, (bridge_lx, bridge_y), (bridge_rx, bridge_y), (0,180,255), 2)

        # Patillas
        cv2.line(frame, (lc[0]-r_x, lc[1]), (lc[0]-r_x-30, lc[1]-5), (0,180,255), 2)
        cv2.line(frame, (rc[0]+r_x, rc[1]), (rc[0]+r_x+30, rc[1]-5), (0,180,255), 2)

    def _draw_mustache(self, frame, lm, w, h):
        """Bigote dibujado con curvas bezier simuladas."""
        nose  = self._pt(lm, 1,   w, h)
        left  = self._pt(lm, 129, w, h)
        right = self._pt(lm, 358, w, h)
        mouth_l = self._pt(lm, 61,  w, h)
        mouth_r = self._pt(lm, 291, w, h)

        cx   = (left[0] + right[0]) // 2
        mw   = int(abs(right[0] - left[0]) * 0.85)
        base_y = nose[1] + int(abs(mouth_l[1] - nose[1]) * 0.35)

        # Lado izquierdo del bigote
        pts_l = []
        for t in np.linspace(0, 1, 20):
            # Curva cuadrática: desde borde izq → centro abajo → nariz izq
            p0 = np.array([cx - mw, base_y - 4])
            p1 = np.array([cx - mw//2, base_y + 12])
            p2 = np.array([cx - 6, base_y + 2])
            pt = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
            pts_l.append(pt.astype(int))

        # Lado derecho del bigote
        pts_r = []
        for t in np.linspace(0, 1, 20):
            p0 = np.array([cx + 6, base_y + 2])
            p1 = np.array([cx + mw//2, base_y + 12])
            p2 = np.array([cx + mw, base_y - 4])
            pt = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
            pts_r.append(pt.astype(int))

        # Borde superior (línea recta)
        top_pts = np.array([[cx-mw, base_y-4],[cx, base_y-1],[cx+mw, base_y-4]], np.int32)

        # Rellenar bigote
        all_pts = np.array(pts_l + pts_r[::-1], dtype=np.int32)
        cv2.fillPoly(frame, [all_pts], (30, 20, 15))

        # Contorno
        pts_curve = np.array(pts_l + pts_r, dtype=np.int32).reshape((-1,1,2))
        cv2.polylines(frame, [pts_curve], False, (80, 50, 30), 2)

        # Línea central (separación del bigote)
        cv2.line(frame, (cx, base_y-1), (cx, base_y+10), (60, 40, 25), 1)

    def process(self, frame_bgr):
        """Aplica el filtro activo y devuelve frame anotado."""
        if self.active_filter == 0:
            return frame_bgr.copy()

        ann = frame_bgr.copy()
        h, w = ann.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._fm.process(rgb)

        if not res.multi_face_landmarks:
            cv2.putText(ann, "Sin cara detectada", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 200), 2)
            return ann

        lm = res.multi_face_landmarks[0].landmark

        if self.active_filter == 1:
            self._draw_necklace(ann, lm, w, h)
        elif self.active_filter == 2:
            self._draw_glasses(ann, lm, w, h)
        elif self.active_filter == 3:
            self._draw_mustache(ann, lm, w, h)

        # Etiqueta del filtro activo
        names = {1:"💎 Collar", 2:"🕶 Lentes", 3:"👨 Bigote"}
        cv2.putText(ann, names[self.active_filter], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
        return ann

    def close(self):
        self._fm.close()


# ═══════════════════════════════════════════════════════════════════════
# SerialManager
# ═══════════════════════════════════════════════════════════════════════
class SerialManager:
    def __init__(self,port=SERIAL_PORT,baud=SERIAL_BAUD):
        self.port=port;self.baud=baud;self._ser=None;self.connected=False
    def connect(self):
        try:
            self._ser=serial.Serial(self.port,self.baud,timeout=ACK_TIMEOUT)
            time.sleep(2);self.connected=True;return True
        except serial.SerialException as e:
            print(f"[Serial] {e}");return False
    def send_command(self,cmd):
        if not self.connected:
            print(f"[Serial] Sim '{cmd}'");time.sleep(0.3);return True
        try:
            self._ser.reset_input_buffer();self._ser.write(cmd.encode())
            deadline=time.time()+ACK_TIMEOUT;buf=b""
            while time.time()<deadline:
                if self._ser.in_waiting:
                    buf+=self._ser.read(self._ser.in_waiting)
                    if b"ACK" in buf:return True
                time.sleep(0.05)
            return False
        except serial.SerialException:return False
    def close(self):
        if self._ser and self._ser.is_open:self._ser.close()
    @staticmethod
    def list_ports():
        return [p.device for p in serial.tools.list_ports.comports()]


# ═══════════════════════════════════════════════════════════════════════
# RampDetector
# ═══════════════════════════════════════════════════════════════════════
class RampDetector:
    MIN_AREA = 2000

    def _detect_any_object(self, frame_bgr):
        """
        Detecta cualquier objeto (sin importar color) usando umbralización
        por saturación/valor para separarlo de un fondo neutro.
        Devuelve el contorno más grande o None.
        """
        hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Combina: bordes (Canny) + diferencia de saturación para detectar
        # cualquier objeto que destaque sobre un fondo relativamente uniforme
        blur   = cv2.GaussianBlur(gray, (7, 7), 0)
        edges  = cv2.Canny(blur, 40, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dil    = cv2.dilate(edges, kernel, iterations=2)
        closed = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, kernel)
        cnts,_ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, 0
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < GENERIC_MIN_AREA:
            return None, 0
        return c, area

    def process(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        bl  = cv2.GaussianBlur(hsv, (9, 9), 0)
        bc, ba, bn = None, 0, None

        # ── 1) Buscar AZUL y AMARILLO específicamente ────────────
        for cn, ranges in COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(bl, lo, hi)
            k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
            cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c    = max(cnts, key=cv2.contourArea)
                area = cv2.contourArea(c)
                if area > self.MIN_AREA and area > ba:
                    ba, bc, bn = area, cn, c

        ann  = frame_bgr.copy()
        dest = None
        label_name = None

        if bc and bn is not None:
            # Objeto azul o amarillo detectado
            dest = COLOR_TO_DEST[bc]
            label_name = bc
            bgr  = DEST_BGR[dest]
            x, y, bw, bh = cv2.boundingRect(bn)
            cv2.rectangle(ann, (x, y), (x+bw, y+bh), bgr, 3)
            cv2.putText(ann, f"{bc}->{dest}({int(ba)}px)",
                        (x, max(y-8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)
        else:
            # ── 2) Ningún azul/amarillo: buscar cualquier otro objeto → C
            c_any, area_any = self._detect_any_object(frame_bgr)
            if c_any is not None:
                dest = "C"
                label_name = "OTRO"
                ba = area_any
                bgr = DEST_BGR[dest]
                x, y, bw, bh = cv2.boundingRect(c_any)
                cv2.rectangle(ann, (x, y), (x+bw, y+bh), bgr, 3)
                cv2.putText(ann, f"OTRO->C({int(area_any)}px)",
                            (x, max(y-8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)

        cv2.putText(ann, "CAM RAMPA", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        return ann, label_name, ba, dest


# ═══════════════════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════════════════
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sistema de Clasificación v5.0")
        self.geometry("1280x800")
        self.resizable(False,False)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Estado navegación
        self._screen     = Screen.MENU
        self._brightness = 1.0
        self._pinch_cd   = 0.0

        # Subnavegación dentro de Lenguaje de Señas
        self._asl_mode      = "SELECTOR" # "SELECTOR" | "LETRAS" | "PALABRAS"
        self._asl_word      = ""         # palabra acumulada en modo Palabras
        self._asl_last_letter = None     # última letra vista (para detectar cambios)
        self._asl_t0_letter = None       # instante desde que se sostiene la letra actual
        self._asl_letter_hold = 1.5      # segundos sosteniendo letra para agregarla
        self._asl_open_fired  = False    # evita repetir el borrado mientras la mano está abierta
        self._asl_pinch_cd  = 0.0        # cooldown del pellizco (borrar última letra)

        # Estado clasificación
        self._clf_state      = St.BLOQUEADO
        self._det_color      = None
        self._det_dest       = None
        self._det_area       = 0
        self._cnt            = {"A":0,"B":0,"C":0}
        self._face_active    = False   # ✌ activa cara en trabajo
        self._op_name        = "—"
        self._op_auth        = False
        self._last_face_t    = 0.0
        self._face_box       = None

        # Colas
        self._q_op    = queue.Queue(maxsize=2)
        self._q_ramp  = queue.Queue(maxsize=2)

        # Módulos
        self._gest    = GestureEngine()
        self._ear_det = EarDetector()
        self._face_rec= FaceRecognizer(OPERATORS_DIR)
        self._body    = BodyAnalyzer()
        self._ramp    = RampDetector()
        self._serial  = SerialManager()
        self._filt    = FaceFilter()
        self._running = True
        self._ramp_active = False

        # Último frame cam0 para análisis corporal
        self._last_frame0 = None
        self._body_info   = {}
        self._ear_val     = 0.0

        self._build_ui()
        self._try_serial()
        threading.Thread(target=self._cap0_thread, daemon=True).start()
        self._update_gui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Barra superior fija
        top = ctk.CTkFrame(self,height=44,fg_color="#080f18",corner_radius=0)
        top.pack(fill="x",side="top")
        self._lbl_screen = ctk.CTkLabel(top,text="● MENÚ",
            font=ctk.CTkFont(size=20,weight="bold"),text_color="#00d4ff")
        self._lbl_screen.pack(side="left",padx=16,pady=6)
        self._lbl_clock = ctk.CTkLabel(top,text="",
            font=ctk.CTkFont(size=11),text_color="#3a5a7a")
        self._lbl_clock.pack(side="right",padx=14)
        self._lbl_serial_st = ctk.CTkLabel(top,text="● Serial —",
            font=ctk.CTkFont(size=11),text_color="#3a5a7a")
        self._lbl_serial_st.pack(side="right",padx=10)
        self._lbl_fingers = ctk.CTkLabel(top,text="✋ —",
            font=ctk.CTkFont(size=13,weight="bold"),text_color="#ffcc00")
        self._lbl_fingers.pack(side="right",padx=16)

        # Contenedor principal (swap de pantallas)
        self._container = ctk.CTkFrame(self,fg_color="transparent")
        self._container.pack(fill="both",expand=True,padx=8,pady=(4,0))

        # Log inferior fijo
        log_bar = ctk.CTkFrame(self,height=72,fg_color="#060d14",corner_radius=0)
        log_bar.pack(fill="x",side="bottom")
        ctk.CTkLabel(log_bar,text="LOG",
            font=ctk.CTkFont(size=9,weight="bold"),text_color="#2a4a6a"
            ).pack(anchor="w",padx=10,pady=(3,0))
        self._log_box = ctk.CTkTextbox(log_bar,height=48,
            font=ctk.CTkFont(family="Courier",size=10),
            fg_color="#060d14",text_color="#5a8a7a",
            state="disabled",wrap="word")
        self._log_box.pack(fill="x",padx=6,pady=(0,4))

        # Construir todas las pantallas
        self._screens = {}
        self._screens[Screen.MENU]     = self._build_screen_menu()
        self._screens[Screen.TRABAJO]  = self._build_screen_trabajo()
        self._screens[Screen.BRILLO]   = self._build_screen_brillo()
        self._screens[Screen.SEÑAS]    = self._build_screen_señas()
        self._screens[Screen.CORPORAL] = self._build_screen_corporal()
        self._screens[Screen.FILTROS]  = self._build_screen_filtros()

        # Mostrar menú al inicio
        self._show_screen(Screen.MENU)

    def _card(self,parent,**kw):
        return ctk.CTkFrame(parent,fg_color="#111c27",corner_radius=10,
                            border_width=1,border_color="#1e3048",**kw)

    def _slbl(self,parent,text,size=10):
        ctk.CTkLabel(parent,text=text,
            font=ctk.CTkFont(size=size,weight="bold"),
            text_color="#3a5a7a").pack(pady=(8,2))

    # ── MENÚ PRINCIPAL ────────────────────────────────────────────
    def _build_screen_menu(self):
        frm = ctk.CTkFrame(self._container,fg_color="transparent")

        # Cámara grande arriba
        top = self._card(frm)
        top.pack(fill="both",expand=True,pady=(0,6))
        ctk.CTkLabel(top,text="CÁMARA — muestra dedos para navegar",
            font=ctk.CTkFont(size=10,weight="bold"),
            text_color="#3a5a7a").pack(pady=(6,2))
        self._menu_cam = ctk.CTkLabel(top,text="",width=640,height=360)
        self._menu_cam.pack(padx=6,pady=(0,6))

        # 5 bloques del menú
        bot = ctk.CTkFrame(frm,fg_color="transparent")
        bot.pack(fill="x",pady=(0,4))
        opts = [
            ("1","Área de Trabajo","#ff5555","Clasificación + Arduino"),
            ("2","Brillo","#ffcc00","Ajusta brillo de cámaras"),
            ("3","Lenguaje de Señas","#55ccff","Reconocimiento ASL"),
            ("4","Análisis Corporal","#55ee99","Ropa · Cabello · Postura"),
            ("5","Filtros AR","#ff88ff","Gorra · Lentes · Bigote"),
        ]
        self._menu_blocks = []
        for i,(num,title,col,sub) in enumerate(opts):
            c = ctk.CTkFrame(bot,fg_color="#0a1520",corner_radius=12,
                             border_width=2,border_color="#1e3048",height=80)
            c.pack(side="left",expand=True,fill="x",padx=3)
            c.pack_propagate(False)
            ctk.CTkLabel(c,text=num,
                font=ctk.CTkFont(size=26,weight="bold"),
                text_color=col).pack(side="left",padx=10)
            txt = ctk.CTkFrame(c,fg_color="transparent")
            txt.pack(side="left",fill="both",expand=True,pady=8)
            ctk.CTkLabel(txt,text=title,
                font=ctk.CTkFont(size=12,weight="bold"),
                text_color="#e0e0e0").pack(anchor="w")
            ctk.CTkLabel(txt,text=sub,
                font=ctk.CTkFont(size=9),
                text_color="#3a5a7a").pack(anchor="w")
            self._menu_blocks.append((c,col))
        return frm

    def _highlight_menu(self, idx):
        """Resalta el bloque del menú según dedos extendidos."""
        for i,(c,col) in enumerate(self._menu_blocks):
            if i == idx-1:
                c.configure(fg_color="#0d2030",border_color=col)
            else:
                c.configure(fg_color="#0a1520",border_color="#1e3048")

    # ── ÁREA DE TRABAJO ───────────────────────────────────────────
    def _build_screen_trabajo(self):
        frm = ctk.CTkFrame(self._container,fg_color="transparent")
        frm.grid_columnconfigure(0,weight=4)
        frm.grid_columnconfigure(1,weight=3)
        frm.grid_rowconfigure(0,weight=1)

        # Columna izquierda: cámaras
        left = self._card(frm)
        left.grid(row=0,column=0,padx=(0,6),pady=0,sticky="nsew")
        left.grid_rowconfigure(1,weight=3); left.grid_rowconfigure(3,weight=2)
        left.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(left,text="CÁMARA OPERADOR",
            font=ctk.CTkFont(size=10,weight="bold"),
            text_color="#3a5a7a").grid(row=0,column=0,pady=(6,2))
        self._trab_cam_op = ctk.CTkLabel(left,text="",width=480,height=300)
        self._trab_cam_op.grid(row=1,column=0,padx=6,pady=2)
        ctk.CTkLabel(left,text="CÁMARA RAMPA",
            font=ctk.CTkFont(size=10,weight="bold"),
            text_color="#3a5a7a").grid(row=2,column=0,pady=(6,2))
        self._trab_cam_ramp = ctk.CTkLabel(left,text="Rampa inactiva",
            width=480,height=200,fg_color="#060d14",corner_radius=8,
            text_color="#2a4a6a",font=ctk.CTkFont(size=13))
        self._trab_cam_ramp.grid(row=3,column=0,padx=6,pady=(2,6))

        # Columna derecha: info
        right = self._card(frm)
        right.grid(row=0,column=1,padx=0,pady=0,sticky="nsew")

        # Estado sistema
        self._slbl(right,"ESTADO SISTEMA",12)
        self._trab_state_lbl = ctk.CTkLabel(right,text=f"● {self._clf_state}",
            font=ctk.CTkFont(size=16,weight="bold"),text_color="#00d4ff")
        self._trab_state_lbl.pack()

        ctk.CTkFrame(right,height=1,fg_color="#1e3048").pack(fill="x",padx=10,pady=8)

        # Operador
        self._slbl(right,"OPERADOR (✌ para activar)")
        self._trab_op_name = ctk.CTkLabel(right,text="—",
            font=ctk.CTkFont(size=16,weight="bold"),text_color="#00d4ff")
        self._trab_op_name.pack()
        self._trab_auth = ctk.CTkLabel(right,text="⬤  NO AUTORIZADO",
            font=ctk.CTkFont(size=12),text_color="#ff4444")
        self._trab_auth.pack(pady=(4,0))

        # Fatiga
        self._slbl(right,"FATIGA (EAR)")
        self._trab_ear = ctk.CTkLabel(right,text="0.00",
            font=ctk.CTkFont(size=26,weight="bold"),text_color="#00ff88")
        self._trab_ear.pack()
        self._trab_ear_bar = ctk.CTkProgressBar(right,width=150,height=8,
            progress_color="#00ff88",fg_color="#1e3048")
        self._trab_ear_bar.set(0); self._trab_ear_bar.pack(pady=(2,0))
        self._trab_fat_warn = ctk.CTkLabel(right,text="",
            font=ctk.CTkFont(size=11,weight="bold"),text_color="#ff4444")
        self._trab_fat_warn.pack()

        ctk.CTkFrame(right,height=1,fg_color="#1e3048").pack(fill="x",padx=10,pady=6)

        # Objeto detectado
        self._slbl(right,"OBJETO DETECTADO")
        self._trab_obj = ctk.CTkLabel(right,text="—",
            font=ctk.CTkFont(size=18,weight="bold"),text_color="#ffcc00")
        self._trab_obj.pack()
        self._trab_dest = ctk.CTkLabel(right,text="Destino: —",
            font=ctk.CTkFont(size=12),text_color="#888")
        self._trab_dest.pack()

        ctk.CTkFrame(right,height=1,fg_color="#1e3048").pack(fill="x",padx=10,pady=6)

        # Contadores
        self._slbl(right,"PRODUCCIÓN")
        grid = ctk.CTkFrame(right,fg_color="transparent")
        grid.pack(padx=8,fill="x")
        self._trab_cnt = {}
        for i,(key,col) in enumerate([("A","#ff5555"),("B","#55cc55"),
                                       ("C","#5599ff"),("T","#ffffff")]):
            sub = ctk.CTkFrame(grid,fg_color="#0a1520",corner_radius=8,
                               border_width=1,border_color="#1e3048")
            sub.grid(row=i//2,column=i%2,padx=3,pady=3,sticky="ew")
            grid.grid_columnconfigure(i%2,weight=1)
            ctk.CTkLabel(sub,text="Total" if key=="T" else f"Lugar {key}",
                font=ctk.CTkFont(size=9),text_color="#3a5a7a").pack(pady=(4,0))
            lbl=ctk.CTkLabel(sub,text="0",
                font=ctk.CTkFont(size=22,weight="bold"),text_color=col)
            lbl.pack(pady=(0,4))
            self._trab_cnt[key]=lbl

        # Botones sim
        ctk.CTkFrame(right,height=1,fg_color="#1e3048").pack(fill="x",padx=10,pady=6)
        bf=ctk.CTkFrame(right,fg_color="transparent"); bf.pack(pady=4)
        ctk.CTkButton(bf,text="▲ Sim OK",width=90,
            fg_color="#0d3320",border_color="#1a6a40",border_width=1,
            text_color="#33ee77",hover_color="#0a4a2a",
            command=lambda:self._on_ok()).pack(side="left",padx=3)
        ctk.CTkButton(bf,text="✕ Cancelar",width=90,
            fg_color="#330d0d",border_color="#6a1a1a",border_width=1,
            text_color="#ee3333",hover_color="#4a0a0a",
            command=lambda:self._on_cancel()).pack(side="left",padx=3)

        ctk.CTkLabel(right,text="Palma extendida → Menú",
            font=ctk.CTkFont(size=9),text_color="#2a4a6a").pack(pady=(6,4))

        return frm

    # ── BRILLO ────────────────────────────────────────────────────
    def _build_screen_brillo(self):
        frm = ctk.CTkFrame(self._container,fg_color="transparent")
        frm.grid_columnconfigure(0,weight=1)
        frm.grid_columnconfigure(1,weight=1)
        frm.grid_rowconfigure(0,weight=1)

        left = self._card(frm)
        left.grid(row=0,column=0,padx=(0,6),pady=0,sticky="nsew")
        ctk.CTkLabel(left,text="CÁMARA (preview)",
            font=ctk.CTkFont(size=10,weight="bold"),
            text_color="#3a5a7a").pack(pady=(8,2))
        self._bri_cam = ctk.CTkLabel(left,text="",width=580,height=420)
        self._bri_cam.pack(padx=6,pady=(0,8))

        right = self._card(frm)
        right.grid(row=0,column=1,padx=0,pady=0,sticky="nsew")
        self._slbl(right,"CONTROL DE BRILLO",14)
        self._bri_val = ctk.CTkLabel(right,text="100%",
            font=ctk.CTkFont(size=60,weight="bold"),text_color="#ffcc00")
        self._bri_val.pack(pady=16)
        self._bri_bar = ctk.CTkProgressBar(right,width=200,height=18,
            progress_color="#ffcc00",fg_color="#1e3048")
        self._bri_bar.set(1.0); self._bri_bar.pack()
        self._bri_pinch = ctk.CTkLabel(right,text="Pellizco: —",
            font=ctk.CTkFont(size=12),text_color="#555")
        self._bri_pinch.pack(pady=8)
        ctk.CTkFrame(right,height=1,fg_color="#1e3048").pack(fill="x",padx=20,pady=12)
        ctk.CTkLabel(right,
            text="Pellizco abierto (>80px) = + brillo\nPellizco cerrado (<30px) = - brillo",
            font=ctk.CTkFont(size=12),text_color="#3a5a7a",justify="center").pack()
        ctk.CTkLabel(right,text="Palma extendida → Menú",
            font=ctk.CTkFont(size=10),text_color="#2a4a6a").pack(pady=(20,0))
        return frm

    # ── LENGUAJE DE SEÑAS ─────────────────────────────────────────
    def _build_screen_señas(self):
        frm = ctk.CTkFrame(self._container,fg_color="transparent")
        frm.grid_columnconfigure(0,weight=1)
        frm.grid_columnconfigure(1,weight=1)
        frm.grid_rowconfigure(1,weight=1)

        # Selector de subopción (1 dedo = Letras, 2 dedos = Palabras)
        sub_bar = ctk.CTkFrame(frm,fg_color="transparent")
        sub_bar.grid(row=0,column=0,columnspan=2,sticky="ew",pady=(0,6))
        self._asl_sub_blocks = []
        for i,(num,title) in enumerate([("1","Letras"),("2","Palabras")]):
            c = ctk.CTkFrame(sub_bar,fg_color="#0a1520",corner_radius=10,
                             border_width=2,border_color="#1e3048",height=44)
            c.pack(side="left",expand=True,fill="x",padx=4)
            c.pack_propagate(False)
            ctk.CTkLabel(c,text=f"{num}. {title}",
                font=ctk.CTkFont(size=12,weight="bold"),
                text_color="#e0e0e0").pack(expand=True)
            self._asl_sub_blocks.append(c)

        left = self._card(frm)
        left.grid(row=1,column=0,padx=(0,6),pady=0,sticky="nsew")
        ctk.CTkLabel(left,text="CÁMARA",
            font=ctk.CTkFont(size=10,weight="bold"),
            text_color="#3a5a7a").pack(pady=(8,2))
        self._asl_cam = ctk.CTkLabel(left,text="",width=580,height=420)
        self._asl_cam.pack(padx=6,pady=(0,8))

        right = self._card(frm)
        right.grid(row=1,column=1,padx=0,pady=0,sticky="nsew")

        # Contenedor de la vista activa (Selector, Letras o Palabras)
        self._asl_view_container = ctk.CTkFrame(right,fg_color="transparent")
        self._asl_view_container.pack(fill="both",expand=True)

        # ── Vista SELECTOR (sub-menú propio) ───────────────────────
        self._asl_view_selector = ctk.CTkFrame(self._asl_view_container,fg_color="transparent")
        self._slbl(self._asl_view_selector,"LENGUAJE DE SEÑAS",14)
        ctk.CTkLabel(self._asl_view_selector,
            text="Elige una subopción",
            font=ctk.CTkFont(size=12),text_color="#6a7a8a").pack(pady=(4,16))

        sel_grid = ctk.CTkFrame(self._asl_view_selector,fg_color="transparent")
        sel_grid.pack(fill="x",padx=10)
        for num,title,sub in [("1","Letras","Letra detectada en vivo"),
                               ("2","Palabras","Acumula letras en una palabra")]:
            row_f = ctk.CTkFrame(sel_grid,fg_color="#0a1520",corner_radius=10,
                                  border_width=1,border_color="#1e3048")
            row_f.pack(fill="x",pady=6)
            ctk.CTkLabel(row_f,text=num,
                font=ctk.CTkFont(size=26,weight="bold"),
                text_color="#00d4ff").pack(side="left",padx=14,pady=10)
            txt_f = ctk.CTkFrame(row_f,fg_color="transparent")
            txt_f.pack(side="left",fill="both",expand=True,pady=8)
            ctk.CTkLabel(txt_f,text=title,
                font=ctk.CTkFont(size=13,weight="bold"),
                text_color="#e0e0e0").pack(anchor="w")
            ctk.CTkLabel(txt_f,text=sub,
                font=ctk.CTkFont(size=10),
                text_color="#3a5a7a").pack(anchor="w")

        ctk.CTkFrame(self._asl_view_selector,height=1,fg_color="#1e3048").pack(
            fill="x",padx=20,pady=(16,10))
        ctk.CTkLabel(self._asl_view_selector,
            text="Solo se aceptan 1, 2 o puño (2s) aquí.\nPuño (2s) → Menú principal",
            font=ctk.CTkFont(size=11),text_color="#2a4a6a",justify="center").pack()

        # ── Vista LETRAS ──────────────────────────────────────────
        self._asl_view_letras = ctk.CTkFrame(self._asl_view_container,fg_color="transparent")
        self._slbl(self._asl_view_letras,"LETRA DETECTADA",14)
        self._asl_letter = ctk.CTkLabel(self._asl_view_letras,text="—",
            font=ctk.CTkFont(size=100,weight="bold"),text_color="#00d4ff")
        self._asl_letter.pack(pady=8)
        self._asl_desc = ctk.CTkLabel(self._asl_view_letras,text="",
            font=ctk.CTkFont(size=13),text_color="#6a7a8a")
        self._asl_desc.pack()
        ctk.CTkFrame(self._asl_view_letras,height=1,fg_color="#1e3048").pack(
            fill="x",padx=20,pady=10)
        self._slbl(self._asl_view_letras,"GUÍA RÁPIDA")
        guide_f = ctk.CTkFrame(self._asl_view_letras,fg_color="#0a1520",corner_radius=8)
        guide_f.pack(padx=12,fill="x")
        for letra,desc in [("A","Puño"),("B","Todos los dedos"),("C","Forma curva"),
                            ("D","Índice + pulgar en medio"),("I","Solo meñique"),
                            ("O","Dedos en círculo"),("X","Índice en gancho"),
                            ("L","Índice+pulgar"),("Y","Pulgar+meñique"),
                            ("1","Solo índice"),("2","Índice+medio"),("3","3 dedos")]:
            r=ctk.CTkFrame(guide_f,fg_color="transparent"); r.pack(fill="x",padx=8,pady=1)
            ctk.CTkLabel(r,text=letra,font=ctk.CTkFont(size=13,weight="bold"),
                text_color="#00d4ff",width=24).pack(side="left")
            ctk.CTkLabel(r,text=desc,font=ctk.CTkFont(size=11),
                text_color="#3a5a7a").pack(side="left",padx=6)
        ctk.CTkLabel(self._asl_view_letras,
            text="Puño (2s) → volver al sub-menú de Señas",
            font=ctk.CTkFont(size=10),text_color="#2a4a6a").pack(pady=(10,4))

        # ── Vista PALABRAS ────────────────────────────────────────
        self._asl_view_palabras = ctk.CTkFrame(self._asl_view_container,fg_color="transparent")
        self._slbl(self._asl_view_palabras,"PALABRA FORMADA",14)
        self._asl_word_lbl = ctk.CTkLabel(self._asl_view_palabras,text="—",
            font=ctk.CTkFont(size=42,weight="bold"),text_color="#00d4ff",
            wraplength=420,justify="center")
        self._asl_word_lbl.pack(pady=(8,4))
        self._asl_word_letter_lbl = ctk.CTkLabel(self._asl_view_palabras,text="",
            font=ctk.CTkFont(size=13),text_color="#6a7a8a")
        self._asl_word_letter_lbl.pack()
        self._asl_word_progress = ctk.CTkProgressBar(self._asl_view_palabras,
            width=200,height=8,progress_color="#00d4ff",fg_color="#1e3048")
        self._asl_word_progress.set(0)
        self._asl_word_progress.pack(pady=(6,0))
        ctk.CTkFrame(self._asl_view_palabras,height=1,fg_color="#1e3048").pack(
            fill="x",padx=20,pady=12)
        ctk.CTkLabel(self._asl_view_palabras,
            text="Sostén una letra 1.5s para agregarla a la palabra",
            font=ctk.CTkFont(size=11),text_color="#3a5a7a",
            wraplength=300,justify="center").pack(pady=(0,8))
        ctk.CTkLabel(self._asl_view_palabras,
            text="Pellizco corto → borrar última letra",
            font=ctk.CTkFont(size=11),text_color="#3a5a7a").pack()
        ctk.CTkLabel(self._asl_view_palabras,
            text="Mano abierta (rápida) → limpiar todo",
            font=ctk.CTkFont(size=11),text_color="#3a5a7a").pack(pady=(0,8))
        ctk.CTkLabel(self._asl_view_palabras,
            text="Puño (2s) → volver al sub-menú de Señas",
            font=ctk.CTkFont(size=10),text_color="#2a4a6a").pack(pady=(4,4))

        self._update_asl_subview()
        return frm

    def _update_asl_subview(self):
        """Actualiza qué subvista de Señas se muestra (Selector, Letras o Palabras)."""
        for c in self._asl_sub_blocks:
            c.configure(fg_color="#0a1520",border_color="#1e3048")
        if self._asl_mode == "LETRAS":
            self._asl_sub_blocks[0].configure(fg_color="#0d2030",border_color="#00d4ff")
        elif self._asl_mode == "PALABRAS":
            self._asl_sub_blocks[1].configure(fg_color="#0d2030",border_color="#00d4ff")

        self._asl_view_selector.pack_forget()
        self._asl_view_letras.pack_forget()
        self._asl_view_palabras.pack_forget()
        if self._asl_mode == "SELECTOR":
            self._asl_view_selector.pack(fill="both",expand=True)
        elif self._asl_mode == "LETRAS":
            self._asl_view_letras.pack(fill="both",expand=True)
        elif self._asl_mode == "PALABRAS":
            self._asl_view_palabras.pack(fill="both",expand=True)

    # ── ANÁLISIS CORPORAL ─────────────────────────────────────────
    def _build_screen_corporal(self):
        frm = ctk.CTkFrame(self._container,fg_color="transparent")
        frm.grid_columnconfigure(0,weight=1)
        frm.grid_columnconfigure(1,weight=1)
        frm.grid_rowconfigure(0,weight=1)

        left = self._card(frm)
        left.grid(row=0,column=0,padx=(0,6),pady=0,sticky="nsew")
        ctk.CTkLabel(left,text="CÁMARA + POSE",
            font=ctk.CTkFont(size=10,weight="bold"),
            text_color="#3a5a7a").pack(pady=(8,2))
        self._corp_cam = ctk.CTkLabel(left,text="",width=580,height=420)
        self._corp_cam.pack(padx=6,pady=(0,8))

        right = self._card(frm)
        right.grid(row=0,column=1,padx=0,pady=0,sticky="nsew")
        self._slbl(right,"ANÁLISIS CORPORAL",14)

        items = [
            ("ROPA","—","#ffaa00"),
            ("CABELLO COLOR","—","#aa88ff"),
            ("CABELLO","—","#aa88ff"),
            ("TONO DE PIEL","—","#ffcc88"),
            ("POSTURA","—","#55ccff"),
        ]
        self._corp_labels = {}
        for title,val,col in items:
            row_f=ctk.CTkFrame(right,fg_color="#0a1520",corner_radius=8,
                               border_width=1,border_color="#1e3048")
            row_f.pack(fill="x",padx=12,pady=4)
            ctk.CTkLabel(row_f,text=title,
                font=ctk.CTkFont(size=9,weight="bold"),
                text_color="#3a5a7a").pack(side="left",padx=10,pady=8)
            lbl=ctk.CTkLabel(row_f,text=val,
                font=ctk.CTkFont(size=14,weight="bold"),text_color=col)
            lbl.pack(side="right",padx=10)
            self._corp_labels[title]=lbl

        ctk.CTkLabel(right,text="Puño 2s → Menú",
            font=ctk.CTkFont(size=10),text_color="#2a4a6a").pack(pady=(16,4))
        return frm

    # ── FILTROS AR ────────────────────────────────────────────────
    def _build_screen_filtros(self):
        frm = ctk.CTkFrame(self._container,fg_color="transparent")
        frm.grid_columnconfigure(0,weight=3)
        frm.grid_columnconfigure(1,weight=2)
        frm.grid_rowconfigure(0,weight=1)

        # Cámara con filtro aplicado
        left = self._card(frm)
        left.grid(row=0,column=0,padx=(0,6),pady=0,sticky="nsew")
        ctk.CTkLabel(left,text="FILTROS AR — CÁMARA EN VIVO",
            font=ctk.CTkFont(size=10,weight="bold"),
            text_color="#3a5a7a").pack(pady=(8,2))
        self._filt_cam = ctk.CTkLabel(left,text="",width=560,height=440)
        self._filt_cam.pack(padx=6,pady=(0,8))

        # Panel derecho: selección y estado
        right = self._card(frm)
        right.grid(row=0,column=1,padx=0,pady=0,sticky="nsew")
        self._slbl(right,"ELIGE UN FILTRO",14)
        ctk.CTkLabel(right,text="con dedos 1, 2 ó 3",
            font=ctk.CTkFont(size=10),text_color="#3a5a7a").pack(pady=(0,8))

        # Bloques de filtros
        self._filt_blocks = []
        filt_opts = [
            ("1","💎  Collar","#ff88ff","Collar con dije"),
            ("2","🕶  Lentes","#88ccff","Lentes de sol"),
            ("3","👨  Bigote","#ffaa44","Bigote clásico"),
        ]
        for num,emoji,col,desc in filt_opts:
            c = ctk.CTkFrame(right,fg_color="#0a1520",corner_radius=10,
                             border_width=2,border_color="#1e3048",height=72)
            c.pack(fill="x",padx=12,pady=4)
            c.pack_propagate(False)
            ctk.CTkLabel(c,text=num,
                font=ctk.CTkFont(size=24,weight="bold"),
                text_color=col).pack(side="left",padx=12)
            tf = ctk.CTkFrame(c,fg_color="transparent")
            tf.pack(side="left",fill="both",expand=True,pady=8)
            ctk.CTkLabel(tf,text=emoji,
                font=ctk.CTkFont(size=13,weight="bold"),
                text_color="#e0e0e0").pack(anchor="w")
            ctk.CTkLabel(tf,text=desc,
                font=ctk.CTkFont(size=10),text_color="#3a5a7a").pack(anchor="w")
            self._filt_blocks.append((c,col))

        ctk.CTkFrame(right,height=1,fg_color="#1e3048").pack(fill="x",padx=12,pady=10)

        # Filtro activo
        self._slbl(right,"FILTRO ACTIVO")
        self._filt_active_lbl = ctk.CTkLabel(right,text="Ninguno",
            font=ctk.CTkFont(size=18,weight="bold"),text_color="#888")
        self._filt_active_lbl.pack(pady=4)

        ctk.CTkFrame(right,height=1,fg_color="#1e3048").pack(fill="x",padx=12,pady=8)
        ctk.CTkLabel(right,
            text="Mantén 1/2/3 dedos (1.5s)\npara cambiar filtro\n0 dedos = sin filtro",
            font=ctk.CTkFont(size=11),text_color="#3a5a7a",justify="center").pack()
        ctk.CTkLabel(right,text="Puño 2s → Menú",
            font=ctk.CTkFont(size=10),text_color="#2a4a6a").pack(pady=(12,4))
        return frm

    def _highlight_filt(self, idx):
        """Resalta bloque de filtro activo (idx 1-3, 0=ninguno)."""
        for i,(c,col) in enumerate(self._filt_blocks):
            if i+1 == idx:
                c.configure(fg_color="#1a0d2a",border_color=col)
            else:
                c.configure(fg_color="#0a1520",border_color="#1e3048")

    # ──────────────────────────────────────────────────────────────
    # NAVEGACIÓN DE PANTALLAS
    # ──────────────────────────────────────────────────────────────
    def _show_screen(self, screen: str):
        # Ocultar todo
        for s,w in self._screens.items():
            w.pack_forget()
        # Mostrar la pedida
        self._screens[screen].pack(fill="both",expand=True)
        self._screen = screen

        # Al entrar a Señas siempre se ve primero el sub-selector
        if screen == Screen.SEÑAS and self._asl_mode != "SELECTOR":
            self._asl_mode = "SELECTOR"
            self._update_asl_subview()

        # Al entrar a Filtros resetear filtro activo
        if screen == Screen.FILTROS:
            self._filt.active_filter = 0
            self._filt_active_lbl.configure(text="Ninguno",text_color="#888")
            self._highlight_filt(0)

        # Activar/desactivar cámara rampa
        if screen == Screen.TRABAJO:
            if not self._ramp_active:
                self._ramp_active = True
                threading.Thread(target=self._cap1_thread,daemon=True).start()
        else:
            self._ramp_active = False

        # Actualizar etiqueta superior
        names = {Screen.MENU:"MENÚ", Screen.TRABAJO:"ÁREA DE TRABAJO",
                 Screen.BRILLO:"BRILLO", Screen.SEÑAS:"LENGUAJE DE SEÑAS",
                 Screen.CORPORAL:"ANÁLISIS CORPORAL",
                 Screen.FILTROS:"FILTROS AR"}
        self._lbl_screen.configure(text=f"● {names.get(screen,'—')}")
        self._log_event(f"Pantalla → {names.get(screen,'—')}")

    # ──────────────────────────────────────────────────────────────
    # HILOS DE CÁMARA
    # ──────────────────────────────────────────────────────────────
    def _cap0_thread(self):
        """Cámara laptop — siempre activa."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        iv = 1.0/MAX_FPS
        while self._running:
            t0=time.time()
            ok,frame=cap.read()
            if not ok:
                frame=np.zeros((FRAME_H,FRAME_W,3),dtype=np.uint8)
            self._last_frame0 = frame.copy()
            try: self._q_op.put_nowait(frame)
            except queue.Full: pass
            time.sleep(max(0.0,iv-(time.time()-t0)))
        cap.release()

    def _cap1_thread(self):
        """Cámara rampa — solo cuando Área de Trabajo activa."""
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        iv = 1.0/MAX_FPS
        while self._running and self._ramp_active:
            t0=time.time()
            ok,frame=cap.read()
            if not ok:
                frame=np.zeros((FRAME_H,FRAME_W,3),dtype=np.uint8)
                cv2.putText(frame,"SIN SEÑAL CAM 1",(120,240),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,180),2)
            ann,color,area,dest=self._ramp.process(frame)
            try: self._q_ramp.put_nowait((ann,color,area,dest))
            except queue.Full: pass
            time.sleep(max(0.0,iv-(time.time()-t0)))
        cap.release()

    # ──────────────────────────────────────────────────────────────
    # LOOP PRINCIPAL GUI
    # ──────────────────────────────────────────────────────────────
    def _update_gui(self):
        self._lbl_clock.configure(text=datetime.now().strftime("%H:%M:%S"))

        # Obtener frame de cam0
        try:
            raw = self._q_op.get_nowait()
        except queue.Empty:
            raw = None

        if raw is not None:
            # Gestos siempre procesados
            ann_gest, gdata = self._gest.process(raw)
            fingers = gdata["fingers"]
            self._lbl_fingers.configure(
                text=f"✋ {fingers}" if fingers>0 else "✋ —")

            # EAR siempre
            self._ear_val = self._ear_det.process(raw)

            # Palma (puño sostenido 2s) → volver al menú
            # Caso especial: si estamos en Señas y dentro de una subopción
            # (Letras/Palabras), el puño vuelve primero al sub-selector
            # de Señas, no directo al menú principal.
            if gdata["palm"]:
                if self._screen == Screen.SEÑAS and self._asl_mode != "SELECTOR":
                    self._asl_mode = "SELECTOR"
                    self._update_asl_subview()
                    self._log_event("Señas → sub-selector")
                elif self._screen != Screen.MENU:
                    self._show_screen(Screen.MENU)

            # Distribuir frame según pantalla activa
            self._dispatch_frame(raw, ann_gest, gdata, fingers)

        # Frame rampa (solo en trabajo)
        if self._screen == Screen.TRABAJO:
            try:
                ann_r,color,area,dest = self._q_ramp.get_nowait()
                img = self._to_ctk(ann_r,480,200)
                self._trab_cam_ramp.configure(image=img,text="")
                self._trab_cam_ramp.image = img
                if color:
                    self._det_color=color; self._det_dest=dest; self._det_area=area
                    dc={"A":"#ff5555","B":"#55cc55","C":"#5599ff"}.get(dest,"#fff")
                    self._trab_obj.configure(text=color,text_color="#ffcc00")
                    self._trab_dest.configure(text=f"Destino: {dest}",text_color=dc)
                else:
                    self._trab_obj.configure(text="Sin objeto",text_color="#444")
                    self._trab_dest.configure(text="Destino: —",text_color="#444")
            except queue.Empty:
                pass

        if self._running:
            self.after(50,self._update_gui)

    def _dispatch_frame(self, raw, ann_gest, gdata, fingers):
        scr = self._screen

        # ── MENÚ ──────────────────────────────────────────────────
        if scr == Screen.MENU:
            img = self._to_ctk(ann_gest, 640, 360)
            self._menu_cam.configure(image=img); self._menu_cam.image=img
            # Resaltar bloque según dedos
            if 1 <= fingers <= 5:
                self._highlight_menu(fingers)
            else:
                self._highlight_menu(0)
            # Navegar con N dedos mantenidos 1.5s
            fn = gdata.get("finger_nav", 0)
            if fn in (1,2,3,4,5):
                dest_map = {1:Screen.TRABAJO, 2:Screen.BRILLO,
                            3:Screen.SEÑAS,   4:Screen.CORPORAL,
                            5:Screen.FILTROS}
                self._show_screen(dest_map[fn])

        # ── TRABAJO ───────────────────────────────────────────────
        elif scr == Screen.TRABAJO:
            # Reconocimiento facial con ✌
            if gdata["peace_conf"]:
                self._face_active = True
                self._log_event("Reconocimiento facial activado")

            if self._face_active:
                now=time.time()
                if now-self._last_face_t >= FACE_INTERVAL:
                    self._last_face_t=now
                    threading.Thread(
                        target=self._run_face,args=(raw.copy(),),daemon=True).start()

            # Dibujar bbox cara si existe
            frame_show = ann_gest.copy()
            if self._face_box:
                x,y,bw,bh=self._face_box
                col=(0,220,80) if self._op_auth else (0,60,220)
                cv2.rectangle(frame_show,(x,y),(x+bw,y+bh),col,2)
                cv2.putText(frame_show,self._op_name,(x,y-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,col,2)

            img=self._to_ctk(frame_show,480,300)
            self._trab_cam_op.configure(image=img); self._trab_cam_op.image=img

            # Actualizar estado clasificación
            clf_cols={St.BLOQUEADO:"#00d4ff",St.DETECCION:"#ffcc00",
                      St.ESPERA:"#ff8800",St.CLASIF:"#00ff88",St.CANCEL:"#ff4444"}
            self._trab_state_lbl.configure(
                text=f"● {self._clf_state}",
                text_color=clf_cols.get(self._clf_state,"#fff"))

            # Operador
            self._trab_op_name.configure(text=self._op_name)
            if self._op_auth:
                self._trab_auth.configure(text="⬤  AUTORIZADO",text_color="#00ff88")
            else:
                self._trab_auth.configure(text="⬤  NO AUTORIZADO",text_color="#ff4444")

            # EAR
            ear=self._ear_val
            self._trab_ear.configure(text=f"{ear:.2f}")
            self._trab_ear_bar.set(min(ear/0.4,1.0))
            if 0<ear<EAR_THRESH:
                self._trab_fat_warn.configure(text="⚠ FATIGA")
                self._trab_ear_bar.configure(progress_color="#ff4444")
            else:
                self._trab_fat_warn.configure(text="")
                self._trab_ear_bar.configure(progress_color="#00ff88")

            # Gestos funcionales
            if gdata["ok_conf"]:  self._on_ok()
            if gdata["cancel_conf"]: self._on_cancel()

        # ── BRILLO ────────────────────────────────────────────────
        elif scr == Screen.BRILLO:
            img=self._to_ctk(ann_gest,580,420)
            self._bri_cam.configure(image=img); self._bri_cam.image=img
            dist=gdata["pinch_dist"]
            self._bri_pinch.configure(text=f"Pellizco: {int(dist)} px")
            now=time.time()
            if now>self._pinch_cd and dist>0:
                if dist>80:
                    self._brightness=min(1.0,self._brightness+0.04)
                    self._pinch_cd=now+0.12
                elif dist<30:
                    self._brightness=max(0.2,self._brightness-0.04)
                    self._pinch_cd=now+0.12
            pct=int(self._brightness*100)
            self._bri_val.configure(text=f"{pct}%")
            self._bri_bar.set(self._brightness)

        # ── SEÑAS ─────────────────────────────────────────────────
        elif scr == Screen.SEÑAS:
            img=self._to_ctk(ann_gest,580,420)
            self._asl_cam.configure(image=img); self._asl_cam.image=img

            if self._asl_mode == "SELECTOR":
                # Sub-menú propio: solo acepta 1, 2 o puño (puño ya se
                # maneja arriba, a nivel global, y nos saca al menú
                # principal porque aquí no hay subopción activa). No se
                # lee ASL en esta pantalla, solo navegación por dedos.
                fn = gdata.get("finger_nav", 0)
                if fn == 1:
                    self._asl_mode = "LETRAS"
                    self._update_asl_subview()
                    self._log_event("Señas → subopción Letras")
                elif fn == 2:
                    self._asl_mode = "PALABRAS"
                    self._update_asl_subview()
                    self._log_event("Señas → subopción Palabras")

            elif self._asl_mode == "LETRAS":
                asl=gdata["asl_letter"]
                descs={"A":"Puño cerrado","B":"Todos los dedos","C":"Forma curva",
                       "D":"Índice + pulgar en medio","I":"Solo meñique",
                       "O":"Dedos en círculo","X":"Índice en gancho",
                       "L":"Índice + pulgar","Y":"Pulgar + meñique",
                       "1":"Solo índice","2":"Índice + medio","3":"3 dedos"}
                if asl:
                    self._asl_letter.configure(text=asl,text_color="#00d4ff")
                    self._asl_desc.configure(text=descs.get(asl,""))
                else:
                    self._asl_letter.configure(text="—",text_color="#2a4a6a")
                    self._asl_desc.configure(text="")

            else:  # ── PALABRAS ──────────────────────────────────
                asl=gdata["asl_letter"]
                now = time.time()

                # Acumular letra sostenida 1.5s
                if asl != self._asl_last_letter:
                    self._asl_last_letter = asl
                    self._asl_t0_letter   = now if asl else None

                if asl and self._asl_t0_letter:
                    hold = now - self._asl_t0_letter
                    pct  = min(hold/self._asl_letter_hold, 1.0)
                    self._asl_word_progress.set(pct)
                    self._asl_word_letter_lbl.configure(
                        text=f"Sosteniendo: {asl}  ({int(pct*100)}%)")
                    if hold >= self._asl_letter_hold:
                        self._asl_word += asl
                        self._asl_t0_letter = now  # evita repetir de inmediato
                        self._log_event(f"Palabra: '{self._asl_word}' (+{asl})")
                else:
                    self._asl_word_progress.set(0)
                    self._asl_word_letter_lbl.configure(text="")

                # Pellizco corto (<30px) → borrar última letra
                dist = gdata["pinch_dist"]
                if 0 < dist < 30 and now > self._asl_pinch_cd:
                    if self._asl_word:
                        self._asl_word = self._asl_word[:-1]
                        self._log_event(f"Palabra: '{self._asl_word}' (borrar)")
                    self._asl_pinch_cd = now + 0.6

                # Pellizco largo (>120px sostenido) → limpiar toda la palabra
                if dist > 120 and now > self._asl_pinch_cd:
                    if self._asl_word:
                        self._asl_word = ""
                        self._log_event("Palabra borrada por completo")
                    self._asl_pinch_cd = now + 1.0

                self._asl_word_lbl.configure(text=self._asl_word or "—")

        # ── CORPORAL ──────────────────────────────────────────────
        elif scr == Screen.CORPORAL:
            # Análisis corporal con MediaPipe Pose
            frame_corp = raw.copy()
            if self._brightness < 1.0:
                frame_corp=cv2.convertScaleAbs(frame_corp,alpha=self._brightness)
            ann_body,binfo=self._body.process(frame_corp)
            # Dibujar landmarks de manos también
            ann_both=ann_body.copy()
            img=self._to_ctk(ann_both,580,420)
            self._corp_cam.configure(image=img); self._corp_cam.image=img
            if binfo:
                self._corp_labels["ROPA"].configure(text=binfo.get("ropa","—"))
                self._corp_labels["CABELLO COLOR"].configure(
                    text=binfo.get("cabello_color","—"))
                co=binfo.get("cabello_orden","—")
                self._corp_labels["CABELLO"].configure(
                    text=co,
                    text_color="#00ff88" if "Ordenado" in co else "#ff8800")
                self._corp_labels["TONO DE PIEL"].configure(
                    text=binfo.get("piel","—"))
                post=binfo.get("postura","—")
                self._corp_labels["POSTURA"].configure(
                    text=post,
                    text_color="#00ff88" if binfo.get("postura_ok") else "#ff4444")

        # ── FILTROS AR ────────────────────────────────────────────
        elif scr == Screen.FILTROS:
            # Cambiar filtro con N dedos mantenidos 1.5s (1,2,3)
            fn = gdata.get("finger_nav", 0)
            if fn in (1, 2, 3):
                self._filt.active_filter = fn
                names_f = {1:"💎 Collar", 2:"🕶 Lentes", 3:"👨 Bigote"}
                self._filt_active_lbl.configure(
                    text=names_f[fn], text_color="#ff88ff")
                self._highlight_filt(fn)
                self._log_event(f"Filtro activo: {names_f[fn]}")
            elif fn == 0 and fingers == 0:
                # Sin dedos extendidos → quitar filtro
                pass  # mantiene el último activo

            # Resaltar bloque hovereado
            if 1 <= fingers <= 3:
                self._highlight_filt(fingers)

            # Aplicar filtro al frame y mostrar
            frame_filt = self._filt.process(raw)
            img = self._to_ctk(frame_filt, 560, 440)
            self._filt_cam.configure(image=img); self._filt_cam.image=img

    # ──────────────────────────────────────────────────────────────
    # RECONOCIMIENTO FACIAL (hilo)
    # ──────────────────────────────────────────────────────────────
    def _run_face(self, frame):
        name,_,box=self._face_rec.recognize(frame)
        self._op_name  = name
        self._op_auth  = name != "Desconocido"
        self._face_box = box

    # ──────────────────────────────────────────────────────────────
    # CLASIFICACIÓN (máquina de estados)
    # ──────────────────────────────────────────────────────────────
    def _on_ok(self):
        if not self._op_auth:
            self._log_event("Operador no autorizado"); return
        s=self._clf_state
        if s==St.BLOQUEADO:
            self._set_clf(St.DETECCION)
        elif s==St.DETECCION:
            if self._det_color:
                self._log_event(f"Objeto: {self._det_color}->{self._det_dest}")
                self._set_clf(St.ESPERA)
            else:
                self._log_event("Sin objeto en rampa")
        elif s==St.ESPERA:
            self._set_clf(St.CLASIF)
            threading.Thread(target=self._do_classify,daemon=True).start()

    def _on_cancel(self):
        s=self._clf_state
        if s in (St.DETECCION,St.ESPERA):
            self._set_clf(St.CANCEL)
            self.after(2000,lambda:self._set_clf(St.BLOQUEADO))
        elif s==St.CLASIF:
            threading.Thread(
                target=lambda:self._serial.send_command("X"),daemon=True).start()
            self._log_event("EMERGENCIA — X al Arduino")
            self._set_clf(St.CANCEL)
            self.after(2000,lambda:self._set_clf(St.BLOQUEADO))

    def _set_clf(self,s):
        self._clf_state=s
        self._log_event(f"Clasificación → {s}")

    def _do_classify(self):
        cmd=self._det_dest
        self._log_event(f"Enviando '{cmd}'...")
        ack=self._serial.send_command(cmd)
        if ack:
            self._cnt[cmd]+=1
            total=sum(self._cnt.values())
            self.after(0,lambda c=cmd,t=total:self._upd_cnt(c,t))
            self._log_event(f"ACK — '{cmd}' OK. Total:{total}")
        else:
            self._log_event(f"Sin ACK para '{cmd}'")
        self.after(600,lambda:self._set_clf(St.BLOQUEADO))

    def _upd_cnt(self,cmd,total):
        self._trab_cnt[cmd].configure(text=str(self._cnt[cmd]))
        self._trab_cnt["T"].configure(text=str(total))

    # ──────────────────────────────────────────────────────────────
    # UTILIDADES
    # ──────────────────────────────────────────────────────────────
    def _try_serial(self):
        ports=SerialManager.list_ports()
        self._log_event(f"Puertos: {', '.join(ports) or '(ninguno)'}")
        if self._serial.connect():
            self._lbl_serial_st.configure(
                text=f"● {SERIAL_PORT} OK",text_color="#00ff88")
            self._log_event(f"Serial OK en {SERIAL_PORT}")
        else:
            self._lbl_serial_st.configure(
                text="● Sin Serial (sim)",text_color="#ff8800")
            self._log_event("Serial NO conectado — simulación activa")

    def _log_event(self,msg):
        ts=datetime.now().strftime("%H:%M:%S")
        self._log_box.configure(state="normal")
        self._log_box.insert("end",f"[{ts}] {msg}\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _to_ctk(self,frame_bgr,w,h):
        if self._brightness<1.0:
            frame_bgr=cv2.convertScaleAbs(frame_bgr,alpha=self._brightness)
        rgb=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
        img=Image.fromarray(rgb)
        return ctk.CTkImage(light_image=img,dark_image=img,size=(w,h))

    def _on_close(self):
        self._running=False; self._ramp_active=False
        self._gest.close(); self._ear_det.close()
        self._body.close(); self._filt.close(); self._serial.close()
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════
if __name__=="__main__":
    app=App(); app.mainloop()
