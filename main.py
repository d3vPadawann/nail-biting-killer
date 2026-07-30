import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import math

# IMPORTAÇÃO DIRETA: Resolve o erro "has no attribute 'solutions'" no Python 3.12
from mediapipe.python.solutions import holistic as mp_holistic

# ─── Constantes ────────────────────────────────────────────────────────────────
THRESHOLD       = 0.08
HEART_THRESHOLD = 0.15
DEBOUNCE_SECS   = 0.5

COLOR_OK    = (80, 220, 0)
COLOR_WARN  = (0, 165, 255)
COLOR_ALERT = (0, 0, 230)
COLOR_BOX   = (0, 215, 255)
COLOR_MOUTH = (255, 255, 0)
COLOR_TIP   = (200, 80, 255)
COLOR_HEART = (147, 20, 255)  # Rosa / Magenta vibrante
FONT        = cv2.FONT_HERSHEY_SIMPLEX

# ─── Estado compartilhado entre threads ───────────────────────────────────────
shared = {
    "is_alert": False,
    "is_heart": False,
    "frame":    None,
    "lock":     threading.Lock(),
}

# ─── Helpers de desenho ────────────────────────────────────────────────────────

def lm_to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def bbox(landmarks, w, h, pad=12):
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    return max(0, min(xs)-pad), max(0, min(ys)-pad), \
           min(w, max(xs)+pad), min(h, max(ys)+pad)

def rounded_rect(img, x1, y1, x2, y2, color, t=2, r=8):
    cv2.line(img, (x1+r, y1), (x2-r, y1), color, t)
    cv2.line(img, (x1+r, y2), (x2-r, y2), color, t)
    cv2.line(img, (x1, y1+r), (x1, y2-r), color, t)
    cv2.line(img, (x2, y1+r), (x2, y2-r), color, t)
    cv2.ellipse(img, (x1+r, y1+r), (r,r), 180, 0, 90, color, t)
    cv2.ellipse(img, (x2-r, y1+r), (r,r), 270, 0, 90, color, t)
    cv2.ellipse(img, (x1+r, y2-r), (r,r),  90, 0, 90, color, t)
    cv2.ellipse(img, (x2-r, y2-r), (r,r),   0, 0, 90, color, t)

def label_box(img, text, x, y, color, fs=0.45, t=1):
    (tw, th), bl = cv2.getTextSize(text, FONT, fs, t)
    cv2.rectangle(img, (x, y-th-5), (x+tw+8, y+bl), color, -1)
    cv2.putText(img, text, (x+4, y-1), FONT, fs, (0,0,0), t, cv2.LINE_AA)

def overlay_text(img, lines, sy=28, fs=0.5):
    for i, line in enumerate(lines):
        y = sy + i * 22
        cv2.putText(img, line, (11, y+1), FONT, fs, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img, line, (10, y),   FONT, fs, (255,255,255), 1, cv2.LINE_AA)

def dist(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def detect_heart_gesture(left_hand_lms, right_hand_lms):
    """Detecta se as duas mãos juntas estão formando o gesto de coração."""
    if left_hand_lms is None or right_hand_lms is None:
        return False, None

    lh = left_hand_lms.landmark
    rh = right_hand_lms.landmark

    # Index 8 (Indicador) e Index 4 (Polegar)
    d_index = dist(lh[8], rh[8])
    d_thumb = dist(lh[4], rh[4])

    if d_index < HEART_THRESHOLD and d_thumb < HEART_THRESHOLD:
        cx_norm = (lh[8].x + rh[8].x + lh[4].x + rh[4].x) / 4.0
        cy_norm = (lh[8].y + rh[8].y + lh[4].y + rh[4].y) / 4.0
        return True, (cx_norm, cy_norm)

    return False, None

def get_heart_points(cx, cy, scale):
    """Gera lista de pontos (x, y) de um coração paramétrico."""
    pts = []
    steps = 60
    for i in range(steps):
        t = 2 * math.pi * i / steps
        # Equação paramétrica do coração
        x = 16 * (math.sin(t) ** 3)
        y = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
        px = int(cx + x * scale)
        py = int(cy + y * scale)
        pts.append([px, py])
    return np.array(pts, dtype=np.int32)

import numpy as np

def draw_giant_heart(img, center_px, now):
    """Desenha um coração gigante animado com brilho e efeito pulsante no frame OpenCV."""
    h, w = img.shape[:2]
    cx, cy = center_px

    # Pulsação suave (heartbeat)
    pulse = 1.0 + 0.12 * math.sin(now * 7.0)
    base_scale = min(w, h) / 38.0 * pulse

    # Overlay para transparência e brilho
    overlay = img.copy()

    # 1. Glow externo (coração maior translúcido)
    pts_glow = get_heart_points(cx, cy, base_scale * 1.25)
    cv2.fillPoly(overlay, [pts_glow], (255, 80, 200))

    # 2. Coração principal preenchido
    pts_main = get_heart_points(cx, cy, base_scale)
    cv2.fillPoly(overlay, [pts_main], (147, 20, 255))

    # 3. Contorno brilhante
    cv2.polylines(overlay, [pts_main], True, (255, 200, 255), 3, cv2.LINE_AA)

    # 4. Núcleo interno destacado
    pts_inner = get_heart_points(cx, cy, base_scale * 0.55)
    cv2.fillPoly(overlay, [pts_inner], (200, 100, 255))

    # Aplica transparência
    alpha = 0.70 + 0.15 * math.sin(now * 7.0)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

    # Texto comemorativo sobre o coração
    msg = "💖 CORACÃO DETECTADO! 💖"
    (tw, th), _ = cv2.getTextSize(msg, FONT, 0.85, 2)
    tx, ty = (w - tw) // 2, max(40, cy - int(base_scale * 16))
    cv2.putText(img, msg, (tx+2, ty+2), FONT, 0.85, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, msg, (tx, ty), FONT, 0.85, (255, 180, 255), 2, cv2.LINE_AA)


# ─── Thread de visão ──────────────────────────────────────────────────────────

def vision_thread():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hand_on_mouth_start = None
    prev_t = time.time()

    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            now   = time.time()
            fps   = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            hand_near_mouth = False
            close_tip_px    = None
            mouth_px        = None

            # ── Rosto ─────────────────────────────────────────────────────────
            if results.face_landmarks:
                lms = results.face_landmarks.landmark
                x1, y1, x2, y2 = bbox(lms, w, h, pad=15)
                rounded_rect(frame, x1, y1, x2, y2, COLOR_BOX)
                label_box(frame, "Face", x1, y1, COLOR_BOX)

                mouth_lm = lms[13]
                mouth_px = lm_to_px(mouth_lm, w, h)
                cv2.circle(frame, mouth_px, 6, COLOR_MOUTH, -1)
                cv2.circle(frame, mouth_px, 8, (0,0,0), 1)
                label_box(frame, "Boca", mouth_px[0]+10, mouth_px[1], COLOR_MOUTH)

            # ── Mãos ──────────────────────────────────────────────────────────
            for hand_lms, hand_name in [
                (results.left_hand_landmarks,  "Mao Esq"),
                (results.right_hand_landmarks, "Mao Dir"),
            ]:
                if hand_lms is None:
                    continue
                lms = hand_lms.landmark

                for idx in [4, 8, 12, 16, 20]:
                    tip = lms[idx]
                    if results.face_landmarks:
                        d = dist(results.face_landmarks.landmark[13], tip)
                        if d < THRESHOLD:
                            hand_near_mouth = True
                            close_tip_px    = lm_to_px(tip, w, h)

                alert_now = shared["is_alert"]
                bc = COLOR_ALERT if (hand_near_mouth and alert_now) else \
                     COLOR_WARN  if hand_near_mouth else COLOR_BOX

                x1, y1, x2, y2 = bbox(lms, w, h)
                rounded_rect(frame, x1, y1, x2, y2, bc)
                label_box(frame, hand_name, x1, y1, bc)

                for idx in [4, 8, 12, 16, 20]:
                    cv2.circle(frame, lm_to_px(lms[idx], w, h), 4, (200,200,255), -1)

            if close_tip_px and mouth_px:
                cv2.circle(frame, close_tip_px, 10, COLOR_TIP, 2)
                cv2.line(frame, mouth_px, close_tip_px, COLOR_TIP, 1)

            # ── Detecção de gesto: Coração com duas mãos ──────────────────────
            is_heart, center_norm = detect_heart_gesture(
                results.left_hand_landmarks, results.right_hand_landmarks
            )
            shared["is_heart"] = is_heart

            if is_heart and center_norm:
                cx_px = int(center_norm[0] * w)
                cy_px = int(center_norm[1] * h)

                # Desenha ligações sutis entre os indicadores e polegares das duas mãos
                lh8 = lm_to_px(results.left_hand_landmarks.landmark[8], w, h)
                rh8 = lm_to_px(results.right_hand_landmarks.landmark[8], w, h)
                lh4 = lm_to_px(results.left_hand_landmarks.landmark[4], w, h)
                rh4 = lm_to_px(results.right_hand_landmarks.landmark[4], w, h)

                cv2.line(frame, lh8, rh8, COLOR_HEART, 2)
                cv2.line(frame, lh4, rh4, COLOR_HEART, 2)

                # Desenha o coração gigante no centro do gesto no frame da câmera
                draw_giant_heart(frame, (cx_px, cy_px), now)

            # ── Debounce ──────────────────────────────────────────────────────
            if hand_near_mouth:
                if hand_on_mouth_start is None:
                    hand_on_mouth_start = now
                elif now - hand_on_mouth_start >= DEBOUNCE_SECS:
                    shared["is_alert"] = True
            else:
                hand_on_mouth_start = None
                shared["is_alert"]  = False

            # ── Barra de progresso ────────────────────────────────────────────
            if hand_near_mouth and hand_on_mouth_start is not None:
                elapsed = min(now - hand_on_mouth_start, DEBOUNCE_SECS)
                bar_w   = int((w - 20) * elapsed / DEBOUNCE_SECS)
                cv2.rectangle(frame, (10, h-25), (w-10, h-10), (60,60,60), -1)
                bc2 = COLOR_ALERT if shared["is_alert"] else COLOR_WARN
                cv2.rectangle(frame, (10, h-25), (10+bar_w, h-10), bc2, -1)
                label_box(frame, f"Debounce {elapsed:.1f}s/{DEBOUNCE_SECS:.0f}s",
                          10, h-28, (70,70,70), fs=0.38)

            # ── HUD / overlay de alerta ────────────────────────────────────────
            status = "ALERTA!" if shared["is_alert"] else \
                     "CORAÇÃO 💖" if is_heart else \
                     "Mao perto" if hand_near_mouth else "OK"
            sc = COLOR_ALERT if shared["is_alert"] else \
                 COLOR_HEART if is_heart else \
                 COLOR_WARN  if hand_near_mouth else COLOR_OK

            cv2.rectangle(frame, (0,0), (w,5), sc, -1)

            if shared["is_alert"]:
                ov = frame.copy()
                cv2.rectangle(ov, (0,0), (w,h), (0,0,180), -1)
                alpha = 0.20 + 0.10 * math.sin(now * 8)
                cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
                msg = "TIRE A MAO DA BOCA!"
                (tw, th), _ = cv2.getTextSize(msg, FONT, 1.0, 2)
                tx, ty = (w-tw)//2, h//2
                cv2.putText(frame, msg, (tx+2, ty+2), FONT, 1.0, (0,0,0),      3, cv2.LINE_AA)
                cv2.putText(frame, msg, (tx,   ty),   FONT, 1.0, (255,255,255), 2, cv2.LINE_AA)

            overlay_text(frame, [
                f"FPS: {fps:.1f}",
                f"Face:    {'SIM' if results.face_landmarks else 'NAO'}",
                f"Mao esq: {'SIM' if results.left_hand_landmarks else 'NAO'}",
                f"Mao dir: {'SIM' if results.right_hand_landmarks else 'NAO'}",
                f"Coração: {'SIM 💖' if is_heart else 'NAO'}",
                f"Status:  {status}",
            ])

            with shared["lock"]:
                shared["frame"] = frame.copy()

    cap.release()


# ─── Painel de controle (janela raiz, sempre visível) ────────────────────────

class ControlPanel:
    """Pequeno painel flutuante sempre visível.
    Permite abrir/fechar a janela de debug sem encerrar o app."""

    BG      = "#0f0f1a"
    BG2     = "#1a1a2e"
    ACCENT  = "#4f8ef7"

    def __init__(self, root: tk.Tk, debug_win: "DebugWindow", alert_win: "AlertWindow", heart_win: "HeartOverlayWindow"):
        self.root      = root
        self.debug_win = debug_win
        self.alert_win = alert_win
        self.heart_win = heart_win

        root.title("HandScanner")
        root.configure(bg=self.BG)
        root.resizable(False, False)
        root.attributes("-topmost", True)

        # ── Cabeçalho ────────────────────────────────────────────────────────
        hdr = tk.Frame(root, bg=self.BG2, height=38)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🖐  HandScanner",
                 bg=self.BG2, fg="#ddeeff",
                 font=("Segoe UI", 11, "bold")).pack(side="left", padx=12, pady=8)

        # ── Indicador de status ───────────────────────────────────────────────
        mid = tk.Frame(root, bg=self.BG, pady=10)
        mid.pack(fill="x", padx=14)

        self._dot = tk.Label(mid, text="●", fg="#44ff88",
                             bg=self.BG, font=("Segoe UI", 18))
        self._dot.pack(side="left")

        self._status_lbl = tk.Label(mid, text="Monitorando...",
                                    fg="#aabbcc", bg=self.BG,
                                    font=("Segoe UI", 10))
        self._status_lbl.pack(side="left", padx=8)

        # ── Botões ────────────────────────────────────────────────────────────
        btn_frame = tk.Frame(root, bg=self.BG, pady=6)
        btn_frame.pack(fill="x", padx=14)

        self._debug_btn = tk.Button(
            btn_frame,
            text="🔍  Mostrar Debug",
            command=self._toggle_debug,
            bg=self.ACCENT, fg="white",
            activebackground="#6aa0ff", activeforeground="white",
            relief="flat", bd=0, padx=10, pady=6,
            font=("Segoe UI", 9, "bold"), cursor="hand2",
        )
        self._debug_btn.pack(fill="x", pady=(0, 6))

        quit_btn = tk.Button(
            btn_frame,
            text="✕  Encerrar app",
            command=self._quit,
            bg="#3a1a1a", fg="#ff6666",
            activebackground="#5a2a2a", activeforeground="#ff8888",
            relief="flat", bd=0, padx=10, pady=5,
            font=("Segoe UI", 9), cursor="hand2",
        )
        quit_btn.pack(fill="x")

        # ── Dica de teclado ───────────────────────────────────────────────────
        tk.Label(root, text="Ctrl+D: debug  ·  Ctrl+Q: sair",
                 bg=self.BG, fg="#333355",
                 font=("Segoe UI", 8)).pack(pady=(2, 8))

        root.bind("<Control-d>", lambda e: self._toggle_debug())
        root.bind("<Control-q>", lambda e: self._quit())

        self._poll()

    # ── Lógica dos botões ─────────────────────────────────────────────────────

    def _toggle_debug(self):
        if self.debug_win.visible:
            self.debug_win.hide()
            self._debug_btn.config(text="🔍  Mostrar Debug")
        else:
            self.debug_win.show()
            self._debug_btn.config(text="🙈  Ocultar Debug")

    def _quit(self):
        self.root.quit()

    # ── Atualização periódica do indicador ────────────────────────────────────

    def _poll(self):
        if shared["is_alert"]:
            self._dot.config(fg="#ff3333")
            self._status_lbl.config(text="ALERTA – Mão na boca!", fg="#ff5555")
        elif shared["is_heart"]:
            self._dot.config(fg="#ff33aa")
            self._status_lbl.config(text="💖 Coração Detectado!", fg="#ff77cc")
        else:
            self._dot.config(fg="#44ff88")
            self._status_lbl.config(text="Monitorando...", fg="#aabbcc")

        # Também gerencia as janelas de overlay a partir daqui
        self.alert_win.update_visibility()
        self.heart_win.update_visibility()

        self.root.after(100, self._poll)


# ─── Janela de debug (pode ser fechada/reaberta) ─────────────────────────────

class DebugWindow:
    def __init__(self, root: tk.Tk):
        self.root    = root
        self.visible = True

        win = tk.Toplevel(root)
        win.title("HandScanner – Debug View")
        win.configure(bg="#1a1a2e")
        win.resizable(False, False)
        # Fechar a janela apenas a OCULTA — não encerra o app
        win.protocol("WM_DELETE_WINDOW", self.hide)
        win.bind("<Control-d>", lambda e: self.hide())
        self._win = win

        # Cabeçalho
        hdr = tk.Frame(win, bg="#16213e")
        hdr.pack(fill="x")
        tk.Label(hdr, text="🔍  Debug View  –  MediaPipe",
                 bg="#16213e", fg="#ddeeff",
                 font=("Segoe UI", 10, "bold")).pack(side="left", padx=12, pady=7)
        tk.Label(hdr, text="Ctrl+D para ocultar",
                 bg="#16213e", fg="#445566",
                 font=("Segoe UI", 8)).pack(side="right", padx=12)

        # Canvas da câmera
        self._canvas = tk.Canvas(win, width=640, height=480,
                                 bg="#000", highlightthickness=0)
        self._canvas.pack()

        # Rodapé de status
        footer = tk.Frame(win, bg="#16213e")
        footer.pack(fill="x")
        self._lbl = tk.Label(footer, text="● Iniciando...",
                             bg="#16213e", fg="#888",
                             font=("Segoe UI", 9))
        self._lbl.pack(side="left", padx=12, pady=6)

        self._photo = None
        self._refresh()

    def _refresh(self):
        with shared["lock"]:
            frame = shared["frame"]

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            self._photo = ImageTk.PhotoImage(image=img)
            self._canvas.create_image(0, 0, anchor="nw", image=self._photo)

        if shared["is_alert"]:
            self._lbl.config(text="🔴 ALERTA – Mão na boca!", fg="#ff4444")
        elif shared["is_heart"]:
            self._lbl.config(text="💖 CORAÇÃO DETECTADO!", fg="#ff55cc")
        elif frame is not None:
            self._lbl.config(text="🟢 Monitorando...", fg="#44ff88")

        # Agenda próxima atualização apenas se a janela estiver visível
        if self.visible:
            self._win.after(33, self._refresh)

    def show(self):
        self.visible = True
        self._win.deiconify()
        self._win.lift()
        self._refresh()   # retoma o loop de atualização

    def hide(self):
        self.visible = False
        self._win.withdraw()


# ─── Janela de alerta fullscreen ─────────────────────────────────────────────

class AlertWindow:
    def __init__(self, root: tk.Tk):
        win = tk.Toplevel(root)
        win.title("ALERTA")
        win.attributes("-fullscreen", True)
        win.attributes("-topmost", True)
        try:
            win.attributes("-alpha", 0.88)
        except Exception:
            pass
        win.configure(bg="red")
        tk.Label(win, text="TIRE A MÃO DA BOCA!",
                 font=("Arial", 60, "bold"),
                 fg="white", bg="red").pack(expand=True)
        win.withdraw()
        self._win = win

    def update_visibility(self):
        if shared["is_alert"]:
            if self._win.state() == "withdrawn":
                self._win.deiconify()
        else:
            if self._win.state() == "normal":
                self._win.withdraw()


# ─── Overlay de Coração Fullscreen ────────────────────────────────────────────

class HeartOverlayWindow:
    """Janela popup/overlay que exibe um coração gigante pulsante no centro da tela
    quando o gesto de duas mãos em forma de coração é reconhecido."""

    def __init__(self, root: tk.Tk):
        win = tk.Toplevel(root)
        win.title("Coração Detectado")
        win.attributes("-fullscreen", True)
        win.attributes("-topmost", True)
        try:
            win.attributes("-alpha", 0.85)
        except Exception:
            pass
        win.configure(bg="#110515")

        self.canvas = tk.Canvas(win, bg="#110515", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        win.withdraw()
        self._win = win
        self._anim_step = 0
        self._is_animating = False

    def update_visibility(self):
        if shared["is_heart"]:
            if self._win.state() == "withdrawn":
                self._win.deiconify()
                self._win.lift()
                if not self._is_animating:
                    self._is_animating = True
                    self._animate()
        else:
            if self._win.state() == "normal":
                self._is_animating = False
                self._win.withdraw()

    def _animate(self):
        if not self._is_animating:
            return

        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 1280
        h = self.canvas.winfo_height() or 720
        cx, cy = w // 2, h // 2

        self._anim_step += 0.12
        pulse = 1.0 + 0.15 * math.sin(self._anim_step)
        scale = (min(w, h) / 32.0) * pulse

        # Desenha brilho externo do coração
        pts_glow = []
        steps = 80
        scale_glow = scale * 1.2
        for i in range(steps):
            t = 2 * math.pi * i / steps
            x = 16 * (math.sin(t) ** 3)
            y = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
            pts_glow.extend([cx + x * scale_glow, cy + y * scale_glow])

        # Desenha o coração principal
        pts = []
        for i in range(steps):
            t = 2 * math.pi * i / steps
            x = 16 * (math.sin(t) ** 3)
            y = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
            pts.extend([cx + x * scale, cy + y * scale])

        self.canvas.create_polygon(pts_glow, fill="#ff33aa", outline="")
        self.canvas.create_polygon(pts, fill="#e6005c", outline="#ff99dd", width=4)

        # Mensagem centralizada
        self.canvas.create_text(
            cx, cy + scale * 18,
            text="💖 CORAÇÃO DETECTADO! 💖",
            font=("Segoe UI", 32, "bold"),
            fill="#ffffff"
        )
        self.canvas.create_text(
            cx, cy + scale * 18 + 45,
            text="Gesto reconhecido com sucesso!",
            font=("Segoe UI", 16),
            fill="#ffb3da"
        )

        if self._is_animating:
            self._win.after(33, self._animate)


# ─── Ponto de entrada ─────────────────────────────────────────────────────────

def main():
    t = threading.Thread(target=vision_thread, daemon=True)
    t.start()

    root = tk.Tk()

    alert_win = AlertWindow(root)
    heart_win = HeartOverlayWindow(root)
    debug_win = DebugWindow(root)
    panel     = ControlPanel(root, debug_win, alert_win, heart_win)

    root.mainloop()


if __name__ == "__main__":
    main()