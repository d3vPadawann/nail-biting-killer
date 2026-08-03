import cv2
import math
import time
import threading
import numpy as np

# IMPORTAÇÃO DIRETA: Resolve o erro "has no attribute 'solutions'" no Python 3.12
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles

# ─── Constantes de aparência ───────────────────────────────────────────────────
THRESHOLD        = 0.08        # mesma distância do main.py
HEART_THRESHOLD  = 0.15        # distância máxima para formar o coração
DEBOUNCE_SECS    = 1.0

COLOR_NORMAL     = (0, 220, 80)    # verde  – tudo ok
COLOR_WARN       = (0, 165, 255)   # laranja – mão próxima, aguardando debounce
COLOR_ALERT      = (0, 0, 230)     # vermelho – alerta ativo!
COLOR_BOX        = (255, 215, 0)   # dourado – bounding boxes
COLOR_MOUTH_PT   = (0, 255, 255)   # ciano   – ponto da boca
COLOR_FINGER_TIP = (255, 80, 200)  # rosa    – ponta do dedo próxima
COLOR_HEART      = (147, 20, 255)  # magenta vibrante – coração
FONT             = cv2.FONT_HERSHEY_SIMPLEX

# ─── Funções auxiliares ────────────────────────────────────────────────────────

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def landmark_to_px(landmark, w, h):
    """Converte coordenadas normalizadas [0,1] em pixels."""
    return int(landmark.x * w), int(landmark.y * h)


def get_bounding_box(landmarks, w, h, padding=10):
    """Retorna (x1, y1, x2, y2) em pixels do bounding box de um conjunto de landmarks."""
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    return (
        max(0, min(xs) - padding),
        max(0, min(ys) - padding),
        min(w, max(xs) + padding),
        min(h, max(ys) + padding),
    )


def draw_rounded_rect(img, x1, y1, x2, y2, color, thickness=2, radius=8):
    """Desenha um retângulo com cantos arredondados."""
    # Lados retos
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    # Cantos arredondados
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90,  color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90,  color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90,  0, 90,  color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0,   0, 90,  color, thickness)


def draw_label_box(img, text, x, y, color, font_scale=0.5, thickness=1):
    """Desenha etiqueta com fundo sólido."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 8, y + baseline), color, -1)
    cv2.putText(img, text, (x + 4, y - 2), FONT, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def draw_overlay_text(img, lines, start_y=30, font_scale=0.55, color=(255, 255, 255)):
    """Escreve várias linhas de texto no canto superior esquerdo com sombra."""
    for i, line in enumerate(lines):
        y = start_y + i * 22
        # sombra
        cv2.putText(img, line, (11, y + 1), FONT, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        # texto
        cv2.putText(img, line, (10, y), FONT, font_scale, color, 1, cv2.LINE_AA)


def detect_heart_gesture(left_hand_lms, right_hand_lms):
    """Detecta se as duas mãos juntas estão formando o gesto de coração."""
    if left_hand_lms is None or right_hand_lms is None:
        return False, None

    lh = left_hand_lms.landmark
    rh = right_hand_lms.landmark

    # Index 8 (Indicador) e Index 4 (Polegar)
    d_index = calculate_distance(lh[8], rh[8])
    d_thumb = calculate_distance(lh[4], rh[4])

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
        x = 16 * (math.sin(t) ** 3)
        y = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
        px = int(cx + x * scale)
        py = int(cy + y * scale)
        pts.append([px, py])
    return np.array(pts, dtype=np.int32)


def draw_giant_heart(img, center_px, now):
    """Desenha um coração gigante animado com brilho e efeito pulsante no frame OpenCV."""
    h, w = img.shape[:2]
    cx, cy = center_px

    pulse = 1.0 + 0.12 * math.sin(now * 7.0)
    base_scale = min(w, h) / 38.0 * pulse

    overlay = img.copy()

    # 1. Glow externo
    pts_glow = get_heart_points(cx, cy, base_scale * 1.25)
    cv2.fillPoly(overlay, [pts_glow], (255, 80, 200))

    # 2. Coração principal
    pts_main = get_heart_points(cx, cy, base_scale)
    cv2.fillPoly(overlay, [pts_main], (147, 20, 255))

    # 3. Contorno
    cv2.polylines(overlay, [pts_main], True, (255, 200, 255), 3, cv2.LINE_AA)

    # 4. Núcleo interno
    pts_inner = get_heart_points(cx, cy, base_scale * 0.55)
    cv2.fillPoly(overlay, [pts_inner], (200, 100, 255))

    alpha = 0.70 + 0.15 * math.sin(now * 7.0)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

    # Texto
    msg = "💖 CORACÃO DETECTADO! 💖"
    (tw, th), _ = cv2.getTextSize(msg, FONT, 0.85, 2)
    tx, ty = (w - tw) // 2, max(40, cy - int(base_scale * 16))
    cv2.putText(img, msg, (tx+2, ty+2), FONT, 0.85, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, msg, (tx, ty), FONT, 0.85, (255, 180, 255), 2, cv2.LINE_AA)


# ─── Loop principal ────────────────────────────────────────────────────────────

def run_debug_view(enable_heart: bool = True):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hand_on_mouth_start = None
    alert_state          = False

    prev_time = time.time()

    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                time.sleep(0.05)
                continue

            # Espelha o frame para uma visualização mais natural
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = holistic.process(image_rgb)

            # ── Calcular FPS ──────────────────────────────────────────────────
            now      = time.time()
            fps      = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            hand_near_mouth  = False
            close_finger_px  = None   # pixel (x,y) da ponta do dedo mais próxima

            # ── Rosto: bounding box + ponto da boca ──────────────────────────
            mouth_px = None
            if results.face_landmarks:
                lms = results.face_landmarks.landmark
                x1, y1, x2, y2 = get_bounding_box(lms, w, h, padding=15)
                draw_rounded_rect(frame, x1, y1, x2, y2, COLOR_BOX, thickness=2)
                draw_label_box(frame, "Face", x1, y1, COLOR_BOX)

                mouth_point = lms[13]   # lábio superior (ponto 13 do Face Mesh)
                mouth_px    = landmark_to_px(mouth_point, w, h)
                cv2.circle(frame, mouth_px, 6, COLOR_MOUTH_PT, -1)
                cv2.circle(frame, mouth_px, 8, (0, 0, 0), 1)
                draw_label_box(frame, "Boca", mouth_px[0] + 10, mouth_px[1], COLOR_MOUTH_PT)

            # ── Mãos: bounding boxes + verificar proximidade com boca ─────────
            hand_labels = [
                (results.left_hand_landmarks,  "Mao Esq"),
                (results.right_hand_landmarks, "Mao Dir"),
            ]

            for hand_landmarks, hand_label in hand_labels:
                if hand_landmarks is None:
                    continue

                lms   = hand_landmarks.landmark
                hx1, hy1, hx2, hy2 = get_bounding_box(lms, w, h, padding=12)

                # Checa pontas dos dedos
                for idx in [4, 8, 12, 16, 20]:
                    tip = lms[idx]
                    if results.face_landmarks:
                        mouth_lm = results.face_landmarks.landmark[13]
                        dist     = calculate_distance(mouth_lm, tip)
                        if dist < THRESHOLD:
                            hand_near_mouth = True
                            close_finger_px = landmark_to_px(tip, w, h)

                # Cor do box muda conforme alerta
                box_color = COLOR_ALERT if (hand_near_mouth and alert_state) else \
                            COLOR_WARN  if hand_near_mouth else \
                            COLOR_BOX
                draw_rounded_rect(frame, hx1, hy1, hx2, hy2, box_color, thickness=2)
                draw_label_box(frame, hand_label, hx1, hy1, box_color)

                # Desenha pontas dos dedos como pontos menores
                for idx in [4, 8, 12, 16, 20]:
                    tip = lms[idx]
                    px  = landmark_to_px(tip, w, h)
                    cv2.circle(frame, px, 4, (200, 200, 255), -1)

            # Destaca a ponta do dedo próxima à boca
            if close_finger_px:
                cv2.circle(frame, close_finger_px, 10, COLOR_FINGER_TIP, 2)
                if mouth_px:
                    cv2.line(frame, mouth_px, close_finger_px, COLOR_FINGER_TIP, 1)

            # ── Detecção do gesto de coração com duas mãos ────────────────────
            if enable_heart:
                is_heart, center_norm = detect_heart_gesture(
                    results.left_hand_landmarks, results.right_hand_landmarks
                )
            else:
                is_heart, center_norm = False, None

            if is_heart and center_norm:
                cx_px = int(center_norm[0] * w)
                cy_px = int(center_norm[1] * h)

                lh8 = landmark_to_px(results.left_hand_landmarks.landmark[8], w, h)
                rh8 = landmark_to_px(results.right_hand_landmarks.landmark[8], w, h)
                lh4 = landmark_to_px(results.left_hand_landmarks.landmark[4], w, h)
                rh4 = landmark_to_px(results.right_hand_landmarks.landmark[4], w, h)

                cv2.line(frame, lh8, rh8, COLOR_HEART, 2)
                cv2.line(frame, lh4, rh4, COLOR_HEART, 2)

                draw_giant_heart(frame, (cx_px, cy_px), now)

            # ── Debounce (igual ao main.py) ───────────────────────────────────
            if hand_near_mouth:
                if hand_on_mouth_start is None:
                    hand_on_mouth_start = now
                elif now - hand_on_mouth_start >= DEBOUNCE_SECS:
                    alert_state = True
            else:
                hand_on_mouth_start = None
                alert_state          = False

            # ── Barra de debounce ─────────────────────────────────────────────
            if hand_near_mouth and hand_on_mouth_start is not None:
                elapsed  = min(now - hand_on_mouth_start, DEBOUNCE_SECS)
                progress = elapsed / DEBOUNCE_SECS
                bar_w    = int((w - 20) * progress)
                cv2.rectangle(frame, (10, h - 25), (w - 10, h - 10), (60, 60, 60), -1)
                bar_color = COLOR_ALERT if alert_state else COLOR_WARN
                cv2.rectangle(frame, (10, h - 25), (10 + bar_w, h - 10), bar_color, -1)
                draw_label_box(frame, f"Debounce: {elapsed:.1f}s / {DEBOUNCE_SECS:.1f}s",
                               10, h - 28, (80, 80, 80), font_scale=0.4)

            # ── Overlay de status (canto superior esquerdo) ───────────────────
            status_str = "ALERTA!" if alert_state else \
                         "CORAÇÃO 💖" if is_heart else \
                         "AVISO - mao perto" if hand_near_mouth else "OK"
            status_color = COLOR_ALERT if alert_state else \
                           COLOR_HEART if is_heart else \
                           COLOR_WARN  if hand_near_mouth else COLOR_NORMAL

            draw_overlay_text(frame, [
                f"FPS: {fps:.1f}",
                f"Face: {'SIM' if results.face_landmarks else 'NAO'}",
                f"Mao esq: {'SIM' if results.left_hand_landmarks else 'NAO'}",
                f"Mao dir: {'SIM' if results.right_hand_landmarks else 'NAO'}",
                f"Coração: {'DESATIVADO' if not enable_heart else ('SIM 💖' if is_heart else 'NAO')}",
                f"Status: {status_str}",
            ])

            # Faixa colorida de status no topo
            cv2.rectangle(frame, (0, 0), (w, 5), status_color, -1)

            # ── Alerta visual: overlay vermelho piscante ──────────────────────
            if alert_state:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
                alpha = 0.25 + 0.1 * math.sin(time.time() * 8)   # pisca
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Texto de alerta centralizado
                msg       = "TIRE A MAO DA BOCA!"
                font_sc   = 1.0
                (tw, th), _ = cv2.getTextSize(msg, FONT, font_sc, 2)
                tx = (w - tw) // 2
                ty = h // 2
                cv2.putText(frame, msg, (tx + 2, ty + 2), FONT, font_sc, (0, 0, 0),   3, cv2.LINE_AA)
                cv2.putText(frame, msg, (tx,     ty),     FONT, font_sc, (255, 255, 255), 2, cv2.LINE_AA)

            # ── Exibe ─────────────────────────────────────────────────────────
            cv2.imshow("HandScanner – Debug View  (ESC para sair)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:   # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HandScanner Debug View")
    parser.add_argument(
        "--no-heart", "--disable-heart",
        dest="enable_heart",
        action="store_false",
        default=True,
        help="Desativa a detecção do gesto de coração"
    )
    parser.add_argument(
        "--heart", "--enable-heart",
        dest="enable_heart",
        action="store_true",
        help="Ativa a detecção do gesto de coração"
    )
    args = parser.parse_args()
    run_debug_view(enable_heart=args.enable_heart)
