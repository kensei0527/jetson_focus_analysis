import cv2
import feat
import torch
import time
import numpy as np

# face-alignment
import face_alignment
from skimage import io

# py-feat (Action Unit )

from feat import Detector

# ========== ブリンク検出用 関数 ==========
def eye_aspect_ratio(landmarks, left_eye_idx, right_eye_idx):
    """
    landmarks: 68点 or それ以上の顔ランドマーク配列 (shape=(68, 2))
    left_eye_idx, right_eye_idx: 左右目のランドマークindex
    EARを計算し、(left_ear, right_ear, ear)を返す。
    """
    def ear_calc(pts):
        # (x,y)をptsに格納している想定
        # EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
        # ここでは 6点式のEyeか、landmarkのindexによって式が変わる
        # 例: 36~41が左目, 42~47が右目 (標準68点モデルの場合)
        p1, p2, p3, p4, p5, p6 = pts
        A = np.linalg.norm(p2 - p6)
        B = np.linalg.norm(p3 - p5)
        C = np.linalg.norm(p1 - p4)
        ear = (A + B) / (2.0 * C) if C > 0 else 0
        return ear

    left_pts = landmarks[left_eye_idx]
    right_pts = landmarks[right_eye_idx]

    left_ear = ear_calc(left_pts)
    right_ear = ear_calc(right_pts)
    ear = (left_ear + right_ear) / 2.0
    return left_ear, right_ear, ear

# ========== 視線速度算出 用 クラス ==========
class GazeVelocityTracker:
    def __init__(self):
        self.prev_center = None
        self.prev_time = None
        self.velocities = []

    def update(self, center, now):
        """
        center: (x, y) 目の中心座標
        now: 現在時刻 (time.time())
        """
        if center is None:
            return None

        if self.prev_center is not None and self.prev_time is not None:
            dt = now - self.prev_time
            if dt > 0:
                dist = np.linalg.norm(np.array(center) - np.array(self.prev_center))
                velocity = dist / dt
                self.velocities.append(velocity)
                return velocity
        self.prev_center = center
        self.prev_time = now
        return None

    def get_average_velocity(self):
        return np.mean(self.velocities) if len(self.velocities) > 0 else 0.0

# ========== メイン スクリプト ==========
def main():
    # 入力動画パス (Webカメラの場合は 0)
    input_video = "my_Docker_app/IMG_7721.mp4"  # 例: "0" でカメラ
    output_video = "output_analyzed.mp4"

    # GPU 対応かどうかを確認
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # face-alignment の初期化
    # 68点ランドマーク (FAN)
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, 
        device=device
    )

    # AU Detector (py-feat)
    # CPU/GPU対応: py-feat がGPUをフル活用するかはモデル依存
    au_detector = Detector(
        face_model="retinaface",  # 顔検出モデル
        landmark_model="mobilefacenet", 
        au_model="xgb",           # xgboost
        emotion_model="resmasknet",
        device=device  # GPU使えるかどうかは環境次第
    )

    # 68点の標準的な目のインデックス (dlib準拠)
    left_eye_idx = [36, 37, 38, 39, 40, 41]  # 6点
    right_eye_idx = [42, 43, 44, 45, 46, 47]

    # 瞬き判定用パラメータ
    EAR_THRESHOLD = 0.22
    MIN_FRAMES_CLOSED = 2  # 連続何フレーム ear < 閾値で瞬きとする
    blink_counter = 0
    blink_total = 0
    last_blink_time = 0.0

    # 視線速度
    gaze_tracker = GazeVelocityTracker()

    # 動画読み込み
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Cannot open video: {input_video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 出力動画
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_bgr = frame.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # face-alignment は PIL or numpy RGBを想定
        # get_landmarks_from_image は複数顔を返す可能性がある
        landmarks_batch = fa.get_landmarks_from_image(frame_rgb)
        
        # py-feat (AU) はファイルパス or PIL 画像で
        # → ピル画像に変換して一時ファイル化 or 直接detectフェーズ
        # ここでは簡易例: OpenCV -> RGB -> PIL
        # ただし py-feat で GPU が効くかどうかはモデル設定依存
        from PIL import Image
        pil_img = Image.fromarray(frame_rgb)

        # AU 推定 (複数顔対応だがここでは1つ目だけ使う例)
        predictions = au_detector.detect_image(pil_img)

        # ========== 瞬き & 視線解析は一人を想定 (landmarks_batch[0]) ==========
        if landmarks_batch and len(landmarks_batch) > 0:
            # 複数顔の場合 [face0, face1, ...]
            landmarks = landmarks_batch[0]  # shape=(68,2) or (n,2)
            # 瞬き
            l_ear, r_ear, ear = eye_aspect_ratio(landmarks, left_eye_idx, right_eye_idx)

            if ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                # 開いたタイミングで瞬き完了とみなす
                if blink_counter >= MIN_FRAMES_CLOSED:
                    blink_total += 1
                blink_counter = 0

            # 視線(簡易): 両目のランドマーク平均で目の中心を計算
            left_center = np.mean(landmarks[left_eye_idx], axis=0)
            right_center = np.mean(landmarks[right_eye_idx], axis=0)
            gaze_center = (left_center + right_center) / 2.0
            now_t = time.time()
            gaze_vel = gaze_tracker.update(gaze_center, now_t)
            if gaze_vel is None:
                gaze_vel = 0.0

            # ランドマーク描画
            for (x,y) in landmarks:
                cv2.circle(frame_bgr, (int(x), int(y)), 2, (0,255,0), -1)

            # 瞳中心描画
            cv2.circle(frame_bgr, (int(gaze_center[0]), int(gaze_center[1])), 3, (0,0,255), -1)

            # 情報表示
            cv2.putText(frame_bgr, f"EAR: {ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame_bgr, f"Blink count: {blink_total}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame_bgr, f"Gaze vel: {gaze_vel:.2f}", (10,110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            # 顔が検出されなかった
            cv2.putText(frame_bgr, "No face", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # ========== AU 推定結果を描画 (py-feat) ==========
        if predictions is not None and len(predictions) > 0:
            # 複数人いる場合 -> predictions.aus など
            # ここでは1人目だけ表示例
            row_0 = predictions.aus.iloc[0].to_dict()  # {"AU01":..., "AU02":...}
            # 例: 全AUのうち、スコアが高いものを表示
            # ここではログっぽくテキスト合成
            au_text = ", ".join([f"{k}:{v:.2f}" for k,v in row_0.items()])
            cv2.putText(frame_bgr, f"AU: {au_text}", (10, height-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # 動画に書き込み
        out.write(frame_bgr)

        # オプションで、表示したい場合
        # cv2.imshow("Analysis", frame_bgr)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("=== Analysis done. Saved video to:", output_video)
    print(f"Total frames: {frame_count}, Blink count: {blink_total}")
    print(f"Average gaze velocity: {gaze_tracker.get_average_velocity():.2f}")

if __name__ == "__main__":
    main()