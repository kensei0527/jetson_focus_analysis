import torch
import face_alignment
from skimage import io
import cv2
import argparse
import os

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # デバイス指定 (GPUが有効なら 'cuda:0')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # FaceAlignment インスタンスの作成
    #   LandmarksType._2D, _2halfD, _3D の３種類から選べます
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)

    # 画像読み込み (サンプルとして外部URLかローカル画像を読み込む)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='Path to input image')
    args = parser.parse_args()

    # ホストでマウントしたファイルパスがコンテナ内で見えるので、そのまま読み込む
    if not os.path.exists(args.input_image):
        print(f"File not found: {args.input_image}")
        return

    # 画像を読み込み
    image_bgr = cv2.imread(args.input_image)
    if image_bgr is None:
        print("cv2.imread failed.")
        return

    # Face Alignment の処理 (仮例)
    # ...
    print("Successfully loaded", args.input_image)

    # ランドマーク推定 (複数の顔が映っていれば複数返る)
    preds = fa.get_landmarks(image_bgr)

    if preds is None or len(preds) == 0:
        print("No faces detected.")
        return

    print(f"Detected {len(preds)} face(s).")
    for i, face_points in enumerate(preds):
        print(f" Face {i}: landmarks shape = {face_points.shape}")
        # face_points は (68, 2) など [landmarks数, 2次元座標] のnumpy配列

    # OpenCVで可視化する例 (optional)
    # 注意: face-alignmentの座標は (x, y)
    image_bgr2 = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR) if len(image_bgr.shape) == 3 else image_bgr
    for face_points in preds:
        for (x, y) in face_points:
            cv2.circle(image_bgr2, (int(x), int(y)), 2, (0, 255, 0), -1)
    cv2.imwrite('output_landmarks.jpg', image_bgr)
    print("Landmarks drawn and saved to output_landmarks.jpg")

if __name__ == "__main__":
    main()