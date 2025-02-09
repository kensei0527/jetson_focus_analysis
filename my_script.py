# my_script.py
import torch
import cv2

if __name__ == "__main__":
    print("Hello from inside the Docker container!")
    print("PyTorch CUDA available:", torch.cuda.is_available())
    print("OpenCV version:", cv2.__version__)