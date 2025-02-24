# Focus Monitoring System for Jetson Orin Nano

## Overview

This is a Focus Monitoring System that runs on the Jetson Orin Nano. The system analyzes real-time camera input to monitor the user's concentration level by detecting facial expressions and gaze direction. By combining face detection, facial landmark extraction, and machine learning inference, the system evaluates indicators such as blink frequency, gaze stability, and subtle changes in expression. The system outputs a concentration level on a 4-class scale (0 to 3), where a higher value indicates a higher degree of focus.

The project leverages the following technology stack:
- OpenCV for video capture and preprocessing
- MediaPipe for facial landmark detection
- PyTorch (optimized for NVIDIA Jetson) for deep learning inference
- RetinaFace for high-precision face detection
- JAANet for extracting facial Action Unit (AU) features
- Flask for providing a simple web interface to display results

The system accumulates facial movement and gaze data over 15-second intervals and uses a multi-layer perceptron (MLP) to classify the overall concentration level in real time.

## Setup Instructions

This section provides detailed instructions for setting up the environment on a Jetson Orin Nano. It assumes that you have a Jetson Orin Nano Developer Kit with JetPack installed (tested on JetPack 5.x with L4T R35 or higher). A connected camera (USB or CSI) is also required. If you prefer using Docker, ensure that the NVIDIA Container Runtime is installed.

### Prerequisites:
- A Jetson Orin Nano with Ubuntu-based Jetson Linux (with JetPack installed)
- An active internet connection and an operational camera device
- Docker and NVIDIA Container Runtime if you opt for a containerized deployment

### Using Docker (Recommended)

To simplify dependency management, the project provides a Dockerfile that includes all necessary libraries.

1. **Install Docker:**
   If Docker is not already installed on your Jetson Orin Nano, follow NVIDIA's guide to install Docker and the NVIDIA Container Runtime. (Often, JetPack installations include these components.)

2. **Obtain the Project:**
   Clone or transfer the complete source code to your Jetson device. The project directory should include the Dockerfile, Python scripts, and model files.

3. **Place Required Files:**
   The following model files must be in place:
   - RetinaFace Pre-trained Model: Place `mobilenet0.25_Final.pth` in the `Pytorch_Retinaface/weights/` directory. This file can be downloaded from the official RetinaFace GitHub repository. (https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1)
   - JAANet Weights: Download the JAANet weights from the official repository and place them in the `jaanet_weights/` directory.
   - MLP Model for Focus Classification: Ensure that the file (e.g., `best_model2.pth`) is available in the project's root or specify its path using the `--mlp_path` argument at runtime.

4. **Build the Docker Image:**
   In the project directory, run:
   ```bash
   cd <project_directory>
   sudo docker build -t focus-monitoring .
   ```
   This process uses an NVIDIA Jetson-optimized PyTorch base image (e.g., `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3`) and installs OpenCV, PyTorch, MediaPipe, face-alignment, RetinaFace, JAANet, and other required libraries. The build may take some time.

5. **Run the Docker Container:**
   Launch the container using the built image. For example, if your USB camera is at `/dev/video0`, run:
   ```bash
   sudo docker run -it --rm --runtime nvidia \
       --network host \
       --device /dev/video0 \
       focus-monitoring:latest \
       python3 main_app2.py --input_video 0 --csv_out /output/au_result.csv
   ```
   In this command:
   - `--runtime nvidia` enables GPU usage
   - `--network host` allows the Flask web service to be accessed from the host
   - `--device /dev/video0` makes the camera accessible within the container
   - The script `main_app2.py` is executed with input from the default camera and outputs results to `/output/au_result.csv`

   Note: For CSI cameras, the device may still be `/dev/video0`, but consult your Jetson camera settings if additional configuration is needed. To save the CSV output to the host, mount a host directory using `-v <host_directory>:/output`.

6. **Verify Startup:**
   Once the container starts, logs will display information about GPU usage (e.g., "Device: cuda"), blink detection counts, gaze speed, and other processing details. A Flask server will also start on port 8080, confirming that the setup is successful.

### Manual Setup (Without Docker)

If you prefer not to use Docker, follow these steps to install the environment directly on the Jetson:

1. **Update System and Install Basic Tools:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-pip python3-dev git libopenblas-dev libopenmpi-dev openmpi-bin libomp-dev python3-scipy
   ```
   Although JetPack includes OpenCV and CUDA, installing the latest pip and build tools is recommended.

2. **Install PyTorch:**
   For the aarch64 architecture on Jetson, use NVIDIA-provided wheel files. For example, on a JetPack 5.1 (L4T R35.2.1) system:
   ```bash
   wget https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
   pip3 install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
   ```
   (Adjust the URL and version based on your JetPack version.)

3. **Install Required Python Packages:**
   ```bash
   pip3 install numpy onnx Cython wheel
   pip3 install matplotlib face-alignment tqdm mediapipe scikit-image scikit-learn flask
   ```
   Note: OpenCV is typically available via JetPack. If you encounter import errors, run `pip3 install opencv-python`.

4. **Install RetinaFace:**
   Clone the RetinaFace repository into your project directory:
   ```bash
   git clone https://github.com/biubug6/Pytorch_Retinaface.git
   ```
   Then, place the `mobilenet0.25_Final.pth` model file in the `Pytorch_Retinaface/weights/` directory.

5. **Install JAANet:**
   Obtain JAANet from its official GitHub repository (e.g., ZhiwenShao/PyTorch-JAANet) and integrate its code and pre-trained weights into your project (for instance, under `/opt/jaanet`). Ensure that all network definitions and weight files are accessible, typically in a folder like `jaanet_weights/`.

6. **Set Environment Variables (if needed):**
   Add the RetinaFace module to your Python path. For example, add the following line to your `~/.bashrc`:
   ```bash
   export PYTHONPATH="/path/to/project/Pytorch_Retinaface:${PYTHONPATH}"
   ```

7. **Run the System:**
   With everything set up, launch the main script:
   ```bash
   python3 main_app2.py --input_video 0 --csv_out ./au_result.csv
   ```
   Ensure that the code does not rely on display functions (like `cv2.imshow`) if you are running in a headless environment, as the system uses Flask for visualization.

Once running, you should see log messages indicating frame processing and focus level classification. The Flask server will be available on port 8080, and the CSV file will log focus results over time.

## Usage

After completing the setup, you can operate the system as follows:

### Basic Execution:
Start the system by running the `main_app2.py` script. If using Docker, the command provided above in the Docker section starts the process. When running directly on the host, execute:

```bash
python3 main_app2.py --input_video 0 --csv_out ./au_result.csv
```

The primary command-line options include:
- `--input_video`: Specifies the video input. The default value (0) uses the default camera. To analyze a video file, supply the file path (e.g., `--input_video sample.mp4`).
- `--csv_out`: Specifies the output CSV file path. The system appends analysis results every 15 seconds to this CSV. In Docker, it is set to `/output/au_result.csv` (with a shared host volume); on a local system, you might use something like `--csv_out ./result.csv`.
- `--mlp_path`: The path to the pre-trained MLP model for focus classification. The default is `best_model2.pth` in the project root. Adjust this if your model file is stored elsewhere.
- Other optional parameters (e.g., `--confidence_threshold`, `--retina_network`, etc.) are provided for fine-tuning. Typically, the defaults are sufficient; refer to the code's `parser.add_argument` section for details.

### System Operation:
Once started, the system processes video frames continuously in a background thread. It performs face detection, facial landmark extraction, and then aggregates features over 15-second intervals to compute the focus level via the MLP classifier. Simultaneously, a Flask web server launches on port 8080 to provide a dashboard with real-time focus metrics.

### Output Data:
1. **Real-Time Monitoring (Web UI):**
   After startup, open a browser on any device in the same network and navigate to `http://<Jetson_IP>:8080` (for example, `http://192.168.1.10:8080`). The dashboard displays the current focus level and a time-series graph showing focus trends.

2. **REST API:**
   The Flask server also exposes JSON-based APIs:
   - `/api/current_label`: Returns the most recent focus classification (e.g., `{ "label": 2 }` where the label ranges from 0 to 3; -1 indicates no data).
   - `/api/live_data`: Returns time-series data from the session in JSON format, for example:
     ```json
     {
       "data": [
         { "time": 15.0, "blink": 2.0, "gaze": 3.4, "label": 2 },
         { "time": 30.0, "blink": 1.0, "gaze": 2.8, "label": 3 },
         ...
       ]
     }
     ```
     Here, `time` represents the elapsed time in seconds, `blink` the average blink count, `gaze` the average gaze speed, and `label` the corresponding focus level.

3. **CSV Log File:**
   The CSV file specified by `--csv_out` logs the aggregated features and classification results every 15 seconds. Each row typically includes a timestamp, average blink count, average gaze speed, and the focus level label, for example:
   ```csv
   time,blink,gaze,label
   15.0,2.0,3.4,2
   30.0,1.0,2.8,3
   ```
   This CSV can be opened in Excel or other analysis tools for further review.

### Stopping the System:
Terminate the system by pressing Ctrl+C in the terminal. The program handles shutdown gracefully, stopping background threads and closing the CSV file (displaying a message like "Done. CSV saved." upon completion).

## System Architecture

This section outlines the internal structure and processing flow of the system:

1. **Video Input and Face Detection:**
   Frames are captured from a camera or video file. RetinaFace is used to detect faces and obtain approximate facial landmarks (e.g., eyes and mouth positions) via bounding boxes.

2. **Facial Landmark Extraction and Alignment:**
   For each detected face, more detailed landmarks (68 points) are extracted using libraries such as face_alignment or MediaPipe Face Mesh. These landmarks are then mapped to a 49-point format required by JAANet. An affine transformation normalizes the face image (e.g., to 176×176 pixels) for further processing.

3. **Action Unit (AU) Feature Extraction:**
   The aligned face image is fed into JAANet to infer the intensities of facial Action Units (AUs). JAANet outputs scores for 12 AUs in both positive and negative directions, forming a 24-dimensional feature vector. These features capture subtle muscle movements related to expression changes, which correlate with focus levels.

4. **Blink Detection:**
   Concurrently, eye landmarks obtained from MediaPipe Face Mesh are used to compute the Eye Aspect Ratio (EAR) from six key points around each eye. A series of frames with EAR values below a threshold is counted as a blink. The blink frequency (blinks per second) serves as an indicator of focus.

5. **Gaze Movement Measurement:**
   The positions of the irises are tracked over successive frames to compute the gaze movement speed. Stable, low-speed gaze generally indicates high focus, while rapid gaze shifts suggest distraction.

6. **Feature Aggregation:**
   Each second, the system aggregates the blink count, average gaze speed, and the 24-dimensional AU features into a 26-dimensional feature vector. A time-series buffer collects 15 such vectors (one per second) to form a 390-dimensional feature vector representing the last 15 seconds of activity.

7. **Focus Classification:**
   Every 15 seconds, the 390-dimensional feature vector is fed into a pre-trained MLP classifier. The MLP outputs scores for 4 classes (0, 1, 2, 3) and selects the class with the highest score as the current focus level. For instance, a label of 3 indicates high concentration, while 0 indicates low focus.

8. **Result Logging and Distribution:**
   The resulting focus level is stored in a global variable and served via Flask's API. The aggregated data (including average blink count, gaze speed, and focus label) is also appended as a new row in a CSV file for later analysis. The Flask dashboard uses this data to provide real-time visual feedback.

### Technology Stack Summary:
- OpenCV: Handles video capture and basic image processing
- MediaPipe (Face Mesh): Detects detailed facial landmarks for blink and gaze analysis
- RetinaFace: Performs fast, accurate face detection on the input frames
- face_alignment: Provides refined 68-point landmark detection to complement RetinaFace
- JAANet: Extracts facial Action Unit features for expression analysis
- PyTorch: Runs the deep learning models (JAANet and MLP classifier) using the GPU for efficient inference
- Flask: Serves a web dashboard and REST API for real-time monitoring

The system is optimized to run in real time on the Jetson Orin Nano, harnessing its GPU capabilities for accelerated processing.

## License

The source code and accompanying materials for this project are provided under the MIT License. This license permits commercial and non-commercial use, modification, and redistribution provided that the original license notice is included.

Note: Third-party libraries and models used in this project (such as RetinaFace, JAANet, MediaPipe, etc.) are distributed under their respective licenses. Please refer to the individual project licenses for further details. The source code in this project is released under the MIT License, and unless otherwise noted, copyright remains with the original authors.

This README is designed to clearly demonstrate the reproducibility and originality of the Focus Monitoring System on the Jetson Orin Nano. Detailed instructions for both Docker and manual setups ensure that users can deploy the system in various environments while achieving consistent results. Enjoy building and experimenting with your real-time focus monitoring solution!for both Docker and manual setups ensure that users can deploy the system in various environments while achieving consistent results.

