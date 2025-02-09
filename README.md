# Focus Analysis System for Jetson

## Overview
This project implements a real-time concentration analysis system for Jetson devices. It combines face detection, blink detection, gaze tracking, and Action Unit (AU) analysis to classify user concentration levels into 4 stages.

## Key Features
- Real-time face detection (RetinaFace)
- Blink detection (MediaPipe Face Mesh)
- Gaze tracking 
- Action Unit (AU) analysis (JAANet)
- Real-time visualization through web interface

## Requirements
- NVIDIA Jetson device
- CUDA support
- Python 3.8
- PyTorch 2.0
- Docker

## Setup
1. Build Docker image:
```bash
docker build -t focus_analysis -f Dockerfile.txt .
```

2. Required directory structure:
```
.
├── Dockerfile.txt
├── main_app2.py
├── templates/
│   └── index.html
├── jaanet_weights/
├── Pytorch_Retinaface/
└── best_model2.pth
```

## Running the Application
1. Launch Docker container:
```bash
docker run --runtime nvidia -it --rm \
    --network host \
    --device /dev/video0:/dev/video0 \
    focus_analysis
```

2. Run the application:
```bash
python3 main_app2.py
```

3. Access in browser:
```
http://localhost:8080
```

## System Architecture
- **Face Detection**: Uses RetinaFace
- **Landmark Detection**: Facial landmark detection using MediaPipe Face Mesh
- **Blink Analysis**: Detection based on EAR (Eye Aspect Ratio)
- **Gaze Analysis**: Calculates gaze velocity and fixation time through iris tracking
- **AU Analysis**: Action Unit detection using JAANet
- **Concentration Classification**: 4-class classification using multi-branch MLP

## Output Classes
- 0: Low concentration
- 1: Normal concentration
- 2: High concentration
- 3: No face, No concentration

## Notes
- Camera access permissions required
- Monitor GPU memory usage
- System may take a few seconds to initialize

## Files Description
- `main_app2.py`: Main application file containing core logic
- `Dockerfile.txt`: Docker configuration for environment setup
- `templates/index.html`: Web interface template
- `jaanet_weights/`: Pre-trained weights for JAANet
- `Pytorch_Retinaface/`: RetinaFace implementation
- `best_model2.pth`: Trained concentration classification model

## Technical Details
- Uses Flask for web server implementation
- Implements real-time processing at approximately 30 FPS
- Features 15-second analysis windows for concentration classification
- Combines multiple ML models for comprehensive analysis

## License
This project is released under the MIT License.

## Acknowledgements
- RetinaFace implementation: biubug6/Pytorch_Retinaface
- JAANet: ZhiwenShao/PyTorch-JAANet
- MediaPipe: Google MediaPipe

## Contributing
Contributions to improve the system are welcome. Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support
For issues and questions, please create an issue in the repository.
