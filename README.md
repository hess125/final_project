# CogHealth Monitor

"The greatest weapon against stress is our ability to choose one thought over another." — William James

## Introduction

CogHealth is an innovative, real-time behavioral stress monitoring system designed to provide users with insights into their cognitive well-being through passive analysis of keyboard and mouse interactions. This system aims to foster a healthier digital work environment by visualizing stress levels and offering timely, actionable advice. Unlike traditional monitoring tools, CogHealth prioritizes user privacy by collecting only timing metadata, never actual keystroke values or click targets.

### Key Features:

*   **Real-time Stress Visualization**: A dynamic 3D visualization that intuitively reflects current stress levels.
    *   **Calm State**: At low stress, a serene **3D Leaf** gently sways, symbolizing focus and tranquility.
    *   **Stressed State**: As stress elevates, the visualization transforms into a **rapidly stretching ball**, indicating heightened cognitive load.
*   **Light/Dark Mode**: A user-friendly toggle allows seamless switching between light and dark themes, with preference persistence across sessions.
*   **Behavioral Feature Extraction**: Analyzes typing speed, keystroke dynamics, error rates, mouse movements, and more to infer stress.
*   **Self-Report Integration**: Allows users to log their perceived stress, providing valuable ground truth for correlation.
*   **Web-Native Deployment**: Designed for easy deployment as a web application, enabling browser-based data collection and real-time feedback without local software installation.

## Directory Structure

```
. 
├── README.md
├── Procfile
├── check_env.py
├── collector.py
├── config.py
├── data/
│   ├── features/
│   ├── raw/
│   └── self_reports.db
├── evaluation.py
├── features.py
├── led.py
├── logs/
│   └── coghealth.log
├── model.py
├── models/
│   ├── autoencoder.h5
│   ├── autoencoder.tflite
│   ├── scaler.pkl
│   └── threshold.json
├── orchestrator.py
├── requirements.txt
├── run.py
├── server.py
├── setup.sh
├── tests/
└── web/
    ├── static/
    └── template/
```

## Installation

To set up the CogHealth Monitor locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/hess125/final_project.git
    cd final_project
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

    *Note: The `requirements.txt` has been optimized for web deployment, removing heavy TensorFlow dependencies. For local development with the original `run.py` (which uses the `collector.py` for local system monitoring), you might need to install `pynput` separately: `pip install pynput`.*

## Demo Instructions

### Local Demo (Original Desktop Agent)

To run the full system with local keyboard/mouse monitoring (requires `pynput`):

1.  Ensure `pynput` is installed (`pip install pynput`).
2.  Run the orchestrator:
    ```bash
    python3 run.py
    ```
3.  Open your web browser and navigate to `http://localhost:5000`.

### Web-Native Demo (Browser-based Monitoring)

For a web-native experience where behavioral data is collected directly in the browser (suitable for cloud deployment):

1.  Ensure you are on the `deployment` branch:
    ```bash
    git checkout deployment
    ```
2.  Start the Flask server:
    ```bash
    python3 server.py
    ```
3.  Open your web browser and navigate to `http://localhost:5000`.

    *Interact with the page (type, move mouse) to see the stress visualization update based on your browser activity. The theme toggle (moon/sun icon) will switch between light and dark modes, and the 3D object will transition between a leaf (low stress) and a stretching ball (high stress).*

## Web Deployment (Render.com Example)

This project is configured for easy deployment to platforms like Render.com. Follow these steps:

1.  **Connect your GitHub repository** to Render.
2.  Select the **`deployment`** branch.
3.  Configure the build and start commands:
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `gunicorn --bind 0.0.0.0:$PORT "server:app"`
    *   **Publish Directory**: Leave empty.
4.  Add an **Environment Variable**:
    *   **Key**: `COGHEALTH_SECRET`
    *   **Value**: A long, random string (e.g., `your-super-secret-key-here`).

Once deployed, the web application will be accessible via the URL provided by Render, offering a live, browser-based stress monitoring experience.
