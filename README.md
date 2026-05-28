# CogHealth Monitor

“It’s not stress that kills us; it is our reaction to it.” – Hans Selye

## Introduction

CogHealth is an innovative, real-time behavioral stress monitoring system designed to provide users with insights into their cognitive well-being through passive analysis of keyboard and mouse interactions. This system was inspired by growing concerns surrounding citizen surveillance and the series *Psycho-Pass* which served an interesting lesson on how behavioral monitoring, even when intended for public safety, can become a tool for oppression, manipulation, and loss of individual autonomy. 

Modern society is increasingly reliant on data collection systems that track user behavior—from keystroke dynamics to mouse movements—often without meaningful transparency or consent. While such data can offer valuable insights into cognitive well-being, it raises critical questions about privacy, autonomy, and the potential for abuse.

CogHealth takes a fundamentally different approach: **empowering individuals rather than surveilling them**. Rather than collecting data for external oversight or control, CogHealth puts the user in command of their own cognitive health insights. The system:

### Key Features:

*   **Real-time Stress Visualization**: A dynamic 3D visualization that intuitively reflects current stress levels.
    *   **Calm State**: At low stress, a serene **3D Leaf** gently sways, symbolizing focus and tranquility.
    *   **Stressed State**: As stress elevates, the visualization transforms into a **rapidly stretching ball**, indicating heightened cognitive load.
*   **Behavioral Feature Extraction**: Analyzes typing speed, keystroke dynamics, error rates, mouse movements, and more to infer stress.
*   **Self-Report Integration**: Allows users to log their perceived stress, providing valuable ground truth for correlation.

## Directory Structure

```
. 
├── README.md
├── AssessmentSubmissions
│   ├── Figures/
│   ├── Forms/
|   ├── ProjectTimelines/
|   ├── Tables/
|   ├── Wireframes/
|   ├── Project A1_32146983.pdf
|   ├── ProjectReportA2_32146983HessaK_Batch2.pdf
│   └── ProjectPosterA3_32146983HessaK_Batch2.pdf
|
├── check_env.py
├── collector.py
├── config.py
├── data/
│   ├── features/
│   ├── raw/
│   └── self_reports.db
|
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
|
├── orchestrator.py
├── requirements.txt
├── run.py
├── server.py
├── setup.sh
├── tests/
└── web/

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

*To run the full system with local keyboard/mouse monitoring (requires `pynput`):*

4.  Ensure `pynput` is installed (`pip install pynput`).

5.  Run the orchestrator:
    ```bash
    python3 run.py
    ```
6.  Open your web browser and navigate to `http://localhost:5000`.

Once deployed, the web application will be accessible via the URL provided by Render, offering a live, browser-based stress monitoring experience.
