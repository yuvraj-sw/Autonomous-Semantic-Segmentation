# 🚀 Real-Time Autonomous Road Perception HUD

A computer vision application utilizing PyTorch and Ultralytics YOLOv8 for real-time object detection and semantic scene partitioning. This pipeline runs asynchronously over video buffers or live hardware feeds to optimize inference speeds and maintain a stable frame rate.

---

## 🛠️ Step-by-Step Installation & First-Time Setup

Because machine learning and computer vision frameworks require specific binary dependencies, this project runs inside an isolated environment using **Python 3.10** to guarantee stability and prevent version mismatches with newer Python interpreters (like 3.12+ or 3.14).

### 1. Download the Package Manager
If you do not have it yet, download and install **Miniconda** (or Anaconda) for Windows:
👉 https://docs.anaconda.com/miniconda/
*(During installation, you can leave all choices on their default settings).*

### 2. Configure Environment and Clear Terms of Service
Open your **Anaconda Prompt** or **Miniconda Prompt** from your Windows Start Menu and copy-paste these commands one by one, hitting Enter after each:

```bash
# 1. Accept Anaconda's channels Terms of Service to allow downloads
conda tos accept --override-channels --channel [https://repo.anaconda.com/pkgs/main](https://repo.anaconda.com/pkgs/main)
conda tos accept --override-channels --channel [https://repo.anaconda.com/pkgs/r](https://repo.anaconda.com/pkgs/r)
conda tos accept --override-channels --channel [https://repo.anaconda.com/pkgs/msys2](https://repo.anaconda.com/pkgs/msys2)

# 2. Create the isolated sandbox environment running Python 3.10
conda create -n road_ai python=3.10 -y

# 3. Activate the fresh sandbox environment
conda activate road_ai

# 4. Navigate directly into your downloaded project folder
cd "C:\Users\yuvia\Downloads\Semantic-Segmentation-AI-main"