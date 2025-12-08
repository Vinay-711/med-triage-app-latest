# MedTriage AI

**MedTriage AI** is a state-of-the-art medical diagnostic assistant designed to streamline patient triage and specific condition analysis. Built with a modern, glassmorphic UI and powered by advanced AI simulation logic, it demonstrates the future of healthcare software.

![MedTriage Banner](https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=1600&q=80) 
*(Note: Placeholder banner)*

## 🚀 Key Features

*   **Intelligent Triage Queue**: Automatically prioritizes patients based on symptom urgency and AI preliminary analysis.
*   **Multi-Modality Support**: Switch seamlessly between **Radiology** (Chest X-Rays) and **Dermatology** (Skin Lesions) modes.
*   **AI-Powered Analysis (Simulation Mode)**:
    *   **Class Detection**: Identifies conditions like Pneumonia, Melanoma, No Finding, etc using simulated high-fidelity responses.
    *   **Uncertainty Estimation**: Provides confidence scores and flags high-uncertainty cases for human review.
    *   **Visual Heatmaps**: Displays AI attention maps overlaid on scans to show regions of interest.
*   **Federated Learning Ready**: Includes endpoints for receiving decentralized privacy-preserving model updates.
*   **Side-by-Side Comparison**: Dedicated mode to compare "Before" and "After" images with difference highlighting.
*   **Automated Reporting**: Generates structured medical reports (Findings, Impression, Recommendations) instantly.
*   **Interactive Assistant**: Built-in chat interface to query specific details about the active case.
*   **Quality Checks**: Validates image quality (brightness, rotation, metal artifacts) before analysis.

## 🛠️ Technology Stack

**Frontend**
*   **React 18** (TypeScript, Vite)
*   **Tailwind CSS** (Styling & Design System)
*   **Lucide React** (Iconography)

**Backend**
*   **Python 3.10+**
*   **FastAPI** (High-performance API framework)
*   **PyTorch / TorchVision** (Deep Learning Logic)
*   **OpenCV & Pillow** (Image Processing)
*   **Google Generative AI** (LLM Integration ready)
*   **Federated Learning** (Simulated Architecture)

---

## 🏁 Getting Started

Follow these steps to set up the project locally.

### Prerequisites
*   Node.js (v18+)
*   Python (v3.10+)

### 1. Backend Setup

The backend handles AI inference (simulated for reliability) and report generation.

```bash
# Navigate to the backend directory
cd backend

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install python-multipart uvicorn fastapi torch torchvision opencv-python

# Run the server
python main.py
```
*The backend runs on `http://localhost:8000` and starts in Simulation Mode by default.*

### 2. Frontend Setup

The frontend is a modern React application.

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```
*The application will open at `http://localhost:5173`.*

---

## 📖 Usage Guide

### 1. Dashboard & Worklist
*   View the **Patient Queue** on the left.
*   Click a patient to load their details (simulated) or use the **Upload Area** to analyze a new file.

### 2. Running Analysis
*   Upload an image (X-Ray or Skin Photo).
*   The system first checks **Image Quality**.
*   Click **Analyze Scan** to generate findings.
*   View the results in the **Findings** tab (Diagnosis, Confidence, Heatmap).
*   Switch to the **Report** tab to see the generated text report.

### 3. Comparison Mode
*   Toggle the **Compare** mode in the header.
*   Upload a "Baseline" image and a "Current" image.
*   Click **Run Comparison** to see a difference analysis and improvement score.

---

## 🤝 Contributing

This project is a demonstration of advanced agentic coding capabilities.

1.  Fork the repository
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
