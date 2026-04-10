# Ceiling-Based Indoor Localization with Reality Model

## Project Overview
This project is a lightweight, infrastructure-free indoor localization web application. It uses a standard smartphone camera to capture ceiling textures and matches them against a local database using the ORB algorithm. 

To overcome pure visual tracking limitations (like Perceptual Aliasing in repetitive corridors), a bespoke backend sequence-checking logic—the **"Reality Model"**—was engineered.

## Core Architecture
* **Frontend:** Glassmorphism UI built with HTML5/CSS3/Vanilla JS (Embedded directly in main.py for zero-setup deployment).
* **Backend:** FastAPI (Asynchronous Python server).
* **Computer Vision:** OpenCV (ORB feature extraction + FLANN matching + RANSAC).
* **Temporal-Spatial Filter (Reality Model):** Enforces human walking speeds (Dynamic Velocity Check) and building topologies (Topological Graph) to intercept physically impossible spatial jumps.

## How to Run (Local Deployment)
1. Install dependencies: `pip install -r requirements.txt`
2. Start the server: `uvicorn main:app --reload`
3. Open your browser and navigate to: `http://127.0.0.1:8000`

## Repository Structure
* `main.py` - FastAPI server, Reality Model validation logic, and frontend endpoints.
* `database.py` - Backend logic for loading the database and handling ORB feature matching.
* `database.json` - Configuration and metadata for the reference map nodes.
* `requirements.txt` - Python package dependencies.
* `/sample_dataset` 
  * `/reference_map` - Contains the base ceiling images used to construct the local map.
  * `/test_queries` - Contains the specific sample images used for the testing scenarios below.

---

## 🧪 Testing Scenarios (Reality Model Evaluation)
To evaluate how the Reality Model handles different spatial constraints, you can upload the sample images provided in the `/sample_dataset/test_queries` directory in specific sequences. 

**Note:** Ensure uploads in each scenario are done within the 60-second active session window.

### Scenario 1: Normal Walking (Baseline)
* **Action:** Upload `1_Normal_A00.jpg`, then upload `1_Normal_A01.jpg`.
* **Expected Result:** Both locations are validated. The trajectory updates normally as the distance and time align with realistic walking speeds.

### Scenario 2: Topological Graph (Valid Junction)
* **Action:** Upload `1_Normal_A00.jpg`, then upload `2_Topo_B00.jpg`.
* **Expected Result:** The system validates the corner turn. The underlying Topological Graph recognizes the legitimate physical connection between Corridor A and Corridor B.

### Scenario 3: Dynamic Velocity Check (Speed Limit)
* **Action:** Upload `1_Normal_A00.jpg`, then immediately upload `3_Speed_A08.jpg`.
* **Expected Result:** Intercepted. The system calculates that moving 8 nodes (approx. 24 meters) in a few seconds exceeds human capability. It displays an "Unrealistic Jump" warning.

### Scenario 4: Perceptual Aliasing (Cross-Building Teleportation)
* **Action:** Upload `1_Normal_A00.jpg`, then upload `4_Teleport_C05.jpg`.
* **Expected Result:** Intercepted. Even if the CV algorithm is confused by identical ceiling textures, the Reality Model identifies the impossible transition across buildings and displays a "Building Changed" warning.