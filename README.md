# News Credibility Estimator Chrome Extension

This Chrome extension uses an AI model to analyze news text and determine its credibility.

## Features
- Analyze text snippets for credibility.
- User-friendly popup UI.
- Backend API powered by a trained DistilBERT model.

## Installation Instructions
1. **Download the extension:**
   - Clone this repository:  
     ```bash
     git clone https://github.com/shamithap/NewsCredibilityDetector
     ```
   - OR download it as a ZIP file from the [Releases](https://github.com/shamithap/NewsCredibilityDetector).

2. **Load the extension:**
   - Open Chrome and navigate to `chrome://extensions/`.
   - Enable **Developer Mode** (toggle in the top-right corner).
   - Click **Load unpacked** and select the unzipped folder.

3. **Run the backend server:**
   - Install required Python dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Start the Flask server:
     ```bash
     python server.py
     ```
   - Ensure the backend server runs on `http://localhost:5001`.

4. **Download model weights and dataset:**
   - [Download Model Weights](https://drive.google.com/file/d/13-zOvZfgX4vRhnr-zvBnvsMrtJbEHQ2I/view?usp=sharing)
   - [Download WELFake Dataset](https://docs.google.com/spreadsheets/d/16ku6OGw9qdOtblocR6a0e4gekcM--aoY/edit?usp=sharing&ouid=106831316560089013652&rtpof=true&sd=true)
   - Place the files in the appropriate locations:
     - `model/final_model.weights.h5`
     - `WELFake_Dataset.xlsx`

## Notes
- The backend must be running for the extension to function.
- Modify the API URL in `popup.js` if hosting the server elsewhere.
