# Car-parking-assistant
AI-powered Smart Parking Assistant that detects available parking slots using YOLOv8 and provides optimal navigation paths to vacant spaces using A* pathfinding.
Smart Parking Assistant 🚗

Smart Parking Assistant is an AI-based parking management system that helps drivers quickly find available parking spaces.

The system uses a custom-trained YOLOv8 model to detect occupied and vacant parking slots from parking lot images or video feeds. Once a vacant slot is identified, an A* Pathfinding algorithm generates the shortest route from the entrance to the selected parking space.

Features
Real-time parking slot detection using YOLOv8
Empty and occupied slot classification
Automatic vacant slot identification
Shortest path generation using A* Algorithm
Streamlit-based interactive user interface
Parking lot visualization with route guidance
Custom trained model (best.pt) for parking slot detection
Tech Stack
Python
YOLOv8 (Ultralytics)
OpenCV
Streamlit
NumPy
A* Pathfinding Algorithm
Use Cases
Smart Parking Management Systems
Shopping Malls
Airports
Corporate Campuses
Smart Cities
Commercial Parking Facilities
How to Run the Project


Prerequisites
Python 3.10+
Git
Virtual Environment (Recommended)
Step 1: Clone the Repository
git clone https://github.com/yourusername/Smart-Parking-Assistant.git

cd Smart-Parking-Assistant
Step 2: Create Virtual Environment
Windows
python -m venv venv

venv\Scripts\activate
Linux / Mac
python3 -m venv venv

source venv/bin/activate
Step 3: Install Dependencies

If you create a clean requirements file:

pip install -r requirements.txt

Or install the main dependencies manually:

pip install streamlit ultralytics opencv-python numpy pathfinding pillow torch torchvision
Step 4: Verify Model File

Ensure the trained model file is present:

best.pt

Project structure:

SmartParkingAssistant/
│
├── app.py
├── best.pt
├── demo.mp4
├── requirements.txt
└── README.md
Step 5: Run the Application
streamlit run app.py
Step 6: Open in Browser

Streamlit will automatically generate a URL:

http://localhost:8501

Open it in your browser.

Step 7: Upload Parking Image/Video
Upload a parking lot image or video.
The system detects:
Empty Slots (Green)
Occupied Slots (Red)
Select a parking slot.
View the shortest navigation path to the available parking space.

