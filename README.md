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

Add this section in your README so recruiters can test the project easily.

Prerequisites
Python 3.10+
Git
Virtual Environment (Recommended)
