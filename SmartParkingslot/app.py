import streamlit as st
import cv2
import numpy as np
import os
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from ultralytics import YOLO
import tempfile
from PIL import Image
import torch
import time
from pathlib import Path
# Set page configuration
st.set_page_config(
    page_title="Smart Parking Navigation System",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
def load_css():
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: #1E88E5;
        }
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            border: none;
            padding: 0.5rem 1rem;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #1565C0;
        }
        .info-box {
            background-color: #5DBAE9;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #1E88E5;
            margin-bottom: 1rem;
        }
        .success-box {
            background-color: #2BC573;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #4CAF50;
            margin-bottom: 1rem;
        }
        .warning-box {
            background-color: #FFF8E1;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #FFC107;
            margin-bottom: 1rem;
        }
        .error-box {
            background-color: #FFEBEE;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #F44336;
            margin-bottom: 1rem;
        }
        .sidebar .sidebar-content {
            background-color: #F5F5F5;
        }
        .upload-btn {
            border: 2px dashed #1E88E5;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            margin-bottom: 1rem;
        }
        .card {
            color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 1rem;
            margin-bottom: 1rem;
            background-color: #019CDD;
        }
        .metric-card {
            background-color: #F5F5F5;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            margin-bottom: 1rem;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #1E88E5;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #616161;
        }
        .sidebar-header {
            font-weight: bold;
            margin-bottom: 0.5rem;
            color: #1E88E5;
        }
    </style>
    """, unsafe_allow_html=True)

# Function to load the YOLOv8 model
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Function to detect parking slots using YOLOv8
def detect_parking_slots(image, model):
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Run detection
    results = model(image)
    
    # Parse results
    empty_slots = []
    occupied_slots = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Get confidence
            conf = float(box.conf[0].cpu().numpy())
            
            # Get class (0: empty, 1: occupied) - adjust based on your model's classes
            cls = int(box.cls[0].cpu().numpy())
            
            if cls == 0:  # Empty slot
                empty_slots.append((x1, y1, x2, y2, "empty", conf))
            else:  # Occupied slot
                occupied_slots.append((x1, y1, x2, y2, "occupied", conf))
    
    return occupied_slots,empty_slots

# Function to create navigation grid with vehicle size consideration
def create_navigation_grid(image_shape, occupied_slots, vehicle_width=30, vehicle_length=60):
    # Create a binary grid where 1 = walkable, 0 = obstacle
    grid_data = np.ones((image_shape[0], image_shape[1]), dtype=int)
    
    # Mark occupied slots as non-walkable with vehicle dimensions buffer
    for slot in occupied_slots:
        x1, y1, x2, y2 = slot[:4]
        
        # Add a margin around obstacles based on vehicle size
        # Use vehicle_width/2 as the side clearance and vehicle_length/2 as front/back clearance
        buffer_x = vehicle_width // 2
        buffer_y = vehicle_length // 2
        
        # Apply buffer
        x1_buffered = max(0, x1 - buffer_x)
        y1_buffered = max(0, y1 - buffer_y)
        x2_buffered = min(image_shape[1] - 1, x2 + buffer_x)
        y2_buffered = min(image_shape[0] - 1, y2 + buffer_y)
        
        grid_data[y1_buffered:y2_buffered, x1_buffered:x2_buffered] = 0
    
    # Apply image processing to ensure corridors are wide enough for vehicles
    # Dilate obstacles to ensure paths are at least as wide as the vehicle
    kernel_size = vehicle_width // 2  # Half vehicle width for the kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    obstacle_map = (1 - grid_data).astype(np.uint8)  # Invert to get obstacles as 1
    dilated_obstacles = cv2.dilate(obstacle_map, kernel, iterations=1)
    
    # Convert back to grid format (1 for walkable, 0 for obstacles)
    grid_data = 1 - dilated_obstacles
    
    # Create the pathfinding grid
    # When creating Grid, the first param is the matrix, which should be
    # a 2D array where 1 is walkable and 0 is blocked
    return Grid(matrix=grid_data.tolist())

# Function to find path with vehicle constraints
def find_path(grid, start_point, end_point):
    # Use A* algorithm for pathfinding
    finder = AStarFinder(diagonal_movement=True)
    
    # Check if points are within grid bounds
    start_x = min(max(start_point[0], 0), grid.width - 1)
    start_y = min(max(start_point[1], 0), grid.height - 1)
    end_x = min(max(end_point[0], 0), grid.width - 1)
    end_y = min(max(end_point[1], 0), grid.height - 1)
    
    # Ensure start and end points are in navigable areas
    # If they're in obstacle cells, find the nearest navigable cells
    # We need to access the node's walkable attribute, not grid.matrix
    if not grid.node(start_x, start_y).walkable:
        # Find nearest navigable point to start
        nearest_start = find_nearest_navigable_point(grid, start_x, start_y)
        if nearest_start:
            start_x, start_y = nearest_start
        else:
            st.error("Cannot find navigable area near entrance. Please adjust entrance position.")
            return []
    
    if not grid.node(end_x, end_y).walkable:
        # Find nearest navigable point to end
        nearest_end = find_nearest_navigable_point(grid, end_x, end_y)
        if nearest_end:
            end_x, end_y = nearest_end
        else:
            st.error("Cannot find navigable area near target. Please select a different parking slot.")
            return []
    
    start = grid.node(start_x, start_y)
    end = grid.node(end_x, end_y)
    
    try:
        path, _ = finder.find_path(start, end, grid)
        
        # Post-process the path to smooth it and make it more suitable for vehicles
        if len(path) > 2:
            path = smooth_path(path)
        
        return path
    except Exception as e:
        st.error(f"Path finding error: {e}")
        return []

# Function to find nearest navigable point
def find_nearest_navigable_point(grid, x, y, max_radius=50):
    """Find the nearest navigable point to the given coordinates."""
    for radius in range(1, max_radius):
        # Check points in increasing radius around the given point
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                # Only check points at the current radius (perimeter)
                if abs(i) == radius or abs(j) == radius:
                    check_x = x + i
                    check_y = y + j
                    
                    # Check if within grid bounds and walkable
                    if (0 <= check_x < grid.width and 
                        0 <= check_y < grid.height and 
                        grid.node(check_x, check_y).walkable):
                        return (check_x, check_y)
    return None

# Function to smooth path for vehicle movement
def smooth_path(path, smoothing_window=3):
    """Apply smoothing to the path to make it more suitable for vehicles."""
    if len(path) < smoothing_window * 2:
        return path
    
    smoothed_path = [path[0]]  # Keep the start point
    
    # Apply moving average to middle points
    for i in range(1, len(path) - 1):
        # Get window of points
        window_start = max(0, i - smoothing_window)
        window_end = min(len(path), i + smoothing_window + 1)
        window = path[window_start:window_end]
        
        # Calculate average position
        avg_x = sum(node.x for node in window) // len(window)
        avg_y = sum(node.y for node in window) // len(window)
        
        # Instead of creating a new Node object, modify the existing node's coordinates
        # This avoids the Node initialization error
        path[i].x = avg_x
        path[i].y = avg_y
        smoothed_path.append(path[i])
    
    smoothed_path.append(path[-1])  # Keep the end point
    
    return smoothed_path

# Function to draw the image with detections and navigation
def draw_detections(image, empty_slots, occupied_slots, entrance_point=None, target_point=None, path=None):
    # Make a copy of the image
    display_img = image.copy()
    
    # Draw empty slots in green
    for slot in empty_slots:
        x1, y1, x2, y2, _, conf = slot
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Label with confidence - larger font, thicker lines
        label = f"Empty {conf:.2f}"
        cv2.putText(display_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw occupied slots in red
    for slot in occupied_slots:
        x1, y1, x2, y2, _, conf = slot
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Label with confidence - larger font, thicker lines
        label = f"Occupied {conf:.2f}"
        cv2.putText(display_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw path if available
    if path and entrance_point and target_point:
        # Draw entrance point as a yellow circle with larger text
        cv2.circle(display_img, entrance_point, 12, (0, 255, 255), -1)
        # Add a black outline to make text more readable
        cv2.putText(display_img, "Entrance", (entrance_point[0]-30, entrance_point[1]-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(display_img, "Entrance", (entrance_point[0]-30, entrance_point[1]-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Draw target point as a magenta circle with larger text
        cv2.circle(display_img, target_point, 12, (255, 0, 255), -1)
        # Add a black outline to make text more readable
        cv2.putText(display_img, "Target", (target_point[0]-30, target_point[1]-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(display_img, "Target", (target_point[0]-30, target_point[1]-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
        
        # Draw path as a thicker blue line
        for i in range(len(path) - 1):
            # Access x and y attributes of the GridNode objects
            pt1 = (path[i].x, path[i].y)
            pt2 = (path[i + 1].x, path[i + 1].y)
            # Make line thicker for better visibility
            cv2.line(display_img, pt1, pt2, (255, 0, 0), 5)
    
    return display_img

def main():
    # Load custom CSS
    load_css()
    
    # App header
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("🅿️ Smart Parking Navigation System")
    with col2:
        st.image("https://img.icons8.com/color/96/000000/car--v1.png", width=70)
    
    st.markdown('<div class="info-box">This system helps you navigate to empty parking spaces using AI detection.</div>', unsafe_allow_html=True)
    
    # Create sidebar
    with st.sidebar:
        st.markdown('<p class="sidebar-header">CONFIGURATION</p>', unsafe_allow_html=True)
        
        # Model selection
        st.markdown("### Model Settings")
        default_model_path = str(Path(__file__).parent / "best.pt")
        model_path = st.text_input("YOLOv8 Model Path", default_model_path)
        st.write("Model path:", model_path)
        conf_threshold = st.slider("Detection Confidence", 0.1, 1.0, 0.25, 0.05)
        
        st.markdown("---")
        
        # Upload image section with custom styling
        st.markdown("### Upload Image")
        st.markdown('<div class="upload-btn">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload parking lot image (top view)", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Information about the app
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        <div style="font-size: 0.85rem;">
        This application uses YOLOv8 to detect empty and occupied parking slots.
        
        Features:
        - AI-based detection
        - Interactive slot selection
        - Automatic path finding
        - Navigation visualization
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'entrance_point' not in st.session_state:
        st.session_state.entrance_point = None
    if 'selected_slot' not in st.session_state:
        st.session_state.selected_slot = None
    if 'navigation_path' not in st.session_state:
        st.session_state.navigation_path = None
    if 'image' not in st.session_state:
        st.session_state.image = None
    if 'empty_slots' not in st.session_state:
        st.session_state.empty_slots = []
    if 'occupied_slots' not in st.session_state:
        st.session_state.occupied_slots = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Main content area
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            img_path = tmp_file.name
        
        # Read the image
        image = cv2.imread(img_path)
        if image is None:
            st.markdown('<div class="error-box">Failed to load image. Please try another file.</div>', unsafe_allow_html=True)
            return
        
        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.session_state.image = image_rgb
        
        # Load YOLOv8 model
        model = load_model(model_path)
        if model is None:
            st.markdown('<div class="error-box">Failed to load model. Please check the model path.</div>', unsafe_allow_html=True)
            return
        
        # Create tabs for different stages
        tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📍 Setup", "🚗 Navigation"])
        
        with tab1:
            st.markdown("### Parking Slot Detection")
            
            # Add a process button
            if st.button("Detect Parking Slots", key="detect_btn"):
                st.session_state.processing = True
                progress_bar = st.progress(0)
                
                # Process image with progress updates
                for i in range(101):
                    progress_bar.progress(i)
                    if i == 20:
                        st.markdown('<div class="info-box">Loading model...</div>', unsafe_allow_html=True)
                    elif i == 40:
                        st.markdown('<div class="info-box">Processing image...</div>', unsafe_allow_html=True)
                    elif i == 60:
                        # Detect parking slots
                        st.session_state.empty_slots, st.session_state.occupied_slots = detect_parking_slots(image_rgb, model)
                        st.markdown('<div class="success-box">Detection completed!</div>', unsafe_allow_html=True)
                    time.sleep(0.02)
                
                st.session_state.processing = False
                
                # Create a metrics display
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(st.session_state.empty_slots)}</div>
                        <div class="metric-label">Empty Slots</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(st.session_state.occupied_slots)}</div>
                        <div class="metric-label">Occupied Slots</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    total = len(st.session_state.empty_slots) + len(st.session_state.occupied_slots)
                    occupancy = round((len(st.session_state.occupied_slots) / total * 100 if total > 0 else 0), 1)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{occupancy}%</div>
                        <div class="metric-label">Occupancy Rate</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Draw detections
                display_img = draw_detections(image_rgb, st.session_state.empty_slots, st.session_state.occupied_slots)
                
                # Display the image in a card
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.image(display_img, caption="Detected Parking Slots", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            else:
                # Show placeholder if detection hasn't been run
                if not st.session_state.processing and len(st.session_state.empty_slots) == 0:
                    st.markdown('<div class="info-box">Click "Detect Parking Slots" to analyze the image</div>', unsafe_allow_html=True)
                    st.image(image_rgb, caption="Uploaded Image", use_column_width=True)
        
        with tab2:
            st.markdown("### Navigation Setup")
            
            if len(st.session_state.empty_slots) > 0:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Set Entrance Point")
                st.markdown("Input the coordinates of the parking lot entrance:")
                
                entrance_col1, entrance_col2 = st.columns(2)
                
                with entrance_col1:
                    entrance_x = st.number_input("Entrance X", 0, image.shape[1]-1, int(image.shape[1] / 2))
                
                with entrance_col2:
                    entrance_y = st.number_input("Entrance Y", 0, image.shape[0]-1, image.shape[0] - 30)
                
                st.session_state.entrance_point = (entrance_x, entrance_y)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Select Parking Slot")
                
                # Create a dictionary of slots for selection
                slot_options = {}
                for i, slot in enumerate(st.session_state.empty_slots):
                    x1, y1, x2, y2, _, conf = slot
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    slot_options[f"Slot #{i+1} (Confidence: {conf:.2f})"] = (slot, (center_x, center_y))
                
                if slot_options:
                    selected_slot_name = st.selectbox("Choose an empty slot:", list(slot_options.keys()))
                    st.session_state.selected_slot, target_point = slot_options[selected_slot_name]
                    
                    # Preview selected slot
                    preview_img = image_rgb.copy()
                    selected_x1, selected_y1, selected_x2, selected_y2, _, _ = st.session_state.selected_slot
                    cv2.rectangle(preview_img, (selected_x1, selected_y1), (selected_x2, selected_y2), (0, 255, 255), 3)
                    
                    # Mark entrance point
                    cv2.circle(preview_img, st.session_state.entrance_point, 12, (255, 0, 0), -1)
                    
                    st.image(preview_img, caption="Selected Slot Preview", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">Please complete detection first in the Detection tab.</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### Navigation Path")
            
            if len(st.session_state.empty_slots) > 0 and st.session_state.entrance_point and st.session_state.selected_slot:
                # Get target point
                slot, target_point = slot_options[selected_slot_name]
                
                # Add vehicle size adjustment
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Vehicle Size Settings")
                col1, col2 = st.columns(2)
                
                with col1:
                    vehicle_width = st.slider("Vehicle Width (pixels)", 20, 100, 40, 5, 
                                              help="Adjust based on the scale of your image. Wider values ensure paths have enough clearance.")
                
                with col2:
                    vehicle_length = st.slider("Vehicle Length (pixels)", 40, 150, 70, 5,
                                              help="Longer values create more clearance in front/behind obstacles.")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Find and display navigation path
                if st.button("Generate Navigation Path", key="navigate_btn"):
                    with st.spinner("Calculating optimal route..."):
                        # Create grid with vehicle size consideration
                        grid = create_navigation_grid(
                            image_rgb.shape, 
                            st.session_state.occupied_slots,
                            vehicle_width=vehicle_width,
                            vehicle_length=vehicle_length
                        )
                        
                        # Find path
                        path = find_path(grid, st.session_state.entrance_point, target_point)
                        
                        # Save path
                        st.session_state.navigation_path = path
                        
                        # Draw navigation
                        if path:
                            navigation_img = draw_detections(
                                image_rgb, 
                                st.session_state.empty_slots, 
                                st.session_state.occupied_slots,
                                st.session_state.entrance_point,
                                target_point,
                                path
                            )
                            
                            # Display the navigation
                            st.markdown('<div class="success-box">Navigation path found successfully! The path is optimized for vehicle size.</div>', unsafe_allow_html=True)
                            
                            # Visualize the vehicle width along the path
                            nav_with_vehicle = navigation_img.copy()
                            if len(path) > 1:
                                for i in range(len(path) - 1):
                                    pt1 = (path[i].x, path[i].y)
                                    pt2 = (path[i + 1].x, path[i + 1].y)
                                    
                                    # Draw vehicle outline along the path
                                    # Calculate the direction vector
                                    dx = pt2[0] - pt1[0]
                                    dy = pt2[1] - pt1[1]
                                    # Normalize
                                    length = max(1, ((dx**2 + dy**2)**0.5))
                                    dx, dy = dx/length, dy/length
                                    # Get perpendicular vector
                                    perp_x, perp_y = -dy, dx
                                    
                                    # Draw vehicle width (perpendicular to path)
                                    half_width = vehicle_width // 2
                                    # Left side of vehicle
                                    left1 = (int(pt1[0] + perp_x * half_width), int(pt1[1] + perp_y * half_width))
                                    left2 = (int(pt2[0] + perp_x * half_width), int(pt2[1] + perp_y * half_width))
                                    # Right side of vehicle
                                    right1 = (int(pt1[0] - perp_x * half_width), int(pt1[1] - perp_y * half_width))
                                    right2 = (int(pt2[0] - perp_x * half_width), int(pt2[1] - perp_y * half_width))
                                    
                                    # Draw vehicle outline (semi-transparent)
                                    cv2.line(nav_with_vehicle, left1, left2, (255, 200, 0), 1)
                                    cv2.line(nav_with_vehicle, right1, right2, (255, 200, 0), 1)
                                    
                                    # Connect front and back
                                    if i == 0:
                                        cv2.line(nav_with_vehicle, left1, right1, (255, 200, 0), 1)
                                    if i == len(path) - 2:
                                        cv2.line(nav_with_vehicle, left2, right2, (255, 200, 0), 1)
                            
                            # Display with vehicle width visualization
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.image(nav_with_vehicle, caption=f"Navigation Path (Vehicle Width: {vehicle_width}px)", use_column_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Path information in a metrics card
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{len(path)}</div>
                                    <div class="metric-label">Path Length (steps)</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                # Estimate distance in arbitrary units
                                distance = sum(
                                    ((path[i+1].x - path[i].x)**2 + (path[i+1].y - path[i].y)**2)**0.5 
                                    for i in range(len(path)-1)
                                )
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{distance:.1f}</div>
                                    <div class="metric-label">Estimated Distance</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Save button
                            if st.button("Download Navigation Image", key="save_btn"):
                                # Convert OpenCV image to PIL
                                pil_img = Image.fromarray(navigation_img)
                                
                                # Save to a temporary file
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                                    img_path = tmp_file.name
                                    pil_img.save(tmp_file.name)
                                
                                # Offer download
                                with open(img_path, "rb") as file:
                                    btn = st.download_button(
                                        label="Click to Download",
                                        data=file,
                                        file_name="parking_navigation.png",
                                        mime="image/png"
                                    )
                        else:
                            st.markdown('<div class="error-box">No path found. Please try a different slot or adjust entrance position.</div>', unsafe_allow_html=True)
                
                # Show instructions if navigation not generated yet
                if not st.session_state.navigation_path:
                    st.markdown('<div class="info-box">Click "Generate Navigation Path" to calculate the route to your selected parking slot.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">Please complete setup in the Setup tab first.</div>', unsafe_allow_html=True)
    else:
        # Show welcome message and instructions
        st.markdown("""
        <div class="card" style="text-align: center; padding: 3rem;">
            <img src="https://img.icons8.com/color/96/000000/parking.png" width="100">
            <h2 style="color: white;">Welcome to Smart Parking Navigation</h2>
            <p>Upload a parking lot image to get started.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Instructions
        st.markdown("### How It Works")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="card" style="height: 200px;">
                <h4>1️⃣ Upload & Detect</h4>
                <p>Upload your parking lot image and run the AI detection to identify empty and occupied parking slots.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="card" style="height: 200px;">
                <h4>2️⃣ Setup Navigation</h4>
                <p>Set the entrance point and select your preferred empty parking slot from the detected options.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="card" style="height: 200px;">
                <h4>3️⃣ Navigate</h4>
                <p>Generate a navigation path that guides you from the entrance to your selected parking slot.</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
