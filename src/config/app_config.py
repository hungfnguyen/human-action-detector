"""
Application Configuration.

Settings for the yoga pose recognition system.
"""

# Application Settings
APP_CONFIG = {
    'app_name': 'Yoga Pose Recognition System',
    'version': '1.0.0',
    'default_model': 'yolov8m-pose.pt',
    'ml_classifier_path': './models/pose_classification.pth',
    'confidence_threshold': 0.5,
    'fps_target': 30,
}

# Pose Classes
POSE_CLASSES = [
    'Downdog',
    'Goddess', 
    'Plank',
    'Tree',
    'Warrior2'
]

# Visualization Settings
VISUALIZATION_CONFIG = {
    'skeleton_thickness': 3,
    'keypoint_radius': 5,
    'font_scale': 0.7,
    'font_thickness': 2,
    
    # Colors (BGR format)
    'colors': {
        'Plank': (0, 255, 0),      # Green
        'Tree': (0, 165, 255),      # Orange
        'Warrior2': (255, 0, 0),    # Blue
        'Goddess': (255, 0, 255),   # Magenta
        'Downdog': (0, 255, 255),   # Yellow
        'default': (255, 255, 255), # White
    },
    
    # UI Settings
    'ui_bg_color': (0, 0, 0),       # Black
    'ui_text_color': (255, 255, 255), # White
    'ui_alpha': 0.7,
}

# COCO Skeleton connections for drawing
COCO_SKELETON = [
    (5, 6),   # Shoulders
    (5, 7),   # Left arm
    (7, 9),   
    (6, 8),   # Right arm
    (8, 10),
    (5, 11),  # Left torso
    (6, 12),  # Right torso
    (11, 12), # Hip
    (11, 13), # Left leg
    (13, 15),
    (12, 14), # Right leg
    (14, 16),
]
