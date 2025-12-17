"""
ML Pose Classifier - Neural Network for pose classification.

Uses trained MLP to classify yoga poses from keypoints.
"""

import torch
import torch.nn as nn
from typing import Tuple, List
import numpy as np


class NeuralNet(nn.Module):
    """
    Simple MLP for pose classification.
    
    Architecture:
        Input (24) → Hidden (256) → ReLU → Output (5)
    """
    
    def __init__(self, input_size: int = 24, hidden_size: int = 256, num_classes: int = 5):
        """
        Initialize neural network.
        
        Args:
            input_size: Number of input features (24 = 12 keypoints × 2)
            hidden_size: Hidden layer size
            num_classes: Number of yoga pose classes (5)
        """
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """Forward pass."""
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


class MLPoseClassifier:
    """
    ML-based pose classifier using trained neural network.
    
    Example:
        >>> classifier = MLPoseClassifier('./models/pose_classification.pth')
        >>> keypoints = [x1, y1, x2, y2, ...]  # 34 values
        >>> pose_name, confidence = classifier.predict(keypoints)
        >>> print(f"{pose_name}: {confidence:.1%}")
    """
    
    def __init__(self, model_path: str):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to trained model weights (.pth file)
        """
        self.model_path = model_path
        self.classes = ['Downdog', 'Goddess', 'Plank', 'Tree', 'Warrior2']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self):
        """Load trained model from checkpoint."""
        try:
            self.model = NeuralNet(input_size=24, hidden_size=256, num_classes=5)
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self.model.eval()  # Set to evaluation mode
            self.model.to(self.device)
            print(f"✅ Loaded ML classifier: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def preprocess(self, keypoints: List[float]) -> torch.Tensor:
        """
        Preprocess keypoints for model input.
        
        Args:
            keypoints: List of 34 values (17 keypoints × 2)
                      Format: [x1, y1, x2, y2, ..., x17, y17]
        
        Returns:
            Tensor of shape (24,) - removes head keypoints (first 5 × 2 = 10 values)
        """
        if len(keypoints) != 34:
            raise ValueError(f"Expected 34 keypoint values, got {len(keypoints)}")
        
        # Remove head keypoints (indices 0-4 = first 10 values)
        # Keep body keypoints (indices 5-16 = last 24 values)
        body_keypoints = keypoints[10:]  # [x6, y6, x7, y7, ..., x17, y17]
        
        if len(body_keypoints) != 24:
            raise ValueError(f"Expected 24 body keypoint values, got {len(body_keypoints)}")
        
        # Convert to tensor
        tensor = torch.tensor(body_keypoints, dtype=torch.float32)
        return tensor.to(self.device)
    
    def predict(self, keypoints: List[float]) -> Tuple[str, float]:
        """
        Predict pose from keypoints.
        
        Args:
            keypoints: List of 34 normalized values (17 keypoints × 2)
        
        Returns:
            Tuple of (pose_name, confidence)
            - pose_name: One of ['Downdog', 'Goddess', 'Plank', 'Tree', 'Warrior2']
            - confidence: Float between 0 and 1
        """
        # Preprocess input
        input_tensor = self.preprocess(keypoints)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension: (1, 24)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)  # (1, 5)
            probabilities = torch.softmax(output, dim=-1)  # Convert logits to probabilities
            confidence, predicted = torch.max(probabilities, dim=-1)
        
        # Get results
        pose_index = predicted.item()
        pose_name = self.classes[pose_index]
        confidence_value = confidence.item()
        
        return pose_name, confidence_value
    
    def __call__(self, keypoints: List[float]) -> str:
        """
        Shortcut for predict() - returns only pose name (for compatibility).
        
        Args:
            keypoints: Can be either:
                - List of 34 values (full keypoints)
                - List of 24 values (body keypoints only)
                - Torch tensor
        
        Returns:
            Pose name string
        """
        # Handle different input formats for backward compatibility
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy().tolist()
        
        # If input is already preprocessed (24 values), pad it
        if len(keypoints) == 24:
            # Pad with zeros for head keypoints
            keypoints = [0.0] * 10 + keypoints
        
        pose_name, _ = self.predict(keypoints)
        return pose_name
