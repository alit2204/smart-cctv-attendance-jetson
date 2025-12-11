"""
Smart CCTV Attendance System with Tailgating Detection
Main package initialization
"""

from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer
from .person_detector import PersonDetector
from .tracker import CentroidTracker
from .database_manager import DatabaseManager
from . import utils

__version__ = "1.0.0"
__author__ = "Umbreen Shah Nawaz, Muhammad Ali Tahir, Hamza Khan"

__all__ = [
    'FaceDetector',
    'FaceRecognizer',
    'PersonDetector',
    'CentroidTracker',
    'DatabaseManager',
    'utils'
]
