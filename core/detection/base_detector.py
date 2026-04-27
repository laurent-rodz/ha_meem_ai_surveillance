from abc import ABC, abstractmethod
import numpy as np
from typing import List
from .face import Face

class BaseDetector(ABC):
    """Abstract base class for face detectors."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = config.get("pipeline", {}).get("device", "cuda")

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Face]:
        """Detect faces in an image.
        
        Args:
            image: BGR image as a numpy array.
            
        Returns:
            List of Face objects.
        """
        pass
