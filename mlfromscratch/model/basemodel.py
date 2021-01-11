# -*- coding: utf-8 -*-
"""Abstract base model"""

from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""
    def __init__(self):

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass