"""
Screening Flow Agent - Integrates tests and questions into conversational flow.
This allows the agent to guide users through the entire screening process.
"""

import logging
from typing import Dict, Any, Optional, List
import cv2
import time

from stimulus.flicker import create_flicker_window, flicker_display
from stimulus.pursuit import run_dot_pursuit
from tracking.facemesh import face_mesh
from analysis.metrics import calculate_metrics
from analysis.risk import assess_concussion_risk

logger = logging.getLogger(__name__)


class ScreeningFlow:
    """Manages the screening test flow within the agent conversation."""
    
    def __init__(self):
        """Initialize screening flow."""
        self.cap = None
        self.flicker_window = None
        self.baseline_metrics = None
        self.flicker_metrics = None
        self.pursuit_metrics = None
        self.symptoms = {}
        self.subjective_score = None
        self.metrics = None
        self.risk_assessment = None
        self.completed_phases = {
            "baseline": False,
            "flicker": False,
            "pursuit": False,
            "symptoms": False,
            "subjective": False
        }
        logger.info("ScreeningFlow initialized")
    
    def initialize_camera(self) -> bool:
        """Initialize webcam."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("Could not open webcam")
                return False
            logger.info("Camera initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def ask_symptom_question(self, symptom_key: str) -> str:
        """Get the question text for a symptom."""
        questions = {
            "headache": "Do you have a headache?",
            "nausea": "Do you feel nauseous?",
            "dizziness": "Do you feel dizzy?",
            "light_sensitivity": "Are you sensitive to light?"
        }
        return questions.get(symptom_key, "")
    
    def process_symptom_answer(self, symptom_key: str, answer: str) -> bool:
        """Process a symptom answer and return True if all symptoms are collected."""
        answer_lower = answer.lower().strip()
        is_yes = answer_lower in ["y", "yes", "true", "1"]
        self.symptoms[symptom_key] = is_yes
        
        # Check if all symptoms collected
        all_symptoms = ["headache", "nausea", "dizziness", "light_sensitivity"]
        if all(key in self.symptoms for key in all_symptoms):
            self.completed_phases["symptoms"] = True
            return True
        return False
    
    def get_next_symptom(self) -> Optional[str]:
        """Get the next symptom that needs to be asked."""
        all_symptoms = ["headache", "nausea", "dizziness", "light_sensitivity"]
        for symptom in all_symptoms:
            if symptom not in self.symptoms:
                return symptom
        return None
    
    def process_subjective_score(self, score_str: str) -> bool:
        """Process subjective feeling score (1-10)."""
        try:
            score = int(score_str.strip())
            if 1 <= score <= 10:
                self.subjective_score = score
                self.completed_phases["subjective"] = True
                return True
        except ValueError:
            pass
        return False
    
    def run_baseline_test(self) -> Dict[str, Any]:
        """Run baseline phase and return metrics."""
        if self.completed_phases["baseline"]:
            return self.baseline_metrics
        
        if not self.cap:
            if not self.initialize_camera():
                return {"error": "Could not initialize camera"}
        
        try:
            from main import run_phase
            self.baseline_metrics = run_phase(
                self.cap, 
                self.flicker_window or create_flicker_window(), 
                "Baseline", 
                duration_sec=8, 
                flicker=False
            )
            self.completed_phases["baseline"] = True
            return self.baseline_metrics
        except Exception as e:
            logger.error(f"Error running baseline test: {e}")
            return {"error": str(e)}
    
    def run_flicker_test(self) -> Dict[str, Any]:
        """Run flicker phase and return metrics."""
        if self.completed_phases["flicker"]:
            return self.flicker_metrics
        
        if not self.completed_phases["baseline"]:
            return {"error": "Baseline test must be completed first"}
        
        try:
            from main import run_phase
            if not self.flicker_window:
                self.flicker_window = create_flicker_window()
            
            self.flicker_metrics = run_phase(
                self.cap,
                self.flicker_window,
                "Flicker",
                duration_sec=15,
                flicker=True
            )
            self.completed_phases["flicker"] = True
            return self.flicker_metrics
        except Exception as e:
            logger.error(f"Error running flicker test: {e}")
            return {"error": str(e)}
    
    def run_pursuit_test(self) -> Dict[str, Any]:
        """Run smooth pursuit phase and return metrics."""
        if self.completed_phases["pursuit"]:
            return self.pursuit_metrics
        
        try:
            self.pursuit_metrics = run_dot_pursuit(duration_sec=12, cap=self.cap)
            self.completed_phases["pursuit"] = True
            return self.pursuit_metrics
        except Exception as e:
            logger.error(f"Error running pursuit test: {e}")
            return {"error": str(e)}
    
    def calculate_final_metrics(self) -> Dict[str, Any]:
        """Calculate final metrics and risk assessment."""
        if not (self.baseline_metrics and self.flicker_metrics):
            return {"error": "Baseline and flicker tests must be completed"}
        
        self.metrics = calculate_metrics(self.baseline_metrics, self.flicker_metrics)
        
        if self.subjective_score is None:
            return {"error": "Subjective score required"}
        
        self.risk_assessment = assess_concussion_risk(
            self.metrics,
            self.symptoms,
            self.pursuit_metrics,
            self.subjective_score
        )
        
        return {
            "metrics": self.metrics,
            "pursuit_metrics": self.pursuit_metrics,
            "symptoms": self.symptoms,
            "subjective_score": self.subjective_score,
            "risk_assessment": self.risk_assessment
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.flicker_window:
            try:
                cv2.destroyWindow(self.flicker_window)
            except:
                pass
        if self.cap:
            try:
                self.cap.release()
            except:
                pass

