"""
Tool functions for the ConcussionSite multi-agent system.
These are callable functions that agents can use to perform actions.
"""

import logging
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def draft_email_for_mckinley(
    metrics: Dict[str, Any],
    risk_assessment: Dict[str, Any],
    symptoms: Dict[str, bool],
    subjective_score: int,
    user_name: Optional[str] = None
) -> Dict[str, str]:
    """
    Draft a professional email to McKinley Health Center requesting evaluation.
    Returns structured JSON for instant display.
    
    Args:
        metrics: Dictionary containing screening metrics
        risk_assessment: Dictionary containing risk level and score
        symptoms: Dictionary of symptom boolean values
        subjective_score: User's subjective feeling score (1-10)
        user_name: Optional user name to include in email
    
    Returns:
        Dictionary with 'subject', 'body', and 'status' keys
    """
    logger.debug("draft_email_for_mckinley called - instant return")
    
    # Build symptom list
    symptom_list = []
    if symptoms.get("headache"):
        symptom_list.append("headache")
    if symptoms.get("nausea"):
        symptom_list.append("nausea")
    if symptoms.get("dizziness"):
        symptom_list.append("dizziness")
    if symptoms.get("light_sensitivity"):
        symptom_list.append("light sensitivity")
    
    symptoms_text = ", ".join(symptom_list) if symptom_list else "none reported"
    
    # Build email body (concise)
    name_line = f"{user_name}\n" if user_name else ""
    
    # Handle mean_error - might be in metrics or pursuit_metrics, and might be missing
    mean_error = metrics.get('mean_error')
    if mean_error is None:
        # Try to get from pursuit_metrics if available
        pursuit_metrics = risk_assessment.get('pursuit_metrics', {})
        mean_error = pursuit_metrics.get('mean_error')
    
    # Format mean_error safely
    if mean_error is not None and isinstance(mean_error, (int, float)):
        tracking_error_str = f"{mean_error:.1f} px"
    else:
        tracking_error_str = "N/A"
    
    body = f"""Dear McKinley Health Center Staff,

I completed a vision-based concussion screening and would like to request an evaluation.

{name_line}Symptoms: {symptoms_text}
Feeling level: {subjective_score}/10

Metrics:
- Blink rates: {metrics['baseline_blink_rate']:.1f} → {metrics['flicker_blink_rate']:.1f} blinks/min
- Eye-closed: {metrics['eye_closed_fraction']:.1%}
- Gaze aversion: {metrics['gaze_off_fraction']:.1%}
- Tracking error: {tracking_error_str}

Risk: {risk_assessment['risk_level']} ({risk_assessment['risk_score']}/10)

I understand this is not a diagnosis, but the results suggest patterns that may warrant evaluation. I'd appreciate discussing these findings with a healthcare provider.

Thank you,
{user_name if user_name else '[Your Name]'}
[Your NetID]"""
    
    result = {
        "subject": "Request for Evaluation - Concussion Screening Results",
        "body": body,
        "status": "ready"
    }
    
    logger.info("Email draft generated instantly")
    return result


def explain_metric(metric_name: str, metric_value: Any, context: Optional[str] = None) -> str:
    """
    Explain a medical metric in simple, supportive language (short version).
    
    Args:
        metric_name: Name of the metric
        metric_value: The value of the metric
        context: Optional additional context
    
    Returns:
        Brief plain-language explanation
    """
    logger.debug(f"explain_metric called: {metric_name} = {metric_value}")
    
    explanations = {
        "baseline_blink_rate": f"Baseline blink rate: {metric_value:.1f} blinks/min. Normal is 15-20. This measures natural blinking.",
        "flicker_blink_rate": f"Flicker blink rate: {metric_value:.1f} blinks/min. Shows how blinking changes with flickering light.",
        "blink_rate_delta": f"Blink rate changed by {metric_value:.1f} blinks/min. Increases may suggest light sensitivity.",
        "eye_closed_fraction": f"Eyes closed {metric_value:.1%} of the time. Higher values may indicate light avoidance.",
        "gaze_off_fraction": f"Looked away {metric_value:.1%} of the time. This may suggest visual discomfort.",
        "mean_error": f"Tracking error: {metric_value:.1f} px. Lower is better. Higher may suggest difficulty following movement."
    }
    
    explanation = explanations.get(metric_name, f"{metric_name}: {metric_value}. This measures eye movement patterns.")
    
    if context:
        explanation += f" {context}"
    
    explanation += " Not a diagnosis - just patterns worth discussing."
    
    logger.info(f"Metric explanation generated for {metric_name}")
    return explanation


def log_tool_call(tool_name: str, args: Dict[str, Any], result: Any) -> None:
    """
    Log tool calls for debugging.
    
    Args:
        tool_name: Name of the tool called
        args: Arguments passed to the tool
        result: Result returned by the tool
    """
    logger.debug(f"TOOL CALL: {tool_name}")
    logger.debug(f"  Args: {args}")
    logger.debug(f"  Result: {str(result)[:100]}...")  # Truncate long results

