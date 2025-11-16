"""
Root Agent for ConcussionSite multi-agent system.
Manages conversation flow and coordinates with tools and child agents.
"""

import logging
from typing import Dict, Any, Optional, List

from agents.setup import create_root_agent, call_agent
from agents.prompt import ROOT_AGENT_SYSTEM_PROMPT
from agents.tools import (
    draft_email_for_mckinley,
    explain_metric,
    log_tool_call
)
from agents.writing_agent import WritingAgent
from agents.testing_agent import TestingAgent

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class RootAgent:
    """Root agent that manages conversation flow and tool calls."""
    
    def __init__(self, metrics: Dict[str, Any], pursuit_metrics: Dict[str, Any],
                 symptoms: Dict[str, bool], subjective_score: int,
                 risk_assessment: Dict[str, Any]):
        """
        Initialize root agent with screening data.
        
        Args:
            metrics: Screening metrics dictionary
            pursuit_metrics: Smooth pursuit metrics dictionary
            symptoms: Symptoms dictionary
            subjective_score: User's subjective feeling score (1-10)
            risk_assessment: Risk assessment dictionary
        """
        self.metrics = metrics
        self.pursuit_metrics = pursuit_metrics
        self.symptoms = symptoms
        self.subjective_score = subjective_score
        self.risk_assessment = risk_assessment
        
        # Build context for agent
        self.context = self._build_context()
        
        # Initialize agent
        system_prompt = f"{ROOT_AGENT_SYSTEM_PROMPT}\n\n{self.context}"
        self.agent_config = create_root_agent(system_prompt)
        
        # Initialize child agents
        self.writing_agent = WritingAgent()
        self.testing_agent = TestingAgent()
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info("Root Agent initialized")
    
    def _build_context(self) -> str:
        """Build concise context string from screening data."""
        pursuit_text = "Not performed."
        if self.pursuit_metrics:
            pursuit_text = f"Error: {self.pursuit_metrics.get('mean_error', 'N/A'):.1f}px"
        
        symptom_list = []
        if self.symptoms.get("headache"):
            symptom_list.append("headache")
        if self.symptoms.get("nausea"):
            symptom_list.append("nausea")
        if self.symptoms.get("dizziness"):
            symptom_list.append("dizziness")
        if self.symptoms.get("light_sensitivity"):
            symptom_list.append("light sensitivity")
        
        symptoms_text = ", ".join(symptom_list) if symptom_list else "none"
        
        context = f"""Results:
- Blink: {self.metrics['baseline_blink_rate']:.1f} → {self.metrics['flicker_blink_rate']:.1f} blinks/min
- Eye-closed: {self.metrics['eye_closed_fraction']:.1%}
- Gaze: {self.metrics['gaze_off_fraction']:.1%}
- Tracking: {pursuit_text}
- Symptoms: {symptoms_text}
- Feeling: {self.subjective_score}/10
- Risk: {self.risk_assessment['risk_level']} ({self.risk_assessment['risk_score']}/10)

Not a diagnosis."""
        
        return context
    
    def process_message(self, user_message: str, current_email_draft: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process a user message and return agent response.
        
        Args:
            user_message: User's message
        
        Returns:
            Dictionary with response and metadata
        """
        try:
            logger.debug(f"Processing message: {user_message[:100]}...")
            
            # Check for exit keywords
            exit_keywords = ["stop", "exit", "quit", "end session", "end", "done"]
            if user_message.lower().strip() in exit_keywords:
                logger.info("User requested to end session")
                return {
                    "response": "Thank you for using ConcussionSite. Take care!",
                    "should_end": True,
                    "tool_called": None
                }
            
            # Check for tool calls based on user intent
            tool_result = self._check_tool_calls(user_message, current_email_draft)
            if tool_result:
                return tool_result
            
            # Build full prompt with conversation history
            full_prompt = self._build_prompt(user_message)
            
            # Call agent
            response = call_agent(
                self.agent_config,
                full_prompt,
                self.conversation_history
            )
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Keep history manageable (last 20 messages)
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            logger.info("Response generated successfully")
            return {
                "response": response,
                "should_end": False,
                "tool_called": None
            }
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return {
                "response": "I encountered an error. Please try again or rephrase your message.",
                "should_end": False,
                "tool_called": None
            }
    
    def _check_tool_calls(self, user_message: str, current_email_draft: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """Check if user message triggers a tool call."""
        message_lower = user_message.lower()
        
        # IMPORTANT: Check for send request FIRST, before edit detection
        # This prevents "send" from being interpreted as an edit request
        if current_email_draft:
            send_phrases = [
                "send", "send it", "send that", "send the email", "send that email",
                "yes send", "yes, send", "go ahead and send", "please send",
                "i want you to send", "send now", "send email"
            ]
            # Check if message is primarily about sending (not editing)
            is_send_request = (
                any(phrase in message_lower for phrase in send_phrases) and
                not any(word in message_lower for word in ["edit", "change", "revise", "update", "modify", "add", "remove"])
            )
            if is_send_request:
                # Don't handle send here - let runner.py handle it after processing
                # Just return None so the message gets processed normally
                return None
        
        # Check if user wants to edit existing email draft
        if current_email_draft:
            edit_keywords = ["change", "edit", "revise", "update", "modify", "add", "remove", "shorten", "tone", "formal", "name", "netid"]
            if any(keyword in message_lower for keyword in edit_keywords):
                logger.info("Email edit request detected")
                try:
                    edited_draft = self.writing_agent.edit_email_draft(
                        current_draft=current_email_draft,
                        user_request=user_message,
                        conversation_history=self.conversation_history[-5:]  # Last 5 messages for context
                    )
                    log_tool_call("edit_email_draft", {"request": user_message}, edited_draft)
                    
                    return {
                        "response": "I've updated the email draft. Review it below. You can ask for more changes or say 'yes, send it' when ready.",
                        "should_end": False,
                        "tool_called": "edit_email",
                        "email_draft": edited_draft
                    }
                except Exception as e:
                    logger.error(f"Error editing email: {e}")
                    return {
                        "response": f"I had trouble editing the email: {str(e)}. Please try rephrasing your request.",
                        "should_end": False,
                        "tool_called": None
                    }
        
        # Check for email draft request - work regardless of risk score if user explicitly asks
        email_keywords = ["email", "mckinley", "referral", "draft", "send"]
        draft_keywords = ["draft", "create", "write", "make", "generate"]
        
        # Check if user wants to draft an email
        wants_email = any(keyword in message_lower for keyword in email_keywords)
        wants_draft = any(keyword in message_lower for keyword in draft_keywords)
        affirmative = any(word in message_lower for word in ["yes", "sure", "ok", "please", "yeah", "yep"])
        
        # Trigger if: (email keyword + draft/affirmative) OR (just "draft me" type request)
        if (wants_email and (wants_draft or affirmative)) or (wants_draft and ("me" in message_lower or "email" in message_lower)):
            logger.info("Email draft tool triggered - instant return")
            try:
                email_result = draft_email_for_mckinley(
                    self.metrics,
                    self.risk_assessment,
                    self.symptoms,
                    self.subjective_score
                )
                log_tool_call("draft_email_for_mckinley", {}, email_result)
                
                return {
                    "response": "Here's your email draft. Review it below. You can ask me to edit it (e.g., 'add my name', 'make it more formal', 'shorten it') or say 'yes, send it' when ready.",
                    "should_end": False,
                    "tool_called": "draft_email",
                    "email_draft": email_result
                }
            except Exception as e:
                logger.error(f"Error drafting email: {e}")
                return {
                    "response": f"I encountered an error while drafting the email: {str(e)}. Please try again.",
                    "should_end": False,
                    "tool_called": None
                }
        
        return None
    
    def _build_prompt(self, user_message: str) -> str:
        """Build concise prompt with context and history."""
        history_text = ""
        if self.conversation_history:
            history_lines = []
            for msg in self.conversation_history[-5:]:  # Last 5 messages only
                role = "U" if msg["role"] == "user" else "A"
                content = msg['content'][:100]  # Truncate long messages
                history_lines.append(f"{role}: {content}")
            history_text = "\n".join(history_lines)
        
        prompt = f"""History:
{history_text}

User: {user_message}

Respond briefly (2-3 sentences max). Be supportive. No diagnosing."""
        
        return prompt
    
    def get_initial_greeting(self) -> str:
        """Get initial greeting message from agent (short version)."""
        logger.debug("Getting initial greeting")
        
        risk_score = self.risk_assessment.get("risk_score", 0)
        risk_level = self.risk_assessment.get("risk_level", "MINIMAL")
        
        greeting = f"Your screening is complete. Your risk level is {risk_level} ({risk_score}/10).\n\nI can help explain your results or answer questions. What would you like to know?"
        
        if risk_score >= 7:
            greeting += "\n\nYour results suggest some concerns. Would you like me to draft an email to McKinley Health Center for evaluation?"
        
        return greeting

