"""
Testing Agent for ConcussionSite multi-agent system.
Handles conversational guidance for screening tests.
"""

import logging
from typing import Dict, Any, Optional
from agents.setup import create_root_agent, call_agent
from agents.prompt import TESTING_AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class TestingAgent:
    """Agent that guides users through screening tests."""
    
    def __init__(self):
        """Initialize testing agent."""
        system_prompt = TESTING_AGENT_SYSTEM_PROMPT
        self.agent_config = create_root_agent(system_prompt)
        logger.info("Testing Agent initialized")
    
    def explain_test(self, test_name: str, user_message: str = "") -> str:
        """
        Explain a screening test to the user.
        
        Args:
            test_name: Name of the test ('flicker' or 'pursuit')
            user_message: Optional user message for context
        
        Returns:
            Explanation string
        """
        logger.debug(f"Explaining test: {test_name}")
        
        test_descriptions = {
            "flicker": "flickering light test that measures how your eyes respond to flashing lights",
            "pursuit": "smooth pursuit test where you follow a moving dot with your eyes"
        }
        
        description = test_descriptions.get(test_name.lower(), "screening test")
        
        prompt = f"""User asked about the {description}.

Explain what this test does in 2-3 short, simple sentences. Be supportive and non-alarming. Ask if they're ready to start."""
        
        try:
            response = call_agent(self.agent_config, prompt, [])
            return response
        except Exception as e:
            logger.error(f"Error explaining test: {e}")
            return f"The {description} helps us understand how your eyes respond. Ready to start? You can stop anytime by saying 'stop'."
    
    def guide_test_start(self, test_name: str) -> str:
        """
        Provide guidance for starting a test.
        
        Args:
            test_name: Name of the test
        
        Returns:
            Guidance message
        """
        logger.debug(f"Guiding start of test: {test_name}")
        
        messages = {
            "flicker": "I'll show you a flickering light. Just look at the screen naturally. The test takes about 30 seconds. Ready?",
            "pursuit": "I'll show you a moving dot. Follow it with your eyes, keeping your head still. This takes about 12 seconds. Ready?"
        }
        
        return messages.get(test_name.lower(), "Ready to start the test? You can stop anytime by saying 'stop'.")

