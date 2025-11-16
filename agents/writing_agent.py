"""
Writing Agent for ConcussionSite multi-agent system.
Handles email drafting and conversational editing.
"""

import logging
from typing import Dict, Any, Optional
from agents.setup import create_root_agent, call_agent
from agents.prompt import WRITING_AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class WritingAgent:
    """Agent that handles email drafting and editing."""
    
    def __init__(self):
        """Initialize writing agent."""
        system_prompt = WRITING_AGENT_SYSTEM_PROMPT
        self.agent_config = create_root_agent(system_prompt)
        logger.info("Writing Agent initialized")
    
    def edit_email_draft(
        self,
        current_draft: Dict[str, str],
        user_request: str,
        conversation_history: Optional[list] = None
    ) -> Dict[str, str]:
        """
        Edit an email draft based on user's conversational request.
        Preserves formatting and spacing.
        
        Args:
            current_draft: Current email draft with 'subject' and 'body'
            user_request: User's editing request (e.g., "make it more formal", "add my name")
            conversation_history: Optional conversation history for context
        
        Returns:
            Updated email draft dictionary
        """
        logger.debug(f"Editing email draft based on: {user_request}")
        
        # Preserve original body structure
        original_body = current_draft.get('body', '')
        original_subject = current_draft.get('subject', '')
        
        # Extract name and NetID if they exist in the request
        user_name = None
        netid = None
        
        # Look for name pattern: "name [Name]" or "name: [Name]"
        import re
        name_match = re.search(r'(?:name|my name)[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', user_request, re.IGNORECASE)
        if name_match:
            user_name = name_match.group(1)
        
        # Look for NetID pattern: "netid [id]" or "net id [id]"
        netid_match = re.search(r'(?:netid|net\s+id)[\s:]+([a-z0-9]+)', user_request, re.IGNORECASE)
        if netid_match:
            netid = netid_match.group(1)
        
        # Handle specific editing requests
        request_lower = user_request.lower()
        
        # If just adding name/NetID, do it directly without LLM
        if (user_name or netid) and not any(word in request_lower for word in ["tone", "formal", "shorten", "change", "revise"]):
            updated_body = original_body
            if user_name:
                # Replace [Your Name] placeholder or add name
                if '[Your Name]' in updated_body:
                    updated_body = updated_body.replace('[Your Name]', user_name)
                elif '[Your NetID]' in updated_body:
                    # Add name before NetID line
                    updated_body = updated_body.replace('[Your NetID]', f'{user_name}\n[Your NetID]')
                else:
                    # Add at the end before NetID
                    updated_body = updated_body.replace('[Your NetID]', f'{user_name}\n[Your NetID]')
            
            if netid:
                updated_body = updated_body.replace('[Your NetID]', netid)
            
            return {
                "subject": original_subject,
                "body": updated_body,
                "status": "edited"
            }
        
        # For other edits, use LLM but with strict formatting instructions
        prompt = f"""Current email draft:

Subject: {original_subject}

Body:
{original_body}

User wants to: {user_request}

IMPORTANT: Preserve the exact formatting, line breaks, and spacing of the original email. Only modify the content as requested. Keep the same structure:
- Greeting line
- Blank line
- Introduction paragraph
- Blank line
- Symptoms line
- Feeling level line
- Blank line
- Metrics section (with bullet points)
- Blank line
- Risk line
- Blank line
- Closing paragraph
- Blank line
- Thank you line
- Blank line
- Name/NetID lines

Return ONLY the revised email in this exact format:
Subject: [subject line]

Body:
[email body with preserved formatting]"""
        
        try:
            response = call_agent(
                self.agent_config,
                prompt,
                conversation_history or []
            )
            
            # Parse with better formatting preservation
            updated_draft = self._parse_email_response(response, current_draft)
            
            # Ensure proper line breaks are preserved
            if updated_draft.get('body'):
                # Normalize line breaks but preserve intentional spacing
                body = updated_draft['body']
                # Replace multiple spaces with single space (except indentation)
                lines = body.split('\n')
                cleaned_lines = []
                for line in lines:
                    # Preserve empty lines
                    if not line.strip():
                        cleaned_lines.append('')
                    else:
                        # Clean up extra spaces but preserve structure
                        cleaned_lines.append(' '.join(line.split()))
                updated_draft['body'] = '\n'.join(cleaned_lines)
            
            logger.info("Email draft edited successfully")
            return updated_draft
            
        except Exception as e:
            logger.error(f"Error editing email draft: {e}")
            return current_draft  # Return original if editing fails
    
    def _parse_email_response(self, response: str, original_draft: Dict[str, str]) -> Dict[str, str]:
        """
        Parse agent response to extract email subject and body.
        Preserves formatting and spacing.
        """
        lines = response.split('\n')
        subject = original_draft.get('subject', '')
        body_lines = []
        in_body = False
        found_subject = False
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Extract subject
            if line_lower.startswith('subject:'):
                subject = line.split(':', 1)[1].strip()
                found_subject = True
                continue
            
            # Start of body
            if line_lower.startswith('body:'):
                in_body = True
                # Add the part after "Body:" if it exists
                if ':' in line and len(line.split(':', 1)[1].strip()) > 0:
                    body_lines.append(line.split(':', 1)[1].strip())
                continue
            
            # If we found subject and haven't started body, look for body start
            if found_subject and not in_body:
                # Skip empty lines between subject and body
                if not line_stripped:
                    continue
                # If we hit non-empty content after subject, it's the body
                in_body = True
            
            # Collect body lines, preserving empty lines for formatting
            if in_body:
                body_lines.append(line.rstrip())  # Preserve left spacing, trim right
        
        # Clean up body - remove leading/trailing empty lines but preserve internal spacing
        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()
        
        # Join with newlines, preserving the structure
        body = '\n'.join(body_lines) if body_lines else ''
        
        # Fallback to original if parsing failed
        if not body or len(body.strip()) < 50:
            body = original_draft.get('body', '')
            subject = original_draft.get('subject', '')
        
        return {
            "subject": subject or original_draft.get('subject', ''),
            "body": body,
            "status": "edited"
        }

