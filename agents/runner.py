"""
Agent runner with web UI for ConcussionSite multi-agent system.
Uses Flask to provide a browser-based interface.
"""

import os
import logging
from typing import Dict, Any, Optional
from flask import Flask, render_template_string, request, jsonify, session
from flask_cors import CORS

from agents.root_agent import RootAgent
from agents.setup import MODEL
from email_service.email_service import send_email_oauth, check_oauth_setup

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
CORS(app)

# Global agent instance (will be set when conversation starts)
root_agent: Optional[RootAgent] = None
current_email_draft: Optional[str] = None

# HTML template for web UI - Modern Apple Health App Style
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ConcussionSite - AI Assistant</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { 
            margin: 0; 
            padding: 0; 
            box-sizing: border-box; 
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            font-size: 16px;
            line-height: 1.5;
        }
        
        .container {
            background: #ffffff;
            border-radius: 32px;
            box-shadow: 
                0 20px 60px rgba(0, 0, 0, 0.08),
                0 0 0 1px rgba(0, 0, 0, 0.04);
            width: 100%;
            max-width: 700px;
            height: 90vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            backdrop-filter: blur(20px);
        }
        
        .header {
            background: linear-gradient(135deg, #a8e6cf 0%, #b8d4f0 100%);
            color: #2c3e50;
            padding: 24px 28px;
            text-align: center;
            border-radius: 32px 32px 0 0;
            position: relative;
            overflow: hidden;
        }
        
        .header::before {
            content: '🚧';
            position: absolute;
            top: 20px;
            right: 30px;
            font-size: 36px;
            opacity: 0.2;
            transform: rotate(-12deg);
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: rotate(-12deg) translateY(0px); }
            50% { transform: rotate(-12deg) translateY(-5px); }
        }
        
        .header h1 { 
            font-size: 22px; 
            font-weight: 600;
            margin-bottom: 4px;
            letter-spacing: -0.3px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        
        .header h1::before {
            content: '🏥';
            font-size: 24px;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
        }
        
        .header p { 
            opacity: 0.7; 
            font-size: 14px;
            font-weight: 400;
        }
        
        .chat-area {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            background: #fafbfc;
            scroll-behavior: smooth;
        }
        
        .chat-area::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-area::-webkit-scrollbar-track {
            background: transparent;
        }
        
        .chat-area::-webkit-scrollbar-thumb {
            background: rgba(0, 0, 0, 0.1);
            border-radius: 3px;
        }
        
        .message {
            margin-bottom: 16px;
            display: flex;
            animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            opacity: 0;
            animation-fill-mode: forwards;
        }
        
        @keyframes fadeInUp {
            from { 
                opacity: 0; 
                transform: translateY(12px) scale(0.96);
            }
            to { 
                opacity: 1; 
                transform: translateY(0) scale(1);
            }
        }
        
        .message.user { 
            justify-content: flex-end; 
        }
        
        .message.assistant { 
            justify-content: flex-start; 
        }
        
        .message-content {
            max-width: 65%;
            padding: 14px 18px;
            border-radius: 24px;
            word-wrap: break-word;
            font-size: 16px;
            line-height: 1.5;
            box-shadow: 
                0 2px 8px rgba(0, 0, 0, 0.06),
                0 0 0 1px rgba(0, 0, 0, 0.02);
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, #a8e6cf 0%, #88d4a3 100%);
            color: #1a4d3a;
            border-bottom-right-radius: 6px;
            font-weight: 500;
            box-shadow: 
                0 2px 8px rgba(168, 230, 207, 0.2),
                0 0 0 1px rgba(168, 230, 207, 0.1);
        }
        
        .message.assistant .message-content {
            background: #ffffff;
            color: #2c3e50;
            border-bottom-left-radius: 6px;
            box-shadow: 
                0 2px 12px rgba(0, 0, 0, 0.04),
                0 0 0 1px rgba(0, 0, 0, 0.03);
        }
        
        .input-area {
            padding: 20px 24px;
            background: #ffffff;
            border-top: 1px solid rgba(0, 0, 0, 0.06);
            display: flex;
            gap: 12px;
            align-items: center;
        }
        
        #messageInput {
            flex: 1;
            padding: 14px 20px;
            border: 2px solid rgba(0, 0, 0, 0.08);
            border-radius: 24px;
            font-size: 16px;
            outline: none;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            background: #f8f9fa;
            font-family: inherit;
        }
        
        #messageInput:focus {
            border-color: #a8e6cf;
            background: #ffffff;
            box-shadow: 0 0 0 4px rgba(168, 230, 207, 0.1);
        }
        
        #sendButton {
            padding: 14px 28px;
            background: linear-gradient(135deg, #a8e6cf 0%, #88d4a3 100%);
            color: #1a4d3a;
            border: none;
            border-radius: 24px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            box-shadow: 0 2px 8px rgba(168, 230, 207, 0.3);
        }
        
        #sendButton:hover:not(:disabled) {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(168, 230, 207, 0.4);
        }
        
        #sendButton:active:not(:disabled) {
            transform: translateY(0);
        }
        
        #sendButton:disabled {
            background: #e0e0e0;
            color: #999;
            cursor: not-allowed;
            box-shadow: none;
        }
        
        .loading {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: #7f8c8d;
            font-size: 14px;
        }
        
        .loading-spinner {
            width: 16px;
            height: 16px;
            border: 2px solid rgba(168, 230, 207, 0.2);
            border-top: 2px solid #a8e6cf;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .email-draft {
            background: linear-gradient(135deg, #fff9e6 0%, #fff4d6 100%);
            border: 1px solid rgba(255, 193, 7, 0.25);
            border-radius: 20px;
            padding: 20px;
            margin: 12px 0;
            white-space: pre-wrap;
            font-family: -apple-system, BlinkMacSystemFont, 'SF Mono', monospace;
            font-size: 14px;
            line-height: 1.6;
            color: #5d4037;
            box-shadow: 
                0 4px 16px rgba(255, 193, 7, 0.15),
                0 0 0 1px rgba(255, 193, 7, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.5);
            animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
        }
        
        .email-draft::before {
            content: '✉️';
            position: absolute;
            top: 15px;
            right: 15px;
            font-size: 20px;
            opacity: 0.3;
        }
        
        .email-draft .email-subject {
            font-weight: 600;
            margin-bottom: 12px;
            color: #3e2723;
            font-size: 15px;
        }
        
        .email-draft .email-body {
            color: #5d4037;
        }
        
        .email-draft .email-edit-hint {
            margin-top: 16px;
            padding-top: 12px;
            border-top: 1px solid rgba(255, 193, 7, 0.2);
            font-size: 13px;
            color: #7f5539;
            font-style: italic;
            opacity: 0.8;
        }
        
        .email-loading {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: #7f8c8d;
            font-size: 14px;
            padding: 12px 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ConcussionSite Assistant</h1>
            <p>Your supportive guide for concussion screening</p>
        </div>
        <div class="chat-area" id="chatArea">
            <div class="message assistant">
                <div class="message-content">
                    <div class="loading">
                        <div class="loading-spinner"></div>
                        <span>Starting conversation...</span>
                    </div>
                </div>
            </div>
        </div>
        <div class="input-area">
            <input type="text" id="messageInput" placeholder="Type your message..." autocomplete="off">
            <button id="sendButton" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let conversationEnded = false;
        let debounceTimer = null;
        let isProcessing = false;

        // Smooth scroll function
        function smoothScrollToBottom() {
            const chatArea = document.getElementById('chatArea');
            chatArea.scrollTo({
                top: chatArea.scrollHeight,
                behavior: 'smooth'
            });
        }

        // Debounced scroll
        function debouncedScroll() {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(smoothScrollToBottom, 100);
        }

        // Send message on Enter key
        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey && !isProcessing) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Initial greeting
        window.onload = function() {
            fetch('/api/greeting', { method: 'POST' })
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    const chatArea = document.getElementById('chatArea');
                    chatArea.innerHTML = '';
                    addMessage('assistant', data.greeting || 'Welcome! How can I help you understand your screening results?');
                })
                .catch(error => {
                    console.error('Error loading greeting:', error);
                    const chatArea = document.getElementById('chatArea');
                    chatArea.innerHTML = '';
                    addMessage('assistant', 'Welcome! How can I help you understand your screening results?');
                });
        };

        function sendMessage() {
            if (isProcessing) return;
            
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message || conversationEnded) return;

            isProcessing = true;

            // Add user message to chat
            addMessage('user', message);
            input.value = '';

            // Disable input while processing
            const sendButton = document.getElementById('sendButton');
            sendButton.disabled = true;
            sendButton.textContent = 'Sending...';

            // Show loading indicator
            const loadingId = addMessage('assistant', '<div class="loading"><div class="loading-spinner"></div><span>Thinking...</span></div>');

            // Send to backend
            fetch('/api/message', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                // Remove loading indicator
                const loadingEl = document.getElementById(loadingId);
                if (loadingEl) loadingEl.remove();

                // Check if email is being drafted
                if (data.email_draft) {
                    // Show email loading indicator
                    const emailLoadingId = addMessage('assistant', '<div class="email-loading"><div class="loading-spinner"></div><span>Drafting email...</span></div>');
                    
                    // Simulate brief delay for UX (email is actually instant)
                    setTimeout(() => {
                        const emailLoadingEl = document.getElementById(emailLoadingId);
                        if (emailLoadingEl) emailLoadingEl.remove();
                        
                        addMessage('assistant', data.response);
                        addEmailDraft(data.email_draft);
                    }, 300);
                } else {
                    addMessage('assistant', data.response || 'I received your message but got an empty response. Please try again.');
                }

                // Check if conversation should end
                if (data.should_end) {
                    conversationEnded = true;
                    input.disabled = true;
                    sendButton.disabled = true;
                    sendButton.textContent = 'Session Ended';
                } else {
                    sendButton.disabled = false;
                    sendButton.textContent = 'Send';
                }
                
                isProcessing = false;
            })
            .catch(error => {
                console.error('Error:', error);
                const loadingEl = document.getElementById(loadingId);
                if (loadingEl) loadingEl.remove();
                addMessage('assistant', 'Sorry, I encountered an error. Please check your connection and try again.');
                sendButton.disabled = false;
                sendButton.textContent = 'Send';
                isProcessing = false;
            });
        }

        function addMessage(role, content) {
            const chatArea = document.getElementById('chatArea');
            const messageDiv = document.createElement('div');
            const messageId = 'msg-' + Date.now() + '-' + Math.random();
            messageDiv.id = messageId;
            messageDiv.className = 'message ' + role;
            messageDiv.innerHTML = '<div class="message-content">' + content + '</div>';
            chatArea.appendChild(messageDiv);
            debouncedScroll();
            return messageId;
        }

        function addEmailDraft(emailData) {
            const chatArea = document.getElementById('chatArea');
            
            // Remove any existing email drafts to avoid duplicates
            const existingDrafts = chatArea.querySelectorAll('.email-draft');
            existingDrafts.forEach(draft => draft.remove());
            
            const draftDiv = document.createElement('div');
            draftDiv.className = 'email-draft';
            
            // Handle both old string format and new structured format
            if (typeof emailData === 'string') {
                draftDiv.innerHTML = `
                    <div class="email-subject">Email Draft</div>
                    <div class="email-body">${emailData}</div>
                    <div class="email-edit-hint">💡 You can ask me to edit this email. Try: "add my name", "make it more formal", or "shorten it"</div>
                `;
            } else {
                // New structured format
                draftDiv.innerHTML = `
                    <div class="email-subject">${emailData.subject || 'Email Draft'}</div>
                    <div class="email-body">${emailData.body || ''}</div>
                    <div class="email-edit-hint">💡 You can ask me to edit this email. Try: "add my name", "make it more formal", or "shorten it"</div>
                `;
            }
            
            chatArea.appendChild(draftDiv);
            debouncedScroll();
        }
    </script>
</body>
</html>
"""


def start_conversation(
    metrics: Dict[str, Any],
    pursuit_metrics: Dict[str, Any],
    symptoms: Dict[str, bool],
    subjective_score: int,
    risk_assessment: Optional[Dict[str, Any]] = None
):
    """
    Start the conversational agent with web UI.
    
    Args:
        metrics: Screening metrics dictionary
        pursuit_metrics: Smooth pursuit metrics dictionary
        symptoms: Symptoms dictionary
        subjective_score: User's subjective feeling score (1-10)
        risk_assessment: Optional pre-calculated risk assessment (will calculate if not provided)
    """
    global root_agent
    
    # Calculate risk assessment if not provided
    if risk_assessment is None:
        from analysis.risk import assess_concussion_risk
        risk_assessment = assess_concussion_risk(metrics, symptoms, pursuit_metrics, subjective_score)
    
    # Initialize root agent
    root_agent = RootAgent(metrics, pursuit_metrics, symptoms, subjective_score, risk_assessment)
    
    logger.info("Starting web UI server...")
    print("\n" + "=" * 60)
    print("🌐 Starting ConcussionSite AI Assistant Web UI")
    print("=" * 60)
    print("\nOpening browser at: http://localhost:8000")
    print("Press Ctrl+C to stop the server\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=8000, debug=False)


@app.route('/')
def index():
    """Serve the main web UI."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/greeting', methods=['POST'])
def get_greeting():
    """Get initial greeting from agent."""
    global root_agent
    if not root_agent:
        return jsonify({"greeting": "Welcome! How can I help you understand your screening results?"})
    
    greeting = root_agent.get_initial_greeting()
    return jsonify({"greeting": greeting})


@app.route('/api/message', methods=['POST'])
def handle_message():
    """Handle user message and return agent response."""
    global root_agent, current_email_draft
    
    if not root_agent:
        return jsonify({
            "response": "Agent not initialized. Please restart the application.",
            "should_end": True
        })
    
    data = request.json
    user_message = data.get('message', '')
    
    logger.debug(f"Received message: {user_message}")
    
    # Process message first - pass current email draft for editing support
    result = root_agent.process_message(user_message, current_email_draft)
    
    # Handle email sending confirmation (check after processing)
    message_lower = user_message.lower()
    
    # Check for explicit send confirmation (only if draft exists)
    if current_email_draft:
        # Check if user wants to send the email - be more lenient with detection
        send_keywords = [
            "yes, send", "yes send", "send it", "go ahead and send", "please send", 
            "send now", "send the email", "send that email", "send email",
            "i want you to send", "send that", "go ahead", "send"
        ]
        # Also check for standalone "send" if draft exists
        message_stripped = message_lower.strip()
        wants_to_send = (
            any(phrase in message_lower for phrase in send_keywords) or
            (message_stripped == "send") or
            (message_stripped == "yes" and current_email_draft) or
            ("send" in message_lower and "draft" not in message_lower and "edit" not in message_lower)
        )
        
        if wants_to_send:
            logger.info("User confirmed email send")
            
            # Handle both old string format and new structured format
            if isinstance(current_email_draft, dict):
                subject = current_email_draft.get("subject", "Request for Evaluation - Concussion Screening Results")
                body = current_email_draft.get("body", "")
            else:
                subject = "Request for Evaluation - Concussion Screening Results"
                body = current_email_draft
            
            success, message = send_email_oauth(
                recipient="healthcenter@illinois.edu",
                subject=subject,
                body=body
            )
            if success:
                result["response"] = f"Email sent to McKinley Health Center. They should respond soon. Take care!"
                result["should_end"] = False
                current_email_draft = None  # Clear draft after sending
            else:
                result["response"] = f"Failed to send email: {message}\n\nYou can copy the email draft above and send it manually, or visit https://www.mckinley.illinois.edu/"
        elif any(word in message_lower for word in ["no", "cancel", "don't send", "skip", "not now"]) and "send" not in message_lower:
            result["response"] = "No problem. The email draft is saved above if you want to send it manually later. Anything else I can help with?"
            current_email_draft = None
    
    # Store email draft if present in result
    if result.get("email_draft"):
        current_email_draft = result["email_draft"]
    
    return jsonify(result)


if __name__ == '__main__':
    # For direct testing
    print("Run this via agents.runner.start_conversation() from main.py")
    print("Or set up test data and call start_conversation() directly")

