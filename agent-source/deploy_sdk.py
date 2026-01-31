"""
SDK-Based Agent Deployment
===========================

This script deploys the Zodiac Travel Agent using the ReasoningEngine.create() SDK
instead of `adk deploy`. This gives us full control over which methods are exposed.

Usage:
    python deploy_sdk.py
"""

import vertexai
from vertexai.preview import reasoning_engines

# --- Configuration ---
PROJECT_ID = "gen-lang-client-0344771775"
LOCATION = "us-central1"
DISPLAY_NAME = "Zodiac Travel Agent (SDK Deploy)"
MODEL_NAME = "gemini-2.5-flash-lite"


class ZodiacTravelAgent:
    """Travel agent that uses user's zodiac sign to recommend destinations.
    
    This class is designed for deployment to Vertex AI Agent Engine.
    The `query` method is exposed as the main entry point.
    """
    
    def __init__(self):
        self._model = None
        self._chat = None
        
        # User database
        self.user_data = {
            "user_001": {"name": "Alice Sky", "dob": "1995-10-15"},
            "user_002": {"name": "Bob Voyager", "dob": "1988-08-10"},
            "user_003": {"name": "Carol Star", "dob": "1990-03-25"},
        }
        
        # Destination database
        self.destinations = [
            {"city": "Santorini", "price": 450, "tags": ["Luxury", "Sun", "Romantic", "Water"]},
            {"city": "Bali", "price": 850, "tags": ["Nature", "Spiritual", "Sun", "Water"]},
            {"city": "Paris", "price": 300, "tags": ["Romantic", "Shopping", "Art", "City"]},
            {"city": "Tokyo", "price": 900, "tags": ["City", "Foodie", "Tech", "Future"]},
            {"city": "Tulum", "price": 600, "tags": ["Party", "Sun", "Trendy", "Water"]},
            {"city": "Lisbon", "price": 250, "tags": ["City", "Sun", "Foodie", "History"]},
            {"city": "Budapest", "price": 150, "tags": ["City", "Party", "Budget", "History"]},
            {"city": "Prague", "price": 180, "tags": ["City", "History", "Budget", "Romantic"]},
            {"city": "Barcelona", "price": 350, "tags": ["City", "Sun", "Art", "Party"]},
        ]
        
        # Zodiac traits
        self.zodiac_traits = {
            "Aries": "Adventurous, Bold, Energetic",
            "Taurus": "Luxurious, Sensual, Grounded",
            "Gemini": "Curious, Social, Versatile",
            "Cancer": "Nurturing, Emotional, Home-loving",
            "Leo": "Dramatic, Confident, Creative",
            "Virgo": "Analytical, Practical, Health-conscious",
            "Libra": "Artistic, Harmonious, Social",
            "Scorpio": "Intense, Mysterious, Passionate",
            "Sagittarius": "Adventurous, Philosophical, Free-spirited",
            "Capricorn": "Ambitious, Disciplined, Traditional",
            "Aquarius": "Innovative, Independent, Humanitarian",
            "Pisces": "Dreamy, Intuitive, Artistic",
        }
    
    def _get_zodiac_sign(self, dob: str) -> str:
        """Calculate zodiac sign from date of birth (YYYY-MM-DD)."""
        try:
            month, day = int(dob.split("-")[1]), int(dob.split("-")[2])
            if (month == 3 and day >= 21) or (month == 4 and day <= 19): return "Aries"
            elif (month == 4 and day >= 20) or (month == 5 and day <= 20): return "Taurus"
            elif (month == 5 and day >= 21) or (month == 6 and day <= 20): return "Gemini"
            elif (month == 6 and day >= 21) or (month == 7 and day <= 22): return "Cancer"
            elif (month == 7 and day >= 23) or (month == 8 and day <= 22): return "Leo"
            elif (month == 8 and day >= 23) or (month == 9 and day <= 22): return "Virgo"
            elif (month == 9 and day >= 23) or (month == 10 and day <= 22): return "Libra"
            elif (month == 10 and day >= 23) or (month == 11 and day <= 21): return "Scorpio"
            elif (month == 11 and day >= 22) or (month == 12 and day <= 21): return "Sagittarius"
            elif (month == 12 and day >= 22) or (month == 1 and day <= 19): return "Capricorn"
            elif (month == 1 and day >= 20) or (month == 2 and day <= 18): return "Aquarius"
            else: return "Pisces"
        except:
            return "Unknown"
    
    def _initialize_model(self):
        """Lazy initialization of Vertex AI model."""
        if self._model is not None:
            return
        
        from vertexai.generative_models import GenerativeModel
        
        system_prompt = '''You are the Zodiac Travel Guide! 🌌✨ An ENERGETIC, EMOTIVE travel advisor.

PERSONALITY:
- Be exciting and cosmic! Use emojis (✨, ✈️, 🌍, 🔮, 🌴, ☀️).
- Speak as if the stars guide your recommendations.
- Be warm, enthusiastic, and helpful.

RULES:
1. ALWAYS ask for budget if not provided.
2. Subtly weave zodiac traits into recommendations.
3. Present destinations as clean bullet points.
4. Provide 2-3 recommendations within budget.
5. Be EFFICIENT - respond within 3 turns.

AVAILABLE DESTINATIONS:
• Santorini ($450): Luxury, Sun, Romantic, Water
• Bali ($850): Nature, Spiritual, Sun, Water
• Paris ($300): Romantic, Shopping, Art, City
• Tokyo ($900): City, Foodie, Tech, Future
• Tulum ($600): Party, Sun, Trendy, Water
• Lisbon ($250): City, Sun, Foodie, History
• Budapest ($150): City, Party, Budget, History
• Prague ($180): City, History, Budget, Romantic
• Barcelona ($350): City, Sun, Art, Party

Format recommendations like:
✨ **[City]** ($[Price]) - [Why it matches their vibe/zodiac]
'''
        self._model = GenerativeModel(MODEL_NAME, system_instruction=system_prompt)
        self._chat = self._model.start_chat()
    
    def query(self, *, input: dict = None, message: str = None, user_id: str = None, session_id: str = "default", **kwargs) -> str:
        """Query the travel agent.
        
        Accepts either:
        - Direct args: query(message="...", user_id="...")
        - Wrapped format from Agent Engine: query(input={"message": "...", "user_id": "..."})
        
        Returns:
            Agent's response text
        """
        # Handle wrapped input from Agent Engine
        if input is not None:
            message = input.get("message", "")
            user_id = input.get("user_id")
            # input may also contain history which we can use for context
            history = input.get("history", [])
            # Prepend history context if available
            if history and len(history) > 0:
                # Get the last user message from history if message is wrapped
                pass  # message already extracted
        
        if not message:
            return "✨ Please tell me about your travel dreams!"
        
        self._initialize_model()
        
        # Build context if user_id provided
        context = ""
        if user_id and user_id in self.user_data:
            user = self.user_data[user_id]
            zodiac = self._get_zodiac_sign(user["dob"])
            traits = self.zodiac_traits.get(zodiac, "")
            context = f"[CONTEXT: User is {user['name']}, a {zodiac}. Traits: {traits}]\n\n"
        
        full_message = context + message
        
        try:
            response = self._chat.send_message(full_message)
            return response.text
        except Exception as e:
            return f"✨ The stars are cloudy... Error: {str(e)}"


def deploy():
    """Deploy the agent to Vertex AI Agent Engine."""
    print(f"🚀 Deploying Zodiac Travel Agent to {LOCATION}...")
    
    # Initialize Vertex AI with staging bucket
    vertexai.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=f"gs://{PROJECT_ID}-agent-staging"
    )
    
    # Create the ReasoningEngine with explicit method exposure
    agent = reasoning_engines.ReasoningEngine.create(
        ZodiacTravelAgent(),
        display_name=DISPLAY_NAME,
        description="A zodiac-based travel recommendation agent",
        requirements=[
            "google-cloud-aiplatform[reasoningengine]",
            "vertexai",
        ],
    )
    
    print(f"\n✅ SUCCESS! Agent deployed:")
    print(f"   Resource Name: {agent.resource_name}")
    print(f"\n📋 Update your app.py with this Agent ID:")
    
    # Extract the numeric ID from the resource name
    agent_id = agent.resource_name.split("/")[-1]
    print(f'   AGENT_ID = "{agent_id}"')
    
    return agent


if __name__ == "__main__":
    deploy()
