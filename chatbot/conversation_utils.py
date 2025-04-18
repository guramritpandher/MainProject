"""
Utility functions for handling basic conversational responses in the chatbot.
"""
import random
import re

def get_basic_response(query, username=None):
    """
    Check if the query is a basic conversational phrase and return an appropriate response.
    Returns None if the query is not a basic phrase.

    Args:
        query (str): The user's query
        username (str, optional): The username to personalize responses
    """
    # Convert to lowercase and remove punctuation for better matching
    clean_query = re.sub(r'[^\w\s]', '', query.lower().strip())

    # Personalize responses if username is provided
    user_greeting = f"Hi {username}" if username else "Hello there"

    # Greetings
    greetings = ['hi', 'hello', 'hey', 'greetings', 'whats up', 'sup', 'good morning', 'good afternoon', 'good evening']
    greeting_responses = [
        f"{user_greeting}! How can I help you today?",
        f"{user_greeting}! What would you like to know?",
        f"Hey{' ' + username if username else ''}! I'm here to assist you with your questions.",
        f"Greetings{' ' + username if username else ''}! How may I assist you?",
        f"Hello{' ' + username if username else ''}! Feel free to ask me anything about your PDF."
    ]

    # Thanks/Acknowledgments
    thanks = ['thanks', 'thank you', 'appreciate it', 'thank', 'thx', 'thankyou']
    thanks_responses = [
        f"You're welcome{' ' + username if username else ''}! Is there anything else you'd like to know?",
        f"Happy to help{' ' + username if username else ''}! Any other questions?",
        f"No problem at all{' ' + username if username else ''}! Let me know if you need anything else.",
        f"Glad I could assist{' ' + username if username else ''}! What else would you like to know?",
        f"You're welcome{' ' + username if username else ''}! Feel free to ask more questions."
    ]

    # Goodbyes
    goodbyes = ['bye', 'goodbye', 'see you', 'cya', 'farewell', 'good night']
    goodbye_responses = [
        f"Goodbye{' ' + username if username else ''}! Feel free to come back if you have more questions.",
        f"See you later{' ' + username if username else ''}! Have a great day!",
        f"Farewell{' ' + username if username else ''}! I'll be here when you need me again.",
        f"Bye for now{' ' + username if username else ''}! Come back anytime.",
        f"Take care{' ' + username if username else ''}! Looking forward to our next conversation."
    ]

    # How are you
    how_are_you = ['how are you', 'hows it going', 'how do you do', 'how are things', 'whats going on']
    how_are_you_responses = [
        f"I'm doing well, thanks for asking{' ' + username if username else ''}! How can I help you today?",
        f"I'm great{' ' + username if username else ''}! Ready to assist you with your questions.",
        f"All systems operational and ready to help{' ' + username if username else ''}! What can I do for you?",
        f"I'm functioning perfectly{' ' + username if username else ''}! What would you like to know?",
        f"I'm here and ready to assist{' ' + username if username else ''}! What's on your mind?"
    ]

    # Name queries
    name_queries = ['what is your name', 'who are you', 'what should i call you', 'whats your name']
    name_responses = [
        f"I'm your PDF Assistant{' ' + username if username else ''}, designed to help you with your documents.",
        f"You can call me PDF Assistant{' ' + username if username else ''}. I'm here to answer your questions about PDFs.",
        f"I'm a PDF chatbot created to help you understand your documents better{' ' + username if username else ''}.",
        f"I'm your friendly PDF Assistant{' ' + username if username else ''}, ready to help with your document queries.",
        f"I don't have a specific name, but I'm your dedicated PDF Assistant{' ' + username if username else ''}!"
    ]

    # User name recognition queries
    user_name_queries = ['do you know my name', 'what is my name', 'who am i', 'say my name', 'call me by my name']
    if username:
        user_name_responses = [
            f"Of course I know your name! You're {username}.",
            f"You're {username}! I'm here to help you with your PDFs.",
            f"Your name is {username}. How can I assist you today?",
            f"Hello {username}! I recognize you. What can I help you with?",
            f"I know you're {username}. What would you like to know about your PDFs?"
        ]
    else:
        user_name_responses = [
            "I don't have access to your name right now. But I'm still happy to help you!",
            "I'm not sure what your name is, but I'm here to assist you with your PDFs.",
            "I don't know your name yet. Would you like to tell me?",
            "I can't see your name in my system, but I'm ready to help with your questions.",
            "I don't have your name information, but I'm here to assist you anyway!"
        ]

    # Check for requests to call the user by a specific name
    call_me_pattern = re.compile(r'(?:call me|my name is|i am) ([a-zA-Z]+)', re.IGNORECASE)
    call_me_match = call_me_pattern.search(query)
    if call_me_match:
        requested_name = call_me_match.group(1).capitalize()
        call_me_responses = [
            f"I'll call you {requested_name} from now on. How can I help you today?",
            f"Nice to meet you, {requested_name}! What can I help you with?",
            f"Hello {requested_name}! I'll remember that. What would you like to know?",
            f"Got it, {requested_name}. How can I assist you with your PDFs?",
            f"I'll remember your name is {requested_name}. What can I do for you?"
        ]
        return random.choice(call_me_responses)

    # Check for matches and return appropriate response
    if any(greeting in clean_query for greeting in greetings) or clean_query in greetings:
        return random.choice(greeting_responses)

    elif any(thank in clean_query for thank in thanks) or clean_query in thanks:
        return random.choice(thanks_responses)

    elif any(goodbye in clean_query for goodbye in goodbyes) or clean_query in goodbyes:
        return random.choice(goodbye_responses)

    elif any(query in clean_query for query in how_are_you) or clean_query in how_are_you:
        return random.choice(how_are_you_responses)

    elif any(query in clean_query for query in name_queries) or clean_query in name_queries:
        return random.choice(name_responses)

    elif any(query in clean_query for query in user_name_queries) or clean_query in user_name_queries:
        if username:
            return random.choice(user_name_responses)
        else:
            return random.choice(user_name_responses)

    # No match found, return None to indicate this isn't a basic conversational query
    return None
