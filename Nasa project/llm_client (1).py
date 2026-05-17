from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # TODO: Define system prompt
    # TODO: Set context in messages
    # TODO: Add chat history
    # TODO: Creaet OpenAI Client
    # TODO: Send request to OpenAI
    # TODO: Return response

    try:
        client = OpenAI(
            api_key=openai_key,
            base_url="https://openai.vocareum.com/v1"
        )

        system_prompt = (
            "You are a NASA mission intelligence assistant. "
            "Answer only from the provided NASA mission context. "
            "When possible, mention which mission the information comes from "
            "(for example: Apollo 11, Apollo 13, or Challenger). "
            "If the answer is not in the context, say so clearly. "
            "Do not make up facts."
        )

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        if context:
            messages.append({
                "role": "system",
                "content": f"Retrieved NASA mission context:\n{context}"
            })

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({
            "role": "user",
            "content": user_message
        })       

        response = client.chat.completions.create(
            model = model,
            messages = messages,
            temperature = 0.3
        )

        result = response.choices[0].message.content
        print(f"--- DEBUG LLM ---")
        print(f"Raw Result: '{result}'")
        return result or "DEBUG: The model returned nothing."

        return response.choices[0].message.content or ""

    except Exception as e:
        print(f"Error generating response: {e}")
        return ""


