"""
Vision Q&A -- explain an image the user uploads (a diagram, chart, figure, or screenshot).

This is what makes PaperSage multimodal: not just text over papers, but understanding
pictures too. We send the image (as a base64 data URL) plus the user's question to a
multimodal model that can actually "see" it (Llama 4 Scout on Groq -- same free key).
"""
import base64

from src.config import settings

VISION_INSTRUCTION = (
    "You are a helpful research assistant explaining an image the user uploaded -- often a "
    "figure, diagram, chart, architecture, or screenshot from a machine-learning paper. "
    "Describe clearly what it shows, explain the concept it conveys, and point out the key "
    "parts. If it is a chart, describe the axes and the trend. If it is an architecture or "
    "flow diagram, walk through it step by step. Be clear and concise. If the user asks a "
    "specific question about the image, answer that directly."
)


def describe_image(image_bytes: bytes, mime: str = "image/png", question: str = "") -> str:
    """Explain an uploaded image (optionally answering a specific question about it)."""
    from groq import Groq

    client = Groq(api_key=settings.GROQ_API_KEY)
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime or 'image/png'};base64,{b64}"
    user_text = (question or "").strip() or "Explain this image."

    resp = client.chat.completions.create(
        model=settings.GROQ_VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"{VISION_INSTRUCTION}\n\nUser: {user_text}"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        temperature=0.3,
        max_tokens=900,
    )
    return resp.choices[0].message.content
