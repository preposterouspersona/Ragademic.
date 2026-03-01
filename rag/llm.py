import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


class GroqLLM:

    def __init__(self, model_name: str = "llama-3.3-70b-versatile", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")

        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,
            max_tokens=1024
        )
        print(f"Groq LLM ready — model: {self.model_name}")

    def generate(self, query: str, context: str) -> str:
        prompt = f"""You are a helpful AI assistant. Use the following context to answer the question accurately and concisely. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"""
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Error generating response: {e}"
