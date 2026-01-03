# src/llm_insights.py
import os
import openai
from typing import Generator

# ----------------------------------------------------------------------
#  CONFIG – put your OpenAI key in .env or set it in the environment
# ----------------------------------------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")   # <-- set this in your env


def get_llm_insight(query: str, df: "pd.DataFrame") -> str:
    """
    One-shot answer (non-streaming). Used for the sidebar Q&A.
    """
    if not openai.api_key:
        return "Warning: OpenAI API key not set."

    prompt = f"""
    You are a retail-analytics assistant. Summarise the data and answer the user.

    Data snapshot:
    • Cities: {len(df):,}
    • Avg CLV: {df['CLV'].mean():,.0f}
    • Avg churn risk: {(df.get('churn_probability', pd.Series([0])) > 0.7).mean()*100:.1f}%
    • Clusters: {df.get('cluster', pd.Series()).nunique()}

    Question: {query}
    """
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM error: {e}"


def stream_llm_response(query: str, df: "pd.DataFrame") -> Generator[str, None, None]:
    """
    Streaming version – yields chunks of text as they arrive.
    Useful for a “type-writer” effect in a modal/chat window.
    """
    if not openai.api_key:
        yield "Warning: OpenAI API key not set."
        return

    prompt = f"""
    You are a retail-analytics assistant. Summarise the data and answer the user.

    Data snapshot:
    • Cities: {len(df):,}
    • Avg CLV: {df['CLV'].mean():,.0f}
    • Avg churn risk: {(df.get('churn_probability', pd.Series([0])) > 0.7).mean()*100:.1f}%
    • Clusters: {df.get('cluster', pd.Series()).nunique()}

    Question: {query}
    """
    try:
        stream = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            content = chunk["choices"][0]["delta"].get("content")
            if content:
                yield content
    except Exception as e:
        yield f"LLM streaming error: {e}"