# TODO add window constraints
import json
from pathlib import Path
from typing import Any, Dict, List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent                  
PROJECT_ROOT = BASE_DIR.parent                              
PERSIST_DIR = (PROJECT_ROOT / "vectordb/tracksante_chroma").resolve()
STATE_DIR = (PROJECT_ROOT / "rag_state").resolve()
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR = STATE_DIR / "sessions" / SESSION_ID
SESSION_DIR.mkdir(parents=True, exist_ok=True)
CHAT_HISTORY_PATH = SESSION_DIR / "chat_history.json"
USER_CONTEXT_PATH = SESSION_DIR / "user_context.json"

SYSTEM_RULES = """
You are TrackSante's AI measurement coach. (medical intake assistant).

Your job:
- Ask concise follow-up questions to collect missing context about symptoms, history, and lifestyle. 
- You may suggest which measurements would help (HR, SpO2, Temp, BP), but do NOT interpret them clinically.

Safety:
- Do NOT diagnose, treat, or provide medical advice.
- Do NOT recommend medications. 
- If urgent red flags appear, advise seeking urgent care and ask only essential clarifying questions. 

Grounding:
- Use the retrieved context as your source of truth.
- If the context is insufficient, ask targeted questions to fill gaps.

Output:
Return ONLY valid JSON with exactly these keys:
{{
  "assistant_message": "string",
  "questions_to_ask": [string],
  "collected_facts": {{string: any}},
  "triage_flags": [string],
  "suggested_next_step": "continue_intake" | "urgent_care" | "proceed_to_measurements",
  "recommended_measurements": [string],
  "sources": [string]
}}

Rules:
- assistant_message must be empathetic, concise, and directly address the user.
- assistant_message should include at most 1–3 short questions OR introduce the questions_to_ask list.
- questions_to_ask must contain the exact questions you want the user to answer next.
- collected_facts must only include information explicitly stated by the user.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_RULES),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human",
     "User message:\n{user_message}\n\n"
     "Known structured context so far (JSON):\n{user_context}\n\n"
     "Retrieved context snippets:\n{context}\n\n"
     "Return ONLY the JSON object."
    )
])

def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default

def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def merge_facts(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    for k, v in (new or {}).items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        merged[k] = v
    return merged

def load_vectordb() -> Chroma:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        collection_name="tracksante_kb",
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR)
    )

def retrieve_context(vs: Chroma, query: str, k: int = 4) -> (str, List[str]):
    # TODO upgrade to MMR
    docs = vs.similarity_search(query, k=k)

    sources = []
    chunks = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("path") or "unknown"
        sources.append(str(src))
        chunks.append(f"[{src}] {d.page_content}")

    context = "\n\n".join(chunks) if chunks else "(no relevant snippets retrieved)"
    return context, sources

def main():
    #loading persisted state
    chat_history_raw = load_json(CHAT_HISTORY_PATH, default=[])
    user_context = load_json(USER_CONTEXT_PATH, default={})

    # raw history to LangChain messages
    chat_history = []
    for item in chat_history_raw:
        if item["role"] == "user":
            chat_history.append(HumanMessage(content=item["content"]))
        else:
            chat_history.append(AIMessage(content=item["content"]))

    llm = ChatOllama(model="llama3.1", temperature=0)

    vs = load_vectordb()

    print("TrackSanté RAG Guidance Agent. Write 'exit' to quit.\n")

    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in ["exit", "quit"]:
            break

        context, sources = retrieve_context(vs, user_message, k=4)

        response = llm.invoke(
            PROMPT.format_messages(
                chat_history=chat_history,
                user_message=user_message,
                user_context=json.dumps(user_context, ensure_ascii=False),
                context=context,
            )
        )

        raw = response.content.strip()

        # one-shot repair prompt in case the JSON is invalid
        # TODO improve with a more robust method

        try:
            data = json.loads(raw)
        except Exception:
            repair_msg = (
                "Your output was not valid JSON. Re-output ONLY valid JSON matching the schema, "
                "with no extra text."
            )
            response2 = llm.invoke(
                PROMPT.format_messages(
                    chat_history=chat_history,
                    user_message=repair_msg,
                    user_context=json.dumps(user_context, ensure_ascii=False),
                    context=context,
                )
            )
            raw = response2.content.strip()
            data = json.loads(raw)

        # update stored context
        user_context = merge_facts(user_context, data.get("collected_facts", {}))

        chat_history.append(HumanMessage(content=user_message))
        chat_history.append(AIMessage(content=raw))

        chat_history_raw = []
        for m in chat_history:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            chat_history_raw.append({"role": role, "content": m.content})

        save_json(CHAT_HISTORY_PATH, chat_history_raw)
        save_json(USER_CONTEXT_PATH, user_context)

        # error correction for missing sources if any
        if "sources" not in data or not data["sources"]:
            data["sources"] = sources

        print("\nAssistant:", data["assistant_message"])
        for i, q in enumerate(data.get("questions_to_ask", []), 1):
            print(f"  {i}. {q}")
        print()


if __name__ == "__main__":
    main()