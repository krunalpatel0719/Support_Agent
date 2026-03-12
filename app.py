"""
Voice-Enabled Knowledge Assistant

A modular support agent that ingests any documentation via Firecrawl,
stores it in a local SQLite database using Memori, and answers questions
with optional text-to-speech responses.

Usage:
    1. Add API keys to .env (or enter in sidebar)
    2. Paste documentation URLs and click "Ingest"
    3. Ask questions in the chat
"""

import json
import os
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional, Iterator, Any
import torch
import streamlit as st
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from firecrawl import FirecrawlApp
from memori import Memori
from openai import OpenAI
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from urllib.parse import urlparse

load_dotenv()

# Path for persisting knowledge sources metadata
KNOWLEDGE_SOURCES_FILE = "./knowledge_sources.json"


def save_knowledge_sources(sources: dict[str, int]) -> None:
    """Persist knowledge sources to JSON file."""
    try:
        with open(KNOWLEDGE_SOURCES_FILE, "w") as f:
            json.dump(sources, f)
    except IOError:
        pass


def load_knowledge_sources() -> dict[str, int]:
    """Load knowledge sources from JSON file."""
    try:
        if os.path.exists(KNOWLEDGE_SOURCES_FILE):
            with open(KNOWLEDGE_SOURCES_FILE, "r") as f:
                return json.load(f)
    except (IOError, json.JSONDecodeError):
        pass
    return {}


@dataclass
class AppContext:
    """
    Central context holding the application's runtime services.

    Attributes:
        memori: Memori instance for semantic memory and retrieval.
        openai_client: OpenAI API client for chat and speech synthesis.
        db_path: Path to the SQLite database storing memories.
        engine: SQLAlchemy engine for connection management.
    """
    memori: Memori
    openai_client: OpenAI
    db_path: str = field(default="./support.sqlite")
    engine: Any = field(default=None)

    def persist(self):
        """Flush any pending writes to storage."""
        adapter = getattr(self.memori.config.storage, "adapter", None)
        if adapter and hasattr(adapter, "commit"):
            adapter.commit()

    def dispose(self):
        """Dispose of database connections to release file locks."""
        if self.engine is not None:
            try:
                self.engine.dispose()
            except Exception:
                pass


@st.cache_resource
def load_whisper_model(model_size: str = "base") -> WhisperModel:
    """
    Load and cache the Whisper model for speech-to-text.

    Uses @st.cache_resource to load the model only once per session,
    avoiding expensive reloads on each transcription request.
    """

    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    else:
        device = "cpu"
        compute_type = "int8"

    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_audio(audio_bytes: bytes, model: WhisperModel) -> str:
    """Transcribe audio bytes to text using faster-whisper."""
    try:
        audio_buffer = BytesIO(audio_bytes)
        segments, info = model.transcribe(audio_buffer, beam_size=5)
        transcript = " ".join(segment.text for segment in segments)
        return transcript.strip()
    except Exception as e:
        st.warning(f"Transcription failed: {e}")
        return ""


def get_api_key(key_name: str) -> str:
    """
    Get API key from session state first, then fall back to environment variable.
    This ensures user-entered keys are session-isolated and not shared across visitors.
    """
    return st.session_state.get(f"api_key_{key_name}", "") or os.getenv(key_name, "")


def create_app_context() -> Optional[AppContext]:
    """
    Bootstrap the application services.

    Creates an SQLite-backed Memori instance and registers the OpenAI client
    so that all chat completions automatically become searchable memories.
    """
    openai_key = get_api_key("OPENAI_API_KEY")
    if not openai_key:
        st.warning("OPENAI_API_KEY not set – add it in sidebar or .env file.")
        return None

    db_path = os.getenv("SQLITE_DB_PATH", "./support.sqlite")

    try:
        engine = create_engine(
            f"sqlite:///{db_path}",
            pool_pre_ping=True,
            connect_args={"check_same_thread": False},
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        session_factory = sessionmaker(bind=engine)
        client = OpenAI(api_key=openai_key)

        mem = Memori(conn=session_factory).openai.register(client)
        mem.attribution(entity_id="user", process_id="knowledge-assistant")
        mem.config.storage.build()

        ctx = AppContext(memori=mem, openai_client=client, db_path=db_path, engine=engine)

        st.session_state.app_context = ctx
        st.session_state.memori = mem
        st.session_state.openai_client = client
        return ctx
    except Exception as e:
        st.warning(f"Setup error: {e}")
        return None


def reset_database(db_path: str) -> tuple[bool, str]:
    """
    Clear all data from database tables and reset session state.

    Instead of deleting the file we drop and recreate all tables.

    Returns:
        Tuple of (success: bool, message: str)
    """
    import gc
    from sqlalchemy import MetaData

    try:
        engine = None
        if "app_context" in st.session_state:
            ctx = st.session_state.app_context
            engine = ctx.engine

        st.session_state.clear()
        gc.collect()

        if engine is not None:
            metadata = MetaData()
            metadata.reflect(bind=engine)
            metadata.drop_all(bind=engine)
            engine.dispose()

        if os.path.exists(KNOWLEDGE_SOURCES_FILE):
            os.remove(KNOWLEDGE_SOURCES_FILE)

        return True, "Database cleared successfully. Reloading..."
    except Exception as e:
        return False, f"Error: {e}"


def synthesize_speech(text: str, client: OpenAI, voice: str = "alloy") -> Optional[BytesIO]:
    """Generate speech audio from text using OpenAI's TTS API."""
    try:
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
        )
        audio_bytes = response.read() if hasattr(response, "read") else response
        return BytesIO(audio_bytes) if isinstance(audio_bytes, bytes) else None
    except Exception as e:
        st.warning(f"Voice synthesis failed: {e}")
        return None


def crawl_urls(urls: list[str], max_pages: int = 50) -> Iterator[dict]:
    """
    Crawl documentation URLs and yield page dicts.

    Uses Firecrawl v2 API - see https://docs.firecrawl.dev
    Each yielded dict has: url, title, content
    """
    firecrawl_key = get_api_key("FIRECRAWL_API_KEY")
    if not firecrawl_key:
        raise RuntimeError("FIRECRAWL_API_KEY not set – add it in sidebar or .env")

    crawler = FirecrawlApp(api_key=firecrawl_key)
    for url in urls:
        try:
            result = crawler.crawl(
                url,
                limit=max_pages,
                scrape_options={"formats": ["markdown"], "onlyMainContent": True},
            )

            pages = result.data if hasattr(result, 'data') else result.get("data", [])

            for page in pages:
                if hasattr(page, 'markdown'):
                    content = page.markdown
                    metadata = page.metadata if hasattr(page, 'metadata') else {}
                else:
                    content = page.get("markdown", "")
                    metadata = page.get("metadata", {})

                if content and len(content.strip()) >= 50:
                    if hasattr(metadata, 'get'):
                        page_url = metadata.get("sourceURL") or metadata.get("url") or url
                        title = metadata.get("title", "Untitled")
                    else:
                        page_url = getattr(metadata, 'sourceURL', None) or getattr(metadata, 'url', url)
                        title = getattr(metadata, 'title', "Untitled")

                    yield {
                        "url": page_url,
                        "title": title,
                        "content": content.strip(),
                    }

        except Exception as e:
            st.warning(f"Crawl failed for {url}: {e}")


def _extract_domain(url: str) -> str:
    """Extract domain from URL."""
    return urlparse(url).netloc or "unknown"


def ingest_to_memory(
    ctx: AppContext,
    pages: Iterator[dict],
    source_name: str = "Documentation",
) -> tuple[int, dict[str, int]]:
    """
    Store documentation pages into Memori via OpenAI completions.

    Args:
        ctx: Application context with OpenAI client.
        pages: Iterator of dicts with keys: url, title, content.
        source_name: Label for the documentation source.

    Returns:
        Tuple of (total_stored, domain_counts) where domain_counts is {"domain": count}.
    """
    stored = 0
    seen_urls: set[str] = set()
    domain_counts: dict[str, int] = {}

    for page in pages:
        url = page["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)

        domain = _extract_domain(url)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        prompt = f"""Remember this documentation for future reference:

**{page["title"]}**
Source: {source_name}
Link: {url}

---
{page["content"]}
---"""

        try:
            ctx.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            stored += 1
        except Exception as e:
            st.warning(f"Could not store '{page['title']}': {e}")

    ctx.persist()
    return stored, domain_counts


def build_system_prompt(company: str, context: str) -> str:
    """
    Construct a system prompt for the support assistant.

    Args:
        company: The company or product name to personalize the prompt.
        context: The relevant documentation snippets to use as context.

    Returns:
        A formatted system prompt string for the chat model.
    """
    return f"""You are a helpful support assistant for {company}.

Use ONLY the provided documentation context to answer questions.
If the answer isn't in the context, say so honestly - don't make things up.
Use a friendly helpful tone, be flexible as in allow some room for general knowledge if it's common sense
and be concise to not confuse the user but do answer the user's question fully if possible.
Cite where you get the information from.
Context:
{context}"""


def main():
    """
    Main Streamlit application entry point.

    Sets up the UI layout with sidebar for configuration and documentation
    ingestion, and the main chat interface supporting both text and voice
    input/output.
    """
    st.set_page_config(
        page_title="Customer Support Agent",
        page_icon="🤖",
        layout="wide",
    )

    if "memori" not in st.session_state:
        create_app_context()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "company_name" not in st.session_state:
        st.session_state.company_name = ""
    if "knowledge_sources" not in st.session_state:
        st.session_state.knowledge_sources = load_knowledge_sources()

    # HEADER
    st.markdown("""
    <h1 style='margin-bottom: 0;'>🤖 Customer Support Agent</h1>
    <p style='margin-top: 4px; opacity: 0.7;'>
        Ingest any documentation, ask questions, get voice answers
    </p>
    """, unsafe_allow_html=True)

    # SIDEBAR
    with st.sidebar:
        with st.expander("⚙️ Settings", expanded=False):
            st.caption("API keys can also be set in .env file")

            openai_key = st.text_input(
                "OpenAI API Key",
                value=get_api_key("OPENAI_API_KEY"),
                type="password",
            )
            firecrawl_key = st.text_input(
                "Firecrawl API Key",
                value=get_api_key("FIRECRAWL_API_KEY"),
                type="password",
            )
            memori_key = st.text_input(
                "Memori API Key (optional)",
                value=get_api_key("MEMORI_API_KEY"),
                type="password",
                help="For cloud features. Leave blank for local-only mode.",
            )
            company = st.text_input(
                "Company/Product Name",
                value=st.session_state.company_name,
                placeholder="e.g., Discord, Stripe, Your App",
            )
            st.session_state.company_name = company.strip()

            if st.button("Apply Settings", use_container_width=True):
                # Store in session_state (isolated per user) instead of os.environ (shared)
                if openai_key:
                    st.session_state.api_key_OPENAI_API_KEY = openai_key
                if firecrawl_key:
                    st.session_state.api_key_FIRECRAWL_API_KEY = firecrawl_key
                if memori_key:
                    st.session_state.api_key_MEMORI_API_KEY = memori_key
                create_app_context()
                st.success("Settings applied!")

        st.divider()

        st.subheader("📥 Add Knowledge")
        st.caption("💡 Use direct article/page URLs for best results")

        doc_urls = st.text_area(
            "Documentation URLs",
            placeholder="https://docs.example.com/guide\nhttps://support.example.com/articles/123",
            height=100,
            label_visibility="collapsed",
        )

        page_limit = st.number_input(
            "Max pages per URL",
            min_value=10,
            max_value=200,
            value=50,
            help="Limit pages crawled from each URL",
        )

        if st.button("🔍 Ingest Documentation", use_container_width=True):
            urls = [u.strip() for u in doc_urls.splitlines() if u.strip()]
            if not urls:
                st.warning("Enter at least one URL")
            elif "app_context" not in st.session_state:
                st.warning("Configure API keys first")
            else:
                with st.spinner(f"Crawling up to {page_limit} pages per URL..."):
                    try:
                        ctx: AppContext = st.session_state.app_context
                        source = st.session_state.company_name or "Documentation"
                        pages = crawl_urls(urls, max_pages=page_limit)
                        count, domain_counts = ingest_to_memory(ctx, pages, source_name=source)

                        for domain, domain_count in domain_counts.items():
                            existing = st.session_state.knowledge_sources.get(domain, 0)
                            st.session_state.knowledge_sources[domain] = existing + domain_count

                        save_knowledge_sources(st.session_state.knowledge_sources)

                        st.success(f"Ingested {count} pages!")
                    except Exception as e:
                        st.error(f"Failed: {e}")

        st.divider()

        with st.expander("📚 Knowledge Sources", expanded=True):
            if st.session_state.knowledge_sources:
                st.caption("Ingested documentation domains:")
                for domain, count in st.session_state.knowledge_sources.items():
                    st.markdown(f"**{domain}** · {count} pages")
            else:
                st.caption("No knowledge sources yet. Ingest documentation above.")

        st.divider()

        st.subheader("🎤 Voice Input")
        enable_voice_input = st.toggle("Enable voice input", value=True)

        if enable_voice_input:
            model_size = st.selectbox(
                "Whisper Model",
                ["tiny", "base", "small", "medium", "large-v3"],
                index=1,
                help="Larger models are more accurate but slower",
            )
            st.session_state.whisper_model_size = model_size

        st.divider()

        st.subheader("🔊 Voice Output")
        enable_voice = st.toggle("Enable voice responses", value=True)
        voice_choice = st.selectbox(
            "Voice",
            ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            index=0,
            disabled=not enable_voice,
        )

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        with st.expander("⚠️ Danger Zone", expanded=False):
            st.warning("**Reset Database**: Permanently deletes all ingested knowledge.")
            confirm_reset = st.checkbox("I understand this cannot be undone")

            if st.button(
                "🗑️ Reset Database",
                type="primary",
                use_container_width=True,
                disabled=not confirm_reset,
            ):
                db_path = os.getenv("SQLITE_DB_PATH", "./support.sqlite")
                success, message = reset_database(db_path)

                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

    # MAIN CHAT AREA
    if "openai_client" not in st.session_state:
        st.info("👈 Add your OpenAI API key in Settings to get started")
        st.stop()

    client: OpenAI = st.session_state.openai_client
    mem: Memori = st.session_state.memori

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = None
    if enable_voice_input:
        model_size = st.session_state.get("whisper_model_size", "base")
        whisper_model = load_whisper_model(model_size)

        if whisper_model is not None:
            audio_data = st.audio_input(
                "🎤 Record your question",
                help="Click to record, click again to stop",
            )

            if audio_data is not None:
                with st.spinner("Transcribing..."):
                    audio_bytes = audio_data.getvalue()
                    user_input = transcribe_audio(audio_bytes, whisper_model)
                    if user_input:
                        st.success(f"Heard: \"{user_input}\"")

    if text_input := st.chat_input("Ask a question about your documentation..."):
        user_input = text_input

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                try:
                    snippets = []
                    if hasattr(mem, "recall"):
                        snippets = mem.recall(user_input, limit=5) or []

                    context = "\n".join(f"- {s}" for s in snippets) if snippets else "(No relevant documentation found)"
                    company = st.session_state.company_name or "the documentation"

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": build_system_prompt(company, context)},
                            {"role": "user", "content": user_input},
                        ],
                    )
                    answer = response.choices[0].message.content or ""

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.markdown(answer)

                    if enable_voice and answer.strip():
                        audio = synthesize_speech(answer, client, voice=voice_choice)
                        if audio:
                            st.audio(audio.getvalue(), format="audio/mp3")

                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.error(error_msg)


if __name__ == "__main__":
    main()
