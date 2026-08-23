import os
import sys
import time
import pandas as pd
import streamlit as st

# Ensure project root is in Python path for 'engines' and 'storage' imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import local RAG Q&A logic
from engines.rag import (
    chunk_for_qa,
    retrieve_relevant_chunks,
    answer_query_with_citations,
)

# Import FastAPI for in-process networking bypass
from fastapi.testclient import TestClient
from main import app as fastapi_app

# ──────────────────────────────────────────────────────────────────────────────
# Config & Setup
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_api_client():
    """Initializes the FastAPI TestClient and keeps model loaded in memory."""
    client = TestClient(fastapi_app)
    client.__enter__()
    return client

api_client = get_api_client()

st.set_page_config(
    page_title="Automotive Compliance Platform",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# Enforce Light White Theme & Reference Dashboard CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');

    /* Global Body Light Theme Enforcement */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f4f6f9 !important;
        color: #1e293b !important;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 96%;
    }

    /* Left Sidebar Custom Dark Slate Style */
    [data-testid="stSidebar"] {
        background-color: #181824 !important;
        border-right: 1px solid #232334 !important;
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 10px 24px 10px;
        border-bottom: 1px solid #2a2a3c;
        margin-bottom: 20px;
    }

    .sidebar-brand-icon {
        background: #f87171;
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 1.2rem;
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .sidebar-brand-text {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.15rem;
        letter-spacing: 0.5px;
        color: #ffffff !important;
    }

    /* Sidebar Radio Navigation Override */
    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 8px;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background: transparent;
        border-radius: 8px;
        padding: 10px 14px;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
        margin-bottom: 4px;
        border: 1px solid transparent;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.05);
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
        background: #f87171 !important;
        color: #ffffff !important;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(248, 113, 113, 0.3);
    }

    /* Top Breadcrumb Header Bar */
    .top-header-bar {
        background: #ffffff;
        border-radius: 12px;
        padding: 14px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        border: 1px solid #e2e8f0;
        margin-bottom: 24px;
    }

    .breadcrumb-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.4rem;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .breadcrumb-sub {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .user-profile-badge {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #f8fafc;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        font-weight: 600;
        font-size: 0.85rem;
        color: #334155;
    }

    /* White Dashboard Stat Cards */
    .stat-card-white {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 14px rgba(0,0,0,0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .stat-card-white:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    }

    .stat-label {
        font-size: 0.8rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }

    .stat-value {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.8rem;
        color: #0f172a;
    }

    .stat-trend {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 4px;
    }

    .trend-up { color: #16a34a; }
    .trend-neutral { color: #2563eb; }

    /* Content Card Container */
    .content-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 14px rgba(0,0,0,0.02);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "certificates" not in st.session_state:
    st.session_state.certificates = []
if "raw_markdown" not in st.session_state:
    st.session_state.raw_markdown = ""

# ──────────────────────────────────────────────────────────────────────────────
# LEFT SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">C</div>
            <div class="sidebar-brand-text">COMPLIANCE OS</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    selected_nav = st.radio(
        "NAVIGATION",
        ["HOME", "CHAT", "DATABASES"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    
    # Connection Indicator in Sidebar Footer
    try:
        api_client.get("/")
        conn_status = "🟢 Online"
    except Exception:
        conn_status = "🔴 Offline"
        
    st.caption(f"Engine Status: **{conn_status}**")
    st.caption("Automotive Compliance Pipeline v3.0")

# ──────────────────────────────────────────────────────────────────────────────
# TOP BREADCRUMB HEADER BAR
# ──────────────────────────────────────────────────────────────────────────────
sub_title_map = {
    "HOME": "HOME > INGESTION DASHBOARD",
    "CHAT": "CHAT > HYBRID RAG COMPLIANCE ASSISTANT",
    "DATABASES": "DATABASES > RELATIONAL & VECTOR STORAGE"
}

st.markdown(
    f"""
    <div class="top-header-bar">
        <div>
            <div class="breadcrumb-sub">{sub_title_map.get(selected_nav, "DASHBOARD")}</div>
            <div class="breadcrumb-title">{selected_nav} PAGE</div>
        </div>
        <div class="user-profile-badge">
            <span>👤 System Administrator</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1: HOME (Document Ingestion & Overview)
# ──────────────────────────────────────────────────────────────────────────────
def _index_document_chunks(data: dict, file_name: str) -> list:
    chunks = []
    if data.get("raw_markdown"):
        chunks = chunk_for_qa(
            markdown_text=data["raw_markdown"],
            file_name=data.get("filename", file_name)
        )
        for cert in data.get("certificates", []):
            summary_text = (
                f"Certificate Summary details:\n"
                f"- Component: {cert.get('component', 'N/A')}\n"
                f"- Supplier: {cert.get('supplier', 'N/A')}\n"
                f"- Country: {cert.get('country', 'N/A')}\n"
                f"- Certif Number: {cert.get('certif_number', 'N/A')}\n"
                f"- Authority: {cert.get('authority', 'N/A')}\n"
                f"- Issue Date: {cert.get('issue_date', 'N/A')}\n"
                f"- Exp Date: {cert.get('exp_date', 'N/A')}"
            )
            chunks.append({
                "file_name": data.get("filename", file_name),
                "document_id": "metadata_summary",
                "page_number": "Summary",
                "content": summary_text
            })
    return chunks


if selected_nav == "HOME":
    # ── Top Row Stat Cards (Matching Dashboard Mockup Layout) ──────────────
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            """
            <div class="stat-card-white">
                <div class="stat-label">System Readiness</div>
                <div class="stat-value">Active</div>
                <div class="stat-trend trend-up">✦ GLM-OCR + Qwen3.8-27B</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="stat-card-white">
                <div class="stat-label">Vector Dimension</div>
                <div class="stat-value">1,024-d</div>
                <div class="stat-trend trend-neutral">BAAI/bge-m3 pgvector</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="stat-card-white">
                <div class="stat-label">RAG Pipeline</div>
                <div class="stat-value">Dual-Path</div>
                <div class="stat-trend trend-up">SQL + Vector Hybrid</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <div class="stat-card-white">
                <div class="stat-label">Auto-Backup</div>
                <div class="stat-value">Enabled</div>
                <div class="stat-trend trend-neutral">pg_dump SQL Sync</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Upload Section ─────────────────────────────────────────────────────
    st.header("📄 Batch Certificate Ingestion")
    st.markdown("Upload compliance certificates (`.pdf`, `.png`, `.jpg`) for layout-aware OCR extraction and vector indexing.")
    
    uploaded_files = st.file_uploader(
        "Select certificate documents",
        type=["pdf", "png", "jpg"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.info(f"📎 Selected {len(uploaded_files)} file(s) for ingestion.")

        if st.button("🚀 Process Batch Ingestion", type="primary"):
            total = len(uploaded_files)
            progress_bar = st.progress(0.0)
            phase_placeholder = st.empty()

            try:
                files_payload = [
                    ("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files
                ]
                ingest_resp = api_client.post("/api/v1/batch/ingest", files=files_payload)
                
                if ingest_resp.status_code == 409:
                    curr_resp = api_client.get("/api/v1/batch/status")
                    if curr_resp.status_code == 200 and "batch_id" in curr_resp.json():
                        batch_id = curr_resp.json()["batch_id"]
                        st.info(f"ℹ️ Attaching to processing batch '{batch_id}'...")
                    else:
                        st.error(f"❌ Batch start failed: {ingest_resp.text[:300]}")
                        ingest_resp.raise_for_status()
                elif ingest_resp.status_code != 200:
                    st.error(f"❌ Batch start failed: {ingest_resp.text[:300]}")
                    ingest_resp.raise_for_status()
                else:
                    batch_id = ingest_resp.json()["batch_id"]

                done = False
                while not done:
                    time.sleep(3)
                    status_resp = api_client.get(f"/api/v1/batch/status/{batch_id}")
                    status = status_resp.json() if status_resp.status_code == 200 else {}

                    phase = status.get("phase", "unknown")
                    ocr_done = status.get("ocr_done", 0)
                    extract_done = status.get("extract_done", 0)
                    skipped = status.get("skipped", 0)
                    failed = status.get("failed", 0)
                    current = status.get("current_file", "") or ""

                    work_units = 2 * max(total, 1)
                    done_units = ocr_done + extract_done + 2 * skipped
                    progress = min(done_units / work_units, 1.0)
                    progress_bar.progress(progress)

                    if phase in ("starting", "unknown"):
                        phase_placeholder.warning(f"⚙️ Initializing batch ({total} files)… {current}")
                    elif phase == "ocr":
                        phase_placeholder.warning(f"🖼️ Phase 1/2 — GLM-OCR ({ocr_done}/{total} files)… {current}")
                    elif phase == "extract":
                        phase_placeholder.info(f"🧠 Phase 2/2 — Extraction ({extract_done}/{total} files)… {current}")
                    elif phase == "done":
                        phase_placeholder.success(f"✅ Ingestion Complete — {extract_done} extracted, {skipped} skipped, {failed} failed.")
                        done = True
                    elif phase == "error":
                        phase_placeholder.error(f"❌ Ingestion failed: {status.get('error', 'unknown error')}")
                        done = True

                progress_bar.progress(1.0)

                if extract_done + skipped > 0:
                    certs = api_client.get("/api/v1/certificates", params={"batch_id": batch_id}).json()
                    st.session_state.certificates = certs.get("certificates", [])
                    if certs.get("raw_markdown"):
                        st.session_state.raw_markdown = certs["raw_markdown"]
                    st.session_state.chunks = _index_document_chunks(
                        {"raw_markdown": certs.get("raw_markdown", ""), "filename": "batch"},
                        "batch",
                    )

            except Exception as e:
                st.error(f"❌ Batch ingestion error: {str(e)}")

    # Display Extracted Data Cards
    if st.session_state.certificates:
        st.markdown("---")
        st.subheader("Extracted Certificates")
        for idx, cert in enumerate(st.session_state.certificates):
            with st.expander(f"Certificate #{idx + 1} - {cert.get('supplier', 'Unknown Supplier')}", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f"**🧩 Component**\n\n#### `{cert.get('component', 'N/A')}`")
                c2.markdown(f"**🏢 Supplier**\n\n#### `{cert.get('supplier', 'N/A')}`")
                c3.markdown(f"**🌍 Country**\n\n#### `{cert.get('country', 'N/A')}`")
                c4.markdown(f"**🔢 Certif Number**\n\n#### `{cert.get('certif_number', 'N/A')}`")
                
                st.markdown("---")
                c5, c6, c7 = st.columns(3)
                c5.markdown(f"**🏛️ Authority**\n\n#### `{cert.get('authority', 'N/A')}`")
                c6.markdown(f"**📅 Issue Date**\n\n#### `{cert.get('issue_date', 'N/A')}`")
                c7.markdown(f"**⏳ Exp Date**\n\n#### `{cert.get('exp_date', 'N/A')}`")

        if st.session_state.raw_markdown:
            with st.expander("🔍 Debug: View Raw OCR Output", expanded=False):
                st.text_area("OCR Markdown", st.session_state.raw_markdown, height=350)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 2: CHAT (RAG Q&A Assistant)
# ──────────────────────────────────────────────────────────────────────────────
elif selected_nav == "CHAT":
    st.header("💬 Hybrid RAG Compliance Assistant")
    st.markdown("Ask natural language questions across all indexed certificates and document chunks.")
    
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_input("Enter your compliance query (e.g., 'How many certificates expire in 2026?' or 'List all Valeo certificates')")
        submitted = st.form_submit_button("Send Query", type="primary")
    
    if submitted and user_query:
        with st.spinner("Analyzing query intent & retrieving context..."):
            try:
                chat_resp = api_client.post("/api/v1/chat", json={"query": user_query})
                if chat_resp.status_code == 200:
                    data = chat_resp.json()
                    answer_text = data.get("answer", "No answer generated.")
                    intent = data.get("intent", "UNSTRUCTURED_RAG")
                    reasoning = data.get("reasoning", "")
                    sources = data.get("sources", [])
                    latency_ms = data.get("latency_ms", 0.0)

                    st.session_state.chat_history.insert(0, {
                        "role": "assistant",
                        "content": answer_text,
                        "intent": intent,
                        "reasoning": reasoning,
                        "sources": sources,
                        "latency_ms": latency_ms,
                    })
                    st.session_state.chat_history.insert(0, {
                        "role": "user",
                        "content": user_query,
                    })
                else:
                    st.error(f"Chat request failed: {chat_resp.text[:300]}")
            except Exception as e:
                st.error(f"Error during Q&A: {str(e)}")

    st.markdown("---")
    
    # Render Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                intent = msg.get("intent", "UNSTRUCTURED_RAG")
                reasoning = msg.get("reasoning", "")
                latency = msg.get("latency_ms", 0.0)

                if intent == "METADATA_QUERY":
                    st.markdown(f"`⚡ SQL Path (Relational Database)` | ⏱️ `{latency:.1f} ms`")
                elif intent == "HYBRID_QUERY":
                    st.markdown(f"`🔗 Dual-Path Hybrid (SQL + Vector RRF)` | ⏱️ `{latency:.1f} ms`")
                else:
                    st.markdown(f"`🔍 Hybrid Vector RAG (Dense + Sparse RRF)` | ⏱️ `{latency:.1f} ms`")

                if reasoning:
                    st.caption(f"**Router Decision:** {reasoning}")

                st.markdown(msg["content"])

                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📚 Retrieved Context Sources ({len(sources)})", expanded=False):
                        for idx, src in enumerate(sources, start=1):
                            fname = src.get("file_name") or src.get("certificate_id") or "Unknown"
                            pages = src.get("pages") or src.get("page_number") or "N/A"
                            supplier = src.get("supplier", "N/A")
                            st.markdown(f"**{idx}. File:** `{fname}` | **Supplier:** `{supplier}` | **Pages:** `{pages}`")
            else:
                st.markdown(msg["content"])


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 3: DATABASES (Database Management & CRUD)
# ──────────────────────────────────────────────────────────────────────────────
elif selected_nav == "DATABASES":
    st.header("🗄️ Relational & Vector Database Management")
    st.markdown("View live database records, perform manual additions/deletions, and manage automated SQL backups.")
    
    colA, colB = st.columns([3, 1])
    
    with colA:
        st.subheader("Persisted Certificate Records")
        if st.button("🔄 Refresh Table Data"):
            pass
            
        try:
            certs_resp = api_client.get("/api/v1/certificates")
            if certs_resp.status_code == 200:
                cert_list = certs_resp.json().get("certificates", [])
                if cert_list:
                    df = pd.DataFrame(cert_list)
                    display_cols = ["certificate_id", "component", "supplier", "country", "certif_number", "authority", "issue_date", "exp_date"]
                    df = df[[c for c in display_cols if c in df.columns]]
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No certificates currently stored in the database.")
            else:
                st.error("Failed to load certificates from API.")
        except Exception as e:
            st.error(f"Error fetching data: {e}")

    with colB:
        st.subheader("🛠️ Management Actions")
        
        with st.expander("➕ Add Manual Record", expanded=False):
            with st.form("add_manual_form", clear_on_submit=True):
                m_comp = st.text_input("Component", "IM3A")
                m_supp = st.text_input("Supplier", "VALEO")
                m_coun = st.text_input("Country", "Bolivia")
                m_cert = st.text_input("Certif Number", "401/2025")
                m_auth = st.text_input("Authority", "ATT")
                m_iss = st.text_input("Issue Date (YYYY-MM-DD)", "2025-06-04")
                m_exp = st.text_input("Exp Date (YYYY-MM-DD)", "2035-06-03")
                
                submitted_add = st.form_submit_button("Save Record", type="primary")
                if submitted_add:
                    payload = {
                        "component": m_comp,
                        "supplier": m_supp,
                        "country": m_coun,
                        "certif_number": m_cert,
                        "authority": m_auth,
                        "issue_date": m_iss if m_iss else None,
                        "exp_date": m_exp if m_exp else None
                    }
                    try:
                        resp = api_client.post("/api/v1/certificates/manual", json=payload)
                        if resp.status_code == 200:
                            st.success(f"Added! ID: {resp.json().get('certificate_id')}")
                        else:
                            st.error(f"Failed to add: {resp.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with st.expander("🗑️ Delete Record", expanded=False):
            with st.form("delete_form", clear_on_submit=True):
                del_id = st.text_input("Certificate ID to delete")
                submitted_del = st.form_submit_button("Delete Record")
                if submitted_del and del_id:
                    try:
                        resp = api_client.delete(f"/api/v1/certificates/{del_id}")
                        if resp.status_code == 200:
                            st.success(f"Deleted {del_id} successfully.")
                        else:
                            st.error(f"Failed to delete: {resp.text}")
                    except Exception as e:
                        st.error(f"Error deleting: {e}")
