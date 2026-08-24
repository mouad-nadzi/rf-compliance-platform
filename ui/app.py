import os
import sys
import time
import pandas as pd
import streamlit as st

# Ensure project root is in Python path for 'core' and 'storage' imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import local RAG Q&A logic
from core.rag import (
    chunk_for_qa,
    retrieve_relevant_chunks,
    answer_query_with_citations,
)

# Import FastAPI for in-process networking bypass
from fastapi.testclient import TestClient
from server.main import app as fastapi_app

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
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────────────────────────────────────
# Enforce Light White Theme & Top Navbar CSS
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

    .main .block-container, [data-testid="stMainBlockContainer"], div.block-container {
        padding-top: 0.2rem !important;
        margin-top: 0rem !important;
        padding-bottom: 2rem;
        max-width: 96%;
    }

    [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > section {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }

    /* Completely Hide Streamlit Left Sidebar */
    [data-testid="stSidebar"], section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Completely Hide Default Streamlit Top Header Bar, Toolbar, Decoration & Footer */
    header[data-testid="stHeader"], [data-testid="stHeader"], .stAppHeader, [data-testid="stDecoration"], [data-testid="stToolbar"], #MainMenu, footer {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }

    /* Top Navigation Bar Styling */
    .top-nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 0;
    }

    .top-nav-icon {
        width: 60px;
        height: 60px;
        object-fit: contain;
    }

    .top-nav-text {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.25rem;
        letter-spacing: 0.5px;
        color: #0f172a !important;
    }

    .top-nav-right {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 12px;
        padding: 4px 0;
    }

    .engine-status-badge {
        display: flex;
        align-items: center;
        gap: 8px;
        background: #ffffff;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        font-size: 0.85rem;
        font-weight: 600;
        color: #334155;
    }

    .status-online {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #16a34a;
        display: inline-block;
        box-shadow: 0 0 6px rgba(22, 163, 74, 0.4);
    }

    .status-offline {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #dc2626;
        display: inline-block;
    }

    /* Top Horizontal Navigation Radio Buttons */
    div[data-testid="stHorizontalBlock"] div[role="radiogroup"] {
        display: flex;
        justify-content: center;
        gap: 8px;
        background: #ffffff;
        padding: 4px 8px;
        border-radius: 30px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }

    div[data-testid="stHorizontalBlock"] div[role="radiogroup"] label {
        background: transparent;
        border-radius: 20px;
        padding: 6px 22px;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
        cursor: pointer;
        border: 1px solid transparent;
        margin: 0 !important;
    }

    div[data-testid="stHorizontalBlock"] div[role="radiogroup"] label:hover {
        background: #f1f5f9;
        color: #0f172a !important;
    }

    div[data-testid="stHorizontalBlock"] div[role="radiogroup"] label[data-checked="true"] {
        background: #f87171 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(248, 113, 113, 0.3);
    }

    div[data-testid="stHorizontalBlock"] div[role="radiogroup"] label[data-checked="true"] * {
        color: #ffffff !important;
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
# TOP NAVIGATION BAR (Replaces Sidebar)
# ──────────────────────────────────────────────────────────────────────────────
try:
    api_client.get("/")
    conn_status = "Online"
    status_class = "status-online"
except Exception:
    conn_status = "Offline"
    status_class = "status-offline"

nav_col1, nav_col2, nav_col3 = st.columns([3, 5, 3])

with nav_col1:
    st.markdown(
        """
        <div class="top-nav-brand">
            <img class="top-nav-icon" src="data:image/webp;base64,UklGRhgeAABXRUJQVlA4TAweAAAv/8RDEP8HOZIkRZLn0HtRktVfKji+zuNTAUIkSbai7OEPDgj868AOFuBu2m0kSYqUPXcavCHv5Tv8GlNDsm23bWPiQBBkJ5hmQdn/6b2XeUMayxdANmlD5qS9AxtauJTLk05nlmEpdQA4POYL+R+I3wM/PcTPC1gRCAQAAEgkgBWJFYnEwAogUNER6Kj4RSUJgSrABQrRVIALpiWEAhQiyRC0wAWTUgQKgUKoKkDBIGgJxXABMvCvQNIAbKIqphMWwbBimDFjhtGmCAQKRBYsaimtNPEOxz8cgKcIAoXBQhPAgIHVNNJkLAREhLpBAOgIhJAUjeLoG3o0YR1UjuAKIRRKNbiIxOFdmEqDGiCuDmCCQ7cYlfiIsLhUqSUYFqWUwgRL/hsPEU+p/kcRuIWgVTNYbUen4U7HZ1WHcg6Y3/coS6mFUoNPpctyVA/8jPSfvS8a+JxlXxZ/T2X6lpeDWHZ6v0Fvg6dfaZn3c/wPrxfn82Gb1tOpb58b765rPL6u6/l9TvjD9n+dlP7/7jNL7N5mZnvppRGUZgEJg04x6LBRAYNGwYYXpZiU0m0rIQ0GYFBKh3Sz5NIwO7szf+xzHvfHPJ+P5+y4r8f7/Yro/wRQXu6DtZJbkBXTP0h+4BxpYZsA2GGFrQD+0sPcD+BnKzwH4Dk9zLJE1PFa4VolVLimh6HrK51kSW+BKPqf///n//8Lf0WPX2J0PumzAZRapvFpjpwJe7Q9WUkGmKDtuQjjatqem7FGU7Q99KDR0zqVhXUbHwvJ5TkVp3VxmOh0zRwLHGY60LvXb6G5PGX6aq3FxVhgciiOlgaAF0xEvqe+6/s7mfjcWwBcLUPyAXCX1mIfgJmh+Aw5XQdNZPoFMFwXipFAvF9nkT0c7i0hcCQaYIiyshON3g3Fkx68SFrLwO+nKJRFjdrKurl005qAGQp0b7PwrKRjMC4fCjq+nXSeXxq4V8o5+2kCgFJ/OUO1ZzgAJPbySvEXM6obEs1ndOkcfUjqofdg/IE/NKsLwfhznwwaYOD6VV9CZ94oVfFZklsPwXuFJKY0gn8jxT/MBSRvJR3rPgjGXwtFWwgm35RBtLZH1yukZf1EBI+Goo4IFsvRt94v1CwUFYQe08vMFuoTimpCW/Qyrwp1CsV4oeN6mcMJAsVvhWKnSB3SzLYT6EohbRYspYBuxvG1yyD2Lwqtv7nLoPQO0s/ufKNc/F0fraSQb399euV7h8SQVtV5dPthMyjXX3D7MU1I+uDSACo/7rCXsy2KAZj+iw7kbG0YvxGwQr6neg7cdMMKV6vAuIU2Y2f3dV5Z9yP4t+Zb+4AbAOLm5zNdYAaCt5d08ZMRDfpnaSQyGwGodlzODggm5jPbwhQEHfOH2TpCsGS6lG1JADAyQx/RAjlHeaU0F0E3k+2Lg2DqBZNNFsF6Gf4KMPxEG+FNNMBSKeOEepsr/S4INzRZdaFBMu7AuIhDFxHjMtoipaFQLwm7FoxKrTjukSgp3cC8IyVry5wRMx4YlE9CDaFHZawOUiKgi6BqBq6TUl4R6si63hDG5fbLaMj5Rsa68jBMaedkvSH0k4yoEkYfkDZykcF8knrCJVD0BufaWAT3PCVhNOdNCVtdCD4hwPlVpKJfBg0xSF6hj6Bfa7qLPxSQQwsEthIz8A5EE/fwUjn1eEs9EB3GoQbBXOtIqmNQGbjf3k1aySiSnj3ByNWXuD0gPo43ivMGK1ANwrFrOWenGcVuJdmOG1mkVdw7eFnIQunoXNuN+HE/EXs4A4dZCzhtWEvB7MMh32M1geQJd8jCR39M1wjsjAU6WIeIsg/5iR/l5vRg7WLEn2H15oxiEVHWISdZ+Qc3ZmXpAx4C8JWl5B4B9wUWvSXWh9izOYVkWH0mgMX6gP0AWivnMmsw79JokffTeW9yiqjnPwAK6gOo01cPO5UTFcfpyKMr7wd78ybxv+fUUM/vZRNfI83lvQzPaQnk3PKuG4ivf4Bk/swZph4d5kLGXJKcufJoFsnNrigWl1/H4hgnVCpaVigXuoS+Ji3r+RkCxXeSBR8S+SCgZ6Gol+KMxl0kS3ZKNUp6IUDa1sOtG7/9wTd/kFVj/v5g1qgGPaNJd+rY+FKD4Q8+e0WKwrN+Htyi3f5sPcjeWjBM6hewD9+rRZEz9a+AJuNWu5ElRzdeLmldIoJP8tlFzBQEb5gpadtnd6XW+ua6NmJnGRi+7JNxIgmizW3COQmikx0yfB/B8PZ+TcTRogjaVMY0CLtP2EMniD8pozmCJv6hh3gLgj/xCoD5kT1UZ8yScMcVDLW1EMdcIo14PTllbeEUuId5zSG6VgfxK0TL8ZpxXOl28CtrKW+EUAcdxGahlJDhnB08yVrHmy7USQexXWgUrz+nGNnhPtYa3ptCv+kgAmVFhvJWcWbbwtk4RpFsXieRCgEdBLUXKHaa5xjJWGoL9AFjAfH9NQU6kh6yaZDkbSTx91ihN8keT8YL3b4mgS6MDjKM9IldpoyuPixGmuORUjnqFiCpj7gE0m7aBHUTiV9CUs83TwFQeTNJvzY3JaH+Ua1AS+Qc6ZNF5F/edfMRkv1PGSN30yiyzc4pQcruINnpa/blJ/lXygDA7ZMagbUuA/SUF+JzjzWoUq721yfJTq/0rgR4Rg7IJGvOhuFbGoG/YTzNInadedVHli1iFBfQB3QL8n5YZenbRgkagZNuo69zS02M3iKN4CcGFS/llq6VylHsok7A+VcFFJl3nnLN195MSnzzIP1vxqe29Ny6NDtcyb91WItBK/UWO+ogZ/GH0sORFW/DcPgajUVPD4LWvBJ+PJGEoHE/aCuegejMqHBjVxwE3Qc0FYeThTAszPBVhnC5KD3FxxBPOR9edADzbz1FBQY2hxdfcd7SUsSA2ze8SOOM1lKcZn0fXlTmlNJSOBM4rcKL4ZwZWgoax9kYXvThfKJ1iNn82qeDoqVsYozxhxdrOGus5c36N8RZoOM3X94/NU21A82zbrS9/mOO/KPVXMNrhUWz2V1t5tnR7CHmcrQuCgDx3zglOOqK/UAWvDqb/9U5ovmjR4+umZaWlvbOxJz3zZ79qxXoAbEJJLipF3uAL2Q9kmPbmsE3m93ILOeqjrbV+0zhm81uZDHnyae/aTJp4sg0xU4Wi3ryzWJQ8xrzTIKtJt0yx0mo+UuyYluw+9jMQ2D3NddLCDpeAp2+R2Suwwrjwf+UiFa5wB6bbYXrY0XuzhD5CPx3r4QokAx4zpsgG2y3WbbAXtuaIhtsl5XSN80uCjX3ErnaojBUPYpMezrWXuaQnS3NdW2C4BMSKGNSsGZesmABN6/QISKixjwMsAJFpwX7zzEKDcb8Hhr/bcB9SWET7cUdbTMZ/UpC2SuDZT6fAHW3Ns8g2OsyOysXyHXNEHlHBtGfX1SKRULDHWTJBuC/RjkvJ/CK3bAC+buluQBXWvcAhQpxz4aEFhaNfY5M6LVKtNte3iZbcW4tDnXPoKB/3AOFx183T5q9lPLb2deU27oOUfdNKUSUb08WWXMb+GMyDagPD70tQUQZBU5cIq4coHFUKMh3kxTWFvba1Vau1YPKBwXpFA+VP0imXQl7/Y7s7GSua4UQLsuyrLOWhPZknHGbl3DBIlJlYdapUJjUZ5VR9pJyzk5WjIHKYw8ZPeuC0g+Yp5fNrLKzWpTruiB2VjEdwa/mD0I9eZirPNz+1Z7uwF6bkI38URRKn0SGnVxQermAaZzl7aU62Vmb3JejnEh1Umt6RQk/U/D0Cjz3HeXB1c9pLb9FPraZjTZyqiTU3sXgTgLU3pJMuw/22tbOYvPlvug5kUcVMwT8aSTalYe31Qc0PG8//tL2UjZgH9m1oPaiWTnSq0DtrlPm+cJe3NF21pByYVkjg9XzqeV8MZ5ru5CzJg9LbACjT1gpYI0DsNdhZB99ofj5lPM1KL4umTazkL28TXbWITdGGfWNJtwktbYAvxGJ75dQI2ADSNpiN3NsZo99HExQ3Y4c5wvJGdPo416q3GGeK+1NX5r3Qnsz77ZQ6fbWz8qVES35Yvjwpj+SYi8n8OKPMGgiDz/YAdDMaxmnNTq1N3tjXp32Zt5C9jEPUj3vzu+lym8dOdpB5sTlDgqHq/BOkpIlVCHbDvvU3Bj874i7280rF2ULeP+arZj/Ed58UrECDsXLuN0qgxTruFtC4tMUJucFWOXiFb3EoiY8tLIHlN5lEUckYRAk1slHyv0D/LhtlHeqHvj9iX8hgVfkuj0gtk3Ep66EiV5Sb08Jj1LeqQPgV8iSQL15+N4mgDnnrEARhKwEXqVbpOAJvFrOvFOBmRKeJpnni/HiT9kFZuaP6PwB/iZS8SjeU5S3JrCm4+MHzlrlcfBnOaXQAB5m2waKLrKAK3LwFO9uUnJxVkJMnpqsV0sBQFyjU5aIKidhPclNr8hz/WYbcD0fiNw8y3tFTbGsapSX5sx0BE1ZZ4V24E8k2Z14mKqir6fLAT7PMJs7ctCO95SawK6f12PD5gsmuvEeBON+Mt/1Ijz3bmnO6TwsUtAzWV9Kwtg7kZrBvOX2cH/ejpgGgOdjp2nmQ7h8lumag9+E5C+VUNWvIKKH4+Qg8QdzeSIYe/NAfYCcrcxy2COGrWY7GMdLPBwC+pyHZ5REv4+RA7yc/S9KgzwdK10GqT6TDALzXrM9CP4rFMoTbl6Zc0qiY8MloXa0iWIjGL/ZQ8U8HZthfNwkCzjFTLbTxSt+IyT0JQ8vqIm8zSSh1I8RmHa8TmpKYrmu5eXYZeQ+b5LZHJfTXO+C35pCeyaRVyifmoi6JspB7ACHWeIiB214zdVUmoWH8nIEahhMIJM246SSqdeBXyk7RPQKDy+riu6MlQN8djPS8jSvRJaSRvBST+fhoMvvAah71ixdOZ+byv+ehKco1DeK82LXqooy6ktCtePmiI8cLOehtZLm8fCBMw8H+Ra++o+fzHq+EKOHqR4BP80RMhrIwwPKosBQlxwU+SWycsPNSz6uoh4S8JEzD4fJXxCr6jPTudISfqTQZ4/lYbmyiH4pLAeuXgETJEQOaBYP5Y8oKNotAVOP5JHx1RMpXIDM3A/8t8iMnSXUdqiL1laVA0y8FElpKQElN6mHpslA/Ffrz+aFocwJwSrsJTNfSeF5TprCOYOHJxRGMR9Iwl17Q5YYQVgpA5i6+axqnpQCwFWs+GhLphROnfXB1+tO52qKvm7JpZEJokXvugFU6BdDpn4d/Plkzm0S7vEqjBytPHKQ0D1yQm9LAVxV0qxZc/Ss4XO//SUfzz9aktVdwwdl5F4sejRSQZTx20/5HWTugrG85GsmoQY8PKsyov3F5QBN0yMmf0qy/KiH8jNooZIApLx0LJIwgiIXVrwf/OfJrAXcvNRbSqPLtSThnTMhSYok0AdKAlz3rRaj+xQFFB4SiBw8k7s706H7xnSF7AK/ZIxp6Ase+qqNsr6UhNRtkZLo4moCXG9cE7peVlXAtOhIQeKN3NzhBwCgyHN+VTj+I2ErmfdqEi8pWm1EW+PkwNPfIS85okCL3IoCSiwWoTuFlYWyeyIEjSgXd7wUjMd5FdEF/Mo+E1EvHuapjn4fIwd4MyYyQn8pC662IvRTEWWhWIHIwIFcnL8agrdQg/duCQvJzGdL8DwFVEfH6kjCewVlpUQYqKdLVUBLEbpTTlkoezUSMMafi+sMwbhrShgE/vsOU1EbHhooj7wvSULhJyIj1DFFWXhYhC7dpyy8648ADKNc3FciaK+CW6kSfidzZ1fiYZ/yiLomyYFrmF9KoYgDrU1TVuwGEaKFY1WFthGAk7m5OkJDVTAU/A/I7E9KmOFUH90ZKweodzoiQs5uZRWFSlFC5N1SU1GFruX6RlJurr5QOwVcTeLFHjWdI42HjjZAGfUlofwaCYUjEETeH0a4lIR+YkRU8KFpSQrCJ7mNuLam35Graym0XwFfgP8imf9PCRXTbYACQ11ykPB3ZISIzjz6VUWXelIyOETk3bPor8G9Pn3dml+OcstJuZHLSCbbDs/yxwqM9lmvgJtX6JAFqCEPA+yA6JfCcoB56ZwikYqc6Wv3/fqENTe371c/VgoGS7D6mV7JMtD93zx6IVjsUrL+OPBfIyvu8fCKnbcFWltVEmZciJxY/Oonbhl3O1RDdPAdGQ3+1XMMdRsUWkjW3wb+mExL0DweetsDxbwpCSU2ihWNWBH9XkoCVqmHzr0roZDv3zyiVU2rlZ7R9wpZ31lLQnuyZnQSL+GCPZDjVY8ceNo5ImN0MlXCAAXR+TE8FPh3T52dwK/utwj15aGxTRBtLCEHeOCWwO1IFv0j4UEV0SYJC/OEpFeUsJiseiuV515lF3ShliRUPhkZo0m8GkpyVuJtzRMyBPy6ZN1BPEyxDcr6UBJSOgQpFtnazyuuJOrL658X5Hwxnmu7hbyjeVhiG0Rb4+QAffyRsEAyK15Nm3iD84K0AH8OWbmDhBoB+6DVYyRh2qEcxSNbNIIFNe3Ni3I5gRd/xFKOd3joYSN0pY4klNsQAbs/z9Yb4H9M1t4hoWymjZC/lyTEttUW/OHiFb1kMZrEQys7IdqSJAf4MKtEhOs+e9iYB6Ue+P3J6ic9vCLXbYV2j5WE6SkRrnfsoRtvSJ6PZeBXyLIczefhe3uhjM8lSYwwOUqwktT0Fa9bXo/ATAldyfrXknlxBxXUbcTdE/bIokBfVwRuDdhllXSzMG9jXo/HwZ/lVAC15OEz9SwAgNjFsoj+KRx5e5FXR0ktwT+TxyOqrIT9pMKYkjzXb6rZCMMxXml0vGqk7Xgcr6mKtify7qI8Hu3Ar09qfJiHd1XTxAir5dHNCZG1zFrgb1HQ2jLgz8/jcboIz71bEd57eFinmAeC7AgBOV71RNDOfg6++4pyHE8WgcRteTxeAv9LUuVmCVX9ahliVOJcKIg2loiYbasMiVNIsYe6vgOZ9wRyGUkX1ZithBdvWD7LYp9etPwtcxyM4yUeVobjPzw8o5askQbtKcQXakUyvDdUeexOm3chdVOOc7XTFPlecUjuTrkMRabcVIICF1tMgavN8SD4r5A6d0goc04pdH1+LEp3pJBnfRjB+A6Kr+bMsQWKr+GPSMylsKC0z+YqO0yx08UrfkMhdD8PL6iF6NzhAJnx0fhIha+U6tZTzrcVF7uBIhL7w4MWZHPtyJR1wG9NKi3o4RW6ohjTrh4TofgViv+Kcl51K64dRSTGBMKDAjbnvmqKf8CvlK0UWsDD6/ZEV+pEJhoprnKMQSuovbEzMjGMwoI0srn6ZEZ/NQkdSK1XUnielfZE/l6RiJgktaUeJcOqapvto8hEgfDgYbvrbIpHwB/pUAx9w8MDNkXUKSny0B1KH7OCDNdA5a7eTopM1KSwIP6SzRWJMsPN0hJ+JNXeLMXDDrui3ZUiDtOUVuswGTdTWenFpPZwbmB4MIFsrimZsR/4k0i9j0qo7bAryvg8wnDZpbDY3tlk7C2hrsTetyhS4bkSHiy2u9VmuJLC85xUkK8yD5ttiwJ9XRGFwVC2q+FJCv4LVH1Xv9Ok/DBuHIUFpX02V9lhhgXgzycVb5Jwt9e2iP4pEkmooqoSCwqQ6GQlFfv8oTUOssEwrnN40IJsrh2Z0Pl4e/4hJTmebs+/ZoYlbdknFEHH27J7hm+B9nyz/QYFu8pO7LfPT+LrnlDtz9t23yC7vHqR7VWT96KCfVa5sE2t181346Jas8wQyY65yL6uJOsf3qbaNSsup9P/aZ4e1vgCugt/13FF4KnaK194suGLsi5PxddX6CwKVodxStcwJOtDGLtf8mordheB4MNhh7cuBBv4NRXnKkA0dnu48R2Eh2oqBkP83jDjSJxYwhk9xWgGLoQXL4DZU0txDdynwot6nElaij9YQ8KL9zi1tBRrWY+FFzU407QUmfGc/eHFZE5TTcH1n49ZisYxCqWHF+05v1rKuWOb77/ULhRH8k5L7WMMpfDyXGmx6U5LzQHq+P87bRCAppaiD4Wq3Qwz6AmXSPIqsvI1ANj+32nrALS1VvpbAlUuUtjZ2h2s8BKydHoxIPHKf6c5hk7vlW0tCrQtYhD34lkKQzdWN5qUnyy+usHEA6QvjOnSp8kn3fJReBrY1W/8p20O0v+P2HnVrxlYM2VMg525qT8rIrFFlk5gXxIA94Hc0+/xADBXIxCohpxjArmmujBcoQ84CePVuaWAx2iIPmBFkG25JUdxo0H6AG8pg+RbFnEcaPpujQaDD9rMqScGPvOHZegjg7gj+gD61Q3A/TdJd/w09MMWvwZknZgBw9gXo2xkXU3krLw5BIdX5Q+E4EZ1AJ5upBNcXn/MjEUk/dQI5Ky+W87SFAQfecsu/C8j+Ec+SZ0qA0jtFSON0ts0mvs7aRMPpsK40AkZp4pCdJJdvAjRl+U0g3H1Y9L0io53EHxUQMIEiP9sDxshfkDGIwg+3KmD2AjRX3lX3IyJ9jCcUVtCeqoAftFB9BHqw+sIZlyWHUS7GK5jvGUQbayDaCQ0l9eKgyN2sB/cbbyeQiN1EJ8Kfczrz8pvB0+yNvHaCo3QQXQRWsR7iuOJsoP1rNW8xULzdRDeSgLv+XinPYxpZIfX3Yz4W7ys0iJ/6iBodUKQpL0kcQ6jgy3Q24zJJHGLwJukh1xdyaDyXpJ58bbQFIc9bHALeQrIoHYuo7dvaiIou8uns1/c5CO5+5MEZmaQTQ4WakNylzeIB2r+7Sct5t6ZRu7xN8k2+3uCxLYh6b7oTNJmBn75snbVes+vJDstMCERQNKDe0h3GtOj8dRpHz6VZZVVfd6pMLNRx2wZROdWbdwbRdrT9iVgWG6TJW7NccGw0lIpetTvENzV3wLXqyK452lNyyMQda0znaMuROPXaFnO3xZCxWyzdYH4+1qWx8D8R4r3QJ/Zb/Ta5ZQyhYE9mouz89PaKmgy51MZv1SCYa0NEryxnIcVdGZAJ78+oDmA9eqZwXlLQj8XgsY/xbsKbl/1nC8JDNMH3A/gWftpD9H4DawrrG/UcwBAVX3AP26UjFbPB5xmrOtFhFDTyQkkcx5XT3Qi8LI+gPZsvk6W9l2MDkh4jLOY1R/MpRyazHBd5d2Z/17JWcOiLUQ/zWmZpRGw+JEvCwEl+2SwbhQTu8fLmsr5mLWP8QFxHUNdyJmySV7mhiWXQqRZ7FCjUI1nArIWp8Cw4nEOPSPkXk/s8pz7WNRI6PYRVj8Ejf1TknNIMpA0SB/RHzlbSFqViKD3ZHLoYwH3s8QvzXmbl1lHoNA+4uaPDYaqTjmvwHCzLuJwggGOy6kLwf4s6p5qNHYZSZzB+ZBH2d/FGQ1fSeznIbpayo04o+q6iK4wHiLljEukJo9ifmj87rR5m7NJ5iucHyQQXeg5uXaDj38iiW8JPSZlJ4zjdREdg7SWsh6iCQ5eSAt6xIqdlRLCUUIDpRwPUl4XcSzRIO6ilCXWoe/FHiZzTxHqIoXeMRqmi6BnDJ4jqReEppO5vVNE5pHJB4okZshZWyLH25naCNpfJ7VWZ5I8QqS/ycj7kscosZXDbDfHCAwjycdaPtCoh5O0mL/HBrs702xEe74blVi4Vt8zZP6dhYNM8crSam6JNxp7nOx0Zd0cyS29pAM98UACULL3eQq58/iGU6Y40Xvi1PnrQ0Z08KluB86RLjQ9/9UAhdzbvwyAio8GQuX91IWcn18KmXY16l4YP+ALjWMOgo5M18x8geB9QrMJgq30MgVcAnGHQ1JPpLxDK/MNRAeFJEkEZ7QyE4SahSIbwnu0Mg8KfRIKKiPiPq+V6SfULSQfibxPWtnjHoGU6yHJnxDMtV+O8+SBTO1JlBz6TmAAhbaTJ8gLJHV9RaDcfr3JllmeEu2yZfgaB+ntCBFtq5qjbGeS+oQLAAod1Zm8hpxfySDHpnc9iPt8H4U+8Fv7R3/0k9y7YbhA63EpPTQX4w2wXwYR+S4FyNqnYfx+SM6l6y0cX6Lo/pA8BuO+kqx/y2VUPxR/eQov1FpsAJBmin6KonuNnglBhgdIdegs7gCoF5K1LgPXH6bJ6PnpwENmOlIsx8RACC4AKOzUWdBDRaefDAkNNviOzLqqPIAiS0xEx+ZXGTkgQKH8DO6tpPV0PDmr2MxuZFZvNeQsctpEZtxziDSqG2H8qFL0qh2CDNX2FAyyTttD4w1GOvU9mS+7gYnnSed7bVtB+p///+f//xM/PdoijjM+Tcy+opjks8L1WbjriB5mOID1VugPoJkeZgqAbVZoA+ATPcze8rHzHFa4WRdp0XoYonSyaDrlJQQ=" alt="Stellantis logo" />
        </div>
        """,
        unsafe_allow_html=True
    )

with nav_col2:
    selected_nav = st.radio(
        "NAVIGATION",
        ["HOME", "CHAT", "DATABASES"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="top_nav_radio"
    )

with nav_col3:
    st.markdown(
        f"""
        <div class="top-nav-right">
            <div class="engine-status-badge">
                <span class="{status_class}"></span>
                <span>Engine: <strong>{conn_status}</strong></span>
            </div>
            <div class="user-profile-badge">
                <span>System Administrator</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

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
                <div class="stat-trend trend-up">GLM-OCR + Qwen3.8-27B</div>
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
    st.header("Batch Certificate Ingestion")
    st.markdown("Upload compliance certificates (`.pdf`, `.png`, `.jpg`) for layout-aware OCR extraction and vector indexing.")
    
    uploaded_files = st.file_uploader(
        "Select certificate documents",
        type=["pdf", "png", "jpg"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.info(f"Selected {len(uploaded_files)} file(s) for ingestion.")

        if st.button("Process Batch Ingestion", type="primary"):
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
                        st.info(f"Attaching to processing batch '{batch_id}'...")
                    else:
                        st.error(f"Batch start failed: {ingest_resp.text[:300]}")
                        ingest_resp.raise_for_status()
                elif ingest_resp.status_code != 200:
                    st.error(f"Batch start failed: {ingest_resp.text[:300]}")
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
                        phase_placeholder.warning(f"Initializing batch ({total} files)… {current}")
                    elif phase == "ocr":
                        phase_placeholder.warning(f"Phase 1/2 — GLM-OCR ({ocr_done}/{total} files)… {current}")
                    elif phase == "extract":
                        phase_placeholder.info(f"Phase 2/2 — Extraction ({extract_done}/{total} files)… {current}")
                    elif phase == "done":
                        phase_placeholder.success(f"Ingestion Complete — {extract_done} extracted, {skipped} skipped, {failed} failed.")
                        done = True
                    elif phase == "error":
                        phase_placeholder.error(f"Ingestion failed: {status.get('error', 'unknown error')}")
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
                st.error(f"Batch ingestion error: {str(e)}")

    # Display Extracted Data Cards
    if st.session_state.certificates:
        st.markdown("---")
        st.subheader("Extracted Certificates")
        for idx, cert in enumerate(st.session_state.certificates):
            with st.expander(f"Certificate #{idx + 1} - {cert.get('supplier', 'Unknown Supplier')}", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f"**Component**\n\n#### `{cert.get('component', 'N/A')}`")
                c2.markdown(f"**Supplier**\n\n#### `{cert.get('supplier', 'N/A')}`")
                c3.markdown(f"**Country**\n\n#### `{cert.get('country', 'N/A')}`")
                c4.markdown(f"**Certif Number**\n\n#### `{cert.get('certif_number', 'N/A')}`")
                
                st.markdown("---")
                c5, c6, c7 = st.columns(3)
                c5.markdown(f"**Authority**\n\n#### `{cert.get('authority', 'N/A')}`")
                c6.markdown(f"**Issue Date**\n\n#### `{cert.get('issue_date', 'N/A')}`")
                c7.markdown(f"**Exp Date**\n\n#### `{cert.get('exp_date', 'N/A')}`")

        if st.session_state.raw_markdown:
            with st.expander("Debug: View Raw OCR Output", expanded=False):
                st.text_area("OCR Markdown", st.session_state.raw_markdown, height=350)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 2: CHAT (RAG Q&A Assistant)
# ──────────────────────────────────────────────────────────────────────────────
elif selected_nav == "CHAT":
    st.header("Hybrid RAG Compliance Assistant")
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
                    st.markdown(f"`SQL Path (Relational Database)` | `{latency:.1f} ms`")
                elif intent == "HYBRID_QUERY":
                    st.markdown(f"`Dual-Path Hybrid (SQL + Vector RRF)` | `{latency:.1f} ms`")
                else:
                    st.markdown(f"`Hybrid Vector RAG (Dense + Sparse RRF)` | `{latency:.1f} ms`")

                if reasoning:
                    st.caption(f"**Router Decision:** {reasoning}")

                st.markdown(msg["content"])

                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"Retrieved Context Sources ({len(sources)})", expanded=False):
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
    st.header("Relational & Vector Database Management")
    st.markdown("View and manage live database records, reference lookup datasets, import/export files, and execute CRUD operations.")
    st.markdown("<br>", unsafe_allow_html=True)

    db_nav_col, db_main_col = st.columns([1.2, 3.8])

    with db_nav_col:
        st.markdown(
            """
            <div style="font-family: 'Outfit', sans-serif; font-size: 0.8rem; font-weight: 700; color: #64748b; letter-spacing: 1px; margin-bottom: 10px; text-transform: uppercase;">
                DATABASE TABLES
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if "selected_db_table" not in st.session_state:
            st.session_state.selected_db_table = "RF Certificates"

        btn_type_certs = "primary" if st.session_state.selected_db_table == "RF Certificates" else "secondary"
        btn_type_auth = "primary" if st.session_state.selected_db_table == "Authorities" else "secondary"
        btn_type_supp = "primary" if st.session_state.selected_db_table == "Suppliers" else "secondary"

        if st.button("RF Certificates", key="btn_nav_certs", type=btn_type_certs, use_container_width=True):
            st.session_state.selected_db_table = "RF Certificates"
            st.rerun()

        if st.button("Authorities", key="btn_nav_auth", type=btn_type_auth, use_container_width=True):
            st.session_state.selected_db_table = "Authorities"
            st.rerun()

        if st.button("Suppliers", key="btn_nav_supp", type=btn_type_supp, use_container_width=True):
            st.session_state.selected_db_table = "Suppliers"
            st.rerun()

        selected_db_table = st.session_state.selected_db_table

        # ── Left-side Table Actions & Management Tools ─────────────────────────
        if selected_db_table == "RF Certificates":
            st.markdown(
                """
                <div style="font-family: 'Outfit', sans-serif; font-size: 0.8rem; font-weight: 700; color: #64748b; letter-spacing: 1px; margin-bottom: 10px; margin-top: 8px; text-transform: uppercase;">
                    TABLE ACTIONS
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if st.button("Refresh Table", use_container_width=True):
                pass

            try:
                csv_exp_resp = api_client.get("/api/v1/certificates/export/csv")
                if csv_exp_resp.status_code == 200:
                    st.download_button(
                        label="Export to CSV",
                        data=csv_exp_resp.content,
                        file_name="certificates_export.csv",
                        mime="text/csv",
                        type="secondary",
                        use_container_width=True
                    )
            except Exception as e_exp:
                st.caption(f"CSV Export offline: {e_exp}")

            try:
                excel_exp_resp = api_client.get("/api/v1/certificates/export/excel")
                if excel_exp_resp.status_code == 200:
                    st.download_button(
                        label="Export to Excel",
                        data=excel_exp_resp.content,
                        file_name="certificates_export.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="secondary",
                        use_container_width=True
                    )
            except Exception as e_exp:
                st.caption(f"Excel Export offline: {e_exp}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                """
                <div style="font-family: 'Outfit', sans-serif; font-size: 0.8rem; font-weight: 700; color: #64748b; letter-spacing: 1px; margin-bottom: 10px; text-transform: uppercase;">
                    MANAGEMENT ACTIONS
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander("Import CSV / Excel", expanded=False):
                uploaded_file = st.file_uploader("Select CSV or Excel file", type=["csv", "xlsx", "xls"], key="file_import_uploader")
                if uploaded_file:
                    try:
                        if uploaded_file.name.lower().endswith(".csv"):
                            df_prev = pd.read_csv(uploaded_file)
                        else:
                            df_prev = pd.read_excel(uploaded_file)

                        st.caption(f"Preview ({len(df_prev)} rows):")
                        st.dataframe(df_prev.head(3), use_container_width=True)
                        
                        if st.button("Import File Records", type="primary", use_container_width=True):
                            uploaded_file.seek(0)
                            mime_type = "text/csv" if uploaded_file.name.lower().endswith(".csv") else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            files_payload = [("file", (uploaded_file.name, uploaded_file.getvalue(), mime_type))]
                            imp_resp = api_client.post("/api/v1/certificates/import/file", files=files_payload)
                            if imp_resp.status_code == 200:
                                r_json = imp_resp.json()
                                st.success(f"Successfully imported {r_json.get('imported_count', 0)} record(s)!")
                            else:
                                st.error(f"Import failed: {imp_resp.text}")
                    except Exception as e_imp:
                        st.error(f"Error reading file: {e_imp}")

            with st.expander("Add Manual Record", expanded=False):
                with st.form("add_manual_form", clear_on_submit=True):
                    m_comp = st.text_input("Component", "IM3A")
                    m_supp = st.text_input("Supplier", "VALEO")
                    m_coun = st.text_input("Country", "Bolivia")
                    m_cert = st.text_input("Certif Number", "401/2025")
                    m_auth = st.text_input("Authority", "ATT")
                    m_iss = st.text_input("Issue Date", "2025-06-04")
                    m_exp = st.text_input("Exp Date", "2035-06-03")
                    m_link = st.text_input("Link (URL)", placeholder="https://...")
                    
                    submitted_add = st.form_submit_button("Save Record", type="primary", use_container_width=True)
                    if submitted_add:
                        payload = {
                            "component": m_comp,
                            "supplier": m_supp,
                            "country": m_coun,
                            "certif_number": m_cert,
                            "authority": m_auth,
                            "issue_date": m_iss if m_iss else None,
                            "exp_date": m_exp if m_exp else None,
                            "cert_link": m_link if m_link else None
                        }
                        try:
                            resp = api_client.post("/api/v1/certificates/manual", json=payload)
                            if resp.status_code == 200:
                                st.success(f"Added! ID: {resp.json().get('certificate_id')}")
                            else:
                                st.error(f"Failed to add: {resp.text}")
                        except Exception as e:
                            st.error(f"Error: {e}")

            with st.expander("Delete Record", expanded=False):
                with st.form("delete_form", clear_on_submit=True):
                    del_id = st.text_input("Certificate ID to delete")
                    submitted_del = st.form_submit_button("Delete Record", use_container_width=True)
                    if submitted_del and del_id:
                        try:
                            resp = api_client.delete(f"/api/v1/certificates/{del_id}")
                            if resp.status_code == 200:
                                st.success(f"Deleted {del_id} successfully.")
                            else:
                                st.error(f"Failed to delete: {resp.text}")
                        except Exception as e:
                            st.error(f"Error deleting: {e}")

    with db_main_col:
        # ──────────────────────────────────────────────────────────────────────
        # TABLE 1: RF CERTIFICATES
        # ──────────────────────────────────────────────────────────────────────
        if selected_db_table == "RF Certificates":
            st.subheader("Persisted RF Certificates Table")
            
            try:
                certs_resp = api_client.get("/api/v1/certificates")
                if certs_resp.status_code == 200:
                    cert_list = certs_resp.json().get("certificates", [])
                    if cert_list:
                        df = pd.DataFrame(cert_list)
                        # Clean issue_date and exp_date strings (strip 00:00:00 timestamps)
                        if "issue_date" in df.columns:
                            df["issue_date"] = df["issue_date"].apply(lambda d: str(d).split()[0] if d and str(d) not in ("None", "nan", "Not Found") else "—")
                        if "exp_date" in df.columns:
                            df["exp_date"] = df["exp_date"].apply(lambda d: str(d).split()[0] if d and str(d) not in ("None", "nan", "Not Found") else "—")

                        # Normalize cert_link to ensure valid public URLs for local stored files
                        if "cert_link" in df.columns:
                            from server.config import PUBLIC_API_URL
                            def _normalize_link(link_val):
                                if not link_val or str(link_val).strip() in ("None", "nan", "", "—"):
                                    return None
                                s_link = str(link_val).strip()

                                # If it's a locally stored file (contains /files/), extract relative path & attach PUBLIC_API_URL
                                if "/files/" in s_link:
                                    rel_path = s_link.split("/files/", 1)[1].lstrip("/")
                                    return f"{PUBLIC_API_URL}/files/{rel_path}"

                                # External web links (e.g. Stellantis portal) stay untouched
                                return s_link

                            df["cert_link"] = df["cert_link"].apply(_normalize_link)

                        base_cols = ["certificate_id", "component", "supplier", "country", "certif_number", "authority", "issue_date", "exp_date", "cert_link"]
                        df = df[[c for c in base_cols if c in df.columns]]
                        
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "cert_link": st.column_config.LinkColumn(
                                    label="Certificate Link",
                                    display_text="open file",
                                    help="Click to open the certificate document in your browser"
                                )
                            }
                        )

                    else:
                        st.info("No certificates currently stored in the database.")
                else:
                    st.error("Failed to load certificates from API.")
            except Exception as e:
                st.error(f"Error fetching data: {e}")

        # ──────────────────────────────────────────────────────────────────────
        # TABLE 2: AUTHORITIES LOOKUP TABLE
        # ──────────────────────────────────────────────────────────────────────
        elif selected_db_table == "Authorities":
            st.subheader("Regulatory Issuing Authorities Reference Table (`authority_lookups`)")
            st.markdown("Master dataset for regulatory issuing authorities, country jurisdictions, standard validity years, and normalization aliases.")
            
            try:
                auth_resp = api_client.get("/api/v1/lookups/authorities")
                if auth_resp.status_code == 200:
                    auth_list = auth_resp.json().get("authorities", [])
                    if auth_list:
                        df_auth = pd.DataFrame(auth_list)
                        
                        # Format aliases list as comma-separated string
                        if "aliases" in df_auth.columns:
                            df_auth["aliases"] = df_auth["aliases"].apply(
                                lambda a: ", ".join(a) if isinstance(a, list) and a else "—"
                            )
                        
                        col_stats1, col_stats2 = st.columns(2)
                        with col_stats1:
                            st.metric("Total Authorities Registered", len(df_auth))
                        with col_stats2:
                            unique_countries = df_auth["country"].nunique() if "country" in df_auth.columns else 0
                            st.metric("Unique Countries", unique_countries)

                        search_auth = st.text_input("Filter Authorities by Name or Country", "", key="search_auth_input")
                        if search_auth.strip():
                            q = search_auth.strip().lower()
                            df_auth = df_auth[
                                df_auth["canonical_authority"].astype(str).str.lower().str.contains(q) |
                                df_auth["country"].astype(str).str.lower().str.contains(q) |
                                df_auth["abbreviation"].astype(str).str.lower().str.contains(q)
                            ]

                        st.dataframe(
                            df_auth,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "id": "ID",
                                "canonical_authority": "Canonical Authority",
                                "abbreviation": "Abbreviation",
                                "country": "Country",
                                "standard_validity_years": "Standard Validity (Years)",
                                "aliases": "Aliases"
                            }
                        )
                    else:
                        st.info("No authority lookup records currently seeded in the database.")
                else:
                    st.error("Failed to load authority lookups from API.")
            except Exception as e_auth:
                st.error(f"Error fetching authority lookups: {e_auth}")

        # ──────────────────────────────────────────────────────────────────────
        # TABLE 3: SUPPLIERS LOOKUP TABLE
        # ──────────────────────────────────────────────────────────────────────
        elif selected_db_table == "Suppliers":
            st.subheader("Global Component Suppliers Reference Table (`supplier_lookups`)")
            st.markdown("Master dataset for OEM component manufacturers, global brand names, and legal entity aliases.")
            
            try:
                supp_resp = api_client.get("/api/v1/lookups/suppliers")
                if supp_resp.status_code == 200:
                    supp_list = supp_resp.json().get("suppliers", [])
                    if supp_list:
                        df_supp = pd.DataFrame(supp_list)
                        
                        # Format aliases list as comma-separated string
                        if "aliases" in df_supp.columns:
                            df_supp["aliases"] = df_supp["aliases"].apply(
                                lambda a: ", ".join(a) if isinstance(a, list) and a else "—"
                            )

                        st.metric("Total Global Suppliers Registered", len(df_supp))

                        search_supp = st.text_input("Filter Suppliers by Name or Alias", "", key="search_supp_input")
                        if search_supp.strip():
                            q = search_supp.strip().lower()
                            df_supp = df_supp[
                                df_supp["canonical_supplier"].astype(str).str.lower().str.contains(q) |
                                df_supp["aliases"].astype(str).str.lower().str.contains(q)
                            ]

                        st.dataframe(
                            df_supp,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "id": "ID",
                                "canonical_supplier": "Canonical Supplier",
                                "aliases": "Aliases"
                            }
                        )
                    else:
                        st.info("No supplier lookup records currently seeded in the database.")
                else:
                    st.error("Failed to load supplier lookups from API.")
            except Exception as e_supp:
                st.error(f"Error fetching supplier lookups: {e_supp}")
