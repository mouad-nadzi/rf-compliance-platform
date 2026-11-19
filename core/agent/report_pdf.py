"""
core/agent/report_pdf.py — Branded Stellantis Renewal Alert PDF Generator

Generates crisp, multi-page A4 PDF renewal reports with the Stellantis header
logo, structured expiring-certificate table, and Stellantis navy footer bar
using Pillow + system TrueType fonts (zero external binary dependencies).
"""

from datetime import date, datetime
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(_BASE_DIR, "data", "agent", "reports")
STELLANTIS_LOGO_PATH = os.path.join(_BASE_DIR, "frontend", "public", "stellantis.png")

_FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# A4 at 150 DPI
PAGE_W = 1240
PAGE_H = 1754
MARGIN_X = 64
CONTENT_W = PAGE_W - 2 * MARGIN_X

NAVY = (36, 56, 129)         # Stellantis #243881
DARK_TEXT = (15, 23, 42)     # #0f172a
MUTED_TEXT = (71, 85, 105)   # #475569
BORDER_GRAY = (203, 213, 225)
ROW_ALT = (248, 250, 252)
WHITE = (255, 255, 255)
URGENT_RED = (185, 28, 28)


def _load_font(path: str, size: int) -> ImageFont.ImageFont:
    try:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    except Exception:
        pass
    return ImageFont.load_default()


def _truncate_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    if draw.textlength(s, font=font) <= max_w:
        return s
    ellipsis = "..."
    while len(s) > 1 and draw.textlength(s + ellipsis, font=font) > max_w:
        s = s[:-1]
    return (s + ellipsis) if s else ""


def _draw_page_chrome(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    font_title: ImageFont.ImageFont,
    font_sub: ImageFont.ImageFont,
    font_footer: ImageFont.ImageFont,
    alert_days_before: int,
    cert_count: int,
    page_num: int,
    total_pages: int,
    generated_str: str,
) -> int:
    """Draw Stellantis header logo, title, divider, and navy footer bar. Returns Y after header."""
    y = 48
    if os.path.exists(STELLANTIS_LOGO_PATH):
        try:
            with Image.open(STELLANTIS_LOGO_PATH) as logo_raw:
                logo = logo_raw.convert("RGBA")
                target_h = 38
                ratio = target_h / max(1, logo.height)
                target_w = max(1, int(logo.width * ratio))
                logo_resized = logo.resize((target_w, target_h), Image.Resampling.LANCZOS)
                img.paste(logo_resized, (MARGIN_X, y), logo_resized)
        except Exception as exc:
            logger.debug(f"Could not render Stellantis logo in PDF: {exc}")
            draw.text((MARGIN_X, y + 4), "STELLANTIS", fill=NAVY, font=font_title)
    else:
        draw.text((MARGIN_X, y + 4), "STELLANTIS", fill=NAVY, font=font_title)

    title = "RF Certification Renewal Alert"
    title_w = draw.textlength(title, font=font_title)
    draw.text((PAGE_W - MARGIN_X - title_w, y), title, fill=NAVY, font=font_title)

    sub = f"Expiring within {alert_days_before} days | {cert_count} certificate{'s' if cert_count != 1 else ''}"
    sub_w = draw.textlength(sub, font=font_sub)
    draw.text((PAGE_W - MARGIN_X - sub_w, y + 32), sub, fill=MUTED_TEXT, font=font_sub)

    y += 66
    draw.line([(MARGIN_X, y), (PAGE_W - MARGIN_X, y)], fill=NAVY, width=3)

    # Bottom Navy Footer Bar
    footer_h = 42
    footer_y = PAGE_H - footer_h - 32
    draw.rectangle([(MARGIN_X, footer_y), (PAGE_W - MARGIN_X, footer_y + footer_h)], fill=NAVY)
    draw.text((MARGIN_X + 16, footer_y + 12), f"Stellantis RF Compliance Platform | Generated {generated_str}", fill=WHITE, font=font_footer)
    page_label = f"Page {page_num} / {total_pages}"
    page_label_w = draw.textlength(page_label, font=font_footer)
    draw.text((PAGE_W - MARGIN_X - 16 - page_label_w, footer_y + 12), page_label, fill=WHITE, font=font_footer)

    return y + 24


def generate_renewal_alert_pdf(
    certs: List[Dict[str, Any]],
    recipient_name: Optional[str] = None,
    alert_days_before: int = 30,
    output_path: Optional[str] = None,
) -> str:
    """
    Render a Stellantis-branded multi-page PDF containing the table of expiring
    certificates and return the absolute file path.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    if not output_path:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(REPORTS_DIR, f"Renewal_Report_{stamp}_{uuid.uuid4().hex[:6]}.pdf")

    font_title = _load_font(_FONT_BOLD, 24)
    font_sub = _load_font(_FONT_REGULAR, 15)
    font_body = _load_font(_FONT_REGULAR, 16)
    font_body_bold = _load_font(_FONT_BOLD, 16)
    font_th = _load_font(_FONT_BOLD, 15)
    font_td = _load_font(_FONT_REGULAR, 14)
    font_td_bold = _load_font(_FONT_BOLD, 14)
    font_footer = _load_font(_FONT_REGULAR, 13)

    columns = [
        ("Component", 260),
        ("Country", 150),
        ("Certif. Number", 200),
        ("Authority", 145),
        ("Supplier", 195),
        ("Exp Date", 162),
    ]
    # Ensure exact fit to CONTENT_W (1112px)
    row_h = 38
    th_h = 42
    footer_top = PAGE_H - 42 - 32
    max_y = footer_top - 24

    # Estimate pagination
    first_page_start_y = 48 + 66 + 24 + 98 + th_h
    next_page_start_y = 48 + 66 + 24 + th_h
    rows_first = max(1, (max_y - first_page_start_y) // row_h)
    rows_next = max(1, (max_y - next_page_start_y) // row_h)

    chunks: List[List[Dict[str, Any]]] = []
    remaining = list(certs or [])
    if not remaining:
        chunks.append([])
    else:
        chunks.append(remaining[:rows_first])
        remaining = remaining[rows_first:]
        while remaining:
            chunks.append(remaining[:rows_next])
            remaining = remaining[rows_next:]

    total_pages = len(chunks)
    generated_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    greeting_target = recipient_name or "Compliance Team"

    pages: List[Image.Image] = []
    for idx, chunk in enumerate(chunks, start=1):
        img = Image.new("RGB", (PAGE_W, PAGE_H), WHITE)
        draw = ImageDraw.Draw(img)

        y = _draw_page_chrome(
            img=img,
            draw=draw,
            font_title=font_title,
            font_sub=font_sub,
            font_footer=font_footer,
            alert_days_before=alert_days_before,
            cert_count=len(certs),
            page_num=idx,
            total_pages=total_pages,
            generated_str=generated_str,
        )

        if idx == 1:
            draw.text((MARGIN_X, y), f"Dear {greeting_target},", fill=DARK_TEXT, font=font_body_bold)
            y += 28
            draw.text(
                (MARGIN_X, y),
                "Please find below the list of RF certificates that need to be renewed as soon as possible.",
                fill=DARK_TEXT,
                font=font_body,
            )
            y += 24
            draw.text(
                (MARGIN_X, y),
                "Please ensure that the renewals are provided before the listed expiration dates.",
                fill=MUTED_TEXT,
                font=font_body,
            )
            y += 42

        # Draw Table Header
        x = MARGIN_X
        draw.rectangle([(MARGIN_X, y), (PAGE_W - MARGIN_X, y + th_h)], fill=NAVY)
        for col_label, col_w in columns:
            draw.text((x + 10, y + 12), col_label, fill=WHITE, font=font_th)
            x += col_w
        y += th_h

        # Draw Table Rows
        for r_idx, cert in enumerate(chunk):
            bg = ROW_ALT if r_idx % 2 == 1 else WHITE
            draw.rectangle([(MARGIN_X, y), (PAGE_W - MARGIN_X, y + row_h)], fill=bg, outline=BORDER_GRAY, width=1)

            exp_val = cert.get("exp_date")
            exp_str = exp_val.isoformat() if isinstance(exp_val, date) else str(exp_val or "")
            days_rem = cert.get("days_remaining")
            exp_color = URGENT_RED if (isinstance(days_rem, int) and days_rem <= 15) else DARK_TEXT

            cell_values = [
                (str(cert.get("component") or ""), DARK_TEXT, font_td_bold),
                (str(cert.get("country") or ""), DARK_TEXT, font_td),
                (str(cert.get("certif_number") or ""), DARK_TEXT, font_td),
                (str(cert.get("authority") or ""), DARK_TEXT, font_td),
                (str(cert.get("supplier") or ""), DARK_TEXT, font_td),
                (exp_str, exp_color, font_td_bold),
            ]

            cx = MARGIN_X
            for (val_str, color, fnt), (_, col_w) in zip(cell_values, columns):
                if cx > MARGIN_X:
                    draw.line([(cx, y), (cx, y + row_h)], fill=BORDER_GRAY, width=1)
                clipped = _truncate_to_width(draw, val_str, fnt, col_w - 18)
                draw.text((cx + 10, y + 11), clipped, fill=color, font=fnt)
                cx += col_w

            y += row_h

        pages.append(img)

    first, *rest = pages
    first.save(output_path, "PDF", resolution=150.0, save_all=True, append_images=rest)
    logger.info(f"Generated Stellantis renewal PDF report ({len(certs)} cert(s), {total_pages} page(s)) -> {output_path}")
    return os.path.abspath(output_path)
