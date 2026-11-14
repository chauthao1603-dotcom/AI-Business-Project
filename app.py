"""PV4 – Giao diện Streamlit cho dự án Virtual Try-On (AI Virtual Fitting Room).

Thiết kế hiện đại chuẩn studio thời trang (tham chiếu phong cách Fitroom.app),
tối ưu trải nghiệm khách hàng:
- Tab 1: 🛍️ Cửa hàng thời trang (Ảnh sản phẩm lớn, rõ nét, thẻ giá VND, bộ lọc)
- Tab 2: 🧥 Buồng thử đồ (Chọn mẫu có sẵn / Tải ảnh lên / Chụp Webcam trực tiếp)
- Tab 3: ✨ Kết quả & Bộ sưu tập (Before/After lớn, tải ảnh HD, lịch sử thử đồ Lookbook)
- Sidebar: Báo cáo kỹ thuật đối chứng & đánh giá mô hình (Dành cho Giảng viên)
"""
from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, UnidentifiedImageError

from src import config
from src.baseline import run_baseline
from src.utils import (
    pair_id,
    standardize_image,
    standardize_mask,
    validate_input_image,
)

# ==============================================================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN HIỆN ĐẠI (FITROOM INSP)
# ==============================================================================
st.set_page_config(
    page_title="FitRoom AI | Phòng Thử Đồ Ảo Thông Minh",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }

    /* Modern Luxury Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #090d16 0%, #1e1b4b 55%, #312e81 100%);
        color: #ffffff;
        padding: 2.2rem 2.6rem;
        border-radius: 22px;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 45px -15px rgba(15, 23, 42, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.22);
        border: 1px solid rgba(165, 180, 252, 0.4);
        color: #c7d2fe;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 4px 14px;
        border-radius: 999px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        margin: 0;
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.25;
        background: linear-gradient(90deg, #ffffff 0%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        margin: 0.6rem 0 0 0;
        font-size: 1.02rem;
        color: #cbd5e1;
        max-width: 820px;
        line-height: 1.55;
    }

    .hero-features {
        display: flex;
        gap: 12px;
        margin-top: 1.2rem;
        flex-wrap: wrap;
    }

    .hero-feat-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.82rem;
        color: #e2e8f0;
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(8px);
        padding: 5px 12px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        font-weight: 500;
    }

    /* Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 4px;
        margin-bottom: 1.4rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 26px;
        border-radius: 12px 12px 0 0;
        font-weight: 700;
        font-size: 1.05rem;
        color: #64748b;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #4f46e5;
        background: #f8fafc;
    }

    .stTabs [aria-selected="true"] {
        color: #4f46e5 !important;
        border-bottom: 3px solid #4f46e5 !important;
    }

    /* Product Card */
    .product-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 14px;
        margin-bottom: 16px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
    }

    .product-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 30px -8px rgba(0, 0, 0, 0.1);
        border-color: #cbd5e1;
    }

    .product-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.6rem 0 0.35rem 0;
        line-height: 1.35;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .product-price {
        font-size: 1.2rem;
        font-weight: 800;
        color: #4f46e5;
        margin-bottom: 0.45rem;
    }

    .product-meta {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 0.75rem;
        font-size: 0.8rem;
        flex-wrap: wrap;
    }

    .pill-cat {
        background: #eef2ff;
        color: #4338ca;
        padding: 3px 10px;
        border-radius: 999px;
        font-weight: 600;
    }

    .pill-color {
        background: #f1f5f9;
        color: #475569;
        padding: 3px 10px;
        border-radius: 999px;
        font-weight: 500;
    }

    /* Lookbook History Card */
    .lookbook-item {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 12px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease;
    }

    .lookbook-item:hover {
        transform: translateY(-3px);
        border-color: #6366f1;
        box-shadow: 0 8px 20px -4px rgba(99, 102, 241, 0.12);
    }

    .lookbook-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.4rem 0 0.2rem 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .lookbook-time {
        font-size: 0.78rem;
        color: #64748b;
        margin-bottom: 0.5rem;
    }

    /* Studio Card Panels */
    .studio-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 16px;
    }
    </style>

    <div class="hero-banner">
        <div class="hero-badge">✨ AI Virtual Fitting Room 2.0</div>
        <h1 class="hero-title">Phòng Thử Đồ Thời Trang Thông Minh</h1>
        <p class="hero-subtitle">
            Trải nghiệm công nghệ thử trang phục trực tuyến thế hệ mới bằng trí tuệ nhân tạo (Diffusion Model).
            Tự tin chọn đồ, xem trước vóc dáng chuẩn xác chỉ trong tích tắc.
        </p>
        <div class="hero-features">
            <span class="hero-feat-tag">⚡ Suy luận thời gian thực</span>
            <span class="hero-feat-tag">🎨 Độ phân giải HD 768x1024</span>
            <span class="hero-feat-tag">📸 Hỗ trợ Webcam & Ảnh tự tải</span>
            <span class="hero-feat-tag">👗 Lưu trữ bộ sưu tập Lookbook</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. CÁC HÀM TIỆN ÍCH DỮ LIỆU & MÔ HÌNH
# ==============================================================================
@st.cache_resource
def get_tryon_fn():
    """Nạp hàm run_tryon chỉ một lần cho cả phiên."""
    return config.get_run_tryon()


def _fallback_baseline(person_img: Image.Image, garment_img: Image.Image, mask: Image.Image) -> Image.Image:
    p = standardize_image(person_img)
    g = standardize_image(garment_img)
    m = standardize_mask(mask)
    bbox = m.getbbox()
    if not bbox:
        return p
    x1, y1, x2, y2 = bbox
    g_resized = g.resize((max(1, x2 - x1), max(1, y2 - y1)), Image.BICUBIC)
    canvas = p.copy()
    canvas.paste(g_resized, (x1, y1))
    return Image.composite(canvas, p, m)


def safe_run_baseline(person_img: Image.Image, garment_img: Image.Image, mask: Image.Image) -> Image.Image:
    try:
        return run_baseline(person_img, garment_img, mask)
    except NotImplementedError:
        return _fallback_baseline(person_img, garment_img, mask)


def _compute_cache_key(
    person_img: Image.Image,
    garment_img: Image.Image,
    person_name: str | None = None,
    garment_name: str | None = None,
) -> str:
    if person_name and garment_name:
        return pair_id(person_name, garment_name)

    h = hashlib.sha1()
    for img in (standardize_image(person_img), standardize_image(garment_img)):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        h.update(buf.getvalue())
    return h.hexdigest()[:16]


def get_result(
    person_img: Image.Image,
    garment_img: Image.Image,
    person_name: str | None = None,
    garment_name: str | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image, float, bool, str]:
    validate_input_image(person_img, "ảnh người")
    validate_input_image(garment_img, "ảnh áo")

    key = _compute_cache_key(person_img, garment_img, person_name, garment_name)
    config.CACHE.mkdir(parents=True, exist_ok=True)
    cache_img_path = config.CACHE / f"{key}.png"
    cache_mask_path = config.CACHE / f"{key}_mask.png"

    t0 = time.perf_counter()
    from_cache = False

    if cache_img_path.exists() and cache_mask_path.exists():
        with Image.open(cache_img_path) as r_im:
            result_img = standardize_image(r_im.copy())
        with Image.open(cache_mask_path) as m_im:
            mask_img = standardize_mask(m_im.copy())
        from_cache = True
    else:
        tryon_fn = get_tryon_fn()
        result_img, mask_img = tryon_fn(person_img, garment_img, mask=None)
        result_img.save(cache_img_path, format="PNG")
        mask_img.save(cache_mask_path, format="PNG")

    baseline_img = safe_run_baseline(person_img, garment_img, mask_img)
    elapsed = time.perf_counter() - t0
    return result_img, baseline_img, mask_img, elapsed, from_cache, key


def load_catalog() -> pd.DataFrame:
    for cat_path in [config.CATALOG, config.REPO_ROOT / "data" / "catalog.csv", config.CATALOG_SAMPLE]:
        if cat_path.exists():
            return pd.read_csv(cat_path)
    return pd.DataFrame(columns=["product_id", "name", "category", "color", "price_vnd", "image_file"])


def make_placeholder_garment(color_name: str) -> Image.Image:
    color_map = {
        "trắng": ((248, 250, 252), (203, 213, 225)),
        "xanh": ((96, 165, 250), (37, 99, 235)),
        "xanh dương": ((96, 165, 250), (37, 99, 235)),
        "xanh lá": ((110, 231, 183), (16, 185, 129)),
        "be": ((231, 213, 188), (180, 155, 125)),
        "đen": ((51, 65, 85), (15, 23, 42)),
        "đỏ": ((248, 113, 113), (185, 28, 28)),
        "hồng": ((249, 168, 212), (219, 39, 119)),
        "vàng": ((253, 224, 71), (202, 138, 4)),
    }
    fill_rgb, border_rgb = color_map.get(str(color_name).strip().lower(), ((148, 163, 184), (71, 85, 105)))
    w, h = config.IMG_SIZE
    img = Image.new("RGB", (w, h), (241, 245, 249))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([36, 36, w - 36, h - 36], radius=32, fill=(255, 255, 255))
    draw.rounded_rectangle([int(w * 0.25), int(h * 0.22), int(w * 0.75), int(h * 0.78)], radius=24, fill=fill_rgb, outline=border_rgb, width=4)
    return img


def load_garment_image(row: pd.Series) -> Image.Image:
    for folder in [
        config.GARMENTS,
        config.REPO_ROOT / "data" / "garments",
        config.REPO_ROOT / "data" / "processed" / "garments",
        config.RAW / "cloth",
    ]:
        img_path = folder / str(row["image_file"])
        if img_path.exists():
            with Image.open(img_path) as im:
                return im.convert("RGB").copy()
    return make_placeholder_garment(str(row.get("color", "")))


def load_demo_persons() -> list[Path]:
    """Tìm tất cả ảnh người mẫu demo từ Drive lẫn trong repo GitHub."""
    candidate_dirs = [
        config.DEMO_PERSONS,
        config.REPO_ROOT / "data" / "processed" / "demo" / "persons",
        config.REPO_ROOT / "data" / "demo" / "persons",
    ]
    found = {}
    for d in candidate_dirs:
        if d.exists():
            for p in sorted(d.glob("*")):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    if p.name not in found:
                        found[p.name] = p
    return list(found.values())


MODEL_DESCRIPTIONS = {
    "00006_00": "Người mẫu 01 (Nữ - Áo thun đen)",
    "00008_00": "Người mẫu 02 (Nữ - Áo thể thao phối sọc)",
    "00013_00": "Người mẫu 03 (Nữ - Áo thun hồng)",
    "00017_00": "Người mẫu 04 (Nữ - Áo thun trắng)",
    "00034_00": "Người mẫu 05 (Nữ - Áo sát nách có túi)",
    "00035_00": "Người mẫu 06 (Nữ - Áo thun trắng cổ tròn)",
    "00055_00": "Người mẫu 07 (Nữ - Áo thun xám)",
    "00057_00": "Người mẫu 08 (Nữ - Áo thun tay dài đen)",
    "00064_00": "Người mẫu 09 (Nữ - Áo thun kẻ sọc ngang)",
    "00067_00": "Người mẫu 10 (Nữ - Áo thun xanh dương)",
    "00069_00": "Người mẫu 11 (Nữ - Áo thun trắng cổ V)",
    "00074_00": "Người mẫu 12 (Nữ - Áo thể thao cam)",
}


def format_person_label(p: Path) -> str:
    stem_name = p.stem
    return MODEL_DESCRIPTIONS.get(stem_name, f"Người mẫu {stem_name}")


def format_vnd(val) -> str:
    try:
        return f"{int(val):,}".replace(",", ".") + " ₫"
    except (ValueError, TypeError):
        return "189.000 ₫"


def image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def read_uploaded_image(uploaded_file, label: str) -> Image.Image:
    if uploaded_file is None:
        raise ValueError(f"Vui lòng tải lên {label}.")
    max_mb = 15
    if hasattr(uploaded_file, "size") and uploaded_file.size > max_mb * 1024 * 1024:
        raise ValueError(f"{label.capitalize()} quá lớn (> {max_mb}MB). Vui lòng chọn ảnh nhẹ hơn.")
    try:
        with Image.open(uploaded_file) as im:
            im.load()
            img = im.convert("RGB")
    except UnidentifiedImageError:
        raise ValueError(f"{label.capitalize()} không đúng định dạng ảnh (chỉ nhận JPG/PNG).")
    except Exception as exc:
        raise ValueError(f"Không thể đọc {label}: {exc}")

    validate_input_image(img, label)
    return img


# ==============================================================================
# 3. KHỞI TẠO SESSION STATE
# ==============================================================================
for k, v in {
    "product_id": None,
    "person_img": None,
    "person_name": None,
    "garment_img": None,
    "garment_name": None,
    "result": None,
    "baseline": None,
    "mask": None,
    "elapsed": None,
    "from_cache": False,
    "pair_key": None,
    "tryon_history": [],  # Bộ sưu tập / Lịch sử thử đồ của khách hàng
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

catalog_df = load_catalog()
demo_persons = load_demo_persons()


def add_to_history(
    person_img: Image.Image,
    garment_img: Image.Image,
    result_img: Image.Image,
    baseline_img: Image.Image,
    mask_img: Image.Image,
    product_id: str | None,
    prod_name: str,
    prod_color: str,
    prod_price: str,
    elapsed: float,
):
    """Lưu trang phục đã thử vào bộ sưu tập lịch sử để người dùng xem lại."""
    # Tránh trùng lặp liên tiếp nếu cùng sản phẩm
    if st.session_state["tryon_history"]:
        last = st.session_state["tryon_history"][-1]
        if last.get("product_id") == product_id and last.get("garment_title") == prod_name:
            return

    item = {
        "id": f"look_{len(st.session_state['tryon_history']) + 1}",
        "timestamp": time.strftime("%H:%M:%S"),
        "product_id": product_id,
        "garment_title": prod_name or "Trang phục tùy chọn",
        "garment_color": prod_color or "",
        "garment_price": prod_price or "",
        "garment_img": garment_img.copy(),
        "person_img": person_img.copy(),
        "result_img": result_img.copy(),
        "baseline_img": baseline_img.copy() if baseline_img else None,
        "mask_img": mask_img.copy() if mask_img else None,
        "elapsed": elapsed,
    }
    st.session_state["tryon_history"].append(item)


# ==============================================================================
# 4. THANH ĐIỀU HƯỚNG TẬP TRUNG KHÁCH HÀNG (3 TABS CHÍNH)
# ==============================================================================
tab_store, tab_tryon, tab_result = st.tabs(
    ["🛍️ Cửa hàng thời trang", "🧥 Buồng thử đồ AI", "✨ Kết quả & Bộ sưu tập"]
)

# ==============================================================================
# 5. TAB 1: CỬA HÀNG THỜI TRANG (CATALOG GIAO DIỆN LỚN & ĐẸP)
# ==============================================================================
with tab_store:
    if catalog_df.empty:
        st.warning("Hiện chưa có sản phẩm nào trong cửa hàng.")
    else:
        # Bộ lọc sản phẩm hiện đại
        st.markdown("#### 🔍 Khám phá bộ sưu tập trang phục")
        f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
        with f_col1:
            search_kw = st.text_input("Tìm kiếm trang phục", placeholder="Nhập tên áo, loại áo...", label_visibility="collapsed")
        with f_col2:
            cats = ["Tất cả danh mục"] + sorted(catalog_df["category"].dropna().unique().tolist())
            sel_cat = st.selectbox("Danh mục", cats, label_visibility="collapsed")
        with f_col3:
            colors = ["Tất cả màu sắc"] + sorted(catalog_df["color"].dropna().unique().tolist())
            sel_col = st.selectbox("Màu sắc", colors, label_visibility="collapsed")

        filtered = catalog_df.copy()
        if search_kw.strip():
            filtered = filtered[filtered["name"].str.contains(search_kw.strip(), case=False, na=False)]
        if sel_cat != "Tất cả danh mục":
            filtered = filtered[filtered["category"] == sel_cat]
        if sel_col != "Tất cả màu sắc":
            filtered = filtered[filtered["color"] == sel_col]

        st.caption(f"Hiển thị **{len(filtered)}** sản phẩm sẵn sàng thử trực tuyến")
        st.write("")

        # Hiển thị 3 cột rộng rãi, ảnh to rõ nét chuẩn thời trang
        cols = st.columns(3, gap="medium")
        for idx, (_, row) in enumerate(filtered.iterrows()):
            with cols[idx % 3]:
                with st.container(border=True):
                    g_img = load_garment_image(row)
                    st.image(g_img, use_container_width=True)

                    price_str = format_vnd(row.get("price_vnd", 189000))
                    st.markdown(
                        f"""
                        <div class="product-title" title="{row['name']}">{row['name']}</div>
                        <div class="product-price">{price_str}</div>
                        <div class="product-meta">
                            <span class="pill-cat">{row['category']}</span>
                            <span class="pill-color">Màu {row['color']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    is_chosen = st.session_state["product_id"] == row["product_id"]
                    btn_label = "✅ Đang trong buồng thử" if is_chosen else "✨ Mặc thử ngay"
                    if st.button(
                        btn_label,
                        key=f"btn_store_{row['product_id']}",
                        type="primary" if is_chosen else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state["product_id"] = row["product_id"]
                        st.toast(f"Đã chọn '{row['name']}'! Chuyển sang buồng thử đồ ngay...")
                        st.rerun()

# ==============================================================================
# 6. TAB 2: BUỒNG THỬ ĐỒ AI (STUDIO 2 CỘT CHUYÊN NGHIỆP)
# ==============================================================================
with tab_tryon:
    st.markdown("#### 🎨 Studio thử đồ trực tuyến")
    col_p, col_g = st.columns(2, gap="large")

    input_person_img: Image.Image | None = None
    input_person_name: str | None = None
    input_garment_img: Image.Image | None = None
    input_garment_name: str | None = None
    active_prod_name = "Trang phục tùy chọn"
    active_prod_color = ""
    active_prod_price = ""

    # Cột 1: Người mẫu
    with col_p:
        with st.container(border=True):
            st.markdown("##### 👤 Bước 1: Chọn ảnh người mẫu")

            p_sources = ["👤 Chọn người mẫu có sẵn", "📤 Tải ảnh từ máy tính", "📸 Chụp từ Camera"]
            p_mode = st.radio("Nguồn ảnh người:", p_sources, horizontal=True, label_visibility="collapsed")

            if p_mode == "👤 Chọn người mẫu có sẵn":
                if not demo_persons:
                    st.warning("Đang quét ảnh người mẫu...")
                else:
                    sel_p_path = st.selectbox(
                        "Danh sách 12 người mẫu chuẩn:",
                        demo_persons,
                        format_func=format_person_label,
                    )
                    try:
                        with Image.open(sel_p_path) as im:
                            input_person_img = im.convert("RGB").copy()
                        validate_input_image(input_person_img, "ảnh người")
                        input_person_name = sel_p_path.name
                        st.image(input_person_img, caption=f"Đang chọn: {format_person_label(sel_p_path)}", use_container_width=True)
                    except Exception as err:
                        st.error(f"Lỗi đọc ảnh người mẫu: {err}")

            elif p_mode == "📤 Tải ảnh từ máy tính":
                up_person = st.file_uploader(
                    "Tải lên ảnh chân dung toàn thân hoặc nửa người (JPG, PNG)",
                    type=["jpg", "jpeg", "png"],
                    key="up_person",
                )
                if up_person is not None:
                    try:
                        input_person_img = read_uploaded_image(up_person, "ảnh người")
                        input_person_name = None
                        st.image(input_person_img, caption="Ảnh chân dung đã tải lên", use_container_width=True)
                    except ValueError as err:
                        st.error(str(err))

            else:  # 📸 Chụp từ Camera
                cam_person = st.camera_input("Chụp ảnh chân dung của bạn (đứng thẳng nhìn vào camera)", key="cam_person")
                if cam_person is not None:
                    try:
                        input_person_img = read_uploaded_image(cam_person, "ảnh chụp từ camera")
                        input_person_name = None
                        st.image(input_person_img, caption="Ảnh chân dung vừa chụp", use_container_width=True)
                    except ValueError as err:
                        st.error(str(err))

    # Cột 2: Trang phục
    with col_g:
        with st.container(border=True):
            st.markdown("##### 👕 Bước 2: Chọn trang phục muốn thử")
            g_mode = st.radio(
                "Nguồn ảnh áo:",
                ["🛍️ Chọn từ Cửa hàng", "📤 Tải ảnh áo từ máy tính"],
                horizontal=True,
                label_visibility="collapsed",
            )
            if g_mode == "🛍️ Chọn từ Cửa hàng":
                if catalog_df.empty:
                    st.warning("Chưa có sản phẩm trong cửa hàng.")
                else:
                    pids = catalog_df["product_id"].tolist()
                    default_idx = pids.index(st.session_state["product_id"]) if st.session_state["product_id"] in pids else 0
                    chosen_pid = st.selectbox(
                        "Chọn sản phẩm trong danh mục:",
                        pids,
                        index=default_idx,
                        format_func=lambda pid: (
                            f"{pid} - {catalog_df.loc[catalog_df['product_id'] == pid, 'name'].values[0]} "
                            f"({catalog_df.loc[catalog_df['product_id'] == pid, 'color'].values[0]}) • "
                            f"{format_vnd(catalog_df.loc[catalog_df['product_id'] == pid, 'price_vnd'].values[0])}"
                        ),
                    )
                    st.session_state["product_id"] = chosen_pid
                    row = catalog_df[catalog_df["product_id"] == chosen_pid].iloc[0]
                    input_garment_img = load_garment_image(row)
                    input_garment_name = str(row["image_file"])
                    active_prod_name = str(row["name"])
                    active_prod_color = str(row["color"])
                    active_prod_price = format_vnd(row.get("price_vnd", 189000))

                    st.image(input_garment_img, caption=f"{active_prod_name} ({active_prod_color})", use_container_width=True)
            else:
                up_garment = st.file_uploader("Tải lên ảnh sản phẩm áo (JPG, PNG)", type=["jpg", "jpeg", "png"], key="up_garment")
                if up_garment is not None:
                    try:
                        input_garment_img = read_uploaded_image(up_garment, "ảnh áo")
                        input_garment_name = None
                        active_prod_name = "Áo tùy chỉnh tải lên"
                        st.image(input_garment_img, caption="Ảnh áo đã tải lên", use_container_width=True)
                    except ValueError as err:
                        st.error(str(err))

    st.write("")
    # Nút bấm Hero Action
    if st.button("✨ Bắt đầu mặc thử ngay với AI (Virtual Try-On)", type="primary", use_container_width=True):
        if input_person_img is None or input_garment_img is None:
            st.error("Vui lòng chọn hoặc tải lên đầy đủ cả ảnh người mẫu và trang phục trước khi thử đồ!")
        else:
            try:
                with st.spinner("AI đang tính toán tạo phom dáng trang phục, vui lòng chờ trong giây lát…"):
                    res_img, base_img, mask_img, elapsed, from_cache, key = get_result(
                        input_person_img,
                        input_garment_img,
                        person_name=input_person_name,
                        garment_name=input_garment_name,
                    )
                st.session_state["person_img"] = standardize_image(input_person_img)
                st.session_state["person_name"] = input_person_name
                st.session_state["garment_img"] = standardize_image(input_garment_img)
                st.session_state["garment_name"] = input_garment_name
                st.session_state["result"] = res_img
                st.session_state["baseline"] = base_img
                st.session_state["mask"] = mask_img
                st.session_state["elapsed"] = elapsed
                st.session_state["from_cache"] = from_cache
                st.session_state["pair_key"] = key

                # Lưu ngay vào lịch sử bộ sưu tập Lookbook
                add_to_history(
                    person_img=input_person_img,
                    garment_img=input_garment_img,
                    result_img=res_img,
                    baseline_img=base_img,
                    mask_img=mask_img,
                    product_id=st.session_state["product_id"],
                    prod_name=active_prod_name,
                    prod_color=active_prod_color,
                    prod_price=active_prod_price,
                    elapsed=elapsed,
                )

                st.success(f"🎉 Mặc thử thành công trong {elapsed:.2f}s! Mời bạn bấm sang tab **✨ Kết quả & Bộ sưu tập** để xem.")
            except ValueError as ve:
                st.error(str(ve))
            except RuntimeError as re_err:
                if "out of memory" in str(re_err).lower():
                    st.error("GPU tạm thời hết bộ nhớ (Out of Memory). Vui lòng thử lại sau giây lát.")
                else:
                    st.error(f"Lỗi khi chạy mô hình: {re_err}")
            except Exception as exc:
                st.error(f"Đã xảy ra lỗi: {exc}")

# ==============================================================================
# 7. TAB 3: KẾT QUẢ & BỘ SƯU TẬP (LOOKBOOK LỊCH SỬ)
# ==============================================================================
with tab_result:
    if st.session_state["result"] is None or st.session_state["person_img"] is None:
        st.info("Chưa có kết quả thử đồ. Mời bạn chọn người mẫu và áo tại tab **🧥 Buồng thử đồ AI** rồi bấm **Bắt đầu mặc thử** nhé!")
    else:
        # Thanh chuyển nhanh sản phẩm khác (giữ nguyên người mẫu)
        with st.container(border=True):
            bar_col1, bar_col2 = st.columns([3, 1])
            with bar_col1:
                if not catalog_df.empty:
                    pids = catalog_df["product_id"].tolist()
                    cur_idx = pids.index(st.session_state["product_id"]) if st.session_state["product_id"] in pids else 0
                    switch_pid = st.selectbox(
                        "🔄 Đổi sang trang phục khác (giữ nguyên người mẫu hiện tại):",
                        pids,
                        index=cur_idx,
                        format_func=lambda pid: (
                            f"{pid} - {catalog_df.loc[catalog_df['product_id'] == pid, 'name'].values[0]} "
                            f"({catalog_df.loc[catalog_df['product_id'] == pid, 'color'].values[0]}) • "
                            f"{format_vnd(catalog_df.loc[catalog_df['product_id'] == pid, 'price_vnd'].values[0])}"
                        ),
                    )
                    if switch_pid != st.session_state["product_id"]:
                        st.session_state["product_id"] = switch_pid
                        new_row = catalog_df[catalog_df["product_id"] == switch_pid].iloc[0]
                        new_g_img = load_garment_image(new_row)
                        new_g_name = str(new_row["image_file"])
                        try:
                            with st.spinner("Đang đổi trang phục cho người mẫu…"):
                                res_img, base_img, mask_img, elapsed, from_cache, key = get_result(
                                    st.session_state["person_img"],
                                    new_g_img,
                                    person_name=st.session_state["person_name"],
                                    garment_name=new_g_name,
                                )
                            st.session_state["garment_img"] = standardize_image(new_g_img)
                            st.session_state["garment_name"] = new_g_name
                            st.session_state["result"] = res_img
                            st.session_state["baseline"] = base_img
                            st.session_state["mask"] = mask_img
                            st.session_state["elapsed"] = elapsed
                            st.session_state["from_cache"] = from_cache
                            st.session_state["pair_key"] = key

                            # Tự động lưu bộ trang phục mới vào lịch sử
                            add_to_history(
                                person_img=st.session_state["person_img"],
                                garment_img=new_g_img,
                                result_img=res_img,
                                baseline_img=base_img,
                                mask_img=mask_img,
                                product_id=switch_pid,
                                prod_name=str(new_row["name"]),
                                prod_color=str(new_row["color"]),
                                prod_price=format_vnd(new_row.get("price_vnd", 189000)),
                                elapsed=elapsed,
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Không thể đổi áo: {exc}")

            with bar_col2:
                cache_text = "⚡ Đọc từ Cache" if st.session_state.get("from_cache") else "🤖 AI CatVTON"
                st.metric("⏱️ Thời gian xử lý", f"{st.session_state['elapsed']:.2f} s", delta=cache_text)

        st.write("")
        # Công tắc bật chế độ đối chứng kỹ thuật (Mặc định tắt cho người dùng)
        show_baseline = st.toggle("🔬 Chế độ so sánh kỹ thuật (Hiện ảnh Baseline đối chứng cho Giảng viên)", value=False)

        if show_baseline:
            c1, c2, c3 = st.columns(3, gap="medium")
            with c1:
                with st.container(border=True):
                    st.markdown("##### 🧍 Ảnh ban đầu")
                    st.image(st.session_state["person_img"], use_container_width=True)
            with c2:
                with st.container(border=True):
                    st.markdown("##### 📐 Mô hình Baseline (Dán áo)")
                    st.image(st.session_state["baseline"], use_container_width=True)
            with c3:
                with st.container(border=True):
                    st.markdown("##### ✨ Mô hình AI CatVTON")
                    st.image(st.session_state["result"], use_container_width=True)
        else:
            # Giao diện Before / After tiêu chuẩn thời trang
            c1, c2 = st.columns(2, gap="large")
            with c1:
                with st.container(border=True):
                    st.markdown("##### 🧍 Ảnh người mẫu ban đầu")
                    st.image(st.session_state["person_img"], use_container_width=True)
            with c2:
                with st.container(border=True):
                    st.markdown("##### ✨ Kết quả mặc thử hoàn chỉnh (AI Studio)")
                    st.image(st.session_state["result"], use_container_width=True)

        # Nút tải ảnh và thông tin
        dl_col1, dl_col2 = st.columns([2, 2])
        with dl_col1:
            st.download_button(
                "⬇️ Tải ảnh trang phục này (HD 768x1024)",
                data=image_to_bytes(st.session_state["result"]),
                file_name=f"fitroom_tryon_{st.session_state.get('product_id', 'look')}.png",
                mime="image/png",
                type="primary",
                use_container_width=True,
            )
        with dl_col2:
            with st.expander("🔍 Xem mask vùng áo thay thế (Kỹ thuật)"):
                st.image(st.session_state["mask"], caption="Vùng màu trắng là vị trí AI bóc tách và vẽ trang phục", width=260)

        # ======================================================================
        # 8. BỘ SƯU TẬP OUTFIT ĐÃ THỬ (LOOKBOOK LỊCH SỬ)
        # ======================================================================
        st.write("")
        st.markdown("---")
        hist_header_col1, hist_header_col2 = st.columns([3, 1])
        with hist_header_col1:
            st.markdown(f"### 👗 Bộ sưu tập Outfit đã thử ({len(st.session_state['tryon_history'])} mẫu)")
            st.caption("Các trang phục bạn từng thử trong phiên này được lưu tự động tại đây để bạn dễ dàng so sánh và chọn lựa.")
        with hist_header_col2:
            if st.session_state["tryon_history"]:
                if st.button("🗑️ Xóa bộ sưu tập", use_container_width=True):
                    st.session_state["tryon_history"] = []
                    st.toast("Đã làm mới bộ sưu tập.")
                    st.rerun()

        if not st.session_state["tryon_history"]:
            st.info("Chưa có trang phục nào trong bộ sưu tập. Hãy thử thêm vài bộ áo khác nhau để ngắm nhìn toàn bộ lookbook của bạn tại đây!")
        else:
            # Hiển thị lưới các outfit đã thử (tối đa 4 cột)
            h_cols = st.columns(4, gap="medium")
            for h_idx, item in enumerate(reversed(st.session_state["tryon_history"])):
                with h_cols[h_idx % 4]:
                    with st.container(border=True):
                        st.image(item["result_img"], use_container_width=True)
                        st.markdown(
                            f"""
                            <div class="lookbook-title" title="{item['garment_title']}">{item['garment_title']}</div>
                            <div style="font-size:0.9rem; font-weight:700; color:#4f46e5;">{item['garment_price']}</div>
                            <div class="lookbook-time">🕒 Thử lúc {item['timestamp']}</div>
                            """,
                            unsafe_allow_html=True,
                        )
                        btn_load_col, btn_dl_col = st.columns([1, 1])
                        with btn_load_col:
                            if st.button("👁️ Xem", key=f"btn_view_{item['id']}", use_container_width=True):
                                st.session_state["result"] = item["result_img"]
                                st.session_state["garment_img"] = item["garment_img"]
                                st.session_state["baseline"] = item["baseline_img"]
                                st.session_state["mask"] = item["mask_img"]
                                st.session_state["product_id"] = item["product_id"]
                                st.session_state["elapsed"] = item["elapsed"]
                                st.rerun()
                        with btn_dl_col:
                            st.download_button(
                                "⬇️",
                                data=image_to_bytes(item["result_img"]),
                                file_name=f"fitroom_{item['id']}.png",
                                mime="image/png",
                                key=f"dl_{item['id']}",
                                use_container_width=True,
                            )


# ==============================================================================
# 9. SIDEBAR: BÁO CÁO ĐÁNH GIÁ MÔ HÌNH (DÀNH CHO GIẢNG VIÊN KHI THUYẾT TRÌNH)
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Thông tin hệ thống")
    st.caption("AI Virtual Fitting Room – Trí tuệ nhân tạo trong Kinh doanh")
    st.write(f"- **Mô hình:** CatVTON Diffusion ({config.IMG_W}x{config.IMG_H})")
    st.write(f"- **Chế độ:** {'Giả lập (Mock)' if config.use_mock() else 'Mô hình thật (GPU T4)'}")
    st.write(f"- **Sản phẩm:** {len(catalog_df)} sản phẩm catalog")
    st.write(f"- **Người mẫu có sẵn:** {len(demo_persons)} mẫu")

    st.markdown("---")
    # Expander đánh giá kỹ thuật học thuật cho giáo viên
    with st.expander("📊 Chỉ số đánh giá học thuật (PV3 - Báo cáo)"):
        st.markdown("##### Bảng chỉ số đánh giá kỹ thuật")
        summary_csv = config.RESULTS / "results_summary.csv"
        if summary_csv.exists():
            try:
                sdf = pd.read_csv(summary_csv)
                st.dataframe(sdf, hide_index=True, use_container_width=True)
            except Exception as e:
                st.caption(f"Lỗi đọc file: {e}")
        else:
            st.caption("Chưa có results_summary.csv")

        figs_dir = config.RESULTS / "figs"
        if figs_dir.exists():
            st.markdown("##### Biểu đồ so sánh định lượng")
            for fig_p in sorted(figs_dir.glob("*.png"))[:4]:
                st.image(str(fig_p), caption=fig_p.name, use_container_width=True)
