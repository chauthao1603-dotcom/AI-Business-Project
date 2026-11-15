from __future__ import annotations

import hashlib
import io
import os
import time

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

if "USE_MOCK" not in os.environ:
    os.environ["USE_MOCK"] = "1"

from src import config
from src.baseline import run_baseline
from src.config import get_run_tryon
from src.utils import (
    pair_id,
    standardize_image,
    standardize_mask,
    validate_input_image,
)

st.set_page_config(
    page_title="Virtual Try-On Studio",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------- TÙY BIẾN CSS: EDITORIAL LUXURY FASHION -----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"], * {
        font-family: 'Montserrat', sans-serif !important;
        border-radius: 0px !important;
    }

    /* Ẩn hoàn toàn Sidebar và nút mở Sidebar */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Giữ nguyên font icon của Streamlit */
    [data-testid="stIconMaterial"], span[class*="material-symbols"], span[class*="material-icons"] {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
    }

    .stApp {
        background-color: #FAF9F6;
        color: #18181B;
    }

    header[data-testid="stHeader"] {
        background: transparent;
        border-top: 3px solid #C5A880;
        height: 2rem !important;
    }

    .block-container, [data-testid="stMainBlockContainer"] {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1280px;
    }

    /* ---------- TABS ---------- */
    [data-testid="stTabs"] [role="tablist"] {
        display: flex !important;
        width: 100% !important;
        gap: 1.5rem !important;
        justify-content: space-evenly !important;
        border-bottom: 1px solid #E4E0D8;
    }
    [data-testid="stTabs"] button[role="tab"] {
        flex: 1 1 auto !important;
        justify-content: center !important;
        height: 48px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        background: transparent !important;
        color: #71717A;
        border: none !important;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #18181B !important;
        border-bottom: 2px solid #C5A880 !important;
    }
    [data-testid="stTabs"] button[role="tab"]:hover {
        color: #C5A880 !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: #C5A880 !important;
    }

    /* ---------- TOAST ---------- */
    [data-testid="stToast"] {
        background: #F4EFEA !important;
        border: 1px solid #EAE6DF !important;
        border-left: 3px solid #C5A880 !important;
        color: #18181B !important;
    }

    /* ---------- BUTTONS ---------- */
    .stButton > button {
        background: #18181B !important;
        color: #FAF9F6 !important;
        border: 1px solid #18181B !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button:hover {
        background: #27272A !important;
        border-color: #C5A880 !important;
        color: #C5A880 !important;
    }
    .stButton > button[kind="secondary"] {
        background: #FFFFFF !important;
        color: #18181B !important;
        border: 1px solid #D4D4D8 !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #F4EFEA !important;
        border-color: #18181B !important;
    }

    /* ---------- BADGES & HEADERS ---------- */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background: #F4EFEA;
        color: #78716C;
        margin-bottom: 6px;
    }
    .badge-gold {
        background: #F8F5EE;
        color: #9A7B4F;
        border: 1px solid #EADBCA;
    }
    .badge-dark {
        background: #18181B;
        color: #FAF9F6;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        color: #18181B;
        margin: 0.8rem 0 0.6rem 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .step-num {
        background: #C5A880;
        color: #FFFFFF;
        font-size: 0.75rem;
        padding: 2px 7px;
        font-weight: 800;
    }

    /* ---------- CARDS & INPUTS ---------- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border: 1px solid #EAE6DF !important;
        padding: 0.75rem;
    }
    [data-testid="stFileUploader"], [data-testid="stCameraInput"] {
        border: 1px dashed #D4CEBF !important;
        background-color: #FFFFFF;
        padding: 0.8rem;
    }
    [data-testid="stRadio"] label p { font-size: 0.82rem; font-weight: 600; }
    [data-testid="stCameraInput"] video,
    [data-testid="stCameraInput"] img { max-height: 320px; object-fit: contain; }
    .status-chip {
        display: inline-block; padding: 4px 10px; font-size: 0.7rem; font-weight: 700;
        letter-spacing: 0.08em; text-transform: uppercase; border: 1px solid #EAE6DF;
    }
    .status-ok  { background: #F8F5EE; color: #9A7B4F; border-color: #C5A880; }
    .status-off { background: #F4EFEA; color: #A1A1AA; }
    .ph-box {
        height: 280px; border: 1px dashed #D4CEBF; background: #FAF9F6; display: flex;
        align-items: center; justify-content: center; color: #A1A1AA; font-size: 0.78rem;
        font-weight: 600; letter-spacing: 0.06em; text-align: center; padding: 0 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------- HÀM TIỆN ÍCH, BASELINE & CACHE -----------------
@st.cache_resource
def load_cached_tryon_fn():
    return get_run_tryon()


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


def get_image_hash(img: Image.Image) -> str:
    with io.BytesIO() as buf:
        standardize_image(img).save(buf, format="PNG")
        return hashlib.sha1(buf.getvalue()).hexdigest()[:12]


def compute_cache_key(
    person_img: Image.Image,
    garment_img: Image.Image,
    person_name: str | None = None,
    garment_name: str | None = None,
) -> str:
    if person_name and garment_name:
        return pair_id(person_name, garment_name)
    return f"{get_image_hash(person_img)}__{get_image_hash(garment_img)}"


def get_result(
    person_img: Image.Image,
    garment_img: Image.Image,
    person_name: str | None = None,
    garment_name: str | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image, float, bool, str]:
    validate_input_image(person_img, "ảnh người")
    validate_input_image(garment_img, "ảnh áo")

    cache_key = compute_cache_key(person_img, garment_img, person_name, garment_name)
    config.CACHE.mkdir(parents=True, exist_ok=True)
    disk_img_path = config.CACHE / f"{cache_key}.png"
    disk_mask_path = config.CACHE / f"{cache_key}_mask.png"

    start_t = time.perf_counter()
    from_cache = False

    if cache_key in st.session_state["tryon_history"]:
        res_img, base_img, mask_img, _ = st.session_state["tryon_history"][cache_key]
        elapsed = time.perf_counter() - start_t
        return res_img, base_img, mask_img, elapsed, True, cache_key

    if disk_img_path.exists() and disk_mask_path.exists():
        with Image.open(disk_img_path) as r_im:
            res_img = standardize_image(r_im.copy())
        with Image.open(disk_mask_path) as m_im:
            mask_img = standardize_mask(m_im.copy())
        from_cache = True
    else:
        run_fn = load_cached_tryon_fn()
        res_img, mask_img = run_fn(person_img, garment_img, mask=None)
        res_img.save(disk_img_path, format="PNG")
        mask_img.save(disk_mask_path, format="PNG")

    base_img = safe_run_baseline(person_img, garment_img, mask_img)
    elapsed = time.perf_counter() - start_t
    st.session_state["tryon_history"][cache_key] = (res_img, base_img, mask_img, elapsed)
    return res_img, base_img, mask_img, elapsed, from_cache, cache_key


def load_catalog_data() -> pd.DataFrame | None:
    if config.CATALOG.exists():
        return pd.read_csv(config.CATALOG)
    elif config.CATALOG_SAMPLE.exists():
        return pd.read_csv(config.CATALOG_SAMPLE)
    return None


def make_placeholder_garment(color_name: str) -> Image.Image:
    color_map = {
        "trắng": ((248, 248, 246), (200, 195, 185)),
        "xanh": ((135, 175, 215), (75, 115, 155)),
        "be": ((225, 210, 188), (175, 150, 120)),
        "đen": ((45, 45, 48), (20, 20, 22)),
        "đỏ": ((195, 70, 70), (140, 40, 40)),
    }
    fill_rgb, border_rgb = color_map.get(str(color_name).strip().lower(), ((210, 205, 195), (140, 130, 115)))
    w, h = config.IMG_SIZE
    img = Image.new("RGB", (w, h), (244, 239, 234))
    draw = ImageDraw.Draw(img)

    left_sleeve = [(int(w * 0.26), int(h * 0.24)), (int(w * 0.11), int(h * 0.42)), (int(w * 0.21), int(h * 0.49)), (int(w * 0.28), int(h * 0.37))]
    right_sleeve = [(int(w * 0.74), int(h * 0.24)), (int(w * 0.89), int(h * 0.42)), (int(w * 0.79), int(h * 0.49)), (int(w * 0.72), int(h * 0.37))]
    draw.polygon(left_sleeve, fill=fill_rgb, outline=border_rgb, width=3)
    draw.polygon(right_sleeve, fill=fill_rgb, outline=border_rgb, width=3)
    draw.rectangle([int(w * 0.25), int(h * 0.22), int(w * 0.75), int(h * 0.78)], fill=fill_rgb, outline=border_rgb, width=3)
    draw.pieslice([int(w * 0.39), int(h * 0.17), int(w * 0.61), int(h * 0.28)], start=0, end=180, fill=(244, 239, 234), outline=border_rgb, width=3)
    return img


def load_garment_from_row(row: pd.Series) -> Image.Image:
    img_name = str(row.get("image_file", ""))
    garment_path = config.GARMENTS / img_name
    if garment_path.exists():
        return _load_rgb(garment_path)
    return make_placeholder_garment(str(row.get("color", "")))


def _load_rgb(file_or_path, label: str = "ảnh") -> Image.Image:
    if hasattr(file_or_path, "size") and file_or_path.size > 15 * 1024 * 1024:
        raise ValueError(f"{label.capitalize()} quá lớn (> 15MB). Vui lòng chọn ảnh nhẹ hơn.")
    try:
        img = Image.open(file_or_path)
        img.load()
        img = ImageOps.exif_transpose(img)
        rgb = img.convert("RGB")
    except UnidentifiedImageError:
        raise ValueError(f"{label.capitalize()} không đúng định dạng hợp lệ (chỉ hỗ trợ JPG, PNG).")
    except Exception as exc:
        raise ValueError(f"Không thể đọc {label}: {exc}")

    validate_input_image(rgb, label)
    return rgb


PANEL_HEIGHT = 540


def panel_header(tag: str, title: str, ready: bool):
    chip = (
        '<span class="status-chip status-ok">ĐÃ SẴN SÀNG</span>'
        if ready
        else '<span class="status-chip status-off">CHƯA CÓ ẢNH</span>'
    )
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-bottom:0.6rem;"><div><span class="badge badge-dark">{tag}</span>'
        f'<span style="margin-left:8px;font-weight:700;font-size:0.9rem;">{title}</span></div>{chip}</div>',
        unsafe_allow_html=True,
    )


# ----------------- KHỞI TẠO SESSION STATE -----------------
for k, v in {
    "selected_product_id": None,
    "product_id": None,
    "person_img": None,
    "person_name": None,
    "garment_img": None,
    "garment_name": None,
    "tryon_history": {},
    "latest_result": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ----------------- HEADER TỐI GIẢN -----------------
st.markdown(
    """
    <div style="background: #18181B; padding: 0.95rem 1.8rem; margin-bottom: 0.8rem; border-left: 5px solid #C5A880; display: flex; align-items: center; gap: 14px;">
        <span class="badge badge-gold" style="margin: 0;">AI FASHION</span>
        <h1 style="color: #FAF9F6; font-size: 1.45rem; font-weight: 800; margin: 0; padding: 0; letter-spacing: 0.04em;">
            VIRTUAL TRY-ON STUDIO
        </h1>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_shop, tab_tryon, tab_result, tab_eval = st.tabs([
    "01. CỬA HÀNG",
    "02. THỬ ĐỒ",
    "03. KẾT QUẢ",
    "04. ĐÁNH GIÁ",
])

df_cat = load_catalog_data()


# =====================================================================
# TAB 1: CỬA HÀNG
# =====================================================================
with tab_shop:
    st.markdown('<div class="section-title"><span class="step-num">01</span> DANH MỤC TRANG PHỤC</div>', unsafe_allow_html=True)

    if df_cat is not None and not df_cat.empty:
        cols = st.columns(4)
        for idx, row in df_cat.iterrows():
            with cols[idx % 4]:
                with st.container(border=True):
                    category_name = row.get("category", "Áo")
                    st.markdown(f'<span class="badge">{category_name}</span>', unsafe_allow_html=True)

                    img_name = str(row.get("image_file", ""))
                    garment_path = config.GARMENTS / img_name
                    if garment_path.exists():
                        st.image(str(garment_path), use_container_width=True)
                    else:
                        st.image(load_garment_from_row(row), use_container_width=True)

                    st.markdown(
                        f"<div style='font-weight: 700; margin-top: 8px; font-size: 0.95rem; color: #18181B;'>{row.get('name', 'Sản phẩm')}</div>"
                        f"<div style='font-size: 0.8rem; color: #71717A; margin-bottom: 10px;'>Màu sắc: <b>{row.get('color', 'Tiêu chuẩn')}</b></div>",
                        unsafe_allow_html=True,
                    )

                    if st.button("CHỌN MẶC THỬ", key=f"shop_btn_{row.get('product_id', idx)}", use_container_width=True):
                        pid = row.get("product_id")
                        st.session_state["selected_product_id"] = pid
                        st.session_state["product_id"] = pid
                        st.session_state["garment_img"] = load_garment_from_row(row)
                        st.session_state["garment_name"] = img_name
                        st.toast(f"Đã chọn: {row.get('name')}")
    else:
        st.warning("Chưa có dữ liệu sản phẩm.")


# =====================================================================
# TAB 2: THỬ ĐỒ
# =====================================================================
with tab_tryon:
    st.markdown('<div class="section-title"><span class="step-num">02</span> CHỌN ẢNH THỬ ĐỒ</div>', unsafe_allow_html=True)

    col_person, col_garment = st.columns(2, gap="large")

    # ------------------------- DỮ LIỆU A: NGƯỜI -------------------------
    with col_person:
        with st.container(border=True, height=PANEL_HEIGHT):
            person_slot = st.empty()

            person_mode = st.radio(
                "Nguồn ảnh người",
                ["Tải file", "Mẫu có sẵn", "Camera"],
                horizontal=True,
                label_visibility="collapsed",
                key="person_mode",
            )

            p_img = None
            p_name = None
            if person_mode == "Camera":
                cam = st.camera_input("Đứng thẳng, thấy rõ nửa thân trên", key="cam_person")
                if cam is not None:
                    try:
                        p_img = _load_rgb(cam, "ảnh người")
                    except ValueError as err:
                        st.error(str(err))

            elif person_mode == "Mẫu có sẵn":
                demo_dir = config.DEMO_PERSONS
                demo_files = (
                    sorted(demo_dir.glob("*.jpg")) + sorted(demo_dir.glob("*.png"))
                ) if demo_dir.exists() else []
                if demo_files:
                    chosen = st.selectbox("Chọn người mẫu", [f.name for f in demo_files], key="demo_person")
                    try:
                        p_img = _load_rgb(demo_dir / chosen, "ảnh người mẫu")
                        p_name = chosen
                        st.image(p_img, width=240)
                    except ValueError as err:
                        st.error(str(err))
                else:
                    st.markdown('<div class="ph-box">CHƯA CÓ ẢNH MẪU</div>', unsafe_allow_html=True)

            else:
                up = st.file_uploader("Ảnh người (JPG, PNG)", type=["jpg", "jpeg", "png"], key="upload_person")
                if up is not None:
                    try:
                        p_img = _load_rgb(up, "ảnh người")
                        st.image(p_img, width=240)
                    except ValueError as err:
                        st.error(str(err))
                else:
                    st.markdown('<div class="ph-box">KÉO THẢ ẢNH NGƯỜI VÀO Ô TRÊN</div>', unsafe_allow_html=True)

            st.session_state["person_img"] = p_img
            st.session_state["person_name"] = p_name
            with person_slot.container():
                panel_header("BƯỚC 1", "ẢNH NGƯỜI MẪU", p_img is not None)

    # ------------------------- DỮ LIỆU B: ÁO ----------------------------
    with col_garment:
        with st.container(border=True, height=PANEL_HEIGHT):
            garment_slot = st.empty()

            garment_mode = st.radio(
                "Nguồn ảnh áo",
                ["Từ cửa hàng", "Tải file"],
                horizontal=True,
                label_visibility="collapsed",
                key="garment_mode",
            )

            g_img = None
            g_name = None
            if garment_mode == "Từ cửa hàng":
                if df_cat is not None and not df_cat.empty:
                    pids = df_cat["product_id"].tolist()
                    cur_pid = st.session_state.get("selected_product_id")
                    def_idx = pids.index(cur_pid) if cur_pid in pids else 0
                    chosen_pid = st.selectbox(
                        "Chọn áo từ cửa hàng",
                        pids,
                        index=def_idx,
                        format_func=lambda pid: f"{df_cat.loc[df_cat['product_id'] == pid, 'name'].values[0]} ({df_cat.loc[df_cat['product_id'] == pid, 'color'].values[0]})",
                    )
                    st.session_state["selected_product_id"] = chosen_pid
                    st.session_state["product_id"] = chosen_pid
                    r = df_cat[df_cat["product_id"] == chosen_pid].iloc[0]
                    g_img = load_garment_from_row(r)
                    g_name = str(r.get("image_file", ""))
                    st.image(g_img, width=240)
                else:
                    st.markdown('<div class="ph-box">CHƯA CÓ SẢN PHẨM</div>', unsafe_allow_html=True)
            else:
                up_g = st.file_uploader("Ảnh áo (JPG, PNG)", type=["jpg", "jpeg", "png"], key="upload_garment")
                if up_g is not None:
                    try:
                        g_img = _load_rgb(up_g, "ảnh áo")
                        st.image(g_img, width=240)
                    except ValueError as err:
                        st.error(str(err))
                else:
                    st.markdown('<div class="ph-box">KÉO THẢ ẢNH ÁO VÀO Ô TRÊN</div>', unsafe_allow_html=True)

            st.session_state["garment_img"] = g_img
            st.session_state["garment_name"] = g_name
            with garment_slot.container():
                panel_header("BƯỚC 2", "ẢNH TRANG PHỤC", g_img is not None)

    # --------------------------- NÚT CHẠY --------------------------------
    st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)
    ready = st.session_state["person_img"] is not None and st.session_state["garment_img"] is not None
    _, col_run, _ = st.columns([2, 1, 2])
    with col_run:
        btn_run = st.button("THỬ ĐỒ NGAY", use_container_width=True, type="primary")

    if btn_run:
        if not ready:
            st.error("Vui lòng chọn đầy đủ cả ảnh người và ảnh áo trước khi bấm Thử đồ!")
        else:
            p_img = st.session_state["person_img"]
            g_img = st.session_state["garment_img"]
            try:
                with st.spinner("Đang sinh ảnh, vui lòng chờ…"):
                    res_img, base_img, mask_img, elapsed, from_cache, cache_key = get_result(
                        p_img,
                        g_img,
                        person_name=st.session_state.get("person_name"),
                        garment_name=st.session_state.get("garment_name"),
                    )

                st.session_state["latest_result"] = {
                    "person": standardize_image(p_img),
                    "person_name": st.session_state.get("person_name"),
                    "garment": standardize_image(g_img),
                    "garment_name": st.session_state.get("garment_name"),
                    "result": res_img,
                    "baseline": base_img,
                    "mask": mask_img,
                    "elapsed": elapsed,
                    "from_cache": from_cache,
                    "key": cache_key,
                }
                st.success("Hoàn tất! Chuyển sang tab '03. KẾT QUẢ' để xem.")
            except ValueError as err:
                st.error(str(err))
            except RuntimeError as rt_err:
                if "out of memory" in str(rt_err).lower():
                    st.error("GPU đã hết bộ nhớ. Vui lòng thử lại sau.")
                else:
                    st.error(f"Lỗi thực thi mô hình: {rt_err}")
            except Exception as exc:
                st.error(f"Đã xảy ra lỗi: {exc}")


# =====================================================================
# TAB 3: KẾT QUẢ
# =====================================================================
with tab_result:
    st.markdown('<div class="section-title"><span class="step-num">03</span> KẾT QUẢ THỬ ĐỒ</div>', unsafe_allow_html=True)

    res_data = st.session_state.get("latest_result")

    if res_data:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                """
                <div style="background: #FFFFFF; border: 1px solid #EAE6DF; padding: 10px; margin-bottom: 10px;">
                    <span class="badge">GỐC</span>
                    <strong style="margin-left: 5px; font-size: 0.85rem;">ẢNH NGƯỜI MẪU</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(res_data["person"], use_container_width=True)

        with c2:
            st.markdown(
                """
                <div style="background: #FFFFFF; border: 1px solid #EAE6DF; padding: 10px; margin-bottom: 10px;">
                    <span class="badge" style="background: #F8F5EE; color: #9A7B4F;">BASELINE</span>
                    <strong style="margin-left: 5px; font-size: 0.85rem;">PHƯƠNG PHÁP CƠ BẢN</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(res_data["baseline"], use_container_width=True)

        with c3:
            st.markdown(
                """
                <div style="background: #FFFFFF; border: 1px solid #C5A880; padding: 10px; margin-bottom: 10px;">
                    <span class="badge badge-gold">AI TRY-ON</span>
                    <strong style="margin-left: 5px; font-size: 0.85rem; color: #C5A880;">CATVTON</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(res_data["result"], use_container_width=True)

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        col_mask, col_info = st.columns(2)
        with col_mask:
            with st.expander("Xem mask vùng áo"):
                st.image(res_data["mask"], width=280)

        with col_info:
            cur_pid = st.session_state.get("selected_product_id")
            if cur_pid and df_cat is not None and not df_cat.empty:
                m_row = df_cat[df_cat["product_id"] == cur_pid]
                if not m_row.empty:
                    pr = m_row.iloc[0]
                    st.markdown(
                        f"""
                        <div style="background:#FFFFFF;border:1px solid #EAE6DF;padding:1rem;">
                            <span class="badge badge-gold">THÔNG TIN SẢN PHẨM</span>
                            <div style="font-weight:700;font-size:0.95rem;margin-top:4px;">{pr.get('name')}</div>
                            <div style="font-size:0.82rem;color:#71717A;margin-top:4px;">
                                Danh mục: <b>{pr.get('category')}</b> · Màu sắc: <b>{pr.get('color')}</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        if df_cat is not None and not df_cat.empty:
            st.markdown("---")
            options = {f"{row['name']} ({row['color']})": row["product_id"] for _, row in df_cat.iterrows()}
            selected_name = st.selectbox("Thử áo khác", list(options.keys()))
            if st.button("ĐỔI ÁO", type="secondary"):
                new_pid = options[selected_name]
                st.session_state["selected_product_id"] = new_pid
                st.session_state["product_id"] = new_pid
                new_row = df_cat[df_cat["product_id"] == new_pid].iloc[0]
                new_g_img = load_garment_from_row(new_row)
                new_g_name = str(new_row.get("image_file", ""))
                try:
                    with st.spinner("Đang sinh ảnh, vui lòng chờ…"):
                        res_img, base_img, mask_img, elapsed, from_cache, cache_key = get_result(
                            res_data["person"],
                            new_g_img,
                            person_name=res_data.get("person_name"),
                            garment_name=new_g_name,
                        )
                    st.session_state["garment_img"] = new_g_img
                    st.session_state["garment_name"] = new_g_name
                    st.session_state["latest_result"] = {
                        "person": res_data["person"],
                        "person_name": res_data.get("person_name"),
                        "garment": standardize_image(new_g_img),
                        "garment_name": new_g_name,
                        "result": res_img,
                        "baseline": base_img,
                        "mask": mask_img,
                        "elapsed": elapsed,
                        "from_cache": from_cache,
                        "key": cache_key,
                    }
                    st.rerun()
                except Exception as exc:
                    st.error(f"Không thể đổi áo: {exc}")
    else:
        st.info("Chưa có kết quả. Vui lòng chọn ảnh và bấm 'THỬ ĐỒ NGAY' tại tab '02. THỬ ĐỒ'.")


# =====================================================================
# TAB 4: ĐÁNH GIÁ
# =====================================================================
with tab_eval:
    st.markdown('<div class="section-title"><span class="step-num">04</span> KẾT QUẢ ĐÁNH GIÁ</div>', unsafe_allow_html=True)

    summary_file = config.RESULTS / "results_summary.csv"
    if summary_file.exists():
        df_summary = pd.read_csv(summary_file)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)
    else:
        st.info("Chưa có kết quả đánh giá")

    figs_dir = config.RESULTS / "figs"
    if figs_dir.exists() and any(figs_dir.iterdir()):
        cols_fig = st.columns(2)
        for idx_f, fig_path in enumerate(sorted(figs_dir.glob("*.png"))):
            with cols_fig[idx_f % 2]:
                st.image(str(fig_path), caption=fig_path.stem, use_container_width=True)