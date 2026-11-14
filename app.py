"""PV4 – Giao diện Streamlit cho dự án Virtual Try-On (Mục 4.1 -> 4.8 trong docs/PHAN_VIEC_v2.md)."""
from __future__ import annotations

import hashlib
import io
import time

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
# 4.1 KHUNG APP & GIAO DIỆN HIỆN ĐẠI
# ==============================================================================
st.set_page_config(
    page_title="AI Virtual Try-On | Phòng Thử Đồ Ảo",
    page_icon="✨",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }
    .hero-banner {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: #ffffff;
        padding: 1.6rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff;
    }
    .hero-banner p {
        margin: 0.35rem 0 0 0;
        font-size: 0.98rem;
        color: #cbd5e1;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        padding: 0 22px;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
        font-size: 1rem;
    }
    .prod-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0f172a;
        margin: 0.4rem 0 0.35rem 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .prod-meta {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.75rem;
        font-size: 0.85rem;
        color: #475569;
    }
    .badge {
        background: #f1f5f9;
        color: #334155;
        padding: 2px 10px;
        border-radius: 999px;
        font-weight: 500;
    }
    </style>

    <div class="hero-banner">
        <h1>✨ Phòng Thử Đồ Thời Trang AI</h1>
        <p>Trải nghiệm mặc thử trang phục trực tuyến nhanh chóng, trực quan chỉ với một cú nhấp chuột</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 4.4 CACHE MÔ HÌNH & CACHE KẾT QUẢ THEO CẶP
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


# ==============================================================================
# HÀM TIỆN ÍCH ĐỌC DỮ LIỆU & MINH HỌA ÁO
# ==============================================================================
def load_catalog() -> pd.DataFrame:
    for cat_path in [config.CATALOG, config.REPO_ROOT / "data" / "catalog.csv", config.CATALOG_SAMPLE]:
        if cat_path.exists():
            return pd.read_csv(cat_path)
    return pd.DataFrame(columns=["product_id", "name", "category", "color", "image_file"])


def make_placeholder_garment(color_name: str) -> Image.Image:
    """Vẽ hình chiếc áo thời trang đẹp mắt khi chưa tải thư mục ảnh thật từ Drive."""
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

    left_sleeve = [(int(w * 0.26), int(h * 0.24)), (int(w * 0.11), int(h * 0.42)), (int(w * 0.21), int(h * 0.49)), (int(w * 0.28), int(h * 0.37))]
    right_sleeve = [(int(w * 0.74), int(h * 0.24)), (int(w * 0.89), int(h * 0.42)), (int(w * 0.79), int(h * 0.49)), (int(w * 0.72), int(h * 0.37))]
    draw.polygon(left_sleeve, fill=fill_rgb, outline=border_rgb, width=4)
    draw.polygon(right_sleeve, fill=fill_rgb, outline=border_rgb, width=4)

    draw.rounded_rectangle(
        [int(w * 0.25), int(h * 0.22), int(w * 0.75), int(h * 0.78)],
        radius=24,
        fill=fill_rgb,
        outline=border_rgb,
        width=4,
    )

    draw.pieslice(
        [int(w * 0.39), int(h * 0.17), int(w * 0.61), int(h * 0.28)],
        start=0,
        end=180,
        fill=(255, 255, 255),
        outline=border_rgb,
        width=4,
    )
    return img


def load_garment_image(row: pd.Series) -> Image.Image:
    for folder in [config.GARMENTS, config.REPO_ROOT / "data" / "garments", config.RAW / "cloth"]:
        img_path = folder / str(row["image_file"])
        if img_path.exists():
            with Image.open(img_path) as im:
                return im.convert("RGB").copy()
    return make_placeholder_garment(str(row["color"]))


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
# KHỞI TẠO SESSION STATE
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
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

catalog_df = load_catalog()

# 4.1 Bốn tab chính
tab_store, tab_tryon, tab_result, tab_eval = st.tabs(
    ["🛍️ Cửa hàng", "🧥 Thử đồ", "✨ Kết quả", "📊 Đánh giá"]
)

# ==============================================================================
# 4.2 TAB CỬA HÀNG
# ==============================================================================
with tab_store:
    if catalog_df.empty:
        st.warning("Hiện chưa có sản phẩm nào trong cửa hàng.")
    else:
        f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
        with f_col1:
            search_kw = st.text_input("🔍 Tìm kiếm sản phẩm", placeholder="Nhập tên áo...")
        with f_col2:
            cats = ["Tất cả"] + sorted(catalog_df["category"].dropna().unique().tolist())
            sel_cat = st.selectbox("Danh mục", cats)
        with f_col3:
            colors = ["Tất cả"] + sorted(catalog_df["color"].dropna().unique().tolist())
            sel_col = st.selectbox("Màu sắc", colors)

        filtered = catalog_df.copy()
        if search_kw.strip():
            filtered = filtered[filtered["name"].str.contains(search_kw.strip(), case=False, na=False)]
        if sel_cat != "Tất cả":
            filtered = filtered[filtered["category"] == sel_cat]
        if sel_col != "Tất cả":
            filtered = filtered[filtered["color"] == sel_col]

        st.write("")
        cols = st.columns(4)
        for idx, (_, row) in enumerate(filtered.iterrows()):
            with cols[idx % 4]:
                with st.container(border=True):
                    g_img = load_garment_image(row)
                    st.image(g_img, use_container_width=True)

                    st.markdown(
                        f"""
                        <div class="prod-title" title="{row['name']}">{row['name']}</div>
                        <div class="prod-meta">
                            <span class="badge">{row['category']}</span>
                            <span>• Màu {row['color']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    is_chosen = st.session_state["product_id"] == row["product_id"]
                    btn_text = "✅ Đang chọn thử" if is_chosen else "Chọn thử ngay"
                    if st.button(
                        btn_text,
                        key=f"btn_store_{row['product_id']}",
                        type="primary" if is_chosen else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state["product_id"] = row["product_id"]
                        st.toast(f"Đã chọn '{row['name']}'! Hãy chuyển sang tab '🧥 Thử đồ'.")
                        st.rerun()

# ==============================================================================
# 4.3 TAB THỬ ĐỒ
# ==============================================================================
with tab_tryon:
    col_p, col_g = st.columns(2, gap="large")

    input_person_img: Image.Image | None = None
    input_person_name: str | None = None
    input_garment_img: Image.Image | None = None
    input_garment_name: str | None = None

    with col_p:
        with st.container(border=True):
            st.markdown("#### 👤 Bước 1: Chọn ảnh người mẫu")
            demo_persons = (
                sorted([p for p in config.DEMO_PERSONS.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
                if config.DEMO_PERSONS.exists()
                else []
            )
            p_sources = ["Tải ảnh từ máy tính", "📸 Chụp từ Camera"] + (["Chọn người mẫu có sẵn"] if demo_persons else [])
            p_mode = st.radio("Nguồn ảnh người:", p_sources, horizontal=True, label_visibility="collapsed")

            if p_mode == "Tải ảnh từ máy tính":
                up_person = st.file_uploader("Tải lên ảnh chân dung đứng thẳng (JPG, PNG)", type=["jpg", "jpeg", "png"], key="up_person")
                if up_person is not None:
                    try:
                        input_person_img = read_uploaded_image(up_person, "ảnh người")
                        input_person_name = None
                        st.image(input_person_img, caption="Ảnh người đã tải lên", width=260)
                    except ValueError as err:
                        st.error(str(err))
            elif p_mode == "📸 Chụp từ Camera":
                cam_person = st.camera_input("Chụp ảnh chân dung của bạn (đứng thẳng nhìn vào camera)", key="cam_person")
                if cam_person is not None:
                    try:
                        input_person_img = read_uploaded_image(cam_person, "ảnh chụp từ camera")
                        input_person_name = None
                        st.image(input_person_img, caption="Ảnh chân dung vừa chụp", width=260)
                    except ValueError as err:
                        st.error(str(err))
            else:
                sel_p_path = st.selectbox("Chọn người mẫu:", demo_persons, format_func=lambda p: f"Người mẫu {p.stem}")
                try:
                    with Image.open(sel_p_path) as im:
                        input_person_img = im.convert("RGB").copy()
                    validate_input_image(input_person_img, "ảnh người")
                    input_person_name = sel_p_path.name
                    st.image(input_person_img, caption=f"Người mẫu: {sel_p_path.stem}", width=260)
                except Exception as err:
                    st.error(f"Lỗi đọc ảnh người mẫu: {err}")

    with col_g:
        with st.container(border=True):
            st.markdown("#### 👕 Bước 2: Chọn trang phục muốn thử")
            g_mode = st.radio(
                "Nguồn ảnh áo:",
                ["Chọn từ Cửa hàng", "Tải ảnh áo từ máy tính"],
                horizontal=True,
                label_visibility="collapsed",
            )
            if g_mode == "Chọn từ Cửa hàng":
                if catalog_df.empty:
                    st.warning("Chưa có sản phẩm trong cửa hàng.")
                else:
                    pids = catalog_df["product_id"].tolist()
                    default_idx = pids.index(st.session_state["product_id"]) if st.session_state["product_id"] in pids else 0
                    chosen_pid = st.selectbox(
                        "Sản phẩm trong Cửa hàng:",
                        pids,
                        index=default_idx,
                        format_func=lambda pid: (
                            f"{catalog_df.loc[catalog_df['product_id'] == pid, 'name'].values[0]} "
                            f"({catalog_df.loc[catalog_df['product_id'] == pid, 'color'].values[0]})"
                        ),
                    )
                    st.session_state["product_id"] = chosen_pid
                    row = catalog_df[catalog_df["product_id"] == chosen_pid].iloc[0]
                    input_garment_img = load_garment_image(row)
                    input_garment_name = str(row["image_file"])
                    st.image(input_garment_img, caption=f"{row['name']} – Màu {row['color']}", width=260)
            else:
                up_garment = st.file_uploader("Tải lên ảnh sản phẩm áo (JPG, PNG)", type=["jpg", "jpeg", "png"], key="up_garment")
                if up_garment is not None:
                    try:
                        input_garment_img = read_uploaded_image(up_garment, "ảnh áo")
                        input_garment_name = None
                        st.image(input_garment_img, caption="Ảnh áo đã tải lên", width=260)
                    except ValueError as err:
                        st.error(str(err))

    st.write("")
    if st.button("✨ Thử đồ ngay", type="primary", use_container_width=True):
        if input_person_img is None or input_garment_img is None:
            st.error("Vui lòng chọn hoặc tải lên đầy đủ cả ảnh người và ảnh áo trước khi bấm Thử đồ!")
        else:
            try:
                with st.spinner("Đang sinh ảnh, vui lòng chờ…"):
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

                st.success(
                    f"🎉 Đã mặc thử hoàn tất trong {elapsed:.2f} giây! "
                    "Mời bạn bấm sang tab **✨ Kết quả** để xem ảnh trước/sau và đổi thử mẫu áo khác."
                )
            except ValueError as ve:
                st.error(str(ve))
            except RuntimeError as re_err:
                if "out of memory" in str(re_err).lower():
                    st.error("GPU đã hết bộ nhớ (Out of Memory). Vui lòng thử lại sau.")
                else:
                    st.error(f"Lỗi khi chạy mô hình: {re_err}")
            except Exception as exc:
                st.error(f"Đã xảy ra lỗi: {exc}")

# ==============================================================================
# 4.5 TAB KẾT QUẢ
# ==============================================================================
with tab_result:
    if st.session_state["result"] is None or st.session_state["person_img"] is None:
        st.info("Chưa có ảnh kết quả. Hãy chọn ảnh và bấm **✨ Thử đồ ngay** ở tab **🧥 Thử đồ** trước nhé!")
    else:
        with st.container(border=True):
            bar_col1, bar_col2 = st.columns([3, 1])
            with bar_col1:
                if not catalog_df.empty:
                    pids = catalog_df["product_id"].tolist()
                    cur_idx = pids.index(st.session_state["product_id"]) if st.session_state["product_id"] in pids else 0
                    switch_pid = st.selectbox(
                        "🔄 Thử áo khác (giữ nguyên ảnh người):",
                        pids,
                        index=cur_idx,
                        format_func=lambda pid: (
                            f"{catalog_df.loc[catalog_df['product_id'] == pid, 'name'].values[0]} "
                            f"({catalog_df.loc[catalog_df['product_id'] == pid, 'color'].values[0]})"
                        ),
                    )
                    if switch_pid != st.session_state["product_id"]:
                        st.session_state["product_id"] = switch_pid
                        new_row = catalog_df[catalog_df["product_id"] == switch_pid].iloc[0]
                        new_g_img = load_garment_image(new_row)
                        new_g_name = str(new_row["image_file"])
                        try:
                            with st.spinner("Đang sinh ảnh, vui lòng chờ…"):
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
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Không thể đổi áo: {exc}")

            with bar_col2:
                st.metric("⏱️ Thời gian xử lý", f"{st.session_state['elapsed']:.2f} giây")

        st.write("")
        # Toggle bật/tắt chế độ đối chứng (Mặc định tắt để hiển thị 2 cột Before/After chuẩn thương mại)
        show_baseline = st.toggle("🔬 Chế độ so sánh kỹ thuật (Hiện ảnh Baseline đối chứng)", value=False)

        if show_baseline:
            c1, c2, c3 = st.columns(3, gap="medium")
            with c1:
                with st.container(border=True):
                    st.markdown("##### 🧍 Ảnh gốc")
                    st.image(st.session_state["person_img"], use_container_width=True)
            with c2:
                with st.container(border=True):
                    st.markdown("##### 📐 Phương pháp Baseline")
                    st.image(st.session_state["baseline"], use_container_width=True)
            with c3:
                with st.container(border=True):
                    st.markdown("##### ✨ Kết quả AI (CatVTON)")
                    st.image(st.session_state["result"], use_container_width=True)
        else:
            # Giao diện Before/After to rõ, đẹp mắt cho khách hàng
            c1, c2 = st.columns(2, gap="large")
            with c1:
                with st.container(border=True):
                    st.markdown("##### 🧍 Ảnh người mẫu ban đầu")
                    st.image(st.session_state["person_img"], use_container_width=True)
            with c2:
                with st.container(border=True):
                    st.markdown("##### ✨ Kết quả mặc thử AI (CatVTON)")
                    st.image(st.session_state["result"], use_container_width=True)

        bot_col1, bot_col2 = st.columns(2, gap="medium")
        with bot_col1:
            with st.expander("🔍 Xem mask vùng áo"):
                st.image(st.session_state["mask"], caption="Vùng màu trắng là vùng áo được AI xử lý thay thế", width=260)

        with bot_col2:
            if st.session_state["product_id"] and not catalog_df.empty:
                matched = catalog_df[catalog_df["product_id"] == st.session_state["product_id"]]
                if not matched.empty:
                    prod = matched.iloc[0]
                    with st.container(border=True):
                        st.markdown("##### 🏷️ Thông tin sản phẩm đang thử")
                        st.markdown(
                            f"""
                            - **Tên sản phẩm:** {prod['name']} (`{prod['product_id']}`)
                            - **Danh mục:** {prod['category']}
                            - **Màu sắc:** {prod['color']}
                            """
                        )

# ==============================================================================
# 4.6 TAB ĐÁNH GIÁ
# ==============================================================================
with tab_eval:
    st.subheader("Bảng chỉ số đánh giá chất lượng mô hình")
    summary_csv = config.RESULTS / "results_summary.csv"
    figs_dir = config.RESULTS / "figs"

    if not summary_csv.exists():
        st.info("Chưa có kết quả đánh giá")
    else:
        try:
            summary_df = pd.read_csv(summary_csv)
            if summary_df.empty:
                st.info("Chưa có kết quả đánh giá")
            else:
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Không thể đọc file đánh giá: {exc}")

    if figs_dir.exists():
        figs = sorted([p for p in figs_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if figs:
            st.markdown("#### Biểu đồ so sánh")
            f_cols = st.columns(min(2, len(figs)))
            for i, fp in enumerate(figs):
                with f_cols[i % len(f_cols)]:
                    st.image(str(fp), caption=fp.name, use_container_width=True)