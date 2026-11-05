"""Test "hợp đồng" giữa các phân việc: chữ ký hàm, chuẩn ảnh/mask, quy ước tên file.

Chạy: pytest -q. Test chỉ đọc cấu trúc file nên chạy được cả khi src/tryon.py còn là khung hoặc
đã import torch (không cần GPU).
"""
import ast
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src import config, utils
from src.tryon_mock import run_tryon as mock_run_tryon

SRC = Path(config.REPO_ROOT) / "src"


def _arg_names(py_file: str, func: str) -> list[str]:
    """Đọc tên tham số của hàm bằng ast, không import file (tránh kéo torch)."""
    tree = ast.parse((SRC / py_file).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            return [a.arg for a in node.args.args]
    raise AssertionError(f"Không thấy hàm {func} trong src/{py_file}")


def _img(size=(500, 700), mode="RGB", color=(120, 60, 200)):
    return Image.new(mode, size, color)


# ---------- chữ ký hàm ----------
def test_run_tryon_signature_is_frozen():
    expected = ["person_img", "garment_img", "mask"]
    assert _arg_names("tryon.py", "run_tryon") == expected
    assert _arg_names("tryon_mock.py", "run_tryon") == expected


def test_run_baseline_signature_is_frozen():
    assert _arg_names("baseline.py", "run_baseline") == ["person_img", "garment_img", "mask"]
    assert _arg_names("baseline.py", "remove_bg") == ["garment_img"]


# ---------- quy ước tên ----------
def test_pair_id_roundtrip():
    pid = utils.pair_id("00006_00.jpg", "outputs/x/00012_00.png")
    assert pid == "00006_00__00012_00"
    assert utils.split_pair_id(pid) == ("00006_00", "00012_00")
    with pytest.raises(ValueError):
        utils.split_pair_id("00006_00")


# ---------- chuẩn ảnh và mask ----------
def test_standardize_image_size_and_mode():
    for src in (_img((500, 700)), _img((2000, 1000)), _img((300, 300), "RGBA"), _img((400, 600), "L", 128)):
        out = utils.standardize_image(src)
        assert out.size == config.IMG_SIZE and out.mode == "RGB"


def test_standardize_mask_is_binary():
    m = Image.fromarray(np.random.default_rng(0).integers(0, 256, (300, 200), dtype=np.uint8))
    out = utils.standardize_mask(m)
    assert out.size == config.IMG_SIZE and out.mode == "L"
    assert set(np.unique(np.asarray(out))) <= {0, 255}


def test_load_dataset_mask_polarity(tmp_path, monkeypatch):
    arr = np.zeros((config.IMG_H, config.IMG_W), dtype=np.uint8)
    arr[200:400, 200:400] = 255                     # file gốc: vùng áo trắng
    f = tmp_path / "m.png"
    Image.fromarray(arr).save(f)
    monkeypatch.setattr(config, "AGNOSTIC_REGION_IS_WHITE", True)
    assert np.asarray(utils.load_dataset_mask(f))[300, 300] == 255
    monkeypatch.setattr(config, "AGNOSTIC_REGION_IS_WHITE", False)   # file gốc ngược chiều -> phải đảo
    assert np.asarray(utils.load_dataset_mask(f))[300, 300] == 0


def test_validate_input_image_messages():
    with pytest.raises(ValueError, match="Chưa có"):
        utils.validate_input_image(None, "ảnh người")
    with pytest.raises(ValueError, match="quá nhỏ"):
        utils.validate_input_image(_img((100, 100)), "ảnh áo")


# ---------- hợp đồng của hàm giả (hàm thật phải cư xử giống) ----------
def test_mock_tryon_contract():
    person, garment = _img((500, 700), color=(200, 200, 200)), _img((300, 400), color=(10, 120, 30))
    result, mask = mock_run_tryon(person, garment)
    assert result.size == config.IMG_SIZE and result.mode == "RGB"
    assert mask.size == config.IMG_SIZE and mask.mode == "L"
    assert set(np.unique(np.asarray(mask))) <= {0, 255}


def test_mock_tryon_keeps_pixels_outside_mask_and_uses_given_mask():
    person, garment = _img((768, 1024), color=(200, 200, 200)), _img((300, 400), color=(10, 120, 30))
    arr = np.zeros((1024, 768), dtype=np.uint8)
    arr[300:600, 200:500] = 255
    result, mask = mock_run_tryon(person, garment, mask=Image.fromarray(arr))
    assert np.array_equal(np.asarray(mask), arr)
    out = np.asarray(result)
    assert (out[0, 0] == 200).all()                  # ngoài mask: giữ nguyên ảnh người
    assert not (out[400, 300] == 200).all()          # trong mask: đã thay đổi


def test_mock_tryon_rejects_bad_input():
    with pytest.raises(ValueError):
        mock_run_tryon(None, _img())
    with pytest.raises(ValueError):
        mock_run_tryon(_img(), _img((50, 50)))


def test_get_run_tryon_switches_with_env(monkeypatch):
    monkeypatch.setenv("USE_MOCK", "1")
    assert config.get_run_tryon() is mock_run_tryon
