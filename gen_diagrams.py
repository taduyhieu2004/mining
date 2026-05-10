"""
Sinh hai sơ đồ minh họa cho Chương 6 của báo cáo:

    outputs/figures/hinh_6_1_kien_truc.png        - Kiến trúc hệ thống (horizontal flow)
    outputs/figures/hinh_6_2_champion_challenger.png - Quy trình Champion-Challenger

Chạy: python gen_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10


# ---------------------------------------------------------------
# Tiện ích vẽ "card" cho từng component
# ---------------------------------------------------------------
def draw_card(
    ax, x, y, w, h, *,
    icon: str, accent: str,
    name: str, caption: str,
    name_fs: int = 10.5, caption_fs: int = 8.5,
    icon_radius: float = 0.30,
    body_color: str = "white",
):
    # Đổ bóng
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.05, y - 0.05), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=0, facecolor="#0000000F",
        )
    )
    # Body
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.3, edgecolor="#CFD8DC", facecolor=body_color,
        )
    )
    # Icon tròn
    cx = x + w / 2
    cy = y + h - icon_radius - 0.22
    ax.add_patch(
        Circle((cx, cy), icon_radius, facecolor=accent,
               edgecolor="white", linewidth=2, zorder=3)
    )
    ax.text(
        cx, cy, icon, ha="center", va="center",
        fontsize=13, color="white", fontweight="bold", zorder=4,
    )
    # Tên
    ax.text(
        x + w / 2, cy - icon_radius - 0.30, name,
        ha="center", va="top",
        fontsize=name_fs, fontweight="bold", color="#1F2933",
    )
    # Caption
    ax.text(
        x + w / 2, y + 0.15, caption,
        ha="center", va="bottom",
        fontsize=caption_fs, color="#52606D",
    )


def draw_arrow(ax, x1, y1, x2, y2, label: str = "",
               color: str = "#37474F",
               label_offset: tuple[float, float] = (0, 0.32)):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=18,
            linewidth=1.7, color=color,
        )
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mx + label_offset[0], my + label_offset[1], label,
            ha="center", va="center",
            fontsize=8.8, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=color, linewidth=0.8),
        )


# ---------------------------------------------------------------
# HÌNH 6.1 — KIẾN TRÚC HỆ THỐNG
# ---------------------------------------------------------------
def gen_architecture_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Tiêu đề
    ax.text(
        9, 9.7,
        "Hình 6.1. Kiến trúc hệ thống dự đoán nguy cơ tiểu đường",
        ha="center", va="center",
        fontsize=14.5, fontweight="bold", color="#1F2933",
    )

    # Boundary ngoài
    sys_x, sys_y, sys_w, sys_h = 0.4, 0.4, 17.2, 8.85
    ax.add_patch(
        FancyBboxPatch(
            (sys_x, sys_y), sys_w, sys_h,
            boxstyle="round,pad=0.05,rounding_size=0.25",
            linewidth=1.4, edgecolor="#37474F",
            facecolor="#F7F9FB",
        )
    )
    ax.text(
        sys_x + 0.3, sys_y + sys_h - 0.3,
        "Hệ thống dự đoán nguy cơ tiểu đường  (BRFSS 2015)",
        ha="left", va="center",
        fontsize=11.5, fontweight="bold", color="#37474F",
    )

    # Boundary pipeline
    pl_x, pl_y, pl_w, pl_h = 0.85, 3.95, 13.15, 3.55
    ax.add_patch(
        FancyBboxPatch(
            (pl_x, pl_y), pl_w, pl_h,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            linewidth=1.3, edgecolor="#7A3FAE",
            facecolor="#FAF6FE", linestyle="--",
        )
    )
    ax.text(
        pl_x + pl_w / 2, pl_y + pl_h - 0.3,
        "Quy trình tái huấn luyện liên tục   —   retrain.py  +  bộ lập lịch cron",
        ha="center", va="center",
        fontsize=10.5, fontweight="bold", color="#5E2A91", style="italic",
    )

    # 5 card pipeline
    card_w, card_h = 2.0, 2.55
    n_cards = 5
    inner_pad = 0.55
    available = pl_w - 2 * inner_pad
    gap = (available - n_cards * card_w) / (n_cards - 1)
    pipeline_card_y = pl_y + 0.25
    cards = [
        dict(icon="≣", accent="#1E5AA8", name="Nguồn dữ liệu",
             caption="diabetes_*.csv\n+ data/incoming/\nKhung tham chiếu"),
        dict(icon="⚙", accent="#7A3FAE", name="Tiền xử lý",
             caption="Kiểm tra cấu trúc\nPhát hiện trôi\nChuẩn hóa Z-score"),
        dict(icon="◆", accent="#C8721A", name="Huấn luyện",
             caption="5 thuật toán\nGridSearchCV cv = 3\nTiêu chí: Recall"),
        dict(icon="±", accent="#A48A1A", name="Đánh giá",
             caption="Recall / Precision\nAccuracy\ntrên tập kiểm thử"),
        dict(icon="★", accent="#37474F", name="Kho mô hình",
             caption="outputs/runs/\nrun_<timestamp>\n+ summary.json"),
    ]
    positions_x = []
    for i, c in enumerate(cards):
        x = pl_x + inner_pad + i * (card_w + gap)
        positions_x.append(x)
        draw_card(ax, x, pipeline_card_y, card_w, card_h, **c)

    # Mũi tên giữa các card pipeline
    actions = ["Nạp dữ liệu", "Tiền xử lý", "Huấn luyện", "Đánh giá"]
    arrow_y = pipeline_card_y + card_h * 0.45
    for i, act in enumerate(actions):
        xa = positions_x[i] + card_w
        xb = positions_x[i + 1]
        draw_arrow(
            ax, xa + 0.05, arrow_y, xb - 0.05, arrow_y,
            label=act, label_offset=(0, 0.65),
        )

    # ----- Mô hình hiện hành + Streamlit (stack dọc bên phải) -----
    side_x = 14.3
    side_w = 2.95
    champion_h = 1.85
    streamlit_h = 2.05
    champion_y = pl_y + 0.7
    streamlit_y = champion_y - streamlit_h - 0.55

    draw_card(
        ax, side_x, champion_y, side_w, champion_h,
        icon="✓", accent="#1F8A4C",
        name="Champion",
        caption="Mô hình đang phục vụ\n(duy nhất 1 phiên bản)\nbest_model.pkl",
        name_fs=10,
    )
    draw_card(
        ax, side_x, streamlit_y, side_w, streamlit_h,
        icon="◐", accent="#E63946",
        name="Giao diện Streamlit",
        caption="Trang Dự đoán\nQuản trị & Tái huấn luyện\nSo sánh & Demo",
        name_fs=10,
    )

    # Mũi tên Promote: Kho mô hình → Champion (ngang).
    # Label đẩy LÊN CAO trên đỉnh Champion để tránh đè tiêu đề card.
    reg_right_x = positions_x[-1] + card_w
    reg_mid_y = pipeline_card_y + card_h / 2
    draw_arrow(
        ax,
        reg_right_x + 0.05, reg_mid_y,
        side_x - 0.05, champion_y + champion_h / 2,
        label="Promote", color="#1F8A4C",
        label_offset=(0, 1.4),
    )
    # Mũi tên Nạp mô hình: Mô hình hiện hành → Streamlit
    draw_arrow(
        ax,
        side_x + side_w / 2, champion_y - 0.02,
        side_x + side_w / 2, streamlit_y + streamlit_h + 0.02,
        label="Nạp mô hình", color="#E63946",
        label_offset=(0.75, 0),
    )

    # ----- Bộ lập lịch (trên cùng pipeline) -----
    cron_x, cron_y, cron_w, cron_h = 1.0, 7.85, 4.0, 0.7
    ax.add_patch(
        FancyBboxPatch(
            (cron_x, cron_y), cron_w, cron_h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            linewidth=1.2, edgecolor="#5E2A91", facecolor="#EDE2FA",
        )
    )
    ax.text(
        cron_x + cron_w / 2, cron_y + cron_h / 2,
        "[ ⟳ ]   Lịch tự động (cron)  /  Bấm thủ công",
        ha="center", va="center", fontsize=10.5,
        color="#5E2A91", fontweight="bold",
    )
    draw_arrow(
        ax,
        cron_x + cron_w / 2, cron_y - 0.02,
        cron_x + cron_w / 2, pl_y + pl_h + 0.02,
        label="Kích hoạt", color="#5E2A91",
        label_offset=(0.7, 0),
    )

    # ----- Người vận hành (góc dưới trái, dưới Data Source) -----
    op_x = positions_x[0]
    op_w = card_w
    op_h = 2.35
    op_y = 1.0
    draw_card(
        ax, op_x, op_y, op_w, op_h,
        icon="P", accent="#1E5AA8",
        name="Người vận hành",
        caption="Upload CSV mới\n→ data/incoming/\nNhấn nút Retrain",
        name_fs=10,
    )
    draw_arrow(
        ax,
        op_x + op_w / 2, op_y + op_h + 0.02,
        op_x + op_w / 2, pipeline_card_y - 0.02,
        label="Tải lên", color="#1E5AA8",
        label_offset=(0.55, 0),
    )

    # ----- Người dùng cuối (dưới Streamlit) -----
    eu_text_y = streamlit_y - 0.45
    ax.text(
        side_x + side_w / 2, eu_text_y,
        "[  Người dùng cuối — bệnh nhân / bác sĩ  ]",
        ha="center", va="center", fontsize=9.5,
        color="#444", fontweight="bold",
    )
    draw_arrow(
        ax,
        side_x + side_w / 2, streamlit_y - 0.02,
        side_x + side_w / 2, eu_text_y + 0.25,
        color="#888",
    )

    # Chú thích nhỏ ở góc dưới phải
    ax.text(
        sys_x + sys_w - 0.3, sys_y + 0.25,
        "Công nghệ: Python · scikit-learn · Streamlit · joblib",
        ha="right", va="center", fontsize=8.5, color="#90A4AE", style="italic",
    )

    out = FIG_DIR / "hinh_6_1_kien_truc.png"
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------
# HÌNH 6.2 — CHAMPION ↔ CHALLENGER
# ---------------------------------------------------------------
def gen_champion_challenger_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 11.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        6.5, 11.05,
        "Hình 6.2. Quy trình ra quyết định Champion – Challenger",
        ha="center", va="center", fontsize=13.5, fontweight="bold",
        color="#1F2933",
    )

    ax.add_patch(
        FancyBboxPatch(
            (0.4, 0.4), 12.2, 10.2,
            boxstyle="round,pad=0.04,rounding_size=0.2",
            linewidth=1.2, edgecolor="#37474F", facecolor="#F7F9FB",
        )
    )

    cards = [
        dict(x=4.0, y=8.55, w=5.0, h=1.95,
             icon="≣", accent="#1E5AA8", name="Bước 1 — Dữ liệu",
             caption="Gộp dữ liệu nền + data/incoming/  →  loại trùng"),
        dict(x=4.0, y=6.30, w=5.0, h=2.00,
             icon="◆", accent="#C8721A", name="Bước 2 — Huấn luyện 5 mô hình",
             caption="GridSearchCV  (cv = 3,  scoring = recall)\n"
                     "Logistic · Cây QĐ · RF · GB · KNN"),
        dict(x=4.0, y=4.05, w=5.0, h=2.00,
             icon="★", accent="#7A3FAE", name="Bước 3 — Chọn Challenger",
             caption="Challenger = mô hình có\nRecall trên tập kiểm thử cao nhất"),
        dict(x=4.0, y=1.65, w=5.0, h=2.15,
             icon="±", accent="#A48A1A", name="Bước 4 — So sánh với Champion",
             caption="Recall (Challenger)   so với   Recall (Champion)\n"
                     "(đọc từ run_metadata.json)",
             name_fs=10),
    ]
    for c in cards:
        draw_card(ax, c["x"], c["y"], c["w"], c["h"],
                  icon=c["icon"], accent=c["accent"],
                  name=c["name"], caption=c["caption"],
                  name_fs=c.get("name_fs", 10.5))

    # Mũi tên dọc giữa các bước
    cx = 6.5
    for y_top, y_bot in [(8.55, 8.30), (6.30, 6.05), (4.05, 3.80)]:
        draw_arrow(ax, cx, y_top, cx, y_bot)

    # Hai nhánh THĂNG CẤP / GIỮ NGUYÊN
    draw_card(
        ax, 0.55, 0.55, 5.0, 1.05,
        icon="✓", accent="#1F8A4C",
        name="PROMOTE  (Challenger ≥ Champion)",
        caption="Cập nhật Champion = Challenger + khung tham chiếu + metadata",
        name_fs=10, icon_radius=0.24,
    )
    draw_card(
        ax, 7.45, 0.55, 5.0, 1.05,
        icon="✕", accent="#B53A3A",
        name="HOLD  (Challenger < Champion)",
        caption="Giữ Champion. Lưu Challenger vào outputs/runs/",
        name_fs=10, icon_radius=0.24,
    )

    # Mũi tên rẽ nhánh từ Bước 4 xuống 2 nhánh
    draw_arrow(ax, 5.0, 1.85, 3.0, 1.65, label="≥",
               color="#1F8A4C", label_offset=(-0.2, 0.25))
    draw_arrow(ax, 8.0, 1.85, 10.0, 1.65, label="<",
               color="#B53A3A", label_offset=(0.2, 0.25))

    out = FIG_DIR / "hinh_6_2_champion_challenger.png"
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p1 = gen_architecture_diagram()
    p2 = gen_champion_challenger_diagram()
    print(f"Đã sinh: {p1}")
    print(f"Đã sinh: {p2}")
