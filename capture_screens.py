"""
Chụp screenshot 3 trang giao diện Streamlit dùng Playwright (headless Chromium).

Yêu cầu Streamlit đang chạy tại http://localhost:8765 trước khi gọi script.

    streamlit run app.py --server.port 8765 &
    python capture_screens.py
"""

from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import sync_playwright


URL = "http://localhost:8765"
OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIEWPORT = {"width": 1440, "height": 900}


def click_radio(page, label_text: str) -> None:
    """Click radio điều hướng theo label tiếng Việt ở sidebar."""
    page.locator(f"label:has-text('{label_text}')").first.click()
    time.sleep(2.0)  # đợi rerun


def main() -> None:
    targets = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport=VIEWPORT, locale="vi-VN")
        page = context.new_page()

        # Trang 1: Dự đoán
        page.goto(URL, wait_until="networkidle")
        time.sleep(3.0)
        # Mặc định trang Dự đoán đã active. Cần scroll xuống cuối form trước khi capture.
        out1 = OUT_DIR / "hinh_7_1_trang_du_doan.png"
        page.screenshot(path=str(out1), full_page=True)
        targets.append(out1)
        print(f"Saved {out1}")

        # Trang 2: Quản trị & Tái huấn luyện
        click_radio(page, "Quản trị & Tái huấn luyện")
        time.sleep(2.0)
        out2 = OUT_DIR / "hinh_7_2_trang_quan_tri.png"
        page.screenshot(path=str(out2), full_page=True)
        targets.append(out2)
        print(f"Saved {out2}")

        # Trang 3: So sánh & Demo
        click_radio(page, "So sánh & Demo")
        time.sleep(3.0)
        # Mở 3 expander demo case để thấy đầy đủ
        for name in [
            "Khỏe mạnh, nguy cơ thấp",
            "Trung niên, nguy cơ trung bình",
            "Cao tuổi, nguy cơ cao",
        ]:
            try:
                page.locator(f"summary:has-text('{name}')").first.click()
                time.sleep(0.6)
            except Exception:
                pass
        time.sleep(1.5)
        out3 = OUT_DIR / "hinh_7_3_trang_so_sanh_demo.png"
        page.screenshot(path=str(out3), full_page=True)
        targets.append(out3)
        print(f"Saved {out3}")

        # Trang 1 sau khi nhấn dự đoán: nạp case "Cao tuổi, nguy cơ cao" và bấm Dự đoán
        click_radio(page, "Dự đoán")
        time.sleep(2.0)
        # Mở expander demo case
        try:
            page.locator("summary:has-text('Nạp nhanh case demo')").first.click()
            time.sleep(0.6)
            page.locator("button:has-text('Cao tuổi, nguy cơ cao')").first.click()
            time.sleep(1.0)
        except Exception as e:
            print(f"Không nạp được demo: {e}")

        # Cuộn xuống và bấm "Dự đoán"
        try:
            btn = page.locator("button:has-text('Dự đoán')").last
            btn.scroll_into_view_if_needed()
            time.sleep(0.4)
            btn.click()
            time.sleep(2.5)
        except Exception as e:
            print(f"Không bấm được nút Dự đoán: {e}")

        out4 = OUT_DIR / "hinh_7_4_ket_qua_du_doan.png"
        page.screenshot(path=str(out4), full_page=True)
        targets.append(out4)
        print(f"Saved {out4}")

        browser.close()

    print("\nĐã chụp:")
    for t in targets:
        print(f"  - {t}")


if __name__ == "__main__":
    main()
