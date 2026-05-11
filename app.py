"""
Streamlit app demo hệ thống dự đoán nguy cơ tiểu đường (BRFSS 2015).

Chạy: streamlit run app.py

Gồm 3 trang:
    1. Dự đoán       - Form 21 đặc trưng -> trả về xác suất + giải thích.
    2. Quản trị      - Upload CSV mới, kích hoạt retrain, xem lịch sử phiên bản.
    3. So sánh & Demo - Bảng so sánh 5 mô hình + 3 case demo dựng sẵn.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import retrain as rt
import scheduler as sch


BASE_DIR = Path(__file__).resolve().parent
CHAMPION_DIR = BASE_DIR / "outputs" / "diabetes_brfss_v1"
INCOMING_DIR = BASE_DIR / "data" / "incoming"
INCOMING_DIR.mkdir(parents=True, exist_ok=True)

if not sch.is_running():
    sch.start()

FEATURE_ORDER = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
    "Sex", "Age", "Education", "Income",
]

GENHLTH_OPTIONS = {
    1: "1 – Rất tốt",
    2: "2 – Tốt",
    3: "3 – Trung bình",
    4: "4 – Kém",
    5: "5 – Rất kém",
}
AGE_OPTIONS = {
    1: "1 – 18-24 tuổi", 2: "2 – 25-29 tuổi", 3: "3 – 30-34 tuổi",
    4: "4 – 35-39 tuổi", 5: "5 – 40-44 tuổi", 6: "6 – 45-49 tuổi",
    7: "7 – 50-54 tuổi", 8: "8 – 55-59 tuổi", 9: "9 – 60-64 tuổi",
    10: "10 – 65-69 tuổi", 11: "11 – 70-74 tuổi", 12: "12 – 75-79 tuổi",
    13: "13 – 80 tuổi trở lên",
}
EDU_OPTIONS = {
    1: "1 – Không đi học",
    2: "2 – Tiểu học",
    3: "3 – Trung học cơ sở",
    4: "4 – Trung học phổ thông / GED",
    5: "5 – Cao đẳng",
    6: "6 – Đại học trở lên",
}
INCOME_OPTIONS = {
    1: "1 – Dưới 10.000 USD/năm",
    2: "2 – 10.000 – 15.000 USD",
    3: "3 – 15.000 – 20.000 USD",
    4: "4 – 20.000 – 25.000 USD",
    5: "5 – 25.000 – 35.000 USD",
    6: "6 – 35.000 – 50.000 USD",
    7: "7 – 50.000 – 75.000 USD",
    8: "8 – Từ 75.000 USD trở lên",
}
YESNO = {0: "Không", 1: "Có"}
SEX_OPTIONS = {0: "Nữ", 1: "Nam"}

FEATURE_LABEL = {
    "HighBP": "Cao huyết áp",
    "HighChol": "Cholesterol cao",
    "CholCheck": "Đã kiểm tra cholesterol trong 5 năm",
    "BMI": "Chỉ số BMI",
    "Smoker": "Đã hút trên 100 điếu thuốc",
    "Stroke": "Đã từng bị đột quỵ",
    "HeartDiseaseorAttack": "Bệnh tim hoặc đau tim",
    "PhysActivity": "Có vận động thể chất 30 ngày qua",
    "Fruits": "Ăn trái cây ≥ 1 lần/ngày",
    "Veggies": "Ăn rau xanh ≥ 1 lần/ngày",
    "HvyAlcoholConsump": "Uống rượu nặng",
    "AnyHealthcare": "Có bảo hiểm y tế",
    "NoDocbcCost": "Không khám bác sĩ vì chi phí",
    "GenHlth": "Tự đánh giá sức khỏe",
    "MentHlth": "Số ngày sức khỏe tinh thần kém (30 ngày qua)",
    "PhysHlth": "Số ngày sức khỏe thể chất kém (30 ngày qua)",
    "DiffWalk": "Khó đi lại / leo cầu thang",
    "Sex": "Giới tính",
    "Age": "Nhóm tuổi",
    "Education": "Trình độ học vấn",
    "Income": "Thu nhập năm",
}

BINARY_FEATURES = {
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk",
}


def humanize_values(vals: dict) -> pd.DataFrame:
    """Đổi dict 21 đặc trưng sang DataFrame có nhãn và giá trị tiếng Việt."""
    rows = []
    for key in FEATURE_ORDER:
        v = vals[key]
        if key in BINARY_FEATURES:
            display = YESNO.get(int(v), str(v))
        elif key == "Sex":
            display = SEX_OPTIONS.get(int(v), str(v))
        elif key == "GenHlth":
            display = GENHLTH_OPTIONS.get(int(v), str(v))
        elif key == "Age":
            display = AGE_OPTIONS.get(int(v), str(v))
        elif key == "Education":
            display = EDU_OPTIONS.get(int(v), str(v))
        elif key == "Income":
            display = INCOME_OPTIONS.get(int(v), str(v))
        else:
            display = str(v)
        rows.append({"Đặc trưng": FEATURE_LABEL.get(key, key), "Giá trị": display})
    return pd.DataFrame(rows)

DEMO_CASES = {
    "Khỏe mạnh, nguy cơ thấp": {
        "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 22,
        "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0,
        "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1, "NoDocbcCost": 0,
        "GenHlth": 1, "MentHlth": 2, "PhysHlth": 1, "DiffWalk": 0,
        "Sex": 0, "Age": 3, "Education": 6, "Income": 7,
    },
    "Trung niên, nguy cơ trung bình": {
        "HighBP": 1, "HighChol": 0, "CholCheck": 1, "BMI": 28,
        "Smoker": 1, "Stroke": 0, "HeartDiseaseorAttack": 0,
        "PhysActivity": 0, "Fruits": 0, "Veggies": 1, "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1, "NoDocbcCost": 0,
        "GenHlth": 3, "MentHlth": 5, "PhysHlth": 8, "DiffWalk": 0,
        "Sex": 1, "Age": 8, "Education": 4, "Income": 4,
    },
    "Cao tuổi, nguy cơ cao": {
        "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 35,
        "Smoker": 1, "Stroke": 0, "HeartDiseaseorAttack": 1,
        "PhysActivity": 0, "Fruits": 0, "Veggies": 0, "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1, "NoDocbcCost": 1,
        "GenHlth": 5, "MentHlth": 15, "PhysHlth": 20, "DiffWalk": 1,
        "Sex": 1, "Age": 11, "Education": 3, "Income": 2,
    },
}


@st.cache_resource
def load_model():
    return joblib.load(CHAMPION_DIR / "best_model.pkl")


@st.cache_data
def load_metadata():
    p = CHAMPION_DIR / "run_metadata.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data
def load_results():
    p = CHAMPION_DIR / "model_results.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def reset_caches():
    st.cache_resource.clear()
    st.cache_data.clear()


def predict_row(values: dict) -> tuple[int, float]:
    """Trả về (label, xác suất thuộc lớp 1)."""
    model = load_model()
    row = np.array([[values[c] for c in FEATURE_ORDER]], dtype=float)
    label = int(model.predict(row)[0])
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(row)[0, 1])
    else:
        try:
            score = float(model.decision_function(row)[0])
            proba = 1.0 / (1.0 + np.exp(-score))
        except Exception:
            proba = float(label)
    return label, proba


def feature_importance() -> pd.Series | None:
    model = load_model()
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if hasattr(estimator, "feature_importances_"):
        return pd.Series(estimator.feature_importances_, index=FEATURE_ORDER).sort_values(
            ascending=False
        )
    if hasattr(estimator, "coef_"):
        return pd.Series(np.abs(estimator.coef_[0]), index=FEATURE_ORDER).sort_values(
            ascending=False
        )
    return None


# ===============================================================
# UI
# ===============================================================

st.set_page_config(
    page_title="Hệ thống dự đoán nguy cơ tiểu đường – BRFSS 2015",
    layout="wide",
)

st.sidebar.title("Điều hướng")
page = st.sidebar.radio(
    "Chọn trang:",
    ["Dự đoán", "Quản trị & Tái huấn luyện", "So sánh & Demo"],
)

meta = load_metadata()
if meta:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Champion**")
    st.sidebar.write(f"Thuật toán: `{meta.get('best_model', 'không có')}`")
    st.sidebar.write(f"Recall: `{meta.get('best_recall', 0):.4f}`")
    st.sidebar.write(f"Số mẫu huấn luyện: `{meta.get('train_shape', ['?', '?'])[0]}`")


# ---------------------------------------------------------------
# Trang 1 - Dự đoán
# ---------------------------------------------------------------
if page == "Dự đoán":
    st.title("Dự đoán nguy cơ mắc bệnh tiểu đường")
    st.caption(
        "Mô hình được huấn luyện trên bộ dữ liệu BRFSS 2015 (50/50 split). "
        "Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán y khoa."
    )

    if "preset" not in st.session_state:
        st.session_state.preset = None

    with st.expander("Nạp nhanh case demo (tuỳ chọn)"):
        cols = st.columns(len(DEMO_CASES))
        for i, name in enumerate(DEMO_CASES):
            if cols[i].button(name, use_container_width=True):
                st.session_state.preset = name

    preset_vals = DEMO_CASES.get(st.session_state.preset, {})

    def gv(key, default):
        return preset_vals.get(key, default)

    def yesno_radio(container, label, key, default):
        return container.radio(
            label, options=[0, 1],
            format_func=YESNO.get,
            index=gv(key, default),
            horizontal=True,
            key=f"radio_{key}",
        )

    with st.form("predict_form"):
        st.subheader("1. Sức khỏe nền")
        c1, c2 = st.columns(2)
        HighBP = yesno_radio(c1, "Cao huyết áp", "HighBP", 0)
        HighChol = yesno_radio(c2, "Cholesterol cao", "HighChol", 0)

        c1, c2 = st.columns(2)
        CholCheck = yesno_radio(
            c1, "Đã kiểm tra cholesterol trong 5 năm gần đây", "CholCheck", 1,
        )
        Stroke = yesno_radio(c2, "Đã từng bị đột quỵ", "Stroke", 0)

        c1, c2 = st.columns(2)
        HeartDiseaseorAttack = yesno_radio(
            c1, "Bệnh tim hoặc đau tim", "HeartDiseaseorAttack", 0,
        )
        DiffWalk = yesno_radio(c2, "Khó đi lại / leo cầu thang", "DiffWalk", 0)

        c1, c2 = st.columns(2)
        BMI = c1.slider("Chỉ số BMI (kg/m²)", 12, 98, value=gv("BMI", 25))
        GenHlth = c2.select_slider(
            "Tự đánh giá sức khỏe tổng quát",
            options=list(GENHLTH_OPTIONS.keys()),
            format_func=GENHLTH_OPTIONS.get,
            value=gv("GenHlth", 3),
        )

        st.subheader("2. Lối sống")
        c1, c2 = st.columns(2)
        Smoker = yesno_radio(
            c1, "Đã hút trên 100 điếu thuốc trong đời", "Smoker", 0,
        )
        PhysActivity = yesno_radio(
            c2, "Có vận động thể chất 30 ngày qua", "PhysActivity", 1,
        )

        c1, c2 = st.columns(2)
        Fruits = yesno_radio(c1, "Ăn trái cây ≥ 1 lần/ngày", "Fruits", 1)
        Veggies = yesno_radio(c2, "Ăn rau xanh ≥ 1 lần/ngày", "Veggies", 1)

        HvyAlcoholConsump = yesno_radio(
            st,
            "Uống rượu nặng (nam ≥ 14 ly/tuần, nữ ≥ 7 ly/tuần)",
            "HvyAlcoholConsump", 0,
        )

        c1, c2 = st.columns(2)
        MentHlth = c1.slider(
            "Số ngày sức khỏe tinh thần kém (30 ngày qua)",
            0, 30, value=gv("MentHlth", 0),
        )
        PhysHlth = c2.slider(
            "Số ngày sức khỏe thể chất kém (30 ngày qua)",
            0, 30, value=gv("PhysHlth", 0),
        )

        st.subheader("3. Nhân khẩu học và tiếp cận y tế")
        c1, c2 = st.columns(2)
        AnyHealthcare = yesno_radio(c1, "Có bảo hiểm y tế", "AnyHealthcare", 1)
        NoDocbcCost = yesno_radio(
            c2, "Không khám bác sĩ vì lý do chi phí", "NoDocbcCost", 0,
        )

        Sex = st.radio(
            "Giới tính", options=[0, 1],
            format_func=SEX_OPTIONS.get,
            index=gv("Sex", 0),
            horizontal=True,
            key="radio_Sex",
        )

        Age = st.selectbox(
            "Nhóm tuổi",
            options=list(AGE_OPTIONS.keys()),
            format_func=AGE_OPTIONS.get,
            index=list(AGE_OPTIONS.keys()).index(gv("Age", 7)),
        )

        c1, c2 = st.columns(2)
        Education = c1.selectbox(
            "Trình độ học vấn",
            options=list(EDU_OPTIONS.keys()),
            format_func=EDU_OPTIONS.get,
            index=list(EDU_OPTIONS.keys()).index(gv("Education", 4)),
        )
        Income = c2.selectbox(
            "Thu nhập năm",
            options=list(INCOME_OPTIONS.keys()),
            format_func=INCOME_OPTIONS.get,
            index=list(INCOME_OPTIONS.keys()).index(gv("Income", 5)),
        )

        submitted = st.form_submit_button(
            "Dự đoán", type="primary", use_container_width=True
        )

    if submitted:
        values = {
            "HighBP": HighBP, "HighChol": HighChol, "CholCheck": CholCheck, "BMI": BMI,
            "Smoker": Smoker, "Stroke": Stroke, "HeartDiseaseorAttack": HeartDiseaseorAttack,
            "PhysActivity": PhysActivity, "Fruits": Fruits, "Veggies": Veggies,
            "HvyAlcoholConsump": HvyAlcoholConsump, "AnyHealthcare": AnyHealthcare,
            "NoDocbcCost": NoDocbcCost, "GenHlth": GenHlth, "MentHlth": MentHlth,
            "PhysHlth": PhysHlth, "DiffWalk": DiffWalk, "Sex": Sex, "Age": Age,
            "Education": Education, "Income": Income,
        }
        label, proba = predict_row(values)

        st.markdown("## Kết quả")
        col1, col2 = st.columns([1, 2])
        with col1:
            if label == 1:
                st.error(
                    f"### Có nguy cơ mắc tiểu đường\n"
                    f"Xác suất: **{proba:.1%}**"
                )
            else:
                st.success(
                    f"### Không có dấu hiệu nguy cơ\n"
                    f"Xác suất mắc bệnh: **{proba:.1%}**"
                )
            st.progress(proba, text=f"Mức nguy cơ: {proba:.1%}")

        with col2:
            fi = feature_importance()
            if fi is not None:
                st.markdown("**Top 7 yếu tố ảnh hưởng toàn cục của mô hình:**")
                top = fi.head(7).reset_index()
                top.columns = ["Đặc trưng", "Tầm quan trọng"]
                st.bar_chart(top.set_index("Đặc trưng"))

        st.info(
            "**Khuyến cáo:** Kết quả này được sinh ra từ mô hình học máy và chỉ "
            "có giá trị tham khảo. Để được chẩn đoán chính xác, vui lòng liên hệ "
            "cơ sở y tế."
        )


# ---------------------------------------------------------------
# Trang 2 - Quản trị
# ---------------------------------------------------------------
elif page == "Quản trị & Tái huấn luyện":
    st.title("Quản trị & Tái huấn luyện")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Champion",
        "Upload & Tái huấn luyện",
        "Lịch sử phiên bản",
        "Lập lịch tự động",
    ])

    with tab1:
        st.subheader("Thông tin Champion")
        if meta:
            c1, c2, c3 = st.columns(3)
            c1.metric("Thuật toán", meta.get("best_model", "—"))
            c2.metric("Recall trên tập kiểm thử", f"{meta.get('best_recall', 0):.4f}")
            c3.metric("Số mẫu", meta.get("n_rows", "—"))

            st.markdown(
                "**Thời điểm huấn luyện gần nhất:** "
                f"`{meta.get('timestamp', 'chưa ghi nhận')}`"
            )

            st.markdown("**Bảng xếp hạng các mô hình ở lần retrain gần nhất:**")
            res = load_results()
            if not res.empty:
                st.dataframe(res, use_container_width=True)
        else:
            st.warning(
                "Chưa có Champion. Hãy bấm Tái huấn luyện ở tab kế bên để "
                "huấn luyện lần đầu."
            )

    with tab2:
        st.subheader("Cập nhật dữ liệu và kích hoạt tái huấn luyện")
        st.markdown(
            "1. *(Tuỳ chọn)* Tải lên các tệp CSV mới có cấu trúc 22 cột "
            "(21 đặc trưng + cột `Diabetes_binary`).\n"
            "2. Có thể **chỉ lưu vào kho dữ liệu** để scheduler tự gom đủ "
            "ngưỡng rồi retrain, hoặc **lưu và retrain ngay**.\n"
            "3. Hệ thống sẽ gộp với dữ liệu gốc, kiểm tra schema/drift, "
            "huấn luyện 5 mô hình, so sánh Recall với Champion."
        )

        if "uploader_key" not in st.session_state:
            st.session_state["uploader_key"] = 0

        uploaded = st.file_uploader(
            "Tệp CSV bổ sung (có thể chọn nhiều tệp)",
            type=["csv"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state['uploader_key']}",
        )
        force = st.checkbox(
            "Force promote dù recall thấp hơn (chỉ dùng khi debug)",
            value=False,
        )

        def _save_uploaded(files) -> list[str]:
            saved: list[str] = []
            for u in files:
                target = INCOMING_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{u.name}"
                target.write_bytes(u.getbuffer())
                saved.append(target.name)
            return saved

        col_save, col_train = st.columns(2)

        if col_save.button(
            "Chỉ lưu vào kho dữ liệu", disabled=not uploaded
        ):
            saved = _save_uploaded(uploaded)
            for name in saved:
                st.write(f"- Đã lưu `{name}`")
            st.success(
                f"Đã lưu {len(saved)} file vào `data/incoming/`. "
                "Scheduler sẽ tự kích hoạt retrain khi tổng số dòng mới "
                f"≥ ngưỡng cấu hình (xem tab 'Lập lịch tự động')."
            )
            st.session_state["uploader_key"] += 1
            st.rerun()

        if col_train.button("Lưu & tái huấn luyện ngay", type="primary"):
            if uploaded:
                saved = _save_uploaded(uploaded)
                for name in saved:
                    st.write(f"- Đã lưu `{name}`")
                st.session_state["uploader_key"] += 1

            with st.spinner(
                "Đang chạy GridSearchCV cho 5 mô hình, vui lòng chờ trong vài phút…"
            ):
                summary = rt.run_retrain(force_promote=force)

            reset_caches()
            st.success("Hoàn tất tái huấn luyện!")
            c1, c2 = st.columns(2)
            c1.metric(
                "Challenger (vừa huấn luyện)",
                summary.challenger_model,
                f"{summary.challenger_recall:.4f}",
            )
            c2.metric(
                "Champion trước đó",
                summary.champion_model or "—",
                f"{summary.champion_recall:.4f}" if summary.champion_recall else "—",
            )
            if summary.promoted:
                st.success(f"**Promote thành công.** {summary.decision_reason}")
            else:
                st.warning(f"**Không promote.** {summary.decision_reason}")

            if summary.drift_columns:
                st.warning("Phát hiện drift ở các cột:")
                st.json(summary.drift_columns)
            if not summary.schema_ok:
                st.error("Schema có vấn đề:")
                st.json(summary.schema_issues)

        st.divider()
        st.markdown("**Các file đang nằm trong `data/incoming/`**")
        incoming_files = sorted(INCOMING_DIR.glob("*.csv"))
        if not incoming_files:
            st.caption("(Trống)")
        else:
            rows = []
            for p in incoming_files:
                try:
                    n_rows = sum(1 for _ in p.open("r", encoding="utf-8")) - 1
                except Exception:
                    n_rows = "—"
                rows.append({
                    "Tệp": p.name,
                    "Số dòng": n_rows,
                    "Cập nhật": datetime.fromtimestamp(
                        p.stat().st_mtime
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tab3:
        st.subheader("Lịch sử các lần tái huấn luyện")
        runs = rt.list_runs()
        if not runs:
            st.info("Chưa có run nào trong thư mục `outputs/runs/`.")
        else:
            df = pd.DataFrame(runs)
            cols = [
                "timestamp", "challenger_model", "challenger_recall",
                "champion_model", "champion_recall", "promoted",
                "n_rows", "new_rows_added",
            ]
            df = df[[c for c in cols if c in df.columns]]
            df = df.rename(columns={
                "timestamp": "Thời điểm",
                "challenger_model": "Challenger",
                "challenger_recall": "Recall Challenger",
                "champion_model": "Champion trước đó",
                "champion_recall": "Recall Champion",
                "promoted": "Đã promote?",
                "n_rows": "Tổng số mẫu",
                "new_rows_added": "Số mẫu thêm mới",
            })
            st.dataframe(df, use_container_width=True)
            st.markdown("**Diễn biến Recall Challenger theo thời gian:**")
            st.line_chart(
                df.set_index("Thời điểm")["Recall Challenger"],
                height=240,
            )

    with tab4:
        st.subheader("Lập lịch tự động (APScheduler)")
        st.caption(
            "Scheduler chạy nền cùng tiến trình Streamlit, định kỳ kiểm tra "
            "thư mục `data/incoming/`. Chỉ kích hoạt retrain khi số dòng mới "
            "kể từ lần huấn luyện gần nhất đạt ngưỡng cấu hình."
        )

        status = sch.get_status()

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Trạng thái",
            "Đang chạy" if status["running"] else "Dừng",
        )
        c2.metric("Chu kỳ (phút)", status["interval_minutes"])
        c3.metric("Ngưỡng dòng mới", status["min_new_rows"])

        c4, c5 = st.columns(2)
        c4.metric(
            "Dòng mới đang chờ",
            status["pending_new_rows"],
            help="Tổng số dòng trong các CSV được thêm vào sau lần retrain gần nhất.",
        )
        c5.metric(
            "Đủ ngưỡng?",
            "Có" if status["pending_new_rows"] >= status["min_new_rows"] else "Chưa",
        )

        st.markdown(
            f"- **Lần kiểm tra gần nhất:** `{status['last_check_at'] or '—'}`\n"
            f"- **Lần kiểm tra kế tiếp:** `{status['next_run_at'] or '—'}`\n"
            f"- **Lần retrain gần nhất do scheduler:** "
            f"`{status['last_retrain_at'] or '—'}`"
        )

        if status["pending_new_files"]:
            with st.expander(
                f"Các file đang chờ ({len(status['pending_new_files'])})"
            ):
                st.write(status["pending_new_files"])

        st.divider()
        st.markdown("**Cấu hình**")

        new_interval = st.number_input(
            "Chu kỳ kiểm tra (phút)",
            min_value=1,
            max_value=24 * 60,
            value=int(status["interval_minutes"]),
            step=1,
        )
        new_threshold = st.number_input(
            "Số dòng mới tối thiểu để kích hoạt retrain",
            min_value=1,
            value=int(status["min_new_rows"]),
            step=10,
        )

        b1, b2, b3 = st.columns(3)
        if b1.button("Lưu cấu hình & khởi động lại"):
            sch.stop()
            sch.start(
                interval_minutes=int(new_interval),
                min_new_rows=int(new_threshold),
            )
            st.success("Đã cập nhật cấu hình scheduler.")
            st.rerun()

        if b2.button("Kiểm tra ngay"):
            sch.run_now()
            st.success(
                "Đã yêu cầu chạy kiểm tra. Bấm 'Làm mới' sau vài giây để xem kết quả."
            )

        if status["running"]:
            if b3.button("Dừng scheduler"):
                sch.stop()
                st.warning("Đã dừng scheduler.")
                st.rerun()
        else:
            if b3.button("Khởi động scheduler"):
                sch.start(
                    interval_minutes=int(new_interval),
                    min_new_rows=int(new_threshold),
                )
                st.success("Đã khởi động scheduler.")
                st.rerun()


# ---------------------------------------------------------------
# Trang 3 - So sánh & Demo
# ---------------------------------------------------------------
else:
    st.title("So sánh mô hình & Demo")

    st.subheader("Bảng xếp hạng 5 mô hình trên tập kiểm thử")
    res = load_results()
    if not res.empty:
        st.dataframe(res, use_container_width=True)
        st.bar_chart(
            res.set_index("model")[["test_recall", "test_precision", "accuracy"]]
        )
    else:
        st.warning(
            "Chưa có tệp `model_results.csv`. Hãy chạy tái huấn luyện trước."
        )

    st.subheader("Tầm quan trọng đặc trưng (Champion)")
    fi = feature_importance()
    if fi is not None:
        st.bar_chart(fi)
    else:
        st.info(
            "Champion không cung cấp tầm quan trọng đặc trưng."
        )

    st.subheader("Ba kịch bản demo")
    for name, vals in DEMO_CASES.items():
        with st.expander(name):
            label, proba = predict_row(vals)
            c1, c2 = st.columns([1, 2])
            with c1:
                if label == 1:
                    st.error(f"Có nguy cơ – {proba:.1%}")
                else:
                    st.success(f"Không có nguy cơ – {proba:.1%}")
            with c2:
                st.dataframe(
                    humanize_values(vals),
                    use_container_width=True,
                    hide_index=True,
                )
