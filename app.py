import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from ro_model import (
    calculate_single_stage,
    calculate_two_stage,
    solve_xr2_for_target_cv,
)

st.set_page_config(
    page_title="Розрахунок зворотного осмосу (RO / ЗОУ)",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Стилізація
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .badge-ok {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-warn {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def make_sankey_single_stage(flows, concs):
    """Створює діаграму Санкі для одноступеневої установки."""
    # Вузли:
    # 0: Вихідна вода (Gi)
    # 1: Змішувач
    # 2: Мембрана
    # 3: Пермеат (Gp)
    # 4: Розділювач концентрату
    # 5: Рециркулят (Gr)
    # 6: Скид (Gs)
    node_labels = [
        f"Вихідна вода (Gi)<br>{flows['Gi']:.1f} м³/год | {concs['Ci']:.0f} мг/л",
        f"Змішувач<br>{flows['Gv']:.1f} м³/год | {concs['Cv']:.1f} мг/л",
        "Мембранний модуль",
        f"Пермеат (Gp)<br>{flows['Gp']:.1f} м³/год | {concs['Cp']:.2f} мг/л",
        f"Концентрат (Gk)<br>{flows['Gk']:.1f} м³/год | {concs['Ck']:.1f} мг/л",
        f"Рециркулят (Gr)<br>{flows['Gr']:.1f} м³/год",
        f"Скид дренаж (Gs)<br>{flows['Gs']:.1f} м³/год",
    ]

    source = [0, 1, 2, 2, 4, 4, 5]
    target = [1, 2, 3, 4, 5, 6, 1]
    value = [
        flows["Gi"],
        flows["Gv"],
        flows["Gp"],
        flows["Gk"],
        flows["Gr"],
        flows["Gs"],
        flows["Gr"],
    ]
    colors = [
        "rgba(59, 130, 246, 0.5)",
        "rgba(37, 99, 235, 0.5)",
        "rgba(16, 185, 129, 0.7)",
        "rgba(245, 158, 11, 0.5)",
        "rgba(217, 119, 6, 0.5)",
        "rgba(239, 68, 68, 0.6)",
        "rgba(217, 119, 6, 0.5)",
    ]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                    color="#2563EB",
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color=colors,
                ),
            )
        ]
    )
    fig.update_layout(
        title_text="Потокова діаграма (Sankey Diagram) розподілу витрат води [м³/год]",
        font_size=12,
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def make_sankey_two_stage(flows, concs):
    """Створює діаграму Санкі для двоступеневої установки."""
    node_labels = [
        f"Вихідна вода (Gii)<br>{flows['Gii']:.1f} м³/год",  # 0
        f"Змішувач 1<br>{flows['Gv']:.1f} м³/год",          # 1
        "Мембрана Ступінь 1",                               # 2
        f"Пермеат 1 (Gp1)<br>{flows['Gp1']:.1f} м³/год",     # 3
        f"Концентрат 1 (Gk1)<br>{flows['Gk1']:.1f} м³/год", # 4
        f"Скид 1 (Gs1)<br>{flows['Gs1']:.1f} м³/год",        # 5
        "Мембрана Ступінь 2",                               # 6
        f"Чистий Пермеат (Gp2)<br>{flows['Gp2']:.1f} м³/год", # 7
        f"Концентрат 2 (Gk2)<br>{flows['Gk2']:.1f} м³/год", # 8
        f"Скид 2 (Gs2)<br>{flows['Gs2']:.1f} м³/год",        # 9
        f"Змішувач рециркуляту (Grr)<br>{flows['Grr']:.1f} м³/год", # 10
    ]

    source = [0, 1, 2, 2, 4, 4, 3, 6, 6, 8, 8, 10]
    target = [1, 2, 3, 4, 10, 5, 6, 7, 8, 10, 9, 1]
    value = [
        flows["Gii"],
        flows["Gv"],
        flows["Gp1"],
        flows["Gk1"],
        flows["Gr1"],
        flows["Gs1"],
        flows["Gp1"],
        flows["Gp2"],
        flows["Gk2"],
        flows["Gr2"],
        flows["Gs2"],
        flows["Grr"],
    ]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=16,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                    color="#4F46E5",
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color="rgba(99, 102, 241, 0.45)",
                ),
            )
        ]
    )
    fig.update_layout(
        title_text="Двоступеневий розподіл потоків води (Sankey Diagram) [м³/год]",
        font_size=11,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# Головний заголовок
st.markdown('<div class="main-title">💧 Розрахунок параметрів зворотного осмосу (RO / ЗОУ)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Веб-додаток для моделювання одноступеневої (ЗО-1) та двоступеневої (ЗОУ-2) схем з рециркуляцією на основі Colab <b>RO-2022.ipynb</b></div>', unsafe_allow_html=True)

# Сайдбар: вибір режиму та налаштувань
with st.sidebar:
    st.header("⚙️ Режим розрахунку")
    mode = st.radio(
        "Оберіть схему:",
        [
            "Одноступенева схема (ЗО-1)",
            "Двоступенева схема (ЗОУ-2)",
            "Параметричний аналіз (графіки)",
            "Теоретичні основи & формули",
        ],
        index=0,
    )

    st.markdown("---")
    st.header("📥 Вхідні параметри")

    if st.button("🔄 Скинути до значень за замовчуванням (Colab)"):
        st.session_state.clear()
        st.rerun()

    # Спільні параметри
    Ci = st.number_input(
        "Солевміст вихідної води Ci (мг/л):",
        min_value=10.0,
        max_value=50000.0,
        value=1000.0,
        step=50.0,
        help="Вихідна мінералізація або концентрація розчинених солей (TDS).",
    )

    Gv = st.number_input(
        "Витрата подачі на мембрану Gv (м³/год):",
        min_value=1.0,
        max_value=10000.0,
        value=100.0,
        step=10.0,
        help="Загальна витрата потоку після змішувача, що надходить на вхід 1-го ступеня.",
    )

    if mode in ["Одноступенева схема (ЗО-1)", "Параметричний аналіз (графіки)"]:
        st.subheader("Параметри ЗО-1")
        Xz = st.slider(
            "Селективність мембрани Xz (частка):",
            min_value=0.50,
            max_value=0.999,
            value=0.99,
            step=0.005,
            format="%.3f",
            help="Коефіцієнт затримання солей (0.99 = 99%).",
        )
        Xp = st.slider(
            "Вихід пермеату модуля Xp (частка):",
            min_value=0.05,
            max_value=0.80,
            value=0.20,
            step=0.01,
            format="%.2f",
            help="Частка пермеату від вхідного потоку мембрани (Gp / Gv).",
        )
        Xr = st.slider(
            "Коефіцієнт рециркуляції Xr (частка):",
            min_value=0.0,
            max_value=0.95,
            value=0.77,
            step=0.01,
            format="%.2f",
            help="Частка концентрату, що повертається на рецикл (Gr / Gk).",
        )

    elif mode == "Двоступенева схема (ЗОУ-2)":
        st.subheader("Ступінь 1 (ЗО-1)")
        col_s1_1, col_s1_2 = st.columns(2)
        with col_s1_1:
            Xz1 = st.number_input("Селективність Xz1:", min_value=0.5, max_value=0.999, value=0.99, step=0.005, format="%.3f")
            Xp1 = st.number_input("Вихід пермеату Xp1:", min_value=0.05, max_value=0.8, value=0.20, step=0.01, format="%.2f")
        with col_s1_2:
            Xr1 = st.number_input("Рецикл Xr1:", min_value=0.0, max_value=0.95, value=0.20, step=0.01, format="%.2f")

        st.subheader("Ступінь 2 (ЗО-2)")
        col_s2_1, col_s2_2 = st.columns(2)
        with col_s2_1:
            Xz2 = st.number_input("Селективність Xz2:", min_value=0.5, max_value=0.999, value=0.99, step=0.005, format="%.3f")
            Xp2 = st.number_input("Вихід пермеату Xp2:", min_value=0.05, max_value=0.8, value=0.20, step=0.01, format="%.2f")
        with col_s2_2:
            if "custom_xr2" not in st.session_state:
                st.session_state["custom_xr2"] = 0.25
            Xr2 = st.number_input(
                "Рецикл Xr2:",
                min_value=0.0,
                max_value=0.95,
                value=float(st.session_state["custom_xr2"]),
                step=0.01,
                format="%.3f",
                key="xr2_input",
            )

        if st.button("🎯 Автопідбір Xr2 для умови Cv == Ci"):
            with st.spinner("Підбір Xr2..."):
                found_xr2 = solve_xr2_for_target_cv(
                    Ci=Ci, Gv=Gv,
                    Xz1=Xz1, Xp1=Xp1, Xr1=Xr1,
                    Xz2=Xz2, Xp2=Xp2, target_cv=Ci
                )
                st.session_state["custom_xr2"] = round(found_xr2, 4)
                st.success(f"Знайдено оптимальне Xr2 = {found_xr2:.4f}")
                st.rerun()


# =====================================================================
# РЕЖИМ 1: ОДНОСТУПЕНЕВА СХЕМА (ЗО-1)
# =====================================================================
if mode == "Одноступенева схема (ЗО-1)":
    res = calculate_single_stage(Ci=Ci, Gv=Gv, Xz=Xz, Xp=Xp, Xr=Xr)
    fl = res["flows"]
    co = res["concentrations"]
    bal = res["balance"]
    kpi = res["kpi"]

    # Верхній блок ключових KPI
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            label="💧 Витрата пермеату (Gp)",
            value=f"{fl['Gp']:.2f} м³/год",
            delta=f"Вихід: {kpi['recovery_system']:.1f}%",
        )
    with col2:
        st.metric(
            label="🧪 Якість пермеату (Cp)",
            value=f"{co['Cp']:.2f} мг/л",
            delta=f"-{kpi['rejection_system']:.2f}% від Ci",
            delta_color="inverse",
        )
    with col3:
        st.metric(
            label="🌊 Забір свіжої води (Gi)",
            value=f"{fl['Gi']:.2f} м³/год",
            delta=f"{fl['Gi']/fl['Gv']*100:.1f}% від Gv",
        )
    with col4:
        st.metric(
            label="🛑 Скид концентрату (Gs)",
            value=f"{fl['Gs']:.2f} м³/год",
            delta=f"Ck: {co['Ck']:.1f} мг/л",
            delta_color="off",
        )
    with col5:
        st.metric(
            label="🔄 Рециркулят (Gr)",
            value=f"{fl['Gr']:.2f} м³/год",
            delta=f"Концентрат Gk: {fl['Gk']:.1f}",
            delta_color="off",
        )

    st.markdown("---")

    # Перевірка матеріального балансу
    st.subheader("⚖️ Аудит матеріального балансу (BBB Test)")
    col_bal1, col_bal2 = st.columns([2, 1])
    with col_bal1:
        st.write(
            f"""
            - **Баланс по воді:** $\\Sigma G_{{in}} = G_i = {bal['water_in']:.4f}$ м³/год  
              $\\Sigma G_{{out}} = G_p + G_s = {bal['water_out']:.4f}$ м³/год  
              $\\Delta_G = {bal['water_delta']:.2e}$ м³/год  
            - **Баланс по солі (розчиненій речовині):**  
              $LP = G_i \\cdot C_i = {bal['salt_in']:.4f}$ г/год  
              $RP = G_p \\cdot C_p + G_s \\cdot C_k = {bal['salt_out']:.4f}$ г/год  
              $\\Delta_{{солі}} = LP - RP = {bal['salt_delta']:.2e}$ г/год
            """
        )
    with col_bal2:
        if bal["is_balance_ok"]:
            st.markdown('<div class="badge-ok">✅ BBB is OK! Матеріальний баланс зійшовся точно.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-warn">⚠️ Увага: нев\'язка балансу перевищує допустиму норму!</div>', unsafe_allow_html=True)
        st.caption(f"Кількість ітерацій до збіжності: **{res['iterations']}**")

    # Діаграма Санкі та Таблиця потоків
    st.markdown("---")
    tab_diag, tab_table, tab_iter = st.tabs(["📊 Діаграма розподілу потоків", "📋 Зведена таблиця потоків", "📈 Динаміка ітерацій"])

    with tab_diag:
        sankey_fig = make_sankey_single_stage(fl, co)
        st.plotly_chart(sankey_fig, use_container_width=True)

    with tab_table:
        df_streams = pd.DataFrame([
            {"Потік": "Вихідна вода (джерело)", "Позначення": "Gi", "Витрата G, м³/год": fl["Gi"], "Солевміст C, мг/л": co["Ci"], "Масовий потік солі, кг/год": fl["Gi"] * co["Ci"] / 1000},
            {"Потік": "Подача на мембрану", "Позначення": "Gv", "Витрата G, м³/год": fl["Gv"], "Солевміст C, мг/л": co["Cv"], "Масовий потік солі, кг/год": fl["Gv"] * co["Cv"] / 1000},
            {"Потік": "Очищена вода (пермеат)", "Позначення": "Gp", "Витрата G, м³/год": fl["Gp"], "Солевміст C, мг/л": co["Cp"], "Масовий потік солі, кг/год": fl["Gp"] * co["Cp"] / 1000},
            {"Потік": "Загальний концентрат", "Позначення": "Gk", "Витрата G, м³/год": fl["Gk"], "Солевміст C, мг/л": co["Ck"], "Масовий потік солі, кг/год": fl["Gk"] * co["Ck"] / 1000},
            {"Потік": "Рециркулят концентрату", "Позначення": "Gr", "Витрата G, м³/год": fl["Gr"], "Солевміст C, мг/л": co["Ck"], "Масовий потік солі, кг/год": fl["Gr"] * co["Ck"] / 1000},
            {"Потік": "Скид концентрату (дренаж)", "Позначення": "Gs", "Витрата G, м³/год": fl["Gs"], "Солевміст C, мг/л": co["Ck"], "Масовий потік солі, кг/год": fl["Gs"] * co["Ck"] / 1000},
        ])
        st.dataframe(df_streams.style.format({
            "Витрата G, м³/год": "{:.2f}",
            "Солевміст C, мг/л": "{:.2f}",
            "Масовий потік солі, кг/год": "{:.4f}",
        }), use_container_width=True)

        csv = df_streams.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Завантажити результати у CSV",
            data=csv,
            file_name="ro_single_stage_results.csv",
            mime="text/csv",
        )

    with tab_iter:
        df_hist = pd.DataFrame(res["history"])
        fig_iter = go.Figure()
        fig_iter.add_trace(go.Scatter(x=df_hist["iter"], y=df_hist["Cv"], mode="lines+markers", name="Cv (вхід мембрани)"))
        fig_iter.add_trace(go.Scatter(x=df_hist["iter"], y=df_hist["Ck"], mode="lines", name="Ck (концентрат)"))
        fig_iter.add_trace(go.Scatter(x=df_hist["iter"], y=df_hist["Cp"], mode="lines", name="Cp (пермеат)"))
        fig_iter.update_layout(
            title="Збіжність концентрацій за ітераціями",
            xaxis_title="Номер ітерації",
            yaxis_title="Концентрація, мг/л",
            hovermode="x unified",
            height=380,
        )
        st.plotly_chart(fig_iter, use_container_width=True)


# =====================================================================
# РЕЖИМ 2: ДВОСТУПЕНЕВА СХЕМА (ЗОУ-2)
# =====================================================================
elif mode == "Двоступенева схема (ЗОУ-2)":
    res2 = calculate_two_stage(
        Ci=Ci, Gv=Gv,
        Xz1=Xz1, Xp1=Xp1, Xr1=Xr1,
        Xz2=Xz2, Xp2=Xp2, Xr2=Xr2,
    )
    fl2 = res2["flows"]
    co2 = res2["concentrations"]
    bal2 = res2["balance"]
    kpi2 = res2["kpi"]

    # Верхній блок метрик ЗОУ-2
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            label="💎 Пермеат 2-го ступеня (Gp2)",
            value=f"{fl2['Gp2']:.2f} м³/год",
            delta=f"Вихід: {kpi2['recovery_system']:.2f}%",
        )
    with col2:
        st.metric(
            label="🧪 Кінцевий солевміст (Cp2)",
            value=f"{co2['Cp2']:.3f} мг/л",
            delta=f"Селективність: {kpi2['rejection_system']:.3f}%",
            delta_color="inverse",
        )
    with col3:
        st.metric(
            label="🌊 Забір вихідної води (Gii)",
            value=f"{fl2['Gii']:.2f} м³/год",
            delta=f"Вхід Ci: {co2['Ci']:.0f} мг/л",
        )
    with col4:
        diff_cv = bal2["cv_ci_diff"]
        st.metric(
            label="🎯 Умова Cv == Ci",
            value=f"{co2['Cv1']:.2f} мг/л",
            delta=f"Δ = {diff_cv:+.2f} мг/л ({bal2['cv_ci_pct']:+.2f}%)",
            delta_color="normal" if abs(diff_cv) < 5.0 else "inverse",
        )
    with col5:
        total_discharge = fl2["Gs1"] + fl2["Gs2"]
        st.metric(
            label="🛑 Сумарний скид (Gs1 + Gs2)",
            value=f"{total_discharge:.2f} м³/год",
            delta=f"Gs1={fl2['Gs1']:.1f}, Gs2={fl2['Gs2']:.1f}",
            delta_color="off",
        )

    st.markdown("---")

    # Баланс для ЗОУ-2
    st.subheader("⚖️ Аудит матеріального балансу ЗОУ-2")
    col_bal1, col_bal2 = st.columns([2, 1])
    with col_bal1:
        st.write(
            f"""
            - **Водний баланс:**  
              $\\Sigma G_{{in}} = G_{{ii}} = {bal2['water_in']:.4f}$ м³/год  
              $\\Sigma G_{{out}} = G_{{p2}} + G_{{s1}} + G_{{s2}} = {bal2['water_out']:.4f}$ м³/год  
              $\\Delta_G = {bal2['water_delta']:.2e}$ м³/год  
            - **Сольовий баланс:**  
              $LP = G_{{ii}} \\cdot C_i = {bal2['salt_in']:.4f}$ г/год  
              $RP = G_{{p2}} \\cdot C_{{p2}} + G_{{s1}} \\cdot C_{{k1}} + G_{{s2}} \\cdot C_{{k2}} = {bal2['salt_out']:.4f}$ г/год  
              $\\Delta_{{солі}} = LP - RP = {bal2['salt_delta']:.2e}$ г/год
            """
        )
    with col_bal2:
        if bal2["is_balance_ok"]:
            st.markdown('<div class="badge-ok">✅ BBB is OK! Двоступеневий баланс дотримано.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-warn">⚠️ Відхилення матеріального балансу!</div>', unsafe_allow_html=True)
        
        if abs(bal2["cv_ci_diff"]) < 1.0:
            st.markdown('<div class="badge-ok" style="margin-top:8px;">🎯 Умова Cv1 ≈ Ci повністю виконана!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-warn" style="margin-top:8px;">ℹ️ Скористайтесь кнопкою автопідбору Xr2 для виконання Cv1 == Ci</div>', unsafe_allow_html=True)

    st.markdown("---")

    tab_diag2, tab_table2, tab_stages2 = st.tabs(["📊 Діаграма потоків ЗОУ-2", "📋 Детальна таблиця потоків", "🔍 Порівняння ступенів 1 і 2"])

    with tab_diag2:
        sankey_fig2 = make_sankey_two_stage(fl2, co2)
        st.plotly_chart(sankey_fig2, use_container_width=True)

    with tab_table2:
        df_streams2 = pd.DataFrame([
            {"Ступінь": "Вхід", "Потік": "Вихідна вода", "Позначення": "Gii", "Витрата G, м³/год": fl2["Gii"], "Солевміст C, мг/л": co2["Ci"], "Солі, кг/год": fl2["Gii"] * co2["Ci"] / 1000},
            {"Ступінь": "1", "Потік": "Вхід на мембрану 1", "Позначення": "Gv", "Витрата G, м³/год": fl2["Gv"], "Солевміст C, мг/л": co2["Cv1"], "Солі, кг/год": fl2["Gv"] * co2["Cv1"] / 1000},
            {"Ступінь": "1", "Потік": "Пермеат 1", "Позначення": "Gp1", "Витрата G, м³/год": fl2["Gp1"], "Солевміст C, мг/л": co2["Cp1"], "Солі, кг/год": fl2["Gp1"] * co2["Cp1"] / 1000},
            {"Ступінь": "1", "Потік": "Концентрат 1", "Позначення": "Gk1", "Витрата G, м³/год": fl2["Gk1"], "Солевміст C, мг/л": co2["Ck1"], "Солі, кг/год": fl2["Gk1"] * co2["Ck1"] / 1000},
            {"Ступінь": "1", "Потік": "Рецикл 1", "Позначення": "Gr1", "Витрата G, м³/год": fl2["Gr1"], "Солевміст C, мг/л": co2["Ck1"], "Солі, кг/год": fl2["Gr1"] * co2["Ck1"] / 1000},
            {"Ступінь": "1", "Потік": "Скид 1", "Позначення": "Gs1", "Витрата G, м³/год": fl2["Gs1"], "Солевміст C, мг/л": co2["Ck1"], "Солі, кг/год": fl2["Gs1"] * co2["Ck1"] / 1000},
            {"Ступінь": "2", "Потік": "Пермеат 2 (кінцевий)", "Позначення": "Gp2", "Витрата G, м³/год": fl2["Gp2"], "Солевміст C, мг/л": co2["Cp2"], "Солі, кг/год": fl2["Gp2"] * co2["Cp2"] / 1000},
            {"Ступінь": "2", "Потік": "Концентрат 2", "Позначення": "Gk2", "Витрата G, м³/год": fl2["Gk2"], "Солевміст C, мг/л": co2["Ck2"], "Солі, кг/год": fl2["Gk2"] * co2["Ck2"] / 1000},
            {"Ступінь": "2", "Потік": "Рецикл 2", "Позначення": "Gr2", "Витрата G, м³/год": fl2["Gr2"], "Солевміст C, мг/л": co2["Ck2"], "Солі, кг/год": fl2["Gr2"] * co2["Ck2"] / 1000},
            {"Ступінь": "2", "Потік": "Скид 2", "Позначення": "Gs2", "Витрата G, м³/год": fl2["Gs2"], "Солевміст C, мг/л": co2["Ck2"], "Солі, кг/год": fl2["Gs2"] * co2["Ck2"] / 1000},
            {"Ступінь": "Змішувач", "Потік": "Сумарний рециркулят", "Позначення": "Grr", "Витрата G, м³/год": fl2["Grr"], "Солевміст C, мг/л": co2["Ckk"], "Солі, кг/год": fl2["Grr"] * co2["Ckk"] / 1000},
        ])
        st.dataframe(df_streams2.style.format({
            "Витрата G, м³/год": "{:.2f}",
            "Солевміст C, мг/л": "{:.3f}",
            "Солі, кг/год": "{:.4f}",
        }), use_container_width=True)

        csv2 = df_streams2.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Завантажити результати ЗОУ-2 у CSV",
            data=csv2,
            file_name="ro_two_stage_results.csv",
            mime="text/csv",
        )

    with tab_stages2:
        col_st1, col_st2 = st.columns(2)
        with col_st1:
            st.markdown("#### 🔹 1-й ступінь (Попереднє знесолення)")
            st.write(f"- Подача на модуль: **{fl2['Gv']:.1f}** м³/год ({co2['Cv1']:.2f} мг/л)")
            st.write(f"- Вихід пермеату: **{fl2['Gp1']:.1f}** м³/год ({co2['Cp1']:.2f} мг/л)")
            st.write(f"- Концентрат: **{fl2['Gk1']:.1f}** м³/год ({co2['Ck1']:.2f} мг/л)")
            st.write(f"- Зниження солевмісту: **{(1 - co2['Cp1']/co2['Cv1'])*100:.2f}%**")
        with col_st2:
            st.markdown("#### 🔹 2-й ступінь (Глибоке знесолення)")
            st.write(f"- Живлення пермеатом 1: **{fl2['Gp1']:.1f}** м³/год ({co2['Cp1']:.2f} мг/л)")
            st.write(f"- Кінцевий надчистий пермеат: **{fl2['Gp2']:.2f}** м³/год (**{co2['Cp2']:.4f}** мг/л)")
            st.write(f"- Концентрат 2: **{fl2['Gk2']:.2f}** м³/год ({co2['Ck2']:.2f} мг/л)")
            st.write(f"- Загальне очищення від джерела: **{(1 - co2['Cp2']/co2['Ci'])*100:.4f}%**")


# =====================================================================
# РЕЖИМ 3: ПАРАМЕТРИЧНИЙ АНАЛІЗ ТА ГРАФІКИ
# =====================================================================
elif mode == "Параметричний аналіз (графіки)":
    st.subheader("📈 Дослідження впливу технологічних параметрів")
    
    analysis_type = st.selectbox(
        "Оберіть змінний параметр:",
        [
            "Вплив коефіцієнта рециркуляції (Xr) на солевміст і витрати",
            "Вплив коефіцієнта виходу пермеату (Xp) на якість води",
            "Вплив селективності мембрани (Xz) на залишковий солевміст",
        ]
    )

    if analysis_type == "Вплив коефіцієнта рециркуляції (Xr) на солевміст і витрати":
        xr_vals = np.linspace(0.0, 0.92, 50)
        data = []
        for x_val in xr_vals:
            r = calculate_single_stage(Ci=Ci, Gv=Gv, Xz=Xz, Xp=Xp, Xr=float(x_val))
            data.append({
                "Xr": x_val,
                "Cv (Вхід на мембрану)": r["concentrations"]["Cv"],
                "Ck (Концентрат)": r["concentrations"]["Ck"],
                "Cp (Пермеат)": r["concentrations"]["Cp"],
                "Gi (Забір води)": r["flows"]["Gi"],
                "Gs (Скид концентрату)": r["flows"]["Gs"],
                "Вихід пермеату, %": r["kpi"]["recovery_system"],
            })
        df_p = pd.DataFrame(data)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig1 = px.line(
                df_p, x="Xr", y=["Cv (Вхід на мембрану)", "Ck (Концентрат)", "Cp (Пермеат)"],
                title="Залежність солевмісту від частки рециркуляту (Xr)",
                labels={"Xr": "Коефіцієнт рециркуляції Xr", "value": "Концентрація солей, мг/л"},
            )
            st.plotly_chart(fig1, use_container_width=True)
        with col_g2:
            fig2 = px.line(
                df_p, x="Xr", y=["Gi (Забір води)", "Gs (Скид концентрату)"],
                title="Залежність витрати забору та скиду від Xr",
                labels={"Xr": "Коефіцієнт рециркуляції Xr", "value": "Витрата, м³/год"},
            )
            fig2.add_trace(go.Scatter(x=df_p["Xr"], y=df_p["Вихід пермеату, %"], yaxis="y2", name="Загальний вихід пермеату, %", line=dict(dash="dot", color="green")))
            fig2.update_layout(
                yaxis2=dict(title="Вихід пермеату, %", overlaying="y", side="right")
            )
            st.plotly_chart(fig2, use_container_width=True)

    elif analysis_type == "Вплив коефіцієнта виходу пермеату (Xp) на якість води":
        xp_vals = np.linspace(0.05, 0.50, 40)
        data = []
        for p_val in xp_vals:
            r = calculate_single_stage(Ci=Ci, Gv=Gv, Xz=Xz, Xp=float(p_val), Xr=Xr)
            data.append({
                "Xp": p_val,
                "Gp (Пермеат, м³/год)": r["flows"]["Gp"],
                "Cp (Солевміст пермеату, мг/л)": r["concentrations"]["Cp"],
                "Ck (Солевміст концентрату, мг/л)": r["concentrations"]["Ck"],
                "Cv (Вхідний солевміст мембрани, мг/л)": r["concentrations"]["Cv"],
            })
        df_xp = pd.DataFrame(data)

        fig_xp = px.line(
            df_xp, x="Xp", y=["Cp (Солевміст пермеату, мг/л)", "Cv (Вхідний солевміст мембрани, мг/л)", "Ck (Солевміст концентрату, мг/л)"],
            title="Зміна концентрацій зі збільшенням навантаження на пермеат (Xp)",
            labels={"Xp": "Вихід пермеату на модуль Xp (частка)", "value": "Концентрація, мг/л"},
        )
        st.plotly_chart(fig_xp, use_container_width=True)

    elif analysis_type == "Вплив селективності мембрани (Xz) на залишковий солевміст":
        xz_vals = np.linspace(0.85, 0.999, 50)
        data = []
        for z_val in xz_vals:
            r = calculate_single_stage(Ci=Ci, Gv=Gv, Xz=float(z_val), Xp=Xp, Xr=Xr)
            data.append({
                "Xz": z_val * 100,
                "Cp (Пермеат, мг/л)": r["concentrations"]["Cp"],
                "Rejection (%)": r["kpi"]["rejection_system"],
            })
        df_xz = pd.DataFrame(data)

        fig_xz = px.line(
            df_xz, x="Xz", y="Cp (Пермеат, мг/л)",
            title="Залежність солевмісту пермеату від селективності мембрани Xz",
            labels={"Xz": "Селективність Xz (%)", "Cp (Пермеат, мг/л)": "Солевміст пермеату, мг/л"},
        )
        st.plotly_chart(fig_xz, use_container_width=True)


# =====================================================================
# РЕЖИМ 4: ТЕОРЕТИЧНІ ОСНОВИ, ФОРМУЛИ ТА P&ID
# =====================================================================
elif mode == "Теоретичні основи & формули":
    st.subheader("📚 Математична модель та принципові технологічні схеми")

    st.markdown(
        """
        ### 1. Елементи розрахункової схеми (з коду Colab)
        
        #### А) Змішувач (`zmishyvach`)
        Об'єднує потік свіжої вихідної води з джерела ($G_i, C_i$) та потік рециркуляту концентрату ($G_r, C_k$):
        $$G_i = G_v - G_r$$
        $$C_v = \\frac{G_r \\cdot C_k + G_i \\cdot C_i}{G_v}$$

        #### Б) Розділювач концентрату (`rozdiluvach`)
        Розподіляє скидний потік концентрату з мембрани ($G_k$) на лінію рециркуляції та лінію дренажу:
        $$G_r = G_k \\cdot X_r$$
        $$G_s = G_k - G_r$$

        #### В) Мембранний блок (`membrana`)
        Моделює розділення на зворотноосмотичній мембрані:
        $$G_p = G_v \\cdot X_p$$
        $$G_k = G_v - G_p$$
        $$C_p = C_v \\cdot (1 - X_z)$$
        $$C_k = \\frac{G_v \\cdot C_v - G_p \\cdot C_p}{G_k}$$

        #### Г) Змішувач рециркулятів для 2-го ступеня (`zmish2`)
        Об'єднує повернення з першого та другого ступенів:
        $$G_r = G_{r1} + G_{r2}$$
        $$C_k = \\frac{G_{r1} \\cdot C_{k1} + G_{r2} \\cdot C_{k2}}{G_r}$$
        """
    )

    st.markdown("---")
    st.subheader("2. Принципова схема одноступеневої установки (ЗО-1)")
    st.markdown(
        """
        ```mermaid
        flowchart LR
            In["Вихідна вода (Gi, Ci)"] --> Mix(("Змішувач"))
            Mix -->|"Gv, Cv"| Mem["Мембранний модуль"]
            Mem -->|"Gp, Cp"| Perm["Пермеат (очищена вода)"]
            Mem -->|"Gk, Ck"| Split{"Розділювач"}
            Split -->|"Gs, Ck"| Drain["Скид у дренаж"]
            Split -->|"Gr, Ck (рецикл)"| Mix
        ```
        """
    )

    st.markdown("---")
    st.subheader("3. Принципова схема двоступеневої установки (ЗОУ-2)")
    st.markdown(
        """
        ```mermaid
        flowchart LR
            In["Вихідна вода (Gii, Ci)"] --> Mix1(("Змішувач 1"))
            Mix1 -->|"Gv, Cv1"| Mem1["Ступінь 1"]
            Mem1 -->|"Gp1, Cp1"| Mem2["Ступінь 2"]
            Mem1 -->|"Gk1, Ck1"| Split1{"Розділювач 1"}
            Split1 -->|"Gs1, Ck1"| Drain1["Скид 1"]
            Split1 -->|"Gr1, Ck1"| MixRec(("Змішувач рециклу"))

            Mem2 -->|"Gp2, Cp2"| Perm2["Ультрачистий пермеат"]
            Mem2 -->|"Gk2, Ck2"| Split2{"Розділювач 2"}
            Split2 -->|"Gs2, Ck2"| Drain2["Скид 2"]
            Split2 -->|"Gr2, Ck2"| MixRec

            MixRec -->|"Grr, Ckk"| Mix1
        ```
        """
    )

    st.markdown("---")
    st.markdown(
        """
        ### 4. Критерій перевірки матеріального балансу
        За законом збереження маси, для сталої роботи системи сумарний вхід речовини дорівнює виходу:
        - **Одноступенева:** $G_i \\cdot C_i = G_p \\cdot C_p + G_s \\cdot C_k$
        - **Двоступенева:** $G_{ii} \\cdot C_i = G_{p2} \\cdot C_{p2} + G_{s1} \\cdot C_{k1} + G_{s2} \\cdot C_{k2}$
        
        У Colab перевірка позначена як `BBB is OK` за умови $|\\Delta| \\le 10^{-4}$.
        """
    )
