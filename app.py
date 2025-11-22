# app.py – VERSIÓN DEFINITIVA ÉPICA – Leonardo Sola 2025
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
from io import BytesIO

# PDF
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.linecharts import HorizontalLineChart

# ===================== FORMATO ARGENTINO =====================
def peso(valor):
    return f"$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# ===================== CONFIG =====================
st.set_page_config(page_title="Calculadora Créditos Argentina - Leonardo Sola", page_icon="Argentina", layout="centered")
st.markdown("<h1 style='text-align: center; color: #1e40af;'>Calculadora de Créditos ARGENTINA 2025</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:1.2rem;'><strong>Francés • Alemán • UVA • 100% argentino</strong></p>", unsafe_allow_html=True)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("Parámetros del Crédito")

    # === MONTO ===
    st.markdown("**Monto del préstamo**")
    monto_raw = st.number_input(
        "Ingresá el monto (sin puntos ni signos)",
        min_value=100_000,
        value=50_000_000,
        step=100_000,
        format="%d",
        label_visibility="collapsed"
    )
    st.markdown(
        f"<p style='font-size:1rem; font-weight:bold; color:#666; margin-top:-5px; text-align:center;'>{peso(monto_raw)}</p>",
        unsafe_allow_html=True
    )

    # === PLAZO ===
    plazo = st.number_input("Plazo", min_value=1, value=30, step=1)
    unidad = st.selectbox("Unidad", ["Años", "Meses"], index=0)
    plazo_meses = plazo * 12 if unidad == "Años" else plazo
    st.markdown(
        f"<p style='font-size:1rem; color:#666; margin-top:-5px; text-align:center;'><strong>{plazo} {unidad.lower()}</strong> → {plazo_meses} meses totales</p>",
        unsafe_allow_html=True
    )

    # === TASA ===
    tna = st.slider("Tasa Nominal Anual (TNA %)", 0.1, 200.0, 48.0, 0.1)
    tem = tna / 100 / 12

    # === SISTEMA ===
    sistema = st.selectbox("Sistema de amortización", [
        "Francés (cuota fija en pesos)",
        "Francés UVA",
        "Alemán (abono capital fijo)"
    ], index=1)

    # === UVA (solo si corresponde) ===
    if "UVA" in sistema:
        st.markdown("---")
        st.subheader("Parámetros UVA")
        uva_hoy = st.number_input("Valor UVA hoy (BCRA)", value=1042.35, step=0.01, format="%.2f")
        infla_porcentaje = st.slider("Inflación mensual proyectada (%)", 0.0, 10.0, 3.5, 0.1)
        infla = infla_porcentaje / 100
        st.markdown(
            f"<p style='font-size:0.95rem; color:#666;'>Inflación anual estimada: <strong>{((1 + infla)**12 - 1)*100:.1f}%</strong></p>",
            unsafe_allow_html=True
        )
    else:
        uva_hoy = infla = None

    # === BOTÓN CALCULAR ===
    st.markdown("---")
    if st.button("CALCULAR TABLA COMPLETA", type="primary", use_container_width=True):
        st.session_state.calculado = True
        st.session_state.monto = monto_raw


# ===================== CÁLCULO =====================
def calcular_amortizacion(monto, plazo_meses, tem, sistema, uva_hoy=None, infla=None):
    tabla = []
    saldo = monto
    total_pagado = 0
    total_interes = 0

    # Francés UVA
    if sistema == "Francés UVA":
        monto_uva = monto / uva_hoy
        cuota_uva = monto_uva * tem * (1 + tem)**plazo_meses / ((1 + tem)**plazo_meses - 1)
        saldo_uva = monto_uva

    # Francés en pesos
    elif sistema == "Francés (cuota fija en pesos)":
        cuota_fija = monto * tem * (1 + tem)**plazo_meses / ((1 + tem)**plazo_meses - 1)

    for mes in range(1, plazo_meses + 1):

        # === FRANCÉS UVA ===
        if sistema == "Francés UVA":
            uva_actual = uva_hoy * (1 + infla)**(mes - 1)
            cuota = cuota_uva * uva_actual
            interes = saldo_uva * tem * uva_actual
            abono_uva = cuota_uva - saldo_uva * tem
            saldo_uva -= abono_uva
            abono = abono_uva * uva_actual
            saldo = saldo_uva * uva_actual

        # === FRANCÉS PESOS ===
        elif sistema == "Francés (cuota fija en pesos)":
            interes = saldo * tem
            abono = cuota_fija - interes
            cuota = cuota_fija
            saldo -= abono

        # === ALEMÁN ===
        else:
            abono = monto / plazo_meses
            interes = saldo * tem
            cuota = abono + interes
            saldo -= abono

        total_pagado += cuota
        total_interes += interes

        tabla.append({
            "Mes": mes,
            "Cuota": round(cuota),
            "Interés": round(interes),
            "Abono": round(abono),
            "Saldo": round(max(saldo, 0))
        })

        if saldo <= 0:
            break

    return pd.DataFrame(tabla), total_pagado, total_interes


# ===================== PDF CON GRÁFICO =====================
def crear_pdf(df, total_pagado, total_interes, sistema, tna, plazo_meses, monto):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=20*mm, bottomMargin=18*mm)
    elements = []
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Center", alignment=TA_CENTER,
            fontSize=18, textColor=colors.HexColor("#1e40af"), spaceAfter=12
        )
    )

    # ===== TITULO =====
    elements.append(Paragraph("Tabla de Amortización - ARGENTINA 2025", styles["Center"]))
    elements.append(Paragraph(f"Generado el {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["Normal"]))
    elements.append(
        Paragraph(
            f"Sistema: {sistema} • TNA: {tna}% • Plazo: {plazo_meses} meses • Monto: {peso(monto)}",
            styles["Normal"]
        )
    )
    elements.append(Spacer(1, 12*mm))

    # ===== GRAFICO MATPLOTLIB (MISMO QUE STREAMLIT) =====
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from reportlab.platypus import Image

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
    ax.plot(df["Mes"], df["Cuota"], label="Cuota Mensual", linewidth=3.5)
    ax.plot(df["Mes"], df["Saldo"], label="Saldo de Capital", linewidth=3.5)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: peso(x)))

    step = 12 if plazo_meses > 60 else 6
    ticks = list(range(1, plazo_meses + 1, step))
    labels = [f"Año {int((m-1)/12)}" if m > 1 else "Mes 1" for m in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)

    ax.set_title("Evolución de Cuota y Saldo de Capital", fontsize=14)
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Pesos Argentinos ($)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', dpi=150)
    img_buffer.seek(0)
    plt.close(fig)

    elements.append(Image(img_buffer, width=250*mm, height=90*mm))
    elements.append(Spacer(1, 10*mm))

    # ===== TABLA =====
    data = [["Mes", "Cuota", "Interés", "Abono Capital", "Saldo"]]
    for _, r in df.iterrows():
        data.append([str(r["Mes"]), peso(r["Cuota"]), peso(r["Interés"]), peso(r["Abono"]), peso(r["Saldo"])])
    data.append(["TOTAL", peso(total_pagado), peso(total_interes), "", ""])

    table = Table(data, colWidths=[30*mm, 55*mm, 55*mm, 55*mm, 55*mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e40af")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9.5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#dbeafe")),
    ]))
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ===================== RESULTADOS =====================
if st.session_state.get("calculado", False):
    df, total_pagado, total_interes = calcular_amortizacion(
        st.session_state.monto, plazo_meses, tem, sistema, uva_hoy, infla
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total a pagar", peso(total_pagado))
    col2.metric("Intereses totales", peso(total_interes))
    col3.metric("Costo financiero", f"{(total_interes/st.session_state.monto*100):.1f}%")

    st.divider()
    st.subheader("Tabla de amortización")

    # ==== FIX PARA OVERFLOW: convertir columnas a texto antes de mostrar ====
    df_display = df.copy()
    for col in ["Cuota", "Interés", "Abono", "Saldo"]:
        df_display[col] = df_display[col].apply(peso)

    st.dataframe(df_display, use_container_width=True)


    # ===================== GRÁFICO =====================
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    ax.plot(df["Mes"], df["Cuota"], label="Cuota Mensual", color="#1d4ed8", linewidth=3.5)
    ax.plot(df["Mes"], df["Saldo"], label="Saldo de Capital", color="#dc2626", linewidth=3.5)

    # Formato peso en eje Y
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: peso(x)))

    ax.set_title("Evolución de Cuota y Saldo de Capital", fontsize=18, fontweight="bold", color="#1e40af", pad=20)
    ax.set_xlabel("Eje X - Tiempo en meses", fontsize=12, fontweight="bold")
    ax.set_ylabel("Eje Y - Pesos Argentinos ($)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=12, loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    # === DESCARGAS ===
    excel_buffer = BytesIO()
    df_export = df.copy()
    for col in ["Cuota", "Interés", "Abono", "Saldo"]:
        df_export[col] = df_export[col].map(peso)
    df_export.to_excel(excel_buffer, index=False, engine='openpyxl')
    excel_buffer.seek(0)

    pdf_buffer = crear_pdf(df, total_pagado, total_interes, sistema, tna, plazo_meses, st.session_state.monto)

    csv_df = df.copy()
    for col in ["Cuota", "Interés", "Abono", "Saldo"]:
        csv_df[col] = csv_df[col].map(peso)
    csv_bytes = csv_df.to_csv(index=False).encode('utf-8-sig')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("Descargar Excel", excel_buffer, "amortizacion_credito.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with col2:
        st.download_button("Descargar PDF (con gráfico)", pdf_buffer, "amortizacion_credito.pdf", "application/pdf")
    with col3:
        st.download_button("Descargar CSV", csv_bytes, "amortizacion.csv", "text/csv")

    st.success("¡Impecable! Ya podés descargar esta información si te fue útil.")

# ===================== FOOTER =====================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; padding: 20px;">
        <p style="font-size:1.1rem;">Creado por <b>Leonardo Sola</b></p>
        <a href="https://github.com/LeoSola12" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" width="40" style="margin:10px; background:white; border-radius:8px; padding:5px">
        </a>
        <a href="https://www.instagram.com/leeeeeeeo_/" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/2111/2111463.png" width="40" style="margin:10px; background:white; border-radius:8px; padding:5px">
        </a>
        <a href="https://x.com/LeoSola7" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/5968/5968830.png" width="40" style="margin:10px; background:white; border-radius:8px; padding:5px">
    </div>
    """,
    unsafe_allow_html=True
)
