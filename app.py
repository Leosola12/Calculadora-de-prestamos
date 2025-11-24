# app.py — Calculadora de Créditos ARGENTINA 2025 (v1.2 — TIR en UI)
# Autor: Leonardo Sola (adaptado y mejorado)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
from io import BytesIO
import math

# PDF (matplotlib -> imagen -> ReportLab)
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import mm

# ===================== UTILIDADES =====================
def peso(valor):
    """Formatea número en formato argentino: $ 1.234.567,89"""
    try:
        v = float(valor)
    except Exception:
        return str(valor)
    s = f"{v:,.2f}"
    return "$ " + s.replace(",", "X").replace(".", ",").replace("X", ".")

def round_centavos(x):
    return float(round(float(x) + 1e-12, 2))

def safe_pow1p(x, n):
    """(1+x)^n de forma numericamente estable usando log1p cuando corresponde"""
    try:
        return (1 + x) ** n
    except OverflowError:
        return math.exp(n * math.log1p(x))

def irr_cashflows(cashflows, guess=0.01, maxiter=200, tol=1e-10):
    """
    Newton-Raphson TIR solver (para flujos mensuales).
    cashflows: numpy array (t=0 ... T)
    retorna r (períodica) en decimal (ej 0.0125 para 1.25%)
    """
    rate = guess
    for _ in range(maxiter):
        denom = (1 + rate) ** np.arange(len(cashflows))
        npv = (cashflows / denom).sum()
        deriv = - (np.arange(len(cashflows)) * cashflows / ((1 + rate) ** (np.arange(len(cashflows)) + 1))).sum()
        if deriv == 0:
            break
        new_rate = rate - npv / deriv
        if not np.isfinite(new_rate):
            break
        if abs(new_rate - rate) < tol:
            return new_rate
        rate = new_rate
    return rate

# ===================== CONFIG UI =====================
st.set_page_config(page_title="Calculadora de Créditos ARGENTINA 2025", page_icon="🇦🇷", layout="wide")
st.markdown("<h1 style='text-align:center; color:#1e40af;'>Calculadora de Créditos — ARGENTINA 2025</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#374151;'>Francés • Francés UVA • Alemán — con CFT y comparador</p>", unsafe_allow_html=True)
st.write("---")

# ===================== SIDEBAR / PARAMS =====================
with st.sidebar:
    st.header("Parámetros del Crédito")

    # Monto
    monto_raw = st.number_input("Monto del préstamo (sin puntos)", min_value=1000, value=50_000_000, step=10_000, format="%d")
    st.markdown(f"<div style='text-align:center; font-weight:600; color:#374151'>{peso(monto_raw)}</div>", unsafe_allow_html=True)

    # Plazo
    plazo = st.number_input("Plazo", min_value=1, value=30, step=1)
    unidad = st.selectbox("Unidad", ["Años", "Meses"], index=0)
    plazo_meses = int(plazo * 12) if unidad == "Años" else int(plazo)
    st.markdown(f"<div style='text-align:center; color:#6b7280'>Plazo: <b>{plazo} {unidad.lower()}</b> → {plazo_meses} meses</div>", unsafe_allow_html=True)

    # TNA -> TEM
    tna = st.slider("Tasa Nominal Anual (TNA %)", min_value=0.0, max_value=200.0, value=15.0, step=0.1)
    tem = float(tna) / 100.0 / 12.0

    # Sistema
    sistema = st.selectbox("Sistema de amortización", [
        "Francés (cuota fija en pesos)",
        "Francés UVA (cuota fija en UVA)",
        "Alemán (abono capital fijo)"
    ], index=0)

    # UVA params
    if "UVA" in sistema:
        st.markdown("---")
        st.subheader("Parámetros UVA")
        uva_hoy = st.number_input("Valor UVA hoy (BCRA)", value=1042.35, step=0.01, format="%.2f")
        infla_porcentaje = st.slider("Inflación mensual proyectada (%)", 0.0, 20.0, 3.5, 0.1)
        infla = infla_porcentaje / 100.0
        st.markdown(f"<div style='color:#6b7280'>Inflación anual estimada: <b>{((1 + infla)**12 - 1)*100:.1f}%</b></div>", unsafe_allow_html=True)
    else:
        uva_hoy = None
        infla = None

    st.markdown("---")
    # Gastos y seguros para CFT
    st.subheader("Gastos / Seguros (opcionales)")
    gastos_upfront = st.number_input("Gastos desembolso (ej: comisiones, escritura)", min_value=0.0, value=0.0, step=100.0, format="%.2f")
    seguro_mensual = st.number_input("Seguro mensual (si aplica)", min_value=0.0, value=0.0, step=100.0, format="%.2f")

    st.markdown("---")
    # Modo demo (ejemplo preconfigurado)
    demo_mode = st.checkbox("Mostrar ejemplo precalculado (50M - TNA 15% - 30 años)", value=False)
    if demo_mode:
        st.success("Modo demo activo: se usarán parámetros de ejemplo en la vista principal.")

    st.markdown("---")
    if st.button("CALCULAR TABLA COMPLETA", type="primary", use_container_width=True):
        st.session_state.calculado = True
        st.session_state.monto = float(monto_raw)
        st.session_state.plazo_meses = plazo_meses
        st.session_state.tem = tem
        st.session_state.sistema = sistema
        st.session_state.uva_hoy = uva_hoy
        st.session_state.infla = infla
        st.session_state.gastos_upfront = float(gastos_upfront)
        st.session_state.seguro_mensual = float(seguro_mensual)
        st.session_state.demo_mode = demo_mode

# Inicializar session_state si no existe
if "calculado" not in st.session_state:
    st.session_state.calculado = False

# Si demo_mode activo y no presionaron calcular, seteamos parámetros de demo en variables de trabajo
if st.session_state.get("demo_mode", False) and not st.session_state.get("calculado", False):
    st.session_state.monto = 50_000_000.0
    st.session_state.plazo_meses = 360
    st.session_state.tem = 0.15 / 12.0
    st.session_state.sistema = "Francés (cuota fija en pesos)"
    st.session_state.uva_hoy = None
    st.session_state.infla = None
    st.session_state.gastos_upfront = 0.0
    st.session_state.seguro_mensual = 0.0
    st.session_state.calculado = True  # mostramos demo automáticamente

# ===================== CÁLCULO ROBUSTO =====================
def calcular_amortizacion(monto, plazo_meses, tem, sistema, uva_hoy=None, infla=None, seguro_mensual=0.0, round_money=True):
    """
    Versión robusta:
    - redondea a centavos
    - evita division por 0 si tem==0
    - ajusta la última cuota para dejar saldo 0 (corrección bancaria)
    - para UVA: calcula en UVA y convierte mes a mes
    - devuelve df, total_pagado, total_interes, lista_pagos_mensuales
    """
    monto = float(monto)
    n = int(plazo_meses)
    tabla = []
    total_pagado = 0.0
    total_interes = 0.0

    # validaciones
    if n <= 0:
        raise ValueError("Plazo debe ser mayor a 0")
    if "UVA" in sistema and (uva_hoy is None or uva_hoy <= 0):
        raise ValueError("Para sistema UVA debe proporcionar valor de UVA > 0")

    # cálculo de cuotas base
    if abs(tem) < 1e-15:
        cuota_fija = monto / n
    else:
        # cuota francés en pesos
        factor = safe_pow1p(tem, n)
        cuota_fija = monto * tem * factor / (factor - 1) if (factor - 1) != 0 else monto / n

    # para UVA
    if "UVA" in sistema:
        monto_uva = monto / uva_hoy
        factor_uva = safe_pow1p(tem, n)
        cuota_uva = monto_uva * tem * factor_uva / (factor_uva - 1) if (factor_uva - 1) != 0 else monto_uva / n
        saldo_uva = monto_uva

    saldo = monto

    for mes in range(1, n + 1):
        # acumulamos seguro mensual si aplica
        seguro_mes = float(seguro_mensual)

        if "UVA" in sistema:
            uva_actual = uva_hoy * safe_pow1p(infla, mes - 1) if infla is not None else uva_hoy
            cuota_mes = cuota_uva * uva_actual + seguro_mes
            interes_mes = saldo_uva * tem * uva_actual
            abono_uva = cuota_uva - saldo_uva * tem
            saldo_uva -= abono_uva
            abono_mes = abono_uva * uva_actual
            saldo = max(saldo_uva * uva_actual, 0.0)
        elif "Francés" in sistema:
            interes_mes = saldo * tem
            abono_mes = cuota_fija - interes_mes
            cuota_mes = cuota_fija + seguro_mes
            saldo -= abono_mes
        else:  # Alemán
            abono_mes = monto / n
            interes_mes = saldo * tem
            cuota_mes = abono_mes + interes_mes + seguro_mes
            saldo -= abono_mes

        if round_money:
            cuota_mes = round_centavos(cuota_mes)
            interes_mes = round_centavos(interes_mes)
            abono_mes = round_centavos(abono_mes)
            saldo = round_centavos(saldo)

        total_pagado += cuota_mes
        total_interes += interes_mes

        tabla.append({
            "Mes": mes,
            "Cuota": cuota_mes,
            "Interés": interes_mes,
            "Abono": abono_mes,
            "Saldo": max(saldo, 0.0)
        })

    # ajuste bancario final: si queda saldo residual pequeño, lo ajustamos en la última cuota
    if tabla:
        ultimo = tabla[-1]
        if ultimo["Saldo"] != 0 and abs(ultimo["Saldo"]) < 1.0:
            diff = ultimo["Saldo"]
            ultimo["Abono"] = round_centavos(ultimo["Abono"] + diff)
            ultimo["Cuota"] = round_centavos(ultimo["Cuota"] + diff)
            ultimo["Saldo"] = 0.0
            total_pagado = round_centavos(sum(row["Cuota"] for row in tabla))
            total_interes = round_centavos(sum(row["Interés"] for row in tabla))

    df = pd.DataFrame(tabla)
    pagos_mensuales = df["Cuota"].tolist()
    return df, total_pagado, total_interes, pagos_mensuales

# ===================== CFT =====================
def calcular_cft_simple(monto, total_pagado, gastos_upfront=0.0):
    """CFT aproximado (no anualizado): comparacion total pagado vs desembolso"""
    return ((total_pagado + gastos_upfront) - monto) / monto * 100.0

def calcular_cft_anualizado(monto, pagos_mensuales, gastos_upfront=0.0):
    """
    CFTEA: calcula la tasa periódica r (TIR mensual) que anula el VAN con flujos:
      CF0 = monto - gastos_upfront  (positivo)
      CFt = - (cuota_t + seguros_t)
    Devuelve r_per% (TIR mensual) y CFTEA% (CFTEA anual).
    """
    cf = [monto - gastos_upfront]  # t=0 (positivo)
    for p in pagos_mensuales:
        cf.append(-float(p))
    cf_arr = np.array(cf, dtype=float)
    r_per = irr_cashflows(cf_arr, guess=0.01)
    if not np.isfinite(r_per):
        return None, None
    cft_anual = (1 + r_per) ** 12 - 1
    return r_per * 100.0, cft_anual * 100.0  # porcentajes

# ===================== CREAR PDF (matplotlib -> imagen -> reportlab) =====================
def crear_pdf(df, total_pagado, total_interes, sistema, tna, plazo_meses, monto):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=15*mm, bottomMargin=12*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Center", alignment=TA_CENTER, fontSize=16, textColor=colors.HexColor("#1e40af"), spaceAfter=8))
    elements = []

    elements.append(Paragraph("Tabla de Amortización - ARGENTINA 2025", styles["Center"]))
    elements.append(Paragraph(f"Generado el {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["Normal"]))
    elements.append(Paragraph(f"Sistema: {sistema} • TNA: {tna}% • Plazo: {plazo_meses} meses • Monto: {peso(monto)}", styles["Normal"]))
    elements.append(Spacer(1, 8*mm))

    # Generar gráfico con matplotlib (idéntico al mostrado en Streamlit)
    fig, ax = plt.subplots(figsize=(10, 3.8), facecolor="white", dpi=150)
    ax.plot(df["Mes"], df["Cuota"], label="Cuota Mensual", linewidth=2.6)
    ax.plot(df["Mes"], df["Saldo"], label="Saldo de Capital", linewidth=2.6)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: peso(x)))
    step = 12 if plazo_meses > 60 else 6
    ticks = list(range(1, plazo_meses + 1, step))
    labels = [f"Año {int((m-1)/12)}" if m > 1 else "Mes 1" for m in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_title("Evolución de Cuota y Saldo de Capital", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    imgbuf = BytesIO()
    fig.savefig(imgbuf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    imgbuf.seek(0)

    # Insertar imagen en PDF
    img = Image(imgbuf, width=250*mm, height=85*mm)
    elements.append(img)
    elements.append(Spacer(1, 6*mm))

    # Tabla de amortización (resumida) en PDF
    data = [["Mes", "Cuota", "Interés", "Abono Capital", "Saldo"]]
    for _, r in df.iterrows():
        data.append([str(int(r["Mes"])), peso(r["Cuota"]), peso(r["Interés"]), peso(r["Abono"]), peso(r["Saldo"])])
    data.append(["TOTAL", peso(total_pagado), peso(total_interes), "", ""])
    table = Table(data, colWidths=[25*mm, 50*mm, 50*mm, 50*mm, 50*mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e40af")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#dbeafe")),
    ]))
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ===================== RESULTADOS & UI =====================
if st.session_state.get("calculado", False):

    # lectura de variables desde session_state (o defaults)
    monto = float(st.session_state.get("monto", monto_raw))
    plazo_meses = int(st.session_state.get("plazo_meses", plazo_meses))
    tem = float(st.session_state.get("tem", tem))
    sistema = st.session_state.get("sistema", sistema)
    uva_hoy = st.session_state.get("uva_hoy", uva_hoy)
    infla = st.session_state.get("infla", infla)
    gastos_upfront = float(st.session_state.get("gastos_upfront", 0.0))
    seguro_mensual = float(st.session_state.get("seguro_mensual", 0.0))

    try:
        df, total_pagado, total_interes, pagos_mensuales = calcular_amortizacion(
            monto, plazo_meses, tem, sistema, uva_hoy, infla, seguro_mensual, round_money=True
        )
    except Exception as e:
        st.error(f"Error en cálculo: {e}")
        st.stop()

    # KPI
    años = plazo_meses / 12.0 if plazo_meses > 0 else 1.0

    # Cálculos de CFT / TEA / CFTEA / CFTNA
    cft_simple = calcular_cft_simple(monto, total_pagado, gastos_upfront)
    r_per_pct, cft_anual_pct = calcular_cft_anualizado(monto, pagos_mensuales, gastos_upfront)
    # CFTNA (aprox) — costo financiero total nominal anual simple (aproximación)
    # Definimos CFTNA_simple = CFT_simple / años (porcentaje anual nominal sin capitalización)
    cftna_simple = cft_simple / años

    # TEA (tasa efectiva anual pura a partir de TEM)
    tea_calc = (1 + tem) ** 12 - 1

    # Mostrar KPIs con tooltips (help)
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    col1.metric("Total a pagar", peso(total_pagado))
    col2.metric("Intereses totales", peso(total_interes))
    col3.metric(
        "CFT aproximado (no anualizado) ",
        f"{cft_simple:.2f}%",
        help="Cálculo simple: (Total pagado + gastos - monto) / monto. Es una aproximación que NO capitaliza ni considera el calendario de pagos."
    )
    col4.metric(
        "TIR → CFTEA (efectiva anual) ",
        f"{cft_anual_pct:.2f}%" if r_per_pct is not None else "—",
        help="CFTEA: calculada a partir de la TIR (Tasa Interna de Retorno) mensual, obtenida resolviendo el VAN = 0 con los flujos del préstamo (desembolso y pagos)."
    )

    # Mostrar TEA y CFTNA en línea debajo
    st.markdown(
        f"**TEA (efectiva anual, desde TEM):** {tea_calc*100:.3f}%  •  **CFTNA (nominal anual aprox):** {cftna_simple:.2f}%"
    )
    st.markdown("*(Pasa el cursor sobre los indicadores con (?) para ver definiciones rápidas)*")

    st.divider()
    st.subheader("Tabla de amortización")

    # FIX OVERFLOW: convertir a strings antes de mostrar (evita pyarrow OverflowError)
    df_display = df.copy()
    for col in ["Cuota", "Interés", "Abono", "Saldo"]:
        df_display[col] = df_display[col].map(peso)

    st.dataframe(df_display, use_container_width=True)

    # GRÁFICO (responsive)
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="white", dpi=120)
    ax.plot(df["Mes"], df["Cuota"], label="Cuota Mensual", color="#1d4ed8", linewidth=2.8)
    ax.plot(df["Mes"], df["Saldo"], label="Saldo de Capital", color="#dc2626", linewidth=2.8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: peso(x)))
    step = 12 if plazo_meses > 60 else 6
    ticks = list(range(1, plazo_meses + 1, step))
    labels = [f"Año {int((m-1)/12)}" if m > 1 else "Mes 1" for m in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_title("Evolución de Cuota y Saldo de Capital", fontsize=12, color="#1e40af")
    ax.set_xlabel("Eje X - Tiempo en meses/años")
    ax.set_ylabel("Eje Y - Pesos ($)")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # DESCARGAS
    # Excel
    excel_buffer = BytesIO()
    df_export = df.copy()
    for col in ["Cuota", "Interés", "Abono", "Saldo"]:
        df_export[col] = df_export[col].map(lambda x: f"{x:.2f}")
    df_export.to_excel(excel_buffer, index=False, engine='openpyxl')
    excel_buffer.seek(0)

    # PDF
    pdf_buffer = crear_pdf(df, total_pagado, total_interes, sistema, tna, plazo_meses, monto)

    # CSV (con formato peso)
    csv_df = df.copy()
    for col in ["Cuota", "Interés", "Abono", "Saldo"]:
        csv_df[col] = csv_df[col].map(peso)
    csv_bytes = csv_df.to_csv(index=False).encode('utf-8-sig')

    colx1, colx2, colx3 = st.columns(3)
    with colx1:
        st.download_button("Descargar Excel (.xlsx)", excel_buffer, "amortizacion_credito.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with colx2:
        st.download_button("Descargar PDF (con gráfico)", pdf_buffer, "amortizacion_credito.pdf", "application/pdf")
    with colx3:
        st.download_button("Descargar CSV", csv_bytes, "amortizacion.csv", "text/csv")

    st.success("¡Listo! Podés descargar los resultados.")
    st.divider()

    # ===================== PANEL DIDACTICO: FÓRMULAS Y VALORES =====================
    with st.expander("🔢 Mostrar fórmulas y cálculo numérico (TEA / CFT / CFTNA / CFTEA)", expanded=False):
        st.markdown("## Definiciones rápidas")
        st.markdown("- **TEA**: Tasa Efectiva Anual (interés puro, sin incluir gastos).")
        st.markdown("- **CFT (aprox)**: Indicador simple que compara el total pagado con el desembolso; **no** anualizado ni capitalizado (útil para una visión rápida).")
        st.markdown("- **CFTNA**: Costo Financiero Total Nominal Anual (aproximación nominal anual). Se usa como indicador nominal para comparar, pero **no** reemplaza a CFTEA.")
        st.markdown("- **CFTEA (recomendado)**: Costo Financiero Total Efectivo Anual. Es la tasa efectiva anual que iguala el desembolso neto y todos los pagos (se calcula a partir de la TIR mensual).")

        st.markdown("### TEA (desde TEM)")
        st.latex(r"\mathrm{TEA} = (1+i)^{12} - 1")
        st.markdown(f"- Con i (mensual) = {tem:.6f} → TEA = **{tea_calc*100:.3f}%**")

        st.markdown("### CFT (aproximado, no anualizado)")
        st.latex(r"\mathrm{CFT} \approx \frac{Total\ pagado - Desembolso}{Desembolso}")
        st.markdown(f"- Total pagado = {peso(total_pagado)} • Desembolso (monto) = {peso(monto)}")
        st.markdown(f"- CFT (aprox) = **{cft_simple:.2f}%**")

        st.markdown("### CFTNA (Costo Financiero Total Nominal Anual — aproximación)")
        st.latex(r"\mathrm{CFTNA_{approx}} \approx \frac{CFT_{aprox}}{A\tilde{n}os} \quad (\text{porcentaje nominal anual, sin capitalizar})")
        st.markdown(f"- Período: {años:.2f} años → CFTNA aprox = **{cftna_simple:.2f}%**")

        st.markdown("### TIR (mensual) y CFTEA (Costo Financiero Total Efectivo Anual)")
        st.markdown("""
        La **TIR (Tasa Interna de Retorno)** es la tasa periódica que hace que el Valor Actual Neto (VAN) del préstamo sea igual a cero.""")

        st.latex(r"""
        \sum_{t=0}^{n} \frac{CF_t}{(1+r)^t} = 0
        """)

        """
        Donde:
        - CF₀ = Monto - Gastos_de_desembolso (entrada positiva)
        - CFt = - (cuota_t + seguro_t) (egresos mensuales negativos)

        Luego:
        """
        st.latex(r"\mathrm{CFTEA} = (1 + r_{mensual})^{12} - 1")

        if r_per_pct is not None:
            st.markdown(f"- **TIR (mensual implícita):** {r_per_pct:.4f}%")
            st.markdown(f"- **CFTEA (efectiva anual):** {cft_anual_pct:.2f}%")
        else:
            st.warning("No se pudo calcular CFTEA (instabilidad numérica).")

        st.markdown("---")
        st.markdown("**Nota**: para que el CFTEA refleje el costo real que el cliente enfrenta, incluimos en CF₀ los *gastos de desembolso* y sumamos *seguros mensuales* a cada cuota antes de calcular la TIR.")

    # ===================== COMPARADOR ENTRE SISTEMAS =====================
    with st.expander("📊 Comparar sistemas (Francés / Francés UVA / Alemán)", expanded=False):
        rows = []
        sistemas_a_probar = [
            "Francés (cuota fija en pesos)",
            "Francés UVA (cuota fija en UVA)",
            "Alemán (abono capital fijo)"
        ]
        for s in sistemas_a_probar:
            try:
                df_s, tp_s, ti_s, pagos_s = calcular_amortizacion(monto, plazo_meses, tem, s, uva_hoy, infla, seguro_mensual, round_money=True)
                cft_s = calcular_cft_simple(monto, tp_s, gastos_upfront)
                rows.append({
                    "Sistema": s,
                    "Cuota inicial": peso(df_s.loc[0, "Cuota"]) if not df_s.empty else "-",
                    "Total pagado": peso(tp_s),
                    "Intereses totales": peso(ti_s),
                    "CFT aprox": f"{cft_s:.2f}%"
                })
            except Exception as e:
                rows.append({"Sistema": s, "Cuota inicial": "—", "Total pagado": "—", "Intereses totales": "—", "CFT aprox": "—"})
        st.table(pd.DataFrame(rows))

    # ===================== EJEMPLO PRECALCULADO (opcional) =====================
    if st.session_state.get("demo_mode", False):
        st.info("Ejemplo precalculado: Monto $50.000.000 | TNA 15% | 30 años (Francés)")
        meses_clave = [1, 12, 60, plazo_meses]
        st.table(df[df["Mes"].isin(meses_clave)].set_index("Mes"))

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
