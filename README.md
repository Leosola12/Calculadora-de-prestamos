🇦🇷 CALCULADORA DE PRESTAMOS – ARGENTINA 2025

Aplicación web desarrollada con Python + Streamlit para simular préstamos hipotecarios y personales con los sistemas más utilizados en Argentina.  
Permite comparar cuotas, intereses, saldos, descargar reportes y visualizar la evolución completa del crédito.

--------------------------------------------------
🧮 SISTEMAS DE AMORTIZACIÓN INCLUIDOS
--------------------------------------------------
- Sistema Francés (cuota fija en pesos)
- Sistema Francés UVA (cuota fija en UVA - bajo proyección de inflación mensual como imput/supuesto)
- Sistema Alemán (amortización fija de capital)

--------------------------------------------------
⭐ CARACTERÍSTICAS PRINCIPALES
--------------------------------------------------
- Muestra:
  * Tabla completa mes a mes
  * Cuota, interés, abono y saldo restante
  * Totales y costo financiero
- Gráfico dinámico con evolución de cuota y saldo
- Descargas disponibles:
  * Excel (.xlsx)
  * CSV
  * PDF con gráfico embebido
- Formato contable argentino ($ 1.234.567,89)

--------------------------------------------------
¿CÓMO FUNCIONA?
--------------------------------------------------
El usuario ingresa:

1) Monto del crédito
2) Plazo (meses o años)
3) TNA / TEM
4) Sistema de amortización
5) Opcional:
   - Valor UVA actual
   - Inflación proyectada mensual

La aplicación calcula:

- Cuota de cada mes
- Interés pagado
- Capital amortizado
- Saldo pendiente
- Totales acumulados

--------------------------------------------------
🛠 TECNOLOGÍAS UTILIZADAS
--------------------------------------------------
- Python 3
- Streamlit
- Pandas
- Matplotlib
- ReportLab (PDF)
- OpenPyXL (Excel)

--------------------------------------------------
📦 ESTRUCTURA DE LA CALCULADORA
--------------------------------------------------

│

├── app.py → Aplicación principal

└── requirements.txt → Dependencias del proyecto

