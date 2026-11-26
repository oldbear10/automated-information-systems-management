"""
Рисунки статті в єдиному стилі (Times New Roman, відтінки сірого) і Excel-файл
з даними та редагованими діаграмами (вимога редакції: діаграми мають бути
доступні для редагування, наприклад в Excel).

Запуск (після run_experiment.py, бо Рис. 3 бере дані з results/summary.csv):
    python make_article_figs.py
Результат:
    results/fig2_article.png      — Рис. 2 статті
    results/figures_data.xlsx     — дані й діаграми Excel для рисунків 2–4
"""

import csv
import os

import matplotlib
from matplotlib import font_manager as fm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook
from openpyxl.chart import Reference, ScatterChart, Series

import simulator as sim

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PHI = sim.NOISE_MODELS["ar"][0]

for path in ("/System/Library/Fonts/Supplemental/Times New Roman.ttf",
             "/Library/Fonts/Times New Roman.ttf"):
    if os.path.exists(path):
        fm.fontManager.addfont(path)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.edgecolor": "0.2",
    "grid.color": "0.85",
})


# ---------------------------------------------------------------------------
# Дані рисунків (ті самі реалізації, що в run_experiment.py)
# ---------------------------------------------------------------------------
def fig2_data():
    t_h = np.arange(3 * 24 * 60) / 60
    a = sim.generate_load("shift", seed=3, phi=PHI)
    b = sim.generate_load("varnoise", seed=1, phi=PHI)
    sl = slice(12 * 60, 14 * 60)
    white = sim.generate_load("svc_10", seed=0, phi=0.0)[sl]
    ar = sim.generate_load("svc_10", seed=0, phi=PHI)[sl]
    return t_h, a, b, np.arange(sl.stop - sl.start), white, ar


def fig3_data():
    rows = [r for r in csv.DictReader(open(os.path.join(OUT, "summary.csv"), encoding="utf-8"))
            if r["noise"] == "ar"]
    out = {}
    for sc in ("svc_5", "svc_15", "varnoise", "shift"):
        rs = [r for r in rows if r["scenario_key"] == sc]
        fixed = sorted((r for r in rs if r["kind"] == "fixed"), key=lambda r: float(r["k"]))
        pick = lambda kind: next(r for r in rs if r["kind"] == kind)
        out[sc] = {
            "fixed": [(float(r["k"]), float(r["replica_hours_mean"]), float(r["slo_violation_pct_mean"]))
                      for r in fixed],
            "reactive": (float(pick("reactive")["replica_hours_mean"]),
                         float(pick("reactive")["slo_violation_pct_mean"])),
            "adaptive": (float(pick("adaptive")["replica_hours_mean"]),
                         float(pick("adaptive")["slo_violation_pct_mean"])),
        }
    return out


def fig4_data():
    seed = 1
    load = sim.generate_load("varnoise", seed=seed, phi=PHI)
    sd = sim.noise_level("varnoise", len(load), np.random.default_rng(seed))
    adapt = sim.Adaptive()
    r = sim.simulate(adapt, load)
    sl = slice(24 * 60, 2 * 24 * 60)
    return (np.arange(len(load))[sl] / 60, load[sl], load[sl] / (sim.MU * sim.TARGET_UTIL),
            r["replicas"][sl], 100 * sd[sl], 100 * np.array(adapt.k_history)[sl])


# ---------------------------------------------------------------------------
# Рис. 2 (PNG)
# ---------------------------------------------------------------------------
def plot_fig2():
    t_h, a, b, m, white, ar = fig2_data()
    fig, ax = plt.subplots(3, 1, figsize=(8.3, 9.2))
    ax[0].plot(t_h, a, color="black", lw=0.5)
    ax[0].set_title("а) Добовий цикл, раптові сплески та зміна режиму з другої доби")
    ax[0].set_xlabel("Час, год")
    ax[1].plot(t_h, b, color="black", lw=0.5)
    ax[1].set_title("б) Мінливість навантаження змінюється в часі (2–12 %)")
    ax[1].set_xlabel("Час, год")
    ax[2].plot(m, white, color="0.6", lw=1.0, label="Незалежний шум")
    ax[2].plot(m, ar, color="black", lw=1.2, label="Автокорельований шум AR(1), φ = 0,8")
    ax[2].set_title("в) Фрагмент тривалістю 2 год, сервіс C (мінливість 10 %)")
    ax[2].set_xlabel("Час, хв")
    ax[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    for x in ax:
        x.set_ylabel("Інтенсивність, запитів/с")
        x.grid(True, lw=0.6)
    fig.tight_layout()
    path = os.path.join(OUT, "fig2_article.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Excel з даними й редагованими діаграмами
# ---------------------------------------------------------------------------
def _scatter(ws, title, x_title, y_title, series, anchor, smooth=False, width=24, height=11):
    ch = ScatterChart()
    ch.title = title
    ch.style = 2
    ch.x_axis.title = x_title
    ch.y_axis.title = y_title
    ch.width, ch.height = width, height
    for name, xref, yref, marker in series:
        s = Series(yref, xref, title=name)
        if marker:
            s.marker.symbol = marker
            s.marker.size = 9
            s.graphicalProperties.line.noFill = True
        else:
            s.marker.symbol = "none"
            s.smooth = smooth
        ch.series.append(s)
    ch.x_axis.delete = False
    ch.y_axis.delete = False
    ws.add_chart(ch, anchor)


def make_excel():
    wb = Workbook()
    info = wb.active
    info.title = "Опис"
    for row in [
        ["Дані та редаговані діаграми до статті «Метод самокоригованого запасу ресурсів для прогнозного "
         "масштабування сервісів у розподілених системах» (Смик Д. Д.)"],
        ["Рис2а, Рис2б, Рис2в — приклади навантаження (Рис. 2)"],
        ["Рис3 — компроміс «ресурси — порушення SLO», AR(1)-шум, середні за 30 прогонами (Рис. 3)"],
        ["Рис4 — динаміка навантаження, реплік і запасу k(t) за змінної мінливості (Рис. 4)"],
        ["Джерело даних: програма https://github.com/oldbear10/automated-information-systems-management"],
    ]:
        info.append(row)
    info.column_dimensions["A"].width = 110

    t_h, a, b, m, white, ar = fig2_data()
    for name, title, y in (("Рис2а", "а) Добовий цикл, раптові сплески та зміна режиму", a),
                           ("Рис2б", "б) Мінливість навантаження змінюється в часі", b)):
        ws = wb.create_sheet(name)
        ws.append(["Час, год", "Інтенсивність, запитів/с"])
        for x, v in zip(t_h, y):
            ws.append([round(float(x), 4), round(float(v), 2)])
        n = len(t_h) + 1
        _scatter(ws, title, "Час, год", "Інтенсивність, запитів/с",
                 [("Інтенсивність запитів", Reference(ws, min_col=1, min_row=2, max_row=n),
                   Reference(ws, min_col=2, min_row=2, max_row=n), None)], "D2")
    ws = wb.create_sheet("Рис2в")
    ws.append(["Час, хв", "Незалежний шум, запитів/с", "Автокорельований шум AR(1), запитів/с"])
    for x, w, r in zip(m, white, ar):
        ws.append([int(x), round(float(w), 2), round(float(r), 2)])
    n = len(m) + 1
    xr = Reference(ws, min_col=1, min_row=2, max_row=n)
    _scatter(ws, "в) Фрагмент 2 год, сервіс C (мінливість 10 %)", "Час, хв", "Інтенсивність, запитів/с",
             [("Незалежний шум", xr, Reference(ws, min_col=2, min_row=2, max_row=n), None),
              ("Автокорельований шум AR(1)", xr, Reference(ws, min_col=3, min_row=2, max_row=n), None)], "E2")

    ws = wb.create_sheet("Рис3")
    titles = {"svc_5": "а) Сервіс B, мінливість 5%", "svc_15": "б) Сервіс D, мінливість 15%",
              "varnoise": "в) Мінливість змінюється в часі", "shift": "г) Сплески та зміна режиму"}
    row0 = 1
    for i, (sc, d) in enumerate(fig3_data().items()):
        ws.cell(row0, 1, titles[sc])
        ws.cell(row0 + 1, 1, "Стратегія")
        ws.cell(row0 + 1, 2, "k")
        ws.cell(row0 + 1, 3, "Ресурси, репліко-год")
        ws.cell(row0 + 1, 4, "Порушення SLO, %")
        r = row0 + 2
        first = r
        for k, res, slo in d["fixed"]:
            ws.cell(r, 1, "Прогноз із фіксованим запасом")
            ws.cell(r, 2, k)
            ws.cell(r, 3, round(res, 2))
            ws.cell(r, 4, round(slo, 3))
            r += 1
        last = r - 1
        for label, (res, slo) in (("Реактивне масштабування HPA", d["reactive"]),
                                  ("Запропонований метод", d["adaptive"])):
            ws.cell(r, 1, label)
            ws.cell(r, 3, round(res, 2))
            ws.cell(r, 4, round(slo, 3))
            r += 1
        _scatter(ws, titles[sc], "Ресурси, репліко-год", "Порушення SLO, %", [
            ("Прогноз із фіксованим запасом", Reference(ws, min_col=3, min_row=first, max_row=last),
             Reference(ws, min_col=4, min_row=first, max_row=last), None),
            ("Реактивне масштабування HPA", Reference(ws, min_col=3, min_row=last + 1, max_row=last + 1),
             Reference(ws, min_col=4, min_row=last + 1, max_row=last + 1), "square"),
            ("Запропонований метод", Reference(ws, min_col=3, min_row=last + 2, max_row=last + 2),
             Reference(ws, min_col=4, min_row=last + 2, max_row=last + 2), "star"),
        ], f"F{row0}", width=16, height=8)
        row0 = r + 4 if r + 4 > row0 + 17 else row0 + 17
    ws.column_dimensions["A"].width = 32

    hours, load, need, rep, var, k = fig4_data()
    ws = wb.create_sheet("Рис4")
    ws.append(["Час, год", "Навантаження, запитів/с", "Потрібно реплік (за фактом)",
               "Виділено методом, реплік", "Реальна мінливість, %", "Запас k(t), %"])
    for row in zip(hours, load, need, rep, var, k):
        ws.append([round(float(row[0]), 4), round(float(row[1]), 2), round(float(row[2]), 2),
                   int(row[3]), round(float(row[4]), 2), round(float(row[5]), 2)])
    n = len(hours) + 1
    xr = Reference(ws, min_col=1, min_row=2, max_row=n)
    ref = lambda c: Reference(ws, min_col=c, min_row=2, max_row=n)
    _scatter(ws, "а) Навантаження (мінливість змінюється в часі)", "Час, год", "Навантаження, запитів/с",
             [("Навантаження", xr, ref(2), None)], "H2", height=8)
    _scatter(ws, "б) Кількість реплік", "Час, год", "Кількість реплік",
             [("Потрібно за фактичним навантаженням", xr, ref(3), None),
              ("Виділено запропонованим методом", xr, ref(4), None)], "H19", height=8)
    _scatter(ws, "в) Самокоригований запас і мінливість навантаження", "Час, год", "%",
             [("Реальна мінливість, %", xr, ref(5), None), ("Запас k(t), %", xr, ref(6), None)], "H36", height=8)

    path = os.path.join(OUT, "figures_data.xlsx")
    wb.save(path)
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("Збережено", plot_fig2())
    print("Збережено", make_excel())
