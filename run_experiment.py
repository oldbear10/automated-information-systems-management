"""
Експеримент: порівняння стратегій масштабування.

Експеримент 1 (сервіси A–D): чи працює ОДНА стратегія без ручного налаштування
    для сервісів з різною мінливістю навантаження?
Експеримент 2 (мінливість змінюється в часі): чи кращий самокоригований запас
    за будь-який фіксований, коли поведінка сервісу змінюється?
Експеримент 3 (сплески + зміна режиму): поведінка в складних умовах.
Експеримент 4 (чутливість): як результати залежать від параметрів методу
    W (вікно пам'яті) і q (квантиль).

Кожен сценарій проганяється для двох моделей шуму навантаження
(незалежний і автокорельований) — N_RUNS випадкових реалізацій (seed 0..N_RUNS-1),
результат: середнє ± стандартне відхилення.

Порівняння з "ідеально підібраним вручну" запасом:
    фіксований k перебирається на сітці 0.00…0.60 (крок 0.05). Для кожного
    сценарію визначаємо, скільки ресурсів потрібно фіксованому запасу, щоб
    досягти ТОГО САМОГО рівня порушень SLO, що й запропонований метод
    (лінійна інтерполяція між сусідніми k). Це найчесніше порівняння: адже
    адміністратор, який знає майбутнє навантаження, підібрав би саме такий k.

Запуск:  python run_experiment.py
Результати: папка results/ (таблиці CSV/Markdown і графіки PNG).
"""

import csv
import os
from collections import defaultdict
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import simulator as sim

N_RUNS = 30
SLO_TARGET = 0.5            # %, вимога до сервісу: порушення SLO не більше 0.5% часу
TABLE_K = (0.1, 0.2, 0.3)   # які фіксовані k показувати в основних таблицях
SERVICES = [k for k in sim.SCENARIOS if k.startswith("svc_")]

# Експеримент 4: чутливість до параметрів методу
SENS_WINDOWS = (30, 60, 120)
SENS_QUANTILES = (0.80, 0.90, 0.95)
SENS_SCENARIOS = SERVICES + ["varnoise"]
SENS_NOISE = "ar"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
METRICS = [
    ("slo_violation_pct", "Порушення SLO, % часу"),
    ("p95_latency_ms", "p95 час відповіді, мс"),
    ("replica_hours", "Ресурси, репліко-год"),
    ("scaling_actions", "Дій масштабування"),
]


# ---------------------------------------------------------------------------
# Прогони (паралельно, по одному завданню на випадкову реалізацію)
# ---------------------------------------------------------------------------
def _kind(st):
    if isinstance(st, sim.Adaptive):
        return "adaptive", None
    if isinstance(st, sim.Predictive):
        return "fixed", st.k
    return "reactive", None


def _task_main(task):
    noise, sc, seed = task
    load = sim.generate_load(sc, seed=seed, phi=sim.NOISE_MODELS[noise][0])
    out = []
    for st in sim.make_strategies():
        r = sim.simulate(st, load)
        kind, k = _kind(st)
        out.append((noise, sc, st.name, kind, k, {m: r[m] for m, _ in METRICS}))
    return out


def _task_sens(task):
    w, q, sc, seed = task
    load = sim.generate_load(sc, seed=seed, phi=sim.NOISE_MODELS[SENS_NOISE][0])
    r = sim.simulate(sim.Adaptive(window=w, quantile=q), load)
    return (w, q, sc, {m: r[m] for m, _ in METRICS})


def _aggregate(key, runs):
    row = dict(key)
    for m, _ in METRICS:
        vals = np.array([r[m] for r in runs])
        row[m + "_mean"], row[m + "_std"] = float(vals.mean()), float(vals.std())
    return row


def run_main(pool):
    tasks = [(n, sc, s) for n in sim.NOISE_MODELS for sc in sim.SCENARIOS for s in range(N_RUNS)]
    groups, info = defaultdict(list), {}
    for chunk in pool.imap_unordered(_task_main, tasks, chunksize=4):
        for noise, sc, name, kind, k, met in chunk:
            groups[(noise, sc, name)].append(met)
            info[(noise, sc, name)] = (kind, k)
    rows = []
    for (noise, sc, name), runs in groups.items():
        kind, k = info[(noise, sc, name)]
        rows.append(_aggregate({"noise": noise, "scenario_key": sc, "scenario": sim.SCENARIOS[sc],
                                "strategy": name, "kind": kind, "k": k}, runs))
    order = {n: i for i, n in enumerate(sim.NOISE_MODELS)}
    sc_order = {s: i for i, s in enumerate(sim.SCENARIOS)}
    kind_order = {"reactive": 0, "fixed": 1, "adaptive": 2}
    rows.sort(key=lambda r: (order[r["noise"]], sc_order[r["scenario_key"]],
                             kind_order[r["kind"]], r["k"] or 0))
    return rows


def run_sensitivity(pool):
    tasks = [(w, q, sc, s) for w in SENS_WINDOWS for q in SENS_QUANTILES
             for sc in SENS_SCENARIOS for s in range(N_RUNS)]
    groups = defaultdict(list)
    for w, q, sc, met in pool.imap_unordered(_task_sens, tasks, chunksize=8):
        groups[(w, q, sc)].append(met)
    return {key: _aggregate({}, runs) for key, runs in groups.items()}


# ---------------------------------------------------------------------------
# Аналіз
# ---------------------------------------------------------------------------
def select(rows, noise, sc, kind=None):
    return [r for r in rows if r["noise"] == noise and r["scenario_key"] == sc
            and (kind is None or r["kind"] == kind)]


def fixed_needed(fixed, target):
    """
    Скільки ресурсів (і який k) потрібно фіксованому запасу, щоб порушення SLO
    не перевищували target. Лінійна інтерполяція між сусідніми k сітки.
    Повертає (None, None), якщо навіть k = 0.60 не досягає target.
    """
    fixed = sorted(fixed, key=lambda r: r["k"])
    prev = None
    for r in fixed:
        if r["slo_violation_pct_mean"] <= target + 1e-9:
            if prev is None:
                return r["replica_hours_mean"], r["k"]
            s0, s1 = prev["slo_violation_pct_mean"], r["slo_violation_pct_mean"]
            w = (s0 - target) / (s0 - s1) if s0 > s1 else 1.0
            res = prev["replica_hours_mean"] + w * (r["replica_hours_mean"] - prev["replica_hours_mean"])
            return res, prev["k"] + w * (r["k"] - prev["k"])
        prev = r
    return None, None


def equal_slo(rows, noise):
    """Запропонований метод vs фіксований k, підібраний на той самий рівень SLO."""
    out = []
    for sc in sim.SCENARIOS:
        a = select(rows, noise, sc, "adaptive")[0]
        res, k = fixed_needed(select(rows, noise, sc, "fixed"), a["slo_violation_pct_mean"])
        delta = None if res is None else 100 * (a["replica_hours_mean"] / res - 1)
        out.append((sc, a["slo_violation_pct_mean"], a["replica_hours_mean"], k, res, delta))
    return out


def baselines(rows, noise):
    """Ідеально підібраний вручну k для кожного сервісу A–D (вимога SLO ≤ SLO_TARGET)."""
    return {s: fixed_needed(select(rows, noise, s, "fixed"), SLO_TARGET) for s in SERVICES}


def robustness(rows, noise):
    """
    Одні налаштування для всіх сервісів A–D:
      - найгірше порушення SLO серед сервісів;
      - середня переплата ресурсів відносно ідеально підібраного вручну k.
    """
    base = baselines(rows, noise)
    shown = [r["strategy"] for r in select(rows, noise, SERVICES[0])
             if r["kind"] != "fixed" or r["k"] in TABLE_K]
    out = []
    for st in shown:
        rs = [r for s in SERVICES for r in select(rows, noise, s) if r["strategy"] == st]
        worst = max(r["slo_violation_pct_mean"] for r in rs)
        over = np.mean([100 * (r["replica_hours_mean"] / base[r["scenario_key"]][0] - 1) for r in rs])
        out.append((st, worst, over))
    return out, base


def sensitivity_table(sens, rows):
    base = baselines(rows, SENS_NOISE)
    vfixed = select(rows, SENS_NOISE, "varnoise", "fixed")
    out = []
    for w in SENS_WINDOWS:
        for q in SENS_QUANTILES:
            svc = [sens[(w, q, s)] for s in SERVICES]
            worst = max(r["slo_violation_pct_mean"] for r in svc)
            over = np.mean([100 * (sens[(w, q, s)]["replica_hours_mean"] / base[s][0] - 1)
                            for s in SERVICES])
            v = sens[(w, q, "varnoise")]
            res, _ = fixed_needed(vfixed, v["slo_violation_pct_mean"])
            dv = None if res is None else 100 * (v["replica_hours_mean"] / res - 1)
            out.append((w, q, worst, over, v["slo_violation_pct_mean"], dv))
    return out


# ---------------------------------------------------------------------------
# Таблиці
# ---------------------------------------------------------------------------
def fmt_delta(d):
    return "недосяжно при k ≤ 0.60" if d is None else f"{d:+.1f}"


def save_tables(rows, sens_rows):
    with open(os.path.join(OUT, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "sensitivity.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["window_min", "quantile", "worst_slo_AD_pct", "overpay_AD_pct",
                    "varnoise_slo_pct", "varnoise_delta_vs_fixed_pct"])
        w.writerows(sens_rows)

    L = [f"# Результати експерименту\n\n{N_RUNS} прогонів на сценарій; середнє ± стандартне відхилення. "
         f"Вимога до SLO для порівняння: ≤ {SLO_TARGET}% часу."]
    for noise, (phi, title) in sim.NOISE_MODELS.items():
        L.append(f"\n## Модель шуму: {title}\n")
        for sc, sc_name in sim.SCENARIOS.items():
            L += [f"\n### {sc_name}\n",
                  "| Стратегія | " + " | ".join(t for _, t in METRICS) + " |",
                  "|---" * (len(METRICS) + 1) + "|"]
            for r in select(rows, noise, sc):
                if r["kind"] == "fixed" and r["k"] not in TABLE_K:
                    continue
                cells = [f"{r[m + '_mean']:.2f} ± {r[m + '_std']:.2f}" for m, _ in METRICS]
                L.append(f"| {r['strategy']} | " + " | ".join(cells) + " |")

        L += ["\n### Порівняння за однакового рівня SLO\n",
              "Скільки ресурсів потрібно фіксованому запасу, щоб досягти того самого рівня "
              "порушень SLO, що й запропонований метод. Від'ємна різниця — метод економніший.\n",
              "| Сценарій | SLO методу, % | Ресурси методу | Потрібний фіксований k | "
              "Ресурси фіксованого k | Різниця ресурсів, % |",
              "|---|---|---|---|---|---|"]
        for sc, slo, res_a, k, res_f, d in equal_slo(rows, noise):
            ks = "—" if k is None else f"{k:.2f}"
            rf = "—" if res_f is None else f"{res_f:.1f}"
            L.append(f"| {sim.SCENARIOS[sc]} | {slo:.2f} | {res_a:.1f} | {ks} | {rf} | {fmt_delta(d)} |")

        rob, base = robustness(rows, noise)
        L += ["\n### Робастність: одні налаштування для сервісів A–D\n",
              "Переплата — відносно запасу k, ідеально підібраного вручну для кожного сервісу "
              f"(найменший k із порушеннями SLO ≤ {SLO_TARGET}%): "
              + ", ".join(f"{sim.SCENARIOS[s].split(':')[0]} — k = {b[1]:.2f}" for s, b in base.items())
              + ".\n",
              "| Стратегія | Найгірше порушення SLO, % | Середня переплата ресурсів, % |",
              "|---|---|---|"]
        L += [f"| {st} | {w:.2f} | {o:+.1f} |" for st, w, o in rob]

    L += [f"\n## Чутливість до параметрів методу ({sim.NOISE_MODELS[SENS_NOISE][1]})\n",
          "| Вікно W, хв | Квантиль q | Найгірше SLO A–D, % | Переплата A–D, % | "
          "Змінна мінливість: SLO, % | Змінна мінливість: різниця з фікс. k, % |",
          "|---|---|---|---|---|---|"]
    L += [f"| {w} | {q:.2f} | {wo:.2f} | {ov:+.1f} | {vs:.2f} | {fmt_delta(dv)} |"
          for w, q, wo, ov, vs, dv in sens_rows]
    text = "\n".join(L)
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


# ---------------------------------------------------------------------------
# Графіки (розрізняються формою маркерів і типом ліній — читаються і в ч/б друці)
# ---------------------------------------------------------------------------
def _tradeoff_panel(ax, rows, noise, sc):
    fixed = sorted(select(rows, noise, sc, "fixed"), key=lambda r: r["k"])
    ax.plot([r["replica_hours_mean"] for r in fixed], [r["slo_violation_pct_mean"] for r in fixed],
            "--o", color="0.45", lw=1, ms=4, label="Прогноз + фіксований запас k")
    for r in fixed:
        if round(r["k"] * 100) % 10 == 0:
            ax.annotate(f"k={r['k']:.1f}", (r["replica_hours_mean"], r["slo_violation_pct_mean"]),
                        textcoords="offset points", xytext=(3, 5), fontsize=7, color="0.3")
    re = select(rows, noise, sc, "reactive")[0]
    ax.errorbar(re["replica_hours_mean"], re["slo_violation_pct_mean"], yerr=re["slo_violation_pct_std"],
                fmt="s", ms=8, color="black", mfc="white", capsize=3, label="Реактивне (HPA)")
    a = select(rows, noise, sc, "adaptive")[0]
    ax.errorbar(a["replica_hours_mean"], a["slo_violation_pct_mean"],
                xerr=a["replica_hours_std"], yerr=a["slo_violation_pct_std"],
                fmt="*", ms=15, color="crimson", mec="black", capsize=3, label="Запропонований метод")
    ax.set_title(sim.SCENARIOS[sc], fontsize=11)
    ax.set_xlabel("Ресурси, репліко-год (менше — краще)")
    ax.set_ylabel("Порушення SLO, % часу (менше — краще)")
    ax.grid(alpha=0.3)


def plot_tradeoff(rows):
    for noise, (_, title) in sim.NOISE_MODELS.items():
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, sc in zip(axes.flat, sim.SCENARIOS):
            _tradeoff_panel(ax, rows, noise, sc)
        axes.flat[0].legend(fontsize=8, loc="upper right")
        fig.suptitle(f"Компроміс «ресурси — якість обслуговування» ({title})")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"tradeoff_{noise}.png"), dpi=150)
        plt.close(fig)

    # Рисунок для статті: 4 найпоказовіші сценарії, автокорельований шум
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, sc in zip(axes.flat, ["svc_5", "svc_15", "varnoise", "shift"]):
        _tradeoff_panel(ax, rows, "ar", sc)
    axes.flat[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_tradeoff.png"), dpi=200)
    plt.close(fig)


def plot_load():
    """Рисунок 2: приклади навантаження у сценаріях експерименту."""
    t_h = np.arange(3 * 24 * 60) / 60
    phi = sim.NOISE_MODELS["ar"][0]
    fig, ax = plt.subplots(3, 1, figsize=(12, 9))
    ax[0].plot(t_h, sim.generate_load("shift", seed=3, phi=phi), color="black", lw=0.5)
    ax[0].set_title("а) Добовий цикл, раптові сплески та зміна режиму з 2-ї доби")
    ax[0].set_xlabel("Час, год")
    ax[1].plot(t_h, sim.generate_load("varnoise", seed=1, phi=phi), color="black", lw=0.5)
    ax[1].set_title("б) Мінливість навантаження змінюється в часі (2–12 %)")
    ax[1].set_xlabel("Час, год")
    sl = slice(12 * 60, 14 * 60)
    m = np.arange(sl.stop - sl.start)
    ax[2].plot(m, sim.generate_load("svc_10", seed=0, phi=0.0)[sl], color="0.55", lw=1,
               label=sim.NOISE_MODELS["white"][1])
    ax[2].plot(m, sim.generate_load("svc_10", seed=0, phi=phi)[sl], color="black", lw=1.2,
               label=sim.NOISE_MODELS["ar"][1])
    ax[2].set_title("в) Фрагмент 2 год, сервіс C (мінливість 10 %): дві моделі шуму")
    ax[2].set_xlabel("Час, хв")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.set_ylabel("Навантаження, запитів/с")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_load.png"), dpi=200)
    plt.close(fig)


def plot_timeline():
    """Рисунок 4: як запас k(t) підлаштовується, коли мінливість навантаження змінюється."""
    seed, phi = 1, sim.NOISE_MODELS["ar"][0]
    load = sim.generate_load("varnoise", seed=seed, phi=phi)
    sd = sim.noise_level("varnoise", len(load), np.random.default_rng(seed))
    adapt = sim.Adaptive()
    r = sim.simulate(adapt, load)
    sl = slice(24 * 60, 2 * 24 * 60)
    hours = np.arange(len(load))[sl] / 60

    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(hours, load[sl], color="black", lw=0.6)
    ax[0].set_ylabel("Навантаження, запитів/с")
    ax[0].set_title("а) Навантаження (мінливість змінюється в часі)")
    ax[1].plot(hours, load[sl] / (sim.MU * sim.TARGET_UTIL), color="0.5", lw=0.6,
               label="Потрібно реплік (за фактичним навантаженням)")
    ax[1].step(hours, r["replicas"][sl], color="black", lw=1.2, label="Виділено запропонованим методом")
    ax[1].set_ylabel("Кількість реплік")
    ax[1].set_title("б) Кількість реплік")
    ax[1].legend(fontsize=8)
    ax[2].plot(hours, 100 * sd[sl], "--", color="0.4", label="Реальна мінливість навантаження, %")
    ax[2].plot(hours, 100 * np.array(adapt.k_history)[sl], color="black", lw=1.2, label="Запас k(t), %")
    ax[2].set_ylabel("%")
    ax[2].set_xlabel("Час, год")
    ax[2].set_title("в) Самокоригований запас слідує за мінливістю навантаження")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_timeline.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with Pool() as pool:
        print("Експерименти 1–3 ...")
        rows = run_main(pool)
        print("Експеримент 4 (чутливість) ...")
        sens = run_sensitivity(pool)
    save_tables(rows, sensitivity_table(sens, rows))
    plot_tradeoff(rows)
    plot_load()
    plot_timeline()
    print(f"\nГотово. Результати в {OUT}")
