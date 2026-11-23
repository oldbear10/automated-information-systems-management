"""
Рисунок 1 статті: схема циклу керування методу самокоригованого запасу.

Враховано зауваження рецензента 1 до статті 2: явна точка початку циклу
і стрілки, що задають напрям виконання етапів.

Запуск:  python make_fig1.py   ->  results/fig1_scheme.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "fig1_scheme.png")

STEPS = [
    ("1. Збір метрик",
     "фактичне навантаження λ(t−1) з системи моніторингу"),
    ("2. Перевірка прогнозу, зробленого h хв тому",
     "відносна помилка  r = (λ − λ̂) / λ̂"),
    ("3. Оновлення запасу",
     "k(t) = q-квантиль помилок r за останні W хв"),
    ("4. Прогноз навантаження (метод Холта)",
     "λ̂(t+h),  h = час запуску репліки + затримка метрик"),
    ("5. Рішення про кількість реплік",
     "N* = max( R(λ̂·(1+k)),  R(λ) )"),
    ("6. Виконання рішення",
     "збільшення — одразу (репліка готова через 3 хв);\nзменшення — після стабілізації 5 хв"),
]

BX0, BW, BH = 0.6, 6.0, 1.25          # ліва межа, ширина і висота блоків етапів
CX = BX0 + BW / 2
YS = [11.6, 9.9, 8.2, 6.5, 4.8, 3.1]  # центри блоків
SX0, SW = 7.8, 2.5                    # блок керованого сервісу
SY0, SH = 4.0, 2.6


def arrow(ax, p0, p1, style="-|>", ls="-", conn="arc3", lw=1.3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14, lw=lw,
                                 linestyle=ls, color="black", connectionstyle=conn,
                                 shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.set_xlim(-0.9, 10.4)
    ax.set_ylim(1.7, 14.1)
    ax.axis("off")

    # Точка початку
    ax.add_patch(FancyBboxPatch((CX - 2.3, 12.95), 4.6, 0.65, boxstyle="round,pad=0.02,rounding_size=0.32",
                                fc="black", ec="black"))
    ax.text(CX, 13.275, "ПОЧАТОК: чергова хвилина t", ha="center", va="center",
            color="white", fontsize=11, fontweight="bold")
    arrow(ax, (CX, 12.95), (CX, YS[0] + BH / 2))

    # Виділення нового елемента методу (етапи 2–3)
    ax.add_patch(Rectangle((BX0 - 0.25, YS[2] - BH / 2 - 0.25), BW + 0.5, YS[1] - YS[2] + BH + 0.5,
                           fill=False, ls="--", lw=1.2, ec="0.35"))
    ax.text(BX0 + BW + 0.32, YS[2] - BH / 2 - 0.18, "самокоригування\nзапасу (новий\nелемент методу)",
            ha="left", va="bottom", fontsize=8.5, style="italic", color="0.25")

    # Етапи
    for i, ((title, body), y) in enumerate(zip(STEPS, YS)):
        new = i in (1, 2)
        ax.add_patch(FancyBboxPatch((BX0, y - BH / 2), BW, BH, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="0.88" if new else "white", ec="black", lw=1.2))
        ax.text(CX, y + 0.27, title, ha="center", va="center", fontsize=10.5, fontweight="bold")
        ax.text(CX, y - 0.22, body, ha="center", va="center", fontsize=9.5)
        if i < len(STEPS) - 1:
            arrow(ax, (CX, y - BH / 2), (CX, YS[i + 1] + BH / 2))

    # Пам'ять прогнозів: етап 4 -> етап 2 (пунктир)
    xl = BX0 - 0.55
    ax.plot([BX0, xl, xl], [YS[3], YS[3], YS[1]], ls="--", lw=1.1, color="black")
    arrow(ax, (xl, YS[1]), (BX0, YS[1]), lw=1.1)
    ax.text(xl - 0.35, (YS[1] + YS[3]) / 2, "прогноз λ̂ зберігається\nі перевіряється через h хв",
            ha="center", va="center", rotation=90, fontsize=8.5,
            bbox=dict(fc="white", ec="none", pad=1))

    # Керований сервіс
    ax.add_patch(FancyBboxPatch((SX0, SY0), SW, SH, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="white", ec="black", lw=1.6))
    ax.text(SX0 + SW / 2, SY0 + SH / 2, "Керований\nсервіс\n(Kubernetes\nDeployment,\nN(t) реплік)",
            ha="center", va="center", fontsize=10, linespacing=1.3)

    # Етап 6 -> сервіс (дія керування)
    sx = SX0 + SW / 2
    arrow(ax, (BX0 + BW, YS[5]), (sx, SY0), conn="angle,angleA=0,angleB=90,rad=0")
    ax.text((BX0 + BW + sx) / 2 + 0.3, YS[5] + 0.15, "зміна кількості\nреплік", ha="center", va="bottom",
            fontsize=8.5)

    # Сервіс -> етап 1 (зворотний зв'язок, наступна хвилина)
    arrow(ax, (sx, SY0 + SH), (BX0 + BW, YS[0]), conn="angle,angleA=90,angleB=0,rad=0")
    ax.text((BX0 + BW + sx) / 2 + 0.3, YS[0] + 0.15, "метрики\n(наступна хвилина t+1)", ha="center",
            va="bottom", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print("Збережено", OUT)


if __name__ == "__main__":
    main()
