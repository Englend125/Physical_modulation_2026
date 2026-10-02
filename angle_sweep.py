"""Сравнение численного решения и законов сохранения в зависимости от угла удара.
Массы и относительная скорость максимальные (как на ползунках): v1 = 4, v2 = -3 м/с, m1 = m2 = 0.6 кг.
Запуск: python angle_sweep.py   (рядом должен лежать billiard_simple.py)"""

import matplotlib.pyplot as plt
import numpy as np

from billiard_simple import FORCE_LAWS, simulate_collision

FORCE_LAW = "Герц, n=1.5"  # или "Гук, n=1"
SPEED1, SPEED2 = 4.0, -3.0  # м/с
MASS1, MASS2 = 0.6, 0.6  # кг
RADIUS = 0.015  # м (минимум ползунка — эффект сильнее всего)
ANGLES = np.arange(0, 86, 1)  # углы удара, °
NUMERIC_STYLE = dict(color="#2a78d6", lw=2, label="численно")
ANALYTIC_STYLE = dict(color="#eb6834", lw=2, ls="--", label="законы сохранения")


def angle_between(vector_a, vector_b):
    cosine = vector_a @ vector_b / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def quantities(velocities):
    """Величины после удара: |v1|, |v2|, угол разлёта, E1, E2, E1 + E2."""
    v1, v2 = velocities[:2], velocities[2:]
    energy1, energy2 = MASS1 * (v1 @ v1) / 2, MASS2 * (v2 @ v2) / 2
    return (
        np.linalg.norm(v1),
        np.linalg.norm(v2),
        angle_between(v1, v2),
        energy1,
        energy2,
        energy1 + energy2,
    )


stiffness, exponent = FORCE_LAWS[FORCE_LAW]
numeric_rows, analytic_rows = [], []
for angle in ANGLES:
    result = simulate_collision(
        SPEED1, SPEED2, MASS1, MASS2, RADIUS, stiffness, exponent, angle
    )
    numeric_rows.append(quantities(result["numeric_velocities"]))
    analytic_rows.append(quantities(result["analytic_velocities"]))
numeric, analytic = np.array(numeric_rows), np.array(analytic_rows)

panels = [  # номер величины, заголовок, единицы
    (0, "Модуль скорости шара 1 после удара |v1'|", "м/с"),
    (1, "Модуль скорости шара 2 после удара |v2'|", "м/с"),
    (2, "Угол между направлениями v1' и v2'", "°"),
    (3, "Кинетическая энергия шара 1 после удара", "Дж"),
    (4, "Кинетическая энергия шара 2 после удара", "Дж"),
    (5, "Суммарная кинетическая энергия после удара", "Дж"),
]
figure, axes_grid = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
for axes, (column, title, unit) in zip(axes_grid.flat, panels):
    axes.plot(ANGLES, numeric[:, column], **NUMERIC_STYLE)
    axes.plot(ANGLES, analytic[:, column], **ANALYTIC_STYLE)
    difference = numeric[:, column] - analytic[:, column]
    i = np.abs(difference).argmax()
    axes.set_title(
        f"{title}\nмакс. разница (числ. − з.сохр.): {difference[i]:+.3g} {unit} при {ANGLES[i]}°",
        loc="left",
        fontsize=10,
    )
    axes.set_ylabel(unit)
    axes.grid(alpha=0.3)
energy_before = MASS1 * SPEED1**2 / 2 + MASS2 * SPEED2**2 / 2
axes_grid[1, 2].set_ylim(
    energy_before * 0.98, energy_before * 1.02
)  # линии совпадают — энергия сохраняется
for axes in axes_grid[1]:
    axes.set_xlabel("угол удара, °")
axes_grid[0, 0].legend(frameon=False, loc="upper right")
figure.suptitle(
    f"{FORCE_LAW}:  v1 = {SPEED1:g}, v2 = {SPEED2:g} м/с,  m1 = m2 = {MASS1:g} кг,  "
    f"R = {RADIUS * 1e3:g} мм,  E₀ = {energy_before:g} Дж",
    fontsize=11,
)
figure.tight_layout()
figure.savefig("angle_sweep.png", dpi=110)
plt.show()
