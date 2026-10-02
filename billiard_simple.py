"""M2. Бильярд: удар двух гладких шаров, F = k·δ^n (n = 1 — Гук, n = 1.5 — Герц).
Численное решение 2-го закона Ньютона сравнивается с законами сохранения."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons, Slider
from scipy.integrate import solve_ivp

INITIAL_GAP = 0.4  # расстояние, которое шары проходят до касания, м
TIME_AFTER_IMPACT = 0.4  # сколько показывать движение после удара, с
FRAME_STEP = 0.02  # шаг по времени между кадрами анимации, с
FORCE_LAWS = {  # название закона: (жёсткость k, показатель n)
    "Герц, n=1.5": (4e8, 1.5),
    "Гук, n=1": (1e7, 1.0),
}


def equations_of_motion(time, state, mass1, mass2, radius, stiffness, exponent):
    """Правая часть системы: state = (x1, y1, x2, y2, vx1, vy1, vx2, vy2)."""
    position1, position2 = state[0:2], state[2:4]
    center_offset = position2 - position1
    center_distance = np.linalg.norm(center_offset)
    overlap = max(2 * radius - center_distance, 0)  # суммарное сжатие шаров
    normal_direction = center_offset / center_distance
    contact_force = stiffness * overlap**exponent * normal_direction  # сила на шар 2
    return np.r_[state[4:8], -contact_force / mass1, contact_force / mass2]


def analytic_velocities_after(velocity1, velocity2, mass1, mass2, normal_direction):
    """Упругий удар гладких шаров: меняются только нормальные компоненты скоростей."""
    normal_speed1, normal_speed2 = (
        velocity1 @ normal_direction,
        velocity2 @ normal_direction,
    )
    total_mass = mass1 + mass2
    new_normal_speed1 = (
        (mass1 - mass2) * normal_speed1 + 2 * mass2 * normal_speed2
    ) / total_mass
    new_normal_speed2 = (
        (mass2 - mass1) * normal_speed2 + 2 * mass1 * normal_speed1
    ) / total_mass
    return (
        velocity1 + (new_normal_speed1 - normal_speed1) * normal_direction,
        velocity2 + (new_normal_speed2 - normal_speed2) * normal_direction,
    )


def simulate_collision(
    speed1, speed2, mass1, mass2, radius, stiffness, exponent, impact_angle_deg
):
    # положение шаров в момент касания
    impact_parameter = (
        2 * radius * np.sin(np.radians(impact_angle_deg))
    )  # прицельное расстояние
    centers_dx = np.sqrt(4 * radius**2 - impact_parameter**2)
    velocity1, velocity2 = np.array([speed1, 0.0]), np.array([speed2, 0.0])
    state_at_contact = np.r_[
        -centers_dx / 2,
        -impact_parameter / 2,
        centers_dx / 2,
        impact_parameter / 2,
        velocity1,
        velocity2,
    ]

    # оценка длительности удара: μu²/2 = k·δmax^(n+1)/(n+1),  τ ≈ 3.2·δmax/u
    reduced_mass = mass1 * mass2 / (mass1 + mass2)
    normal_approach_speed = (speed1 - speed2) * centers_dx / (2 * radius)
    max_overlap_estimate = (
        (exponent + 1) * reduced_mass * normal_approach_speed**2 / (2 * stiffness)
    ) ** (1 / (exponent + 1))
    contact_time_estimate = 3.2 * max_overlap_estimate / normal_approach_speed

    # численное решение уравнений Ньютона во время контакта
    contact_times = np.linspace(0, 2 * contact_time_estimate, 200)
    contact_states = solve_ivp(
        equations_of_motion,
        (0, contact_times[-1]),
        state_at_contact,
        t_eval=contact_times,
        args=(mass1, mass2, radius, stiffness, exponent),
        rtol=1e-9,
        atol=1e-12,
        max_step=contact_time_estimate / 200,
    ).y
    state_after = contact_states[:, -1]

    # вся траектория: равномерно до удара, удар (численно), равномерно после
    def uniform_motion(start_state, time_offsets):
        velocities_only = np.r_[start_state[4:8], 0, 0, 0, 0]
        return start_state[:, None] + velocities_only[:, None] * time_offsets

    time_before_contact = INITIAL_GAP / (speed1 - speed2)
    times_before = np.linspace(-time_before_contact, 0, 30, endpoint=False)
    times_after = contact_times[-1] + np.linspace(0, TIME_AFTER_IMPACT, 30)[1:]
    all_times = np.r_[times_before, contact_times, times_after]
    all_states = np.hstack(
        [
            uniform_motion(state_at_contact, times_before),
            contact_states,
            uniform_motion(state_after, times_after - contact_times[-1]),
        ]
    )
    frame_times = np.arange(-time_before_contact, times_after[-1], FRAME_STEP)
    frame_states = np.array(
        [np.interp(frame_times, all_times, row) for row in all_states]
    )

    # энергия во время удара
    center_distances = np.hypot(
        contact_states[2] - contact_states[0], contact_states[3] - contact_states[1]
    )
    overlap = np.clip(2 * radius - center_distances, 0, None)
    kinetic_energy = (
        mass1 * (contact_states[4] ** 2 + contact_states[5] ** 2)
        + mass2 * (contact_states[6] ** 2 + contact_states[7] ** 2)
    ) / 2
    elastic_energy = stiffness * overlap ** (exponent + 1) / (exponent + 1)

    # ответ по законам сохранения (нормаль — линия центров в момент касания)
    normal_at_contact = np.array([centers_dx, impact_parameter]) / (2 * radius)
    analytic_velocity1, analytic_velocity2 = analytic_velocities_after(
        velocity1, velocity2, mass1, mass2, normal_at_contact
    )

    return dict(
        frame_states=frame_states,
        all_states=all_states,
        contact_times_ms=contact_times * 1e3,
        kinetic_energy=kinetic_energy,
        elastic_energy=elastic_energy,
        numeric_velocities=state_after[4:8],
        analytic_velocities=np.r_[analytic_velocity1, analytic_velocity2],
        max_overlap=overlap.max(),
    )


# ---------------- интерфейс ----------------
if __name__ == "__main__":  # интерфейс запускается только при прямом запуске файла
    figure = plt.figure(figsize=(14, 8))
    table_axes = figure.add_axes([0.30, 0.08, 0.40, 0.85])
    energy_axes = figure.add_axes([0.76, 0.64, 0.22, 0.30])
    info_axes = figure.add_axes([0.73, 0.01, 0.26, 0.56])
    info_axes.axis("off")

    # колонка начальных параметров
    figure.text(0.02, 0.95, "Удар", weight="bold")
    impact_type_selector = RadioButtons(
        figure.add_axes([0.02, 0.84, 0.16, 0.10]), ["Лобовой", "Под углом"]
    )
    figure.text(0.02, 0.80, "Закон силы", weight="bold")
    force_law_selector = RadioButtons(
        figure.add_axes([0.02, 0.69, 0.16, 0.10]), list(FORCE_LAWS)
    )
    figure.text(0.02, 0.64, "Начальные параметры", weight="bold")

    slider_settings = [  # ключ, подпись, минимум, максимум, начальное значение
        ("speed1", "v1, м/с", 0.1, 4, 2),
        ("speed2", "v2, м/с", -3, 3, 0),
        ("mass1", "m1, кг", 0.05, 0.6, 0.17),
        ("mass2", "m2, кг", 0.05, 0.6, 0.17),
        ("radius_mm", "R, мм", 15, 40, 28.6),
        ("impact_angle", "угол, °", 0, 80, 30),
    ]
    sliders = {}
    for row, (key, label, minimum, maximum, initial) in enumerate(slider_settings):
        slider_axes = figure.add_axes([0.08, 0.58 - 0.06 * row, 0.12, 0.03])
        sliders[key] = Slider(slider_axes, label, minimum, maximum, valinit=initial)
    sliders["impact_angle"].ax.set_visible(
        False
    )  # угол виден только при ударе «под углом»

    animation_state = {}

    def recalculate(_=None):
        params = {key: slider.val for key, slider in sliders.items()}
        radius = params["radius_mm"] / 1000
        stiffness, exponent = FORCE_LAWS[force_law_selector.value_selected]
        is_angled = impact_type_selector.value_selected == "Под углом"
        sliders["impact_angle"].ax.set_visible(is_angled)
        impact_angle = params["impact_angle"] if is_angled else 0

        for axes in (table_axes, energy_axes, info_axes):
            axes.clear()
        info_axes.axis("off")
        if params["speed1"] <= params["speed2"]:
            animation_state.clear()
            table_axes.set_title("Шары не столкнутся: нужно v1 > v2")
            figure.canvas.draw_idle()
            return

        result = simulate_collision(
            params["speed1"],
            params["speed2"],
            params["mass1"],
            params["mass2"],
            radius,
            stiffness,
            exponent,
            impact_angle,
        )
        frames, trajectory = result["frame_states"], result["all_states"]

        # стол: траектории, шары, векторы скорости
        table_axes.set_aspect("equal", adjustable="datalim")
        table_axes.margins(0.1)
        table_axes.set_title("Стол (реальное время)")
        table_axes.plot(trajectory[0], trajectory[1], color="C0", lw=1, alpha=0.4)
        table_axes.plot(trajectory[2], trajectory[3], color="C1", lw=1, alpha=0.4)
        ball1 = table_axes.add_patch(
            plt.Circle(frames[0:2, 0], radius, color="C0", label="шар 1")
        )
        ball2 = table_axes.add_patch(
            plt.Circle(frames[2:4, 0], radius, color="C1", label="шар 2")
        )
        velocity_arrows = table_axes.quiver(  # длина стрелки = путь за 0.2 с
            frames[[0, 2], 0],
            frames[[1, 3], 0],
            frames[[4, 6], 0],
            frames[[5, 7], 0],
            color=["C0", "C1"],
            angles="xy",
            scale_units="xy",
            scale=5,
            width=0.004,
        )
        table_axes.legend(loc="lower right")

        # энергия во время удара
        times_ms = result["contact_times_ms"]
        kinetic, elastic = result["kinetic_energy"], result["elastic_energy"]
        energy_axes.plot(times_ms, kinetic, label="кинетическая")
        energy_axes.plot(times_ms, elastic, label="упругая")
        energy_axes.plot(times_ms, kinetic + elastic, "k--", label="полная")
        energy_axes.set(title="Энергия во время удара, Дж", xlabel="t от касания, мс")
        energy_axes.legend(fontsize=8)

        # сравнение с законами сохранения
        def format_vector(vector):
            return f"({vector[0]:+.3f}, {vector[1]:+.3f})"

        def angle_between(vector_a, vector_b):
            """Угол между направлениями движения, °; не определён, если шар стоит."""
            norm_a, norm_b = np.linalg.norm(vector_a), np.linalg.norm(vector_b)
            if norm_a < 1e-6 or norm_b < 1e-6:
                return "  —  (шар стоит)"
            cosine = np.clip(vector_a @ vector_b / (norm_a * norm_b), -1, 1)
            return f"{np.degrees(np.arccos(cosine)):6.2f}°"

        numeric, analytic = result["numeric_velocities"], result["analytic_velocities"]
        speed_before = (abs(params["speed1"]), abs(params["speed2"]))
        speed_numeric = (np.linalg.norm(numeric[:2]), np.linalg.norm(numeric[2:]))
        speed_analytic = (np.linalg.norm(analytic[:2]), np.linalg.norm(analytic[2:]))

        def ball_energies(v1, v2):
            """Кинетические энергии шаров E1 = m1·v1²/2, E2 = m2·v2²/2 и их сумма, Дж."""
            e1 = params["mass1"] * (v1 @ v1) / 2
            e2 = params["mass2"] * (v2 @ v2) / 2
            return e1, e2, e1 + e2

        def energy_row(label, energies, total_before):
            e1, e2, total = energies
            error = total - total_before  # погрешность: E_после − E_до
            return (
                f"{label} {e1:7.4f} {e2:7.4f} {total:12.9f}\n"
                f"{'':15}ΔE = {error:+.2e} Дж ({error / total_before:+.1e})"
            )

        energy_before = ball_energies(
            np.array([params["speed1"], 0.0]), np.array([params["speed2"], 0.0])
        )
        energy_numeric = ball_energies(numeric[:2], numeric[2:])
        energy_analytic = ball_energies(analytic[:2], analytic[2:])
        contact_duration_ms = times_ms[np.nonzero(elastic)[0][-1]]
        info_axes.text(
            0,
            1,
            "Скорости после удара, м/с\n"
            "      численно          законы сохр.\n"
            f"v1  {format_vector(numeric[:2])}  {format_vector(analytic[:2])}\n"
            f"v2  {format_vector(numeric[2:])}  {format_vector(analytic[2:])}\n\n"
            "Модуль скорости |v|, м/с\n"
            "      до удара  после: числ.  з. сохр.\n"
            f"|v1|  {speed_before[0]:7.3f}  {speed_numeric[0]:11.3f}  {speed_analytic[0]:8.3f}\n"
            f"|v2|  {speed_before[1]:7.3f}  {speed_numeric[1]:11.3f}  {speed_analytic[1]:8.3f}\n\n"
            "Угол между направлениями v1' и v2'\n"
            f"  численно:          {angle_between(numeric[:2], numeric[2:])}\n"
            f"  законы сохранения: {angle_between(analytic[:2], analytic[2:])}\n\n"
            "Кинетическая энергия, Дж\n"
            "                   E1      E2        сумма\n"
            f"до удара       {energy_before[0]:7.4f} {energy_before[1]:7.4f} "
            f"{energy_before[2]:12.9f}\n"
            f"{energy_row('после: числ.  ', energy_numeric, energy_before[2])}\n"
            f"{energy_row('после: з.сохр.', energy_analytic, energy_before[2])}\n\n"
            f"k = {stiffness:.0e}, n = {exponent}\n"
            f"τ удара = {contact_duration_ms:.3f} мс\n"
            f"δmax    = {result['max_overlap'] * 1e3:.3f} мм",
            va="top",
            family="monospace",
            fontsize=8,
        )

        animation_state.update(
            frames=frames,
            ball1=ball1,
            ball2=ball2,
            velocity_arrows=velocity_arrows,
            frame_index=0,
        )
        figure.canvas.draw_idle()

    def animate_frame(_):
        if not animation_state:
            return
        frames, index = animation_state["frames"], animation_state["frame_index"]
        animation_state["ball1"].center = frames[0:2, index]
        animation_state["ball2"].center = frames[2:4, index]
        arrows = animation_state["velocity_arrows"]
        arrows.set_offsets(np.c_[frames[[0, 2], index], frames[[1, 3], index]])
        arrows.set_UVC(frames[[4, 6], index], frames[[5, 7], index])
        animation_state["frame_index"] = (index + 1) % frames.shape[1]

    # пересчёт с задержкой 0.15 с после последнего движения ползунка
    recalculation_timer = figure.canvas.new_timer(interval=150)
    recalculation_timer.single_shot = True
    recalculation_timer.add_callback(recalculate)
    for slider in sliders.values():
        slider.on_changed(
            lambda _: (recalculation_timer.stop(), recalculation_timer.start())
        )
    impact_type_selector.on_clicked(recalculate)
    force_law_selector.on_clicked(recalculate)

    recalculate()
    animation = FuncAnimation(
        figure, animate_frame, interval=FRAME_STEP * 1000, cache_frame_data=False
    )
    plt.show()
