"""
Симулятор автоматизованого масштабування сервісу в розподіленій системі.

Ідея: є сервіс (наприклад, мікросервіс у Kubernetes), на який надходять запити.
Сервіс працює на N копіях (репліках). Кожна репліка обробляє обмежену кількість
запитів за секунду. Якщо реплік замало — зростає час відповіді (порушення SLO).
Якщо реплік забагато — даремно витрачаються ресурси (гроші).

Нова репліка стартує не миттєво, а через STARTUP_DELAY хвилин. Саме через це
реактивне масштабування "запізнюється".

Порівнюються стратегії керування:
  1) Reactive (HPA)        — як стандартний Kubernetes HPA: поточне навантаження.
  2) Прогноз + запас k=..  — прогноз навантаження + ФІКСОВАНИЙ запас k,
                             який адміністратор має підібрати вручну.
  3) Adaptive (наш)        — прогноз + запас k, який система САМА
                             підлаштовує за своїми ж помилками прогнозу.
Стратегії 2 і 3 — гібридні: не менше реплік, ніж дав би реактивний HPA.
"""

import math
from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# Параметри моделі сервісу
# ---------------------------------------------------------------------------
MU = 50.0               # запитів/с, які обробляє одна репліка
STARTUP_DELAY = 3       # хв, за які нова репліка стає готовою
TARGET_UTIL = 0.70      # цільове завантаження реплік (однакове для всіх стратегій)
BASE_LATENCY = 50.0     # мс, час відповіді ненавантаженого сервісу
SLO_LATENCY = 200.0     # мс, допустимий час відповіді (SLO)
MAX_LATENCY = 2000.0    # мс, обмеження зверху (тайм-аут)
SCALE_DOWN_WINDOW = 5   # хв, стабілізація перед зменшенням (як у Kubernetes HPA)
MIN_REPLICAS, MAX_REPLICAS = 1, 200


def latency_ms(load, replicas):
    """Час відповіді за моделлю черги: зростає різко, коли завантаження -> 100%."""
    if replicas <= 0:
        return MAX_LATENCY
    rho = load / (replicas * MU)
    if rho >= 0.975:
        return MAX_LATENCY
    return min(BASE_LATENCY / (1.0 - rho), MAX_LATENCY)


def replicas_for(load):
    """Скільки реплік треба, щоб навантаження load дало цільове завантаження."""
    n = math.ceil(max(load, 0.0) / (MU * TARGET_UTIL))
    return int(min(max(n, MIN_REPLICAS), MAX_REPLICAS))


# ---------------------------------------------------------------------------
# Прогнозування: метод Холта (подвійне експоненційне згладжування)
# Простий і зрозумілий: оцінює "рівень" і "тренд" навантаження.
# ---------------------------------------------------------------------------
class HoltForecaster:
    def __init__(self, alpha=0.5, beta=0.2):
        self.alpha, self.beta = alpha, beta
        self.level = None
        self.trend = 0.0

    def update(self, y):
        if self.level is None:
            self.level = y
            return
        prev = self.level
        self.level = self.alpha * y + (1 - self.alpha) * (self.level + self.trend)
        self.trend = self.beta * (self.level - prev) + (1 - self.beta) * self.trend

    def forecast(self, h):
        return max(self.level + h * self.trend, 0.0)


# ---------------------------------------------------------------------------
# Стратегії масштабування
# Кожна стратегія отримує спостережене навантаження (із затримкою метрик 1 хв)
# і повертає бажану кількість реплік.
# ---------------------------------------------------------------------------
class Reactive:
    """Аналог Kubernetes HPA: рішення лише за поточним навантаженням."""
    name = "Reactive (HPA)"

    def decide(self, observed_load):
        return replicas_for(observed_load)


class Predictive:
    """
    Прогноз на час запуску репліки + ФІКСОВАНИЙ запас k.
    Гібрид: беремо більше з двох рішень — прогнозного і реактивного,
    щоб раптовий сплеск, якого прогноз не побачив, все одно був оброблений.
    """

    def __init__(self, k):
        self.k = k
        self.name = f"Прогноз + запас k={k:.2f}"
        self.fc = HoltForecaster()
        self.h = STARTUP_DELAY + 1  # горизонт = час старту + затримка метрик

    def predictive_load(self, observed_load):
        self.fc.update(observed_load)
        return self.fc.forecast(self.h)

    def decide(self, observed_load):
        pred = self.predictive_load(observed_load)
        return max(replicas_for(pred * (1 + self.k)), replicas_for(observed_load))


class Adaptive(Predictive):
    """
    ЗАПРОПОНОВАНИЙ МЕТОД: самокоригований запас.

    Система запам'ятовує, наскільки її прогнози НЕДООЦІНЮВАЛИ реальне
    навантаження протягом останніх WINDOW хвилин, і бере запас k такий,
    щоб покрити QUANTILE (напр. 90%) цих недооцінок.

      помилка r = (факт - прогноз) / прогноз
      k(t)     = 90-й процентиль помилок r за останнє вікно (від 0 до K_MAX)

    Спокійне навантаження -> помилки малі -> k малий -> економія ресурсів.
    Неспокійне / сплески   -> помилки великі -> k росте -> менше порушень SLO.
    """

    WINDOW = 60      # хв пам'яті про помилки (типове значення)
    QUANTILE = 0.90  # яку частку недооцінок покриваємо (типове значення)
    K_MAX = 1.0      # верхня межа запасу (не більше +100%)

    def __init__(self, window=WINDOW, quantile=QUANTILE):
        super().__init__(k=0.0)
        self.window, self.quantile = window, quantile
        self.name = "Adaptive (запропонований)"
        self.errors = deque(maxlen=window)
        self.pending = deque()  # прогнози, що чекають на перевірку фактом
        self.k_history = []

    def decide(self, observed_load):
        # 1) перевіряємо прогноз, зроблений h кроків тому, з фактичним значенням
        if len(self.pending) == self.h:
            predicted = self.pending.popleft()
            self.errors.append((observed_load - predicted) / max(predicted, 1.0))
        # 2) оновлюємо запас k за пам'яттю помилок
        if len(self.errors) >= 10:
            self.k = float(np.clip(np.quantile(self.errors, self.quantile), 0.0, self.K_MAX))
        self.k_history.append(self.k)
        # 3) прогноз + адаптований запас (і реактивна страховка, як у Predictive)
        pred = self.predictive_load(observed_load)
        self.pending.append(pred)
        return max(replicas_for(pred * (1 + self.k)), replicas_for(observed_load))


# ---------------------------------------------------------------------------
# Генерація навантаження (сценарії експерименту)
# ---------------------------------------------------------------------------
def noise_level(scenario, minutes, rng):
    """
    Мінливість (шум) навантаження у кожну хвилину, частка від середнього.
    Різні сервіси мають різну мінливість, і вона може змінюватися в часі.
    """
    if scenario.startswith("svc_"):                 # сервіс зі сталою мінливістю
        return np.full(minutes, int(scenario[4:]) / 100)
    if scenario == "varnoise":                      # мінливість змінюється
        sd, i = np.empty(minutes), 0                # періоди по 2–6 год
        while i < minutes:
            d = rng.integers(120, 360)
            sd[i:i + d] = rng.choice([0.02, 0.05, 0.12])
            i += d
        return sd
    return np.full(minutes, 0.05)


def ar1_noise(n, phi, rng):
    """
    Шум зі стандартним відхиленням 1 і автокореляцією phi між сусідніми хвилинами.
      phi = 0   — "білий" шум: кожна хвилина незалежна (спрощення);
      phi = 0.8 — шум "хвилями": відхилення тримається кілька хвилин, як у
                  реальних трасах навантаження.
    """
    z = rng.normal(0, 1, n)
    if phi == 0:
        return z
    e = np.empty(n)
    e[0] = z[0]
    s = math.sqrt(1 - phi * phi)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + s * z[i]
    return e


NOISE_MODELS = {
    "white": (0.0, "незалежний (білий) шум"),
    "ar": (0.8, "автокорельований шум, φ = 0.8"),
}


def generate_load(scenario, minutes=3 * 24 * 60, seed=0, phi=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(minutes)
    day = 24 * 60
    # добовий цикл: мінімум уночі, пік удень
    load = 400 - 250 * np.cos(2 * np.pi * (t - 3 * 60) / day)

    if scenario == "shift":
        # раптові сплески (напр. розсилка, новина): ~5 на добу
        for _ in range(rng.poisson(5 * minutes / day)):
            start = rng.integers(0, minutes - 40)
            amp = rng.uniform(0.5, 1.2)          # +50..120% до навантаження
            rise, hold = rng.integers(2, 6), rng.integers(10, 30)
            prof = np.concatenate([np.linspace(0, 1, rise), np.ones(hold), np.linspace(1, 0, 10)])
            end = min(start + len(prof), minutes)
            load[start:end] *= 1 + amp * prof[: end - start]

    if scenario == "shift":
        # зміна режиму роботи: з 2-ї доби навантаження зростає в 1.7 раза
        ramp = np.clip((t - day) / 60, 0, 1)
        load *= 1 + 0.7 * ramp

    sd = noise_level(scenario, minutes, rng)
    return np.maximum(load * (1 + ar1_noise(minutes, phi, rng) * sd), 1.0)


SCENARIOS = {
    "svc_2": "Сервіс A: мінливість 2%",
    "svc_5": "Сервіс B: мінливість 5%",
    "svc_10": "Сервіс C: мінливість 10%",
    "svc_15": "Сервіс D: мінливість 15%",
    "varnoise": "Мінливість змінюється в часі",
    "shift": "Сплески + зміна режиму",
}


# ---------------------------------------------------------------------------
# Симуляція одного прогону
# ---------------------------------------------------------------------------
def simulate(strategy, load):
    minutes = len(load)
    ready = replicas_for(load[0])
    starting = deque()               # (хвилина готовності, кількість)
    desired_hist = deque(maxlen=SCALE_DOWN_WINDOW)

    lat = np.zeros(minutes)
    rep = np.zeros(minutes, dtype=int)
    actions = 0

    for t in range(minutes):
        # репліки, що завершили запуск
        while starting and starting[0][0] <= t:
            ready += starting.popleft()[1]

        lat[t] = latency_ms(load[t], ready)
        rep[t] = ready

        observed = load[t - 1] if t > 0 else load[0]   # метрики із затримкою 1 хв
        desired = strategy.decide(observed)
        desired_hist.append(desired)

        total = ready + sum(c for _, c in starting)
        if desired > total:                              # масштабування вгору
            starting.append((t + STARTUP_DELAY, desired - total))
            actions += 1
        else:                                            # вниз — лише стабільно
            target = max(desired_hist)
            if target < total and not starting:
                ready = max(target, MIN_REPLICAS)
                actions += 1

    return {
        "slo_violation_pct": 100.0 * np.mean(lat > SLO_LATENCY),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "replica_hours": rep.sum() / 60.0,
        "scaling_actions": actions,
        "latency": lat,
        "replicas": rep,
    }


# Сітка фіксованих запасів: з неї обирається "ідеально підібраний вручну" k
# для кожного сервісу (найдешевший k, що виконує вимогу до SLO).
K_GRID = [round(0.05 * i, 2) for i in range(13)]  # 0.00, 0.05, ..., 0.60


def make_strategies():
    return [Reactive()] + [Predictive(k) for k in K_GRID] + [Adaptive()]
