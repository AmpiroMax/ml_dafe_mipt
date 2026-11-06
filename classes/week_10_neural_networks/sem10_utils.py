"""Рисунки занятия 10 «Нейронная сеть с нуля» и слои сети для лекции.

Функции рисования только рисуют: сеть обучают и эксперименты проводят в ноутбуке,
а сюда передают картинки, истории обучения и посчитанные числа. В конце файла —
слои сети на numpy, которые лекция использует, чтобы заглянуть внутрь сети;
на семинаре их пишут сами.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (8, 4.5),
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
})

BLUE, BLACK, OCHRE, GREY = "#0072CE", "#0F1418", "#C98A3C", "#9CA3AF"
CURVE_COLORS = [BLUE, OCHRE, BLACK, GREY, "#4DAF4A", "#B5524B"]

LOWER_IS_BETTER = ("потеря", "ошибк", "время")
NO_COLOR = ("эпох", "параметров", "шаг", "ширина", "значений", "градиент")


def show_table(frame, digits=4):
    """Таблица с заливкой: лучшее значение ярче; для потерь, ошибок и времени лучше меньшее."""
    numeric = list(frame.select_dtypes("number").columns)
    plain = [c for c in numeric if any(key in str(c) for key in NO_COLOR)]
    lower = [c for c in numeric if c not in plain and any(key in str(c) for key in LOWER_IS_BETTER)]
    higher = [c for c in numeric if c not in plain and c not in lower]

    styled = frame.style.format(precision=digits, subset=numeric)
    if higher:
        styled = styled.background_gradient(cmap="Greens", subset=higher)
    if lower:
        styled = styled.background_gradient(cmap="Greens_r", subset=lower)
    return styled


def show_digits(images, labels, predicted=None, n_columns=10, title=None):
    """Цифры 28×28 в ряд; подпись — истинная метка, при predicted — и предсказанная."""
    n_images = len(images)
    n_rows = int(np.ceil(n_images / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(1.3 * n_columns, 1.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for index, ax in enumerate(axes):
        ax.axis("off")
        if index >= n_images:
            continue
        ax.imshow(images[index].reshape(28, 28), cmap="gray_r")
        caption = f"{labels[index]}" if predicted is None else f"{labels[index]} → {predicted[index]}"
        color = BLACK if predicted is None or predicted[index] == labels[index] else "#B5524B"
        ax.set_title(caption, fontsize=11, color=color)
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_histories(histories, accuracy_ylim=None):
    """Слева — потеря на обучении по эпохам, справа — accuracy на обучении (пунктир) и валидации."""
    fig, (ax_loss, ax_accuracy) = plt.subplots(1, 2, figsize=(15, 4.6))
    for (name, history), color in zip(histories.items(), CURVE_COLORS):
        epochs = np.arange(1, len(history["потеря на обучении"]) + 1)
        ax_loss.plot(epochs, history["потеря на обучении"], color=color, lw=2, label=name)
        ax_accuracy.plot(epochs, history["accuracy на обучении"], color=color, lw=1.2, ls="--")
        ax_accuracy.plot(epochs, history["accuracy на валидации"], color=color, lw=2, label=name)
    ax_loss.set_xlabel("эпоха")
    ax_loss.set_ylabel("кросс-энтропия на обучении")
    ax_loss.set_yscale("log")
    ax_loss.legend()
    ax_accuracy.set_xlabel("эпоха")
    ax_accuracy.set_ylabel("accuracy")
    ax_accuracy.set_title("пунктир — обучение, сплошная — валидация", fontsize=11)
    if accuracy_ylim is not None:
        ax_accuracy.set_ylim(*accuracy_ylim)
    ax_accuracy.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_confusion(y_true, y_predicted, n_classes=10):
    """Матрица ошибок: строка — истинная цифра, столбец — предсказанная; диагональ — верные ответы."""
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, predicted in zip(y_true, y_predicted):
        matrix[true, predicted] += 1
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(off_diagonal, cmap="Oranges")
    for row in range(n_classes):
        for column in range(n_classes):
            value = matrix[row, column]
            color = BLACK if row != column else BLUE
            ax.text(column, row, value, ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xlabel("предсказанная цифра")
    ax.set_ylabel("истинная цифра")
    ax.set_title("матрица ошибок; цвет — число ошибок, синие числа — верные ответы", fontsize=11)
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.show()


def plot_first_layer(weights, n_neurons=32, n_columns=8):
    """Веса первых нейронов первого слоя как картинки 28×28: синий — минус, оранжевый — плюс."""
    n_rows = int(np.ceil(n_neurons / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(1.6 * n_columns, 1.6 * n_rows))
    for index, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if index >= n_neurons:
            continue
        image = weights[:, index].reshape(28, 28)
        limit = np.abs(image).max()
        ax.imshow(image, cmap="PuOr_r", vmin=-limit, vmax=limit)
        ax.set_title(f"нейрон {index}", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_bars(values, xlabel, title=None, reference=None, reference_label=None):
    """Горизонтальные столбцы {подпись: значение}, лучший — темнее."""
    names = list(values)
    numbers = [values[name] for name in names]
    best = int(np.argmax(numbers))
    colors = [BLUE if index == best else GREY for index in range(len(names))]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(names) + 1.2))
    ax.barh(names, numbers, color=colors)
    for index, number in enumerate(numbers):
        ax.text(number, index, f" {number:.4f}", va="center", fontsize=10)
    if reference is not None:
        ax.axvline(reference, color=BLACK, ls="--", lw=1, label=reference_label)
        ax.legend(loc="lower right")
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    low = min(numbers) - 0.02
    ax.set_xlim(max(0.0, low), 1.0)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


# --- теория: вычислительные графы, активации, атаки -----------------------------
GREEN, RED = "#2E7D32", "#C62828"


def draw_graph(nodes, edges, title=None, size=(13, 5), step=None, ax=None, explanations=None, panel="right"):
    """Вычислительный граф в духе cs231n.

    nodes: {имя: (x, y, подпись, вид)}, вид — "input" (переменная) или "op" (операция);
    edges: [(откуда, куда, значение на прямом проходе, градиент или None)] или с пятым элементом —
    буквенным именем величины на ребре; тогда подписи выглядят как «q = 3» и «∂f/∂q = −4».
    explanations: [(заголовок, текст прямого прохода, текст обратного прохода)] в том же порядке,
    что edges; на текущем шаге пояснение выводится справа от графа (panel="right") или под ним ("below").

    step — сколько шагов показать: None — все; 0 — пустой граф; 1..E — первые step значений
    прямого прохода слева направо; E + 1..2E — все значения и первые step − E градиентов справа
    налево (E — число ребер). Так рисуют анимацию: один вызов на одно состояние.
    """
    order = sorted(range(len(edges)), key=lambda index: nodes[edges[index][1]][0])    # ребра слева направо
    n_edges = len(edges)
    if step is None:
        step = 2 * n_edges
    forward_shown = set(order[:min(step, n_edges)])
    backward_shown = set(order[::-1][:max(0, step - n_edges)])

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=size)
    current = None                                                  # ребро текущего шага
    if 1 <= step <= n_edges:
        current = order[step - 1]
    elif step > n_edges:
        current = order[::-1][step - n_edges - 1]
    for name, (x, y, label, kind) in nodes.items():
        if kind == "op":
            ax.add_patch(plt.Circle((x, y), 0.32, facecolor="white", edgecolor=BLACK, lw=1.5, zorder=3))
            ax.text(x, y, label, ha="center", va="center", fontsize=13, zorder=4)
        else:
            ax.text(x, y, label, ha="right", va="center", fontsize=14, zorder=4)
    for index, edge in enumerate(edges):
        source, target, value, gradient = edge[:4]
        name = edge[4] if len(edge) > 4 else None
        x0, y0 = nodes[source][0], nodes[source][1]
        x1, y1 = nodes[target][0], nodes[target][1]
        start_x = x0 + (0.32 if nodes[source][3] == "op" else 0.1)
        end_x = x1 - (0.32 if nodes[target][3] == "op" else 0.0)
        active_forward = index in forward_shown
        active_backward = index in backward_shown
        is_current = index == current and explanations is not None
        color = GREY if not (active_forward or active_backward) else BLACK
        if is_current:
            color = GREEN if step <= n_edges else RED
        ax.annotate("", xy=(end_x, y1), xytext=(start_x, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4 if color == GREY else 2.2 if is_current else 1.8,
                                    shrinkA=0, shrinkB=0))
        fraction = 0.35 if nodes[source][3] == "input" and y0 != y1 else 0.5     # у сходящихся ребер — ближе к началу
        middle_x = start_x + fraction * (end_x - start_x)
        middle_y = y0 + fraction * (y1 - y0)
        slanted = abs(y1 - y0) > 0.3
        offset = 0.26 if slanted else 0.14                                       # у наклонных ребер подписи дальше
        label_size = 14 if name else 12
        if value is not None and active_forward:
            value_text = f"${name} = {value:g}$" if name else f"{value:g}"
            ax.text(middle_x, middle_y + offset, value_text, color=GREEN, ha="center", va="bottom", fontsize=label_size,
                    weight="bold", bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85))
        if gradient is not None and active_backward:
            gradient_text = (r"$\frac{\partial f}{\partial " + name + r"} = " + f"{gradient:g}$") if name else f"{gradient:g}"
            ax.text(middle_x, middle_y - offset, gradient_text, color=RED, ha="center", va="top", fontsize=label_size,
                    weight="bold", bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85))
    xs = [x for x, _, _, _ in nodes.values()]
    ys = [y for _, y, _, _ in nodes.values()]
    if explanations is not None:
        panel_x = max(xs) + 0.9
        if current is None:
            text = ("Прямой проход: слева направо считаем\nзначения операций.\n\nОбратный проход: справа налево считаем\n"
                    r"$\frac{\partial f}{\partial\,\cdot}$ по величине на каждом ребре:" "\n"
                    "то, что пришло справа, умножаем\nна локальную производную узла.")
            color = BLACK
        else:
            label, forward_text, backward_text = explanations[current]
            if step <= n_edges:
                text = f"Ребро {label}, прямой проход\n\n{forward_text}"
                color = GREEN
            else:
                text = f"Ребро {label}, обратный проход\n\n{backward_text}"
                color = RED
        box = dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor=color, lw=1.5)
        if panel == "right":
            ax.text(panel_x, max(ys) + 0.6, text, ha="left", va="top", fontsize=15, color=BLACK, linespacing=2.3, bbox=box)
            ax.set_xlim(min(xs) - 1.2, panel_x + 10.0)
        else:
            ax.text(min(xs) - 0.6, min(ys) - 1.3, text, ha="left", va="top", fontsize=15, color=BLACK, linespacing=2.3, bbox=box)
            ax.set_xlim(min(xs) - 1.2, max(xs) + 1.0)
            ax.set_ylim(min(ys) - 6.2, max(ys) + 0.9)
    else:
        ax.set_xlim(min(xs) - 1.2, max(xs) + 1.0)
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    if step == 0:
        caption = "граф: переменные слева, результат справа"
    elif step <= n_edges:
        caption = f"прямой проход: шаг {step} из {n_edges}, зеленым — значения"
    else:
        caption = f"обратный проход: шаг {step - n_edges} из {n_edges}, красным — градиенты"
    ax.text(min(xs) - 1.1, min(ys) - 0.8, caption, fontsize=10, color=GREY)
    if title:
        ax.set_title(title, fontsize=12)
    if own_figure:
        plt.tight_layout()
        plt.show()


def _figure_png(draw, size):
    """Нарисовать draw(ax) в новую фигуру и вернуть ее PNG; фигура закрывается, inline-бэкенд ее не видит."""
    import io
    fig, ax = plt.subplots(figsize=size)
    draw(ax)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    png = buffer.getvalue()
    return png


def _stepper(n_steps, label, render):
    """Слайдер с кнопками; картинка — виджет Image, в который пишут PNG текущего состояния."""
    import ipywidgets as widgets
    from IPython.display import display

    slider = widgets.IntSlider(value=0, min=0, max=n_steps, step=1, description=label, continuous_update=False,
                               layout=widgets.Layout(width="420px"))
    back = widgets.Button(description="◀ назад", layout=widgets.Layout(width="90px"))
    forward = widgets.Button(description="вперед ▶", layout=widgets.Layout(width="90px"))
    image = widgets.Image(format="png", value=render(0))
    back.on_click(lambda _: setattr(slider, "value", max(0, slider.value - 1)))
    forward.on_click(lambda _: setattr(slider, "value", min(n_steps, slider.value + 1)))
    slider.observe(lambda change: setattr(image, "value", render(change["new"])), names="value")
    display(widgets.VBox([widgets.HBox([back, slider, forward]), image]))


def animate_graph(nodes, edges, title=None, size=(13, 5), explanations=None, panel="right"):
    """Виджет: слайдер и кнопки переключают состояние графа из draw_graph; одна картинка на состояние."""
    def render(step):
        png = _figure_png(lambda ax: draw_graph(nodes, edges, title=title, step=step, explanations=explanations,
                                                 panel=panel, ax=ax), size)
        return png

    _stepper(2 * len(edges), "шаг", render)


def plot_network_function():
    """Схема: сеть — функция входа x и весов W, потеря — функция ее выхода и ответа y."""
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.6)
    boxes = [
        (1.0, 2.2, "вход $x$\nкартинка, 784 числа", "#E3F2FD"),
        (4.8, 2.2, "сеть $f(x;\\, W)$\nслои с весами $W$", "#FFF3E0"),
        (8.4, 2.2, "выход $\\hat{y}$\nвероятности 10 классов", "#E3F2FD"),
        (11.6, 2.2, "потеря\n$L(\\hat{y},\\, y)$", "#FCE4EC"),
    ]
    for x, y, text, color in boxes:
        ax.text(x, y, text, ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor=BLACK))
    for x0, x1 in [(2.4, 3.4), (6.2, 7.0), (9.8, 10.7)]:
        ax.annotate("", xy=(x1, 2.2), xytext=(x0, 2.2), arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.5))
    ax.text(4.8, 0.65, "веса $W$\n≈ 100 000 чисел", ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=OCHRE))
    ax.annotate("", xy=(4.8, 1.55), xytext=(4.8, 1.05), arrowprops=dict(arrowstyle="-|>", color=OCHRE, lw=1.5))
    ax.text(11.6, 0.65, "правильный ответ $y$", ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=OCHRE))
    ax.annotate("", xy=(11.6, 1.6), xytext=(11.6, 1.0), arrowprops=dict(arrowstyle="-|>", color=OCHRE, lw=1.5))
    ax.text(6.5, 3.4, "$L(x, W, y) = L(f(x;\\, W),\\, y)$ — функция трех аргументов", ha="center", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_activations(functions):
    """functions: {имя: (функция, производная)} — слева графики, справа производные."""
    z = np.linspace(-5, 5, 500)
    fig, (ax_value, ax_derivative) = plt.subplots(1, 2, figsize=(15, 4.6))
    for (name, (function, derivative)), color in zip(functions.items(), CURVE_COLORS):
        ax_value.plot(z, function(z), color=color, lw=2, label=name)
        ax_derivative.plot(z, derivative(z), color=color, lw=2, label=name)
    ax_value.set_title("функция активации $\\varphi(z)$")
    ax_derivative.set_title("производная $\\varphi'(z)$")
    for ax in (ax_value, ax_derivative):
        ax.axhline(0, color=GREY, lw=0.8)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_xlabel("z")
        ax.legend()
    ax_value.set_ylim(-1.5, 3)
    plt.tight_layout()
    plt.show()


def plot_decision_maps(models, X, y, titles):
    """Карты классов обученных моделей на плоскости: цвет фона — предсказанный класс."""
    low, high = X.min(axis=0) - 0.3, X.max(axis=0) + 0.3
    grid_1, grid_2 = np.meshgrid(np.linspace(low[0], high[0], 300), np.linspace(low[1], high[1], 300))
    grid = np.column_stack([grid_1.ravel(), grid_2.ravel()])
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.6))
    for ax, model, title in zip(np.atleast_1d(axes), models, titles):
        probability = model.predict_proba(grid)[:, 1].reshape(grid_1.shape)
        ax.contourf(grid_1, grid_2, probability, levels=20, cmap="PuOr_r", alpha=0.6, vmin=0, vmax=1)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="PuOr_r", s=10, edgecolors="white", lw=0.3)
        accuracy = np.mean(model.predict(X) == y)
        ax.set_title(f"{title}\naccuracy {accuracy:.2f}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    plt.tight_layout()
    plt.show()


def show_adversarial(images, perturbations, adversarial, labels, before, after, epsilon):
    """По строке на пример: исходная цифра, добавка ε · sign(∇x L), атакованная цифра."""
    n = len(images)
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.6 * n))
    for row in range(n):
        panels = [
            (images[row], f"исходная: {labels[row]} → {before[row]}", "gray_r", (0, 1)),
            (perturbations[row], f"добавка, ε = {epsilon}", "PuOr_r", (-epsilon, epsilon)),
            (adversarial[row], f"после атаки → {after[row]}", "gray_r", (0, 1)),
        ]
        for ax, (image, title, cmap, (low, high)) in zip(axes[row], panels):
            ax.imshow(image.reshape(28, 28), cmap=cmap, vmin=low, vmax=high)
            ax.set_title(title, fontsize=10, color=RED if "атаки" in title and after[row] != labels[row] else BLACK)
            ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_curve(x, values, xlabel, ylabel, title=None):
    """Одна кривая с точками."""
    plt.figure(figsize=(8, 4.2))
    plt.plot(x, values, "o-", color=BLUE, lw=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.show()


def show_saliency(images, saliency, labels):
    """Исходные цифры и модуль градиента оценки верного класса по пикселям."""
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(1.9 * n, 4.2))
    for column in range(n):
        axes[0, column].imshow(images[column].reshape(28, 28), cmap="gray_r")
        axes[0, column].set_title(f"цифра {labels[column]}", fontsize=10)
        axes[1, column].imshow(saliency[column].reshape(28, 28), cmap="Oranges")
        for row in range(2):
            axes[row, column].axis("off")
    axes[1, 0].set_title("|∂ оценки / ∂ пиксель|", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_architecture(sizes, activation="ReLU"):
    """Схема полносвязной сети: столбцы нейронов по слоям, между ними веса и активация."""
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.axis("off")
    n_layers = len(sizes)
    x_positions = np.linspace(0.8, 12.2, n_layers)
    shown = [size if size <= 10 else 9 for size in sizes]      # до 10 нейронов рисуем все, иначе 9 с многоточием
    centers = []
    for x, size, count in zip(x_positions, sizes, shown):
        ys = np.linspace(1.0, 3.6, count)
        if count < size:
            ys = np.concatenate([ys[:count // 2], ys[count // 2 + 1:]])   # дырка под многоточие
            ax.text(x, (ys[count // 2 - 1] + ys[count // 2]) / 2, "⋮", ha="center", va="center", fontsize=16)
        for y in ys:
            ax.add_patch(plt.Circle((x, y), 0.14, facecolor="white", edgecolor=BLACK, lw=1.2, zorder=3))
        centers.append(ys)
    for k in range(n_layers - 1):
        for y0 in centers[k]:
            for y1 in centers[k + 1]:
                ax.plot([x_positions[k] + 0.14, x_positions[k + 1] - 0.14], [y0, y1], color=GREY, lw=0.35, alpha=0.6)
        middle = (x_positions[k] + x_positions[k + 1]) / 2
        label = f"$W_{k + 1}$: {sizes[k]} × {sizes[k + 1]}\n$b_{k + 1}$: {sizes[k + 1]}"
        if k < n_layers - 2:
            label += f"\n{activation}"
        else:
            label += "\nсофтмакс"
        ax.text(middle, 4.25, label, ha="center", va="bottom", fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=OCHRE))
    titles = ["вход\n784 пикселя"] + [f"скрытый слой {k}\n{size} нейронов" for k, size in enumerate(sizes[1:-1], start=1)] + ["выход\n10 оценок"]
    for x, title in zip(x_positions, titles):
        ax.text(x, 0.6, title, ha="center", va="top", fontsize=11)
    n_parameters = sum(sizes[k] * sizes[k + 1] + sizes[k + 1] for k in range(n_layers - 1))
    ax.set_title(f"полносвязная сеть {' → '.join(map(str, sizes))}: {n_parameters:,} параметров".replace(",", " "), fontsize=12)
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.4, 5.4)
    plt.tight_layout()
    plt.show()


def show_saliency_pairs(images, attacked, saliency_clean, saliency_attacked, labels, predicted_clean, predicted_attacked):
    """По столбцу на пример: исходная цифра и ее карта важности, атакованная цифра и ее карта."""
    n = len(images)
    fig, axes = plt.subplots(4, n, figsize=(1.9 * n, 8.0))
    for column in range(n):
        axes[0, column].imshow(images[column].reshape(28, 28), cmap="gray_r")
        axes[0, column].set_title(f"{labels[column]} → {predicted_clean[column]}", fontsize=10,
                                  color=BLACK if predicted_clean[column] == labels[column] else RED)
        axes[1, column].imshow(saliency_clean[column].reshape(28, 28), cmap="Oranges")
        axes[2, column].imshow(attacked[column].reshape(28, 28), cmap="gray_r")
        axes[2, column].set_title(f"{labels[column]} → {predicted_attacked[column]}", fontsize=10,
                                  color=BLACK if predicted_attacked[column] == labels[column] else RED)
        axes[3, column].imshow(saliency_attacked[column].reshape(28, 28), cmap="Oranges")
        for row in range(4):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            axes[row, column].grid(False)
    for row, name in enumerate(["исходная", "важность пикселей", "после атаки", "важность пикселей"]):
        axes[row, 0].set_ylabel(name, fontsize=10)
    plt.tight_layout()
    plt.show()


# --- прямой проход по слоям: анимация -------------------------------------------
def _layer_columns(sizes, n_show=6):
    """Для каждого слоя — список y-координат показанных нейронов; длинные слои сокращены до n_show."""
    columns = []
    for size in sizes:
        count = min(size, n_show)
        ys = np.linspace(0.6, 4.4, count)
        columns.append((ys, count < size))
    return columns


def draw_forward_pass(stages, sizes, step=None, ax=None, size=(16, 6.4)):
    """Прямой проход сети по стадиям: вход → z₁ = xW₁ + b₁ → h₁ = φ(z₁) → … → софтмакс.

    stages: [(подпись, вектор значений для одного объекта)] в порядке вычисления; длина вектора —
    число нейронов слоя. step — сколько стадий показать (None — все). Показанные стадии рисуют
    значения в кружках и окрашивают их; непоказанные — пустые серые.
    """
    n_stages = len(stages)
    if step is None:
        step = n_stages
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=size)
    xs = np.linspace(0.8, 15.2, n_stages)
    columns = _layer_columns([len(values) for _, values in stages])
    for k, ((label, values), (ys, cut), x) in enumerate(zip(stages, columns, xs)):
        active = k < step
        current = k == step - 1
        values = np.asarray(values, dtype=float)
        shown_values = values[:len(ys)] if not cut else np.concatenate([values[:len(ys) // 2], values[-(len(ys) - len(ys) // 2):]])
        limit = max(np.abs(values).max(), 1e-9)
        for y, value in zip(ys, shown_values):
            if active:
                strength = abs(value) / limit
                face = (OCHRE if value >= 0 else BLUE) if strength > 0.02 else "white"
                ax.add_patch(plt.Circle((x, y), 0.26, facecolor=face, edgecolor=RED if current else BLACK,
                                        lw=2.0 if current else 1.0, alpha=0.25 + 0.75 * strength if face != "white" else 1.0, zorder=3))
                ax.text(x, y, f"{value + 0.0:.2f}".replace("-0.00", "0.00"), ha="center", va="center", fontsize=8.5, zorder=4)
            else:
                ax.add_patch(plt.Circle((x, y), 0.26, facecolor="white", edgecolor=GREY, lw=1.0, zorder=3))
        if cut:
            ax.text(x, (ys[len(ys) // 2 - 1] + ys[len(ys) // 2]) / 2, "⋮", ha="center", va="center", fontsize=14)
        ax.text(x, 5.05, label, ha="center", va="bottom", fontsize=10.5, color=RED if current else BLACK,
                weight="bold" if current else "normal")
        ax.text(x, 0.1, f"{len(values)}", ha="center", va="top", fontsize=9, color=GREY)
        if k + 1 < n_stages:
            ax.annotate("", xy=(xs[k + 1] - 0.35, 2.5), xytext=(x + 0.35, 2.5),
                        arrowprops=dict(arrowstyle="-|>", color=BLACK if k + 1 < step else GREY, lw=1.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.4, 5.9)
    ax.set_aspect("equal")
    ax.axis("off")
    caption = "граф сети: столбцы — стадии вычисления, число под столбцом — размер вектора" if step == 0 \
        else f"стадия {step} из {n_stages}: {stages[step - 1][0]}; оранжевый — положительные значения, синий — отрицательные"
    ax.text(0.1, -0.35, caption, fontsize=10, color=GREY)
    if own:
        plt.tight_layout()
        plt.show()


def animate_forward_pass(stages, sizes, size=(16, 6.4)):
    """Виджет: слайдер и кнопки переключают стадию прямого прохода из draw_forward_pass."""
    def render(step):
        png = _figure_png(lambda ax: draw_forward_pass(stages, sizes, step=step, ax=ax), size)
        return png

    _stepper(len(stages), "стадия", render)


def plot_softmax_bars(logits, probabilities, label):
    """Слева оценки z, справа вероятности softmax(z); верный класс подсвечен."""
    fig, (ax_z, ax_p) = plt.subplots(1, 2, figsize=(13, 3.8))
    classes = np.arange(len(logits))
    colors = [OCHRE if k == label else GREY for k in classes]
    ax_z.bar(classes, logits, color=colors)
    ax_z.set_title("оценки $z$ — выход последнего линейного слоя")
    ax_z.set_xticks(classes)
    ax_z.axhline(0, color=BLACK, lw=0.8)
    ax_p.bar(classes, probabilities, color=colors)
    ax_p.set_title(f"softmax(z): вероятности классов, сумма {probabilities.sum():.2f}")
    ax_p.set_xticks(classes)
    ax_p.set_ylim(0, 1)
    for ax in (ax_z, ax_p):
        ax.set_xlabel("класс")
    plt.tight_layout()
    plt.show()


def plot_init_comparison(stage_names, scale_by_init, max_probability_by_init):
    """Слева — типичный размер значений на каждой стадии прямого прохода (логарифмическая шкала),
    справа — гистограмма наибольшей вероятности софтмакса по картинкам; по цвету на инициализацию."""
    fig, (ax_scale, ax_probability) = plt.subplots(1, 2, figsize=(14, 4.4))
    colors = [GREY, "#B5524B", BLUE, OCHRE]
    positions = np.arange(len(stage_names))
    for color, (name, scales) in zip(colors, scale_by_init.items()):
        ax_scale.plot(positions, scales, marker="o", lw=2, color=color, label=name)
        last = scales[-1]
        last_label = f"{last:.0f}" if last >= 100 else f"{last:.2f}"
        ax_scale.text(positions[-1] + 0.1, last, last_label, va="center", fontsize=9, color=color)
    ax_scale.set_yscale("log")
    ax_scale.set_xticks(positions)
    ax_scale.set_xticklabels(stage_names)
    ax_scale.set_xlim(-0.3, len(stage_names) - 0.2)
    ax_scale.set_ylabel("корень из среднего квадрата значений")
    ax_scale.set_title("типичный размер значений по слоям")

    bins = np.linspace(0.0, 1.0, 41)
    for color, (name, values) in zip(colors, max_probability_by_init.items()):
        ax_probability.hist(values, bins=bins, color=color, alpha=0.75, label=name)
    ax_probability.set_xlabel("наибольшая вероятность софтмакса на картинке")
    ax_probability.set_ylabel("число картинок")
    ax_probability.set_title("насколько уверена необученная сеть")
    ax_probability.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


# --- слои сети для лекции ------------------------------------------------------
# Те же модули, что пишут на семинаре: полносвязный слой, активации, софтмакс
# с кросс-энтропией, контейнер Sequential, градиентный спуск и сборка сети.

class Linear:
    """Полносвязный слой: Y = X W + b."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator):
        self.W = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))     # инициализация He
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        output = X @ self.W + self.b
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self.dW = self.X.T @ grad_output
        self.db = grad_output.sum(axis=0)
        grad_input = grad_output @ self.W.T
        return grad_input

    def parameters(self) -> list:
        pairs = [(self.W, self.dW), (self.b, self.db)]
        return pairs

class ReLU:
    """ReLU(z) = max(z, 0) поэлементно."""

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.mask = X > 0
        output = X * self.mask
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_input = grad_output * self.mask
        return grad_input

    def parameters(self) -> list:
        return []

class Sigmoid:
    """σ(z) = 1 / (1 + e^(−z)) поэлементно."""

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.output = 0.5 * (1.0 + np.tanh(0.5 * X))          # то же, что 1 / (1 + e^(−z)), без переполнения
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_input = grad_output * self.output * (1.0 - self.output)
        return grad_input

    def parameters(self) -> list:
        return []

class LeakyReLU:
    """max(z, α z) поэлементно."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.slope = np.where(X > 0, 1.0, self.alpha)
        output = X * self.slope
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_input = grad_output * self.slope
        return grad_input

    def parameters(self) -> list:
        return []

class SoftmaxCrossEntropy:
    """Средняя кросс-энтропия софтмакса оценок по объектам."""

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        shifted = logits - logits.max(axis=1, keepdims=True)        # устойчивость: exp не переполняется
        exp_scores = np.exp(shifted)
        self.probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        self.y = y

        n = len(y)
        true_class_probability = self.probabilities[np.arange(n), y]
        loss = float(-np.mean(np.log(true_class_probability + 1e-12)))
        return loss

    def backward(self) -> np.ndarray:
        n = len(self.y)
        grad_logits = self.probabilities.copy()
        grad_logits[np.arange(n), self.y] -= 1.0
        grad_logits = grad_logits / n
        return grad_logits

class Sequential:
    """Слои по очереди: forward слева направо, backward справа налево."""

    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, X: np.ndarray) -> np.ndarray:
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> list:
        pairs = [pair for layer in self.layers for pair in layer.parameters()]
        return pairs


class SGD:
    """Градиентный спуск: параметр ← параметр − lr · градиент, на месте."""

    def __init__(self, lr: float):
        self.lr = lr

    def step(self, parameters: list) -> None:
        for parameter, gradient in parameters:
            parameter -= self.lr * gradient

ACTIVATIONS = {"relu": ReLU, "sigmoid": Sigmoid, "leaky_relu": LeakyReLU}


def init_weights(n_in: int, n_out: int, method: str, rng: np.random.Generator) -> np.ndarray:
    """Начальные веса слоя n_in → n_out: he, xavier, normal (σ = 1), constant (все 0.01), zeros."""
    if method == "he":
        W = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
    elif method == "xavier":
        W = rng.normal(0.0, np.sqrt(2.0 / (n_in + n_out)), size=(n_in, n_out))
    elif method == "normal":
        W = rng.normal(0.0, 1.0, size=(n_in, n_out))
    elif method == "constant":
        W = np.full((n_in, n_out), 0.01)
    elif method == "zeros":
        W = np.zeros((n_in, n_out))
    else:
        raise ValueError(f"неизвестная инициализация: {method}")
    return W


def make_network(hidden_sizes: list, activation: str, init: str, rng: np.random.Generator,
                 n_in: int = 784, n_out: int = 10) -> Sequential:
    """Полносвязная сеть n_in → hidden_sizes → n_out: после каждого скрытого слоя — активация."""
    layers = []
    previous = n_in
    for size in hidden_sizes:
        layer = Linear(previous, size, rng)
        layer.W = init_weights(previous, size, init, rng)
        layers.append(layer)
        layers.append(ACTIVATIONS[activation]())
        previous = size
    output = Linear(previous, n_out, rng)
    output.W = init_weights(previous, n_out, init, rng)
    layers.append(output)
    network = Sequential(layers)
    return network
