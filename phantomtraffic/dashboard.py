import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd

# THEME
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Phantom Traffic Simulator")
app.geometry("1500x900")
app.minsize(1300, 800)

# LOAD BENCHMARK
def load_benchmark():
    data = {
        "Serial": 0.0,
        "OpenMP": 0.0,
        "CUDA": 0.0,
        "MPI": 0.0
    }

    try:
        with open("benchmark.txt", "r") as f:
            for line in f:
                line = line.strip()

                if not line or "=" not in line:
                    continue

                key, value = line.split("=")

                if key in data:
                    data[key] = float(value)

    except FileNotFoundError:
        pass

    return data

bench = load_benchmark()

backend_colors = {
    "Serial": "#4ea1ff",
    "OpenMP": "#4caf50",
    "CUDA": "#ff9800",
    "MPI": "#9c6bff"
}

def change_backend(choice):
    backend_label.configure(
        text=choice,
        text_color=backend_colors[choice]
    )

    # Update thread/process information
    if choice == "Serial":
        threads_label.configure(text="Threads: 1")
    elif choice == "OpenMP":
        threads_label.configure(text="Threads: 16")
    elif choice == "CUDA":
        threads_label.configure(text="Threads: GPU")
    elif choice == "MPI":
        threads_label.configure(text="Threads: 4 processes")

    # Update status based on available benchmark data
    idx = labels.index(choice)

    if runtime[idx] > 0:
        status_info.configure(
            text="Status: Measured",
            text_color="#4caf50"
        )
    else:
        status_info.configure(
            text="Status: Pending",
            text_color="#ff9800"
        )

labels = ["Serial", "OpenMP", "CUDA", "MPI"]

runtime = [
    bench["Serial"],
    bench["OpenMP"],
    bench["CUDA"],
    bench["MPI"]
]

speedup = [
    1.0,
    runtime[0] / runtime[1] if runtime[1] > 0 else 0,
    runtime[0] / runtime[2] if runtime[2] > 0 else 0,
    runtime[0] / runtime[3] if runtime[3] > 0 else 0
]

# MAIN LAYOUT
top_frame = ctk.CTkFrame(app, fg_color="transparent")
top_frame.pack(fill="both", expand=True, padx=10, pady=10)

# LEFT PANEL
left = ctk.CTkFrame(top_frame, width=250, corner_radius=15)
left.pack(side="left", fill="y", padx=(0, 10))

ctk.CTkLabel(
    left,
    text="Control Panel",
    font=("Segoe UI", 24, "bold")
).pack(pady=(20, 15))

# Backend selector
ctk.CTkLabel(
    left,
    text="Backend",
    font=("Segoe UI", 16, "bold")
).pack(pady=(5, 8))

backend_var = ctk.StringVar(value="OpenMP")

backend_menu = ctk.CTkOptionMenu(
    left,
    values=["Serial", "OpenMP", "CUDA", "MPI"],
    variable=backend_var,
    width=170,
    command=change_backend
)
backend_menu.pack(padx=25, pady=(0, 15))

# Info card
info = ctk.CTkFrame(left, corner_radius=12)
info.pack(fill="x", padx=18, pady=(5, 15))

ctk.CTkLabel(
    info,
    text="Selected Backend",
    font=("Segoe UI", 13, "bold")
).pack(anchor="w", padx=12, pady=(10, 2))

backend_label = ctk.CTkLabel(
    info,
    text="OpenMP",
    font=("Segoe UI", 20, "bold"),
    text_color="#4caf50"
)
backend_label.pack(anchor="w", padx=12)

threads_label = ctk.CTkLabel(
    info,
    text=f"Threads: - ",
    text_color="#bbbbbb"
)
threads_label.pack(anchor="w", padx=12, pady=(6, 0))

road_info = ctk.CTkLabel(info, text="Road: 100 cells", text_color="#bbbbbb")
road_info.pack(anchor="w", padx=12)

veh_info = ctk.CTkLabel(info, text="Vehicles: 30", text_color="#bbbbbb")
veh_info.pack(anchor="w", padx=12)

speed_info = ctk.CTkLabel(info, text="Max Speed: 5", text_color="#bbbbbb")
speed_info.pack(anchor="w", padx=12)

status_info = ctk.CTkLabel(info, text="Status: Ready", text_color="#4ea1ff")
status_info.pack(anchor="w", padx=12, pady=(0, 10))

ctk.CTkLabel(
    left,
    text="Simulation Parameters",
    font=("Segoe UI", 16, "bold")
).pack(pady=(20, 10))

def labeled_entry(parent, label, default):
    ctk.CTkLabel(parent, text=label).pack(anchor="w", padx=25, pady=(10, 4))
    entry = ctk.CTkEntry(parent)
    entry.insert(0, default)
    entry.pack(fill="x", padx=25)
    return entry

road_entry = labeled_entry(left, "Road Length (cells)", "100")
veh_entry = labeled_entry(left, "Number of Vehicles", "30")
speed_entry = labeled_entry(left, "Max Speed (cells/step)", "5")

# CENTER PANEL
center = ctk.CTkFrame(top_frame, corner_radius=15)
center.pack(side="left", fill="both", expand=True, padx=(0, 10))

ctk.CTkLabel(
    center,
    text="Traffic Visualization",
    font=("Segoe UI", 24, "bold")
).pack(pady=(15, 5))

# Legend
legend_frame = ctk.CTkFrame(center, fg_color="transparent")
legend_frame.pack(pady=(0, 5))

legend_colors = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]
legend_labels = ["0", "1", "2", "3", "5"]

for c, t in zip(legend_colors, legend_labels):
    dot = ctk.CTkLabel(
        legend_frame,
        text="●",
        text_color=c,
        font=("Segoe UI", 18)
    )
    dot.pack(side="left", padx=(8, 2))
    ctk.CTkLabel(legend_frame, text=t).pack(side="left", padx=(0, 6))

# Matplotlib figure
fig, ax = plt.subplots(figsize=(7.2, 7.2))
fig.patch.set_facecolor("#1f1f1f")
ax.set_facecolor("#1f1f1f")
ax.set_aspect("equal")
ax.axis("off")

outer = plt.Circle((0, 0), 1.02, edgecolor="white", facecolor="none", linewidth=2)
inner = plt.Circle((0, 0), 0.92, edgecolor="#777777", facecolor="none", linewidth=1)

ax.add_patch(outer)
ax.add_patch(inner)

ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-1.15, 1.15)

ROAD = 100
N = 30
MAX_SPEED = 5

positions = np.linspace(0, ROAD - 1, N)
velocities = np.zeros(N)

theta = 2 * np.pi * positions / ROAD

scat = ax.scatter(
    np.cos(theta),
    np.sin(theta),
    c=velocities,
    cmap="plasma",
    vmin=0,
    vmax=MAX_SPEED,
    s=160,
    edgecolors="white",
    linewidths=1.2
)

canvas = FigureCanvasTkAgg(fig, master=center)
canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

# SIMULATION CONTROL
running = False
paused = False
frame_count = 0

def refresh_info():
    road_info.configure(text=f"Road: {road_entry.get()} cells")
    veh_info.configure(text=f"Vehicles: {veh_entry.get()}")
    speed_info.configure(text=f"Max Speed: {speed_entry.get()}")

def reset_simulation():
    global ROAD, N, MAX_SPEED
    global positions, velocities, frame_count

    ROAD = int(road_entry.get())
    N = int(veh_entry.get())
    MAX_SPEED = int(speed_entry.get())

    positions = np.linspace(0, ROAD - 1, N)
    velocities = np.zeros(N)
    frame_count = 0

    theta = 2 * np.pi * positions / ROAD
    coords = np.column_stack((np.cos(theta), np.sin(theta)))

    scat.set_offsets(coords)
    scat.set_array(velocities)

    refresh_info()

    status.configure(
        text=f"Ready   |   Vehicles: {N}   |   Average Speed: 0.00"
    )

    canvas.draw_idle()

def start_simulation():
    global running, paused
    running = True
    paused = False
    status_info.configure(text="Status: Running", text_color="#00d4ff")

def pause_simulation():
    global paused
    paused = not paused

    if paused:
        status_info.configure(text="Status: Paused", text_color="#ff9800")
    else:
        status_info.configure(text="Status: Running", text_color="#00d4ff")

def stop_simulation():
    global running, paused
    running = False
    paused = False
    status_info.configure(text="Status: Ready", text_color="#4ea1ff")
    reset_simulation()

def step_simulation():
    global running, paused
    running = True
    paused = False
    update(0)
    running = False

# ANIMATION
def update(frame):
    global positions, velocities, frame_count

    if not running or paused:
        return scat,

    frame_count += 1

    for i in range(N):
        velocities[i] = min(velocities[i] + 1, MAX_SPEED)

        nxt = (i + 1) % N
        gap = (positions[nxt] - positions[i] - 1) % ROAD

        velocities[i] = min(velocities[i], gap)

        if velocities[i] > 0 and np.random.rand() < 0.25:
            velocities[i] -= 1

    positions[:] = (positions + velocities) % ROAD

    order = np.argsort(positions)
    positions[:] = positions[order]
    velocities[:] = velocities[order]

    theta = 2 * np.pi * positions / ROAD
    coords = np.column_stack((np.cos(theta), np.sin(theta)))

    scat.set_offsets(coords)
    scat.set_array(velocities)

    avg_speed = np.mean(velocities)

    status.configure(
        text=f"Timestep: {frame_count}   |   Vehicles: {N}   |   Average Speed: {avg_speed:.2f}   |   Running"
    )

    return scat,

ani = FuncAnimation(fig, update, interval=90, blit=True)

# Controls
controls = ctk.CTkFrame(center, fg_color="transparent")
controls.pack(pady=(5, 12))

ctk.CTkButton(controls, text="▶ Start", width=90, command=start_simulation).pack(side="left", padx=6)
ctk.CTkButton(controls, text="⏸ Pause", width=90, command=pause_simulation).pack(side="left", padx=6)
ctk.CTkButton(controls, text="⏹ Reset", width=90, command=stop_simulation).pack(side="left", padx=6)
ctk.CTkButton(controls, text="Step", width=80, command=step_simulation).pack(side="left", padx=6)

# RIGHT PANEL
right = ctk.CTkFrame(top_frame, width=420, corner_radius=15)
right.pack(side="right", fill="y")

colors = ["#4ea1ff", "#4caf50", "#ff9800", "#9c6bff"]

# Runtime chart
ctk.CTkLabel(
    right,
    text="Runtime Comparison",
    font=("Segoe UI", 18, "bold")
).pack(pady=(18, 8))

runtime_fig, runtime_ax = plt.subplots(figsize=(4.4, 3.0))
runtime_fig.patch.set_facecolor("#1f1f1f")
runtime_ax.set_facecolor("#1f1f1f")

bars = runtime_ax.bar(labels, runtime, color=colors)

runtime_ax.tick_params(colors="white")
runtime_ax.set_ylabel("ms", color="white")

for spine in runtime_ax.spines.values():
    spine.set_color("white")

for b, v in zip(bars, runtime):
    runtime_ax.text(
        b.get_x() + b.get_width() / 2,
        v + max(runtime) * 0.03,
        f"{v:.2f}",
        ha="center",
        color="white",
        fontsize=9
    )

runtime_canvas = FigureCanvasTkAgg(runtime_fig, master=right)
runtime_canvas.get_tk_widget().pack(padx=8, pady=5)

# Speedup chart
ctk.CTkLabel(
    right,
    text="Speedup Comparison",
    font=("Segoe UI", 18, "bold")
).pack(pady=(15, 8))

speed_fig, speed_ax = plt.subplots(figsize=(4.4, 3.0))
speed_fig.patch.set_facecolor("#1f1f1f")
speed_ax.set_facecolor("#1f1f1f")

speed_ax.plot(labels, speedup, marker="o", linewidth=2.5, color="#00d4ff")

speed_ax.tick_params(colors="white")
speed_ax.set_ylabel("Speedup (x)", color="white")

for spine in speed_ax.spines.values():
    spine.set_color("white")

for x, y in zip(labels, speedup):
    speed_ax.text(x, y + 0.15, f"{y:.2f}x", color="white", ha="center", fontsize=9)

speed_canvas = FigureCanvasTkAgg(speed_fig, master=right)
speed_canvas.get_tk_widget().pack(padx=8, pady=5)

# BOTTOM TABLE

bottom = ctk.CTkFrame(app, corner_radius=15)
bottom.pack(fill="x", padx=10, pady=(0, 10))

ctk.CTkLabel(
    bottom,
    text="Performance Summary",
    font=("Segoe UI", 18, "bold")
).pack(pady=(10, 5))

table = ctk.CTkTextbox(bottom, height=140, font=("Consolas", 13))
table.pack(fill="x", padx=15, pady=(0, 10))

df = pd.DataFrame({
    "Backend": labels,
    "Runtime (ms)": runtime,
    "Throughput": [round(1000 / x, 2) if x > 0 else 0 for x in runtime],
    "Speedup": speedup,
    "Status": ["Measured" if x > 0 else "Pending" for x in runtime]
})

table.insert("1.0", df.to_string(index=False))
table.configure(state="disabled")

# STATUS BAR
status = ctk.CTkLabel(
    app,
    text="Ready   |   Vehicles: 30   |   Average Speed: 0.00",
    anchor="w",
    font=("Segoe UI", 12)
)
status.pack(fill="x", padx=15, pady=(0, 8))

# Initialize
reset_simulation()
change_backend(backend_var.get())

app.mainloop()