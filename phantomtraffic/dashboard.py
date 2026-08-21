import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import subprocess
import os
import re
import shutil

# THEME
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()

app.title("Phantom Traffic Simulator")

app.geometry("1500x900")
app.minsize(1300, 800)

# PROJECT DIRECTORY
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

EXECUTABLE_CANDIDATES = [
    os.path.join(BASE_DIR, "cmake-build-debug", "phantomtraffic.exe"),
    os.path.join(BASE_DIR, "cmake-build-release", "phantomtraffic.exe"),
    os.path.join(BASE_DIR, "build", "phantomtraffic.exe"),

    os.path.join(PROJECT_ROOT, "x64", "Release", "phantomtraffic.exe"),
    os.path.join(PROJECT_ROOT, "x64", "Debug", "phantomtraffic.exe"),
    os.path.join(BASE_DIR, "phantomtraffic.exe"),
]

DEFAULT_OPENMP_THREADS = 16
DEFAULT_MPI_PROCESSES = 4

# ============================================================
# GLOBAL DATA
# ============================================================

labels = ["Serial", "OpenMP", "CUDA", "MPI"]

backend_colors = {
    "Serial": "#4ea1ff",
    "OpenMP": "#4caf50",
    "CUDA": "#ff9800",
    "MPI": "#9c6bff"
}

colors = ["#4ea1ff", "#4caf50", "#ff9800", "#9c6bff"]

# ============================================================
# BENCHMARK DATA
# ============================================================

def find_executable():
    return next(
        (path for path in EXECUTABLE_CANDIDATES if os.path.exists(path)),
        None
    )

def parse_benchmark_output(output):
    """Read lines emitted by main.cpp: RESULT <backend> <milliseconds>."""
    data = {label: 0.0 for label in labels}
    pattern = re.compile(
        r"^RESULT\s+(Serial|OpenMP|CUDA|MPI)\s+([0-9]+(?:\.[0-9]+)?)$"
    )

    for line in output.splitlines():
        match = pattern.match(line.strip())
        if match:
            data[match.group(1)] = float(match.group(2))

    return data

bench = {label: 0.0 for label in labels}

runtime = [
    bench["Serial"],
    bench["OpenMP"],
    bench["CUDA"],
    bench["MPI"]
]

speedup = [
    1.0 if runtime[0] > 0 else 0,
    runtime[0] / runtime[1] if runtime[0] > 0 and runtime[1] > 0 else 0,
    runtime[0] / runtime[2] if runtime[0] > 0 and runtime[2] > 0 else 0,
    runtime[0] / runtime[3] if runtime[0] > 0 and runtime[3] > 0 else 0
]

def update_benchmark_data(data):
    global bench, runtime, speedup

    bench = data

    runtime = [
        bench["Serial"],
        bench["OpenMP"],
        bench["CUDA"],
        bench["MPI"]
    ]

    if runtime[0] > 0:
        speedup = [
            1.0,
            runtime[0] / runtime[1] if runtime[1] > 0 else 0,
            runtime[0] / runtime[2] if runtime[2] > 0 else 0,
            runtime[0] / runtime[3] if runtime[3] > 0 else 0
        ]
    else:
        speedup = [0, 0, 0, 0]

    update_charts()
    update_table()
    change_backend(backend_var.get())

# ============================================================
# RUN C++ BENCHMARK
# ============================================================

def run_cpp_benchmark():
    try:
        road_length = int(road_entry.get())
        num_vehicles = int(veh_entry.get())
        max_speed = int(speed_entry.get())

        openmp_threads = int(openmp_entry.get())
        mpi_processes = int(mpi_entry.get())

        if road_length <= 0:
            raise ValueError("Road length must be greater than 0.")

        if num_vehicles <= 0:
            raise ValueError("Number of vehicles must be greater than 0.")

        if max_speed <= 0:
            raise ValueError("Maximum speed must be greater than 0.")

        if openmp_threads <= 0:
            raise ValueError("OpenMP threads must be greater than 0.")

        if mpi_processes <= 0:
            raise ValueError("MPI processes must be greater than 0.")

        reset_simulation()

        status_info.configure(
            text="Status: Running Benchmark...",
            text_color="#00d4ff"
        )

        status.configure(
            text="Running C++ performance test..."
        )

        app.update()

        executable_path = find_executable()

        if executable_path is None:
            status_info.configure(
                text="Status: EXE Not Found",
                text_color="#ff4444"
            )

            status.configure(
                text="ERROR: phantomtraffic.exe was not found."
            )

            print("ERROR: phantomtraffic.exe NOT FOUND")
            print("Expected locations:")
            print("\n".join(EXECUTABLE_CANDIDATES))

            return

        # ----------------------------------------------------
        # BASE COMMAND
        # ----------------------------------------------------

        command = [
            executable_path,
            str(road_length),
            str(num_vehicles),
            str(max_speed)
        ]

        # ----------------------------------------------------
        # ENVIRONMENT
        # ----------------------------------------------------

        env = os.environ.copy()

        # OpenMP thread count
        env["OMP_NUM_THREADS"] = str(openmp_threads)

        # ----------------------------------------------------
        # MPI
        # ----------------------------------------------------

        selected_backend = backend_var.get()

        if selected_backend == "MPI":
            mpiexec = shutil.which("mpiexec")

            if mpiexec is None:
                status_info.configure(
                    text="Status: MPI Not Found",
                    text_color="#ff4444"
                )

                status.configure(
                    text="ERROR: mpiexec was not found."
                )

                return

            command = [
                mpiexec,
                "-n",
                str(mpi_processes),
                executable_path,
                str(road_length),
                str(num_vehicles),
                str(max_speed)
            ]

        print("\n======================================")
        print("RUNNING C++ BENCHMARK")
        print("======================================")
        print("Backend:", selected_backend)
        print("Executable:", executable_path)
        print("Road Length:", road_length)
        print("Vehicles:", num_vehicles)
        print("Max Speed:", max_speed)
        print("OpenMP Threads:", openmp_threads)
        print("MPI Processes:", mpi_processes)
        print("Command:", command)

        # ----------------------------------------------------
        # RUN C++ PROGRAM
        # ----------------------------------------------------

        result = subprocess.run(
            command,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            env=env
        )

        # ----------------------------------------------------
        # ERROR
        # ----------------------------------------------------

        if result.returncode != 0:
            print("\n========== C++ ERROR ==========")
            print(result.stderr)

            status_info.configure(
                text="Status: Benchmark Error",
                text_color="#ff4444"
            )

            status.configure(
                text="C++ benchmark failed. Check the terminal output."
            )

            return

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        print("\n========== C++ OUTPUT ==========")
        print(result.stdout)

        measured = parse_benchmark_output(result.stdout)

        if not any(measured.values()):
            raise RuntimeError(
                "The executable completed but did not return RESULT lines. "
                "Rebuild it after updating main.cpp."
            )

        update_benchmark_data(measured)

        status_info.configure(
            text="Status: Benchmark Complete",
            text_color="#4caf50"
        )

        status.configure(
            text=(
                f"Benchmark Complete   |   "
                f"Vehicles: {num_vehicles}   |   "
                f"Road: {road_length}   |   "
                f"Max Speed: {max_speed}   |   "
                f"OpenMP: {openmp_threads} threads   |   "
                f"MPI: {mpi_processes} processes"
            )
        )

    except ValueError as e:
        status_info.configure(
            text="Status: Invalid Parameters",
            text_color="#ff4444"
        )

        status.configure(
            text=f"Invalid parameters: {e}"
        )

    except Exception as e:
        print("\n========== PYTHON ERROR ==========")
        print(e)

        status_info.configure(
            text="Status: Error",
            text_color="#ff4444"
        )

        status.configure(
            text=f"Error: {e}"
        )

# ============================================================
# UPDATE CHARTS
# ============================================================

def update_charts():
    runtime_ax.clear()
    runtime_ax.set_facecolor("#1f1f1f")

    bars = runtime_ax.bar(labels, runtime, color=colors)
    runtime_ax.tick_params(colors="white")
    runtime_ax.set_ylabel("Runtime (ms)", color="white")
    runtime_ax.set_title("Runtime Comparison", color="white", fontsize=11)
    runtime_ax.grid(axis="y", alpha=0.15)

    for spine in runtime_ax.spines.values():
        spine.set_color("white")

    max_runtime = max(runtime)
    if max_runtime <= 0:
        max_runtime = 1

    for bar, value in zip(bars, runtime):
        if value > 0:
            runtime_ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max_runtime * 0.03,
                f"{value:.2f}",
                ha="center",
                color="white",
                fontsize=9
            )

    runtime_ax.set_ylim(0, max_runtime * 1.2)

    speed_ax.clear()
    speed_ax.set_facecolor("#1f1f1f")
    speed_ax.plot(
        labels,
        speedup,
        marker="o",
        linewidth=2.5,
        color="#00d4ff"
    )
    speed_ax.tick_params(colors="white")
    speed_ax.set_ylabel("Speedup (x)", color="white")
    speed_ax.set_title("Speedup Comparison", color="white", fontsize=11)
    speed_ax.grid(axis="y", alpha=0.15)

    for spine in speed_ax.spines.values():
        spine.set_color("white")

    max_speedup = max(speedup)
    if max_speedup <= 0:
        max_speedup = 1

    for x, y in zip(labels, speedup):
        if y > 0:
            speed_ax.text(
                x,
                y + max_speedup * 0.05,
                f"{y:.2f}x",
                color="white",
                ha="center",
                fontsize=9
            )

    speed_ax.set_ylim(0, max_speedup * 1.25)

    runtime_fig.tight_layout()
    speed_fig.tight_layout()
    runtime_canvas.draw()
    speed_canvas.draw()

# ============================================================
# UPDATE TABLE
# ============================================================

def update_table():
    table.configure(state="normal")
    table.delete("1.0", "end")

    throughput = [
        round(1000 / x, 2) if x > 0 else 0
        for x in runtime
    ]

    status_values = [
        "Measured" if x > 0 else "Pending"
        for x in runtime
    ]

    df = pd.DataFrame({
        "Backend": labels,
        "Runtime (ms)": [round(x, 2) for x in runtime],
        "Throughput": throughput,
        "Speedup": [round(x, 2) for x in speedup],
        "Status": status_values
    })

    table.insert("1.0", df.to_string(index=False))
    table.configure(state="disabled")

# ============================================================
# BACKEND SELECTION
# ============================================================

def change_backend(choice):
    backend_label.configure(
        text=choice,
        text_color=backend_colors[choice]
    )

    if choice == "Serial":
        threads_label.configure(text="Threads: 1")
    elif choice == "OpenMP":
        # Reflect whatever the user has typed into the OpenMP Threads
        # entry instead of a hardcoded value.
        threads_value = openmp_entry.get().strip()
        threads_label.configure(
            text=f"Threads: {threads_value if threads_value else DEFAULT_OPENMP_THREADS}"
        )
    elif choice == "CUDA":
        threads_label.configure(text="Threads: GPU")
    elif choice == "MPI":
        # Reflect whatever the user has typed into the MPI Processes
        # entry instead of a hardcoded value.
        processes_value = mpi_entry.get().strip()
        threads_label.configure(
            text=f"Processes: {processes_value if processes_value else DEFAULT_MPI_PROCESSES}"
        )

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

# ============================================================
# MAIN LAYOUT
# ============================================================

top_frame = ctk.CTkFrame(app, fg_color="transparent")
top_frame.pack(fill="both", expand=True, padx=10, pady=10)

# ============================================================
# LEFT PANEL
# ============================================================

left = ctk.CTkScrollableFrame(top_frame, width=250, corner_radius=15)
left.pack(side="left", fill="y", padx=(0, 10))

ctk.CTkLabel(
    left,
    text="Control Panel",
    font=("Segoe UI", 24, "bold")
).pack(pady=(20, 15))

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

# ============================================================
# INFO CARD
# ============================================================

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
    text="Threads: 16",
    text_color="#bbbbbb"
)
threads_label.pack(anchor="w", padx=12, pady=(6, 0))

road_info = ctk.CTkLabel(
    info,
    text="Road: 100 cells",
    text_color="#bbbbbb"
)
road_info.pack(anchor="w", padx=12)

veh_info = ctk.CTkLabel(
    info,
    text="Vehicles: 30",
    text_color="#bbbbbb"
)
veh_info.pack(anchor="w", padx=12)

speed_info = ctk.CTkLabel(
    info,
    text="Max Speed: 5",
    text_color="#bbbbbb"
)
speed_info.pack(anchor="w", padx=12)

status_info = ctk.CTkLabel(
    info,
    text="Status: Ready",
    text_color="#4ea1ff"
)
status_info.pack(anchor="w", padx=12, pady=(0, 10))

# ============================================================
# SIMULATION PARAMETERS
# ============================================================

ctk.CTkLabel(
    left,
    text="Simulation Parameters",
    font=("Segoe UI", 16, "bold")
).pack(pady=(20, 10))

def labeled_entry(parent, label, default):
    ctk.CTkLabel(parent, text=label).pack(
        anchor="w",
        padx=25,
        pady=(10, 4)
    )

    entry = ctk.CTkEntry(parent)
    entry.insert(0, default)
    entry.pack(fill="x", padx=25)
    return entry

road_entry = labeled_entry(left, "Road Length (cells)", "100")
veh_entry = labeled_entry(left, "Number of Vehicles", "30")
speed_entry = labeled_entry(left, "Max Speed (cells/step)", "5")

openmp_entry = labeled_entry(
    left,
    "OpenMP Threads",
    str(DEFAULT_OPENMP_THREADS)
)

mpi_entry = labeled_entry(
    left,
    "MPI Processes",
    str(DEFAULT_MPI_PROCESSES)
)

# ------------------------------------------------------------
# Live sync: keep the "Selected Backend" info card in the
# dashboard up to date as the user edits the OpenMP Threads /
# MPI Processes entries, without needing to run a benchmark.
# ------------------------------------------------------------

def on_openmp_entry_change(event=None):
    if backend_var.get() == "OpenMP":
        change_backend("OpenMP")

def on_mpi_entry_change(event=None):
    if backend_var.get() == "MPI":
        change_backend("MPI")

openmp_entry.bind("<KeyRelease>", on_openmp_entry_change)
mpi_entry.bind("<KeyRelease>", on_mpi_entry_change)

ctk.CTkButton(
    left,
    text="⚡ Run Performance Test",
    width=200,
    height=40,
    command=run_cpp_benchmark
).pack(pady=(20, 10))

# ============================================================
# CENTER PANEL
# ============================================================

center = ctk.CTkFrame(top_frame, corner_radius=15)
center.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 10)
)

ctk.CTkLabel(
    center,
    text="Traffic Visualization",
    font=("Segoe UI", 24, "bold")
).pack(pady=(15, 5))

# ============================================================
# LEGEND
# ============================================================

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

    ctk.CTkLabel(
        legend_frame,
        text=t
    ).pack(side="left", padx=(0, 6))

# ============================================================
# TRAFFIC VISUALIZATION
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 7.2))
fig.patch.set_facecolor("#1f1f1f")
ax.set_facecolor("#1f1f1f")
ax.set_aspect("equal")
ax.axis("off")

outer = plt.Circle(
    (0, 0),
    1.02,
    edgecolor="white",
    facecolor="none",
    linewidth=2
)

inner = plt.Circle(
    (0, 0),
    0.92,
    edgecolor="#777777",
    facecolor="none",
    linewidth=1
)

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
canvas.get_tk_widget().pack(
    fill="both",
    expand=True,
    padx=5,
    pady=5
)

# ============================================================
# SIMULATION CONTROL
# ============================================================

running = False
paused = False
frame_count = 0

def refresh_info():
    road_info.configure(text=f"Road: {road_entry.get()} cells")
    veh_info.configure(text=f"Vehicles: {veh_entry.get()}")
    speed_info.configure(text=f"Max Speed: {speed_entry.get()}")

def reset_simulation():
    global ROAD, N, MAX_SPEED, positions, velocities, frame_count

    try:
        ROAD = int(road_entry.get())
        N = int(veh_entry.get())
        MAX_SPEED = int(speed_entry.get())

        if ROAD <= 0 or N <= 0 or MAX_SPEED <= 0:
            return
    except ValueError:
        return

    positions = np.linspace(0, ROAD - 1, N)
    velocities = np.zeros(N)
    frame_count = 0

    theta = 2 * np.pi * positions / ROAD
    coords = np.column_stack((np.cos(theta), np.sin(theta)))

    scat.set_offsets(coords)
    scat.set_array(velocities)
    scat.set_clim(0, MAX_SPEED)

    refresh_info()

    status.configure(
        text=f"Ready   |   Vehicles: {N}   |   Average Speed: 0.00"
    )

    canvas.draw_idle()

def start_simulation():
    global running, paused
    running = True
    paused = False

    status_info.configure(
        text="Status: Running",
        text_color="#00d4ff"
    )

def pause_simulation():
    global paused
    paused = not paused

    if paused:
        status_info.configure(
            text="Status: Paused",
            text_color="#ff9800"
        )
    else:
        status_info.configure(
            text="Status: Running",
            text_color="#00d4ff"
        )

def stop_simulation():
    global running, paused
    running = False
    paused = False

    status_info.configure(
        text="Status: Ready",
        text_color="#4ea1ff"
    )

    reset_simulation()

def step_simulation():
    global running, paused
    running = True
    paused = False
    update(0)
    running = False

def update(frame):
    global positions, velocities, frame_count

    if not running or paused:
        return scat,

    frame_count += 1

    for i in range(N):
        velocities[i] = min(velocities[i] + 1, MAX_SPEED)

        nxt = (i + 1) % N

        gap = (
            positions[nxt] - positions[i] - 1
        ) % ROAD

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
        text=(
            f"Timestep: {frame_count}   |   Vehicles: {N}   |   "
            f"Average Speed: {avg_speed:.2f}   |   Running"
        )
    )

    return scat,

ani = FuncAnimation(
    fig,
    update,
    interval=90,
    blit=True
)

# ============================================================
# SIMULATION BUTTONS
# ============================================================

controls = ctk.CTkFrame(center, fg_color="transparent")
controls.pack(pady=(5, 12))

ctk.CTkButton(
    controls,
    text="▶ Start",
    width=90,
    command=start_simulation
).pack(side="left", padx=6)

ctk.CTkButton(
    controls,
    text="⏸ Pause",
    width=90,
    command=pause_simulation
).pack(side="left", padx=6)

ctk.CTkButton(
    controls,
    text="⏹ Reset",
    width=90,
    command=stop_simulation
).pack(side="left", padx=6)

ctk.CTkButton(
    controls,
    text="Step",
    width=80,
    command=step_simulation
).pack(side="left", padx=6)

# ============================================================
# RIGHT PANEL
# ============================================================

right = ctk.CTkFrame(top_frame, width=420, corner_radius=15)
right.pack(side="right", fill="y")

ctk.CTkLabel(
    right,
    text="Runtime Comparison",
    font=("Segoe UI", 18, "bold")
).pack(pady=(18, 8))

runtime_fig, runtime_ax = plt.subplots(figsize=(4.4, 3.0))
runtime_fig.patch.set_facecolor("#1f1f1f")
runtime_ax.set_facecolor("#1f1f1f")

runtime_canvas = FigureCanvasTkAgg(runtime_fig, master=right)
runtime_canvas.get_tk_widget().pack(padx=8, pady=5)

ctk.CTkLabel(
    right,
    text="Speedup Comparison",
    font=("Segoe UI", 18, "bold")
).pack(pady=(15, 8))

speed_fig, speed_ax = plt.subplots(figsize=(4.4, 3.0))
speed_fig.patch.set_facecolor("#1f1f1f")
speed_ax.set_facecolor("#1f1f1f")

speed_canvas = FigureCanvasTkAgg(speed_fig, master=right)
speed_canvas.get_tk_widget().pack(padx=8, pady=5)

# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

ctk.CTkLabel(
    right,
    text="Performance Summary",
    font=("Segoe UI", 18, "bold")
).pack(
    pady=(12, 5)
)

table = ctk.CTkTextbox(
    right,
    height=125,
    font=("Consolas", 11)
)

table.pack(
    fill="x",
    padx=8,
    pady=(0, 10)
)

# ============================================================
# STATUS BAR
# ============================================================

status = ctk.CTkLabel(
    app,
    text="Ready   |   Vehicles: 30   |   Average Speed: 0.00",
    anchor="w",
    font=("Segoe UI", 12)
)
status.pack(fill="x", padx=15, pady=(0, 8))

# ============================================================
# INITIALIZE DASHBOARD
# ============================================================

reset_simulation()
update_charts()
update_table()
change_backend(backend_var.get())

# ============================================================
# START APPLICATION
# ============================================================

app.mainloop()