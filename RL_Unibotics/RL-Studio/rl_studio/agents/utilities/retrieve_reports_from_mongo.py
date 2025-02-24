from pymongo import MongoClient
from datetime import datetime
import tkinter as tk
from tkinter import ttk
import yaml
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['training_db']
collection = db['training_results']

# Query by date range
start_date = datetime(2024, 12, 1)
end_date = datetime(2025, 2, 25)

query = {"timestamp": {"$gte": start_date, "$lt": end_date}}
documents = list(collection.find(query))  # Convert cursor to list for multiple iterations

# Function to decode base64 images
def decode_base64_image(base64_string):
    img_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(img_data))

# --- Create a Single Tkinter Window ---
root = tk.Tk()
root.title("Training Results")

# Create Notebook (tabs)
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

# Iterate through all retrieved documents and create a new tab for each
for index, doc in enumerate(documents):
    tab = ttk.Frame(notebook)
    notebook.add(tab, text=f'Result {index + 1}')  # Name tabs dynamically

    frame = ttk.Frame(tab)
    frame.pack(fill='both', expand=True)

    # --- Display YAML Config ---
    config_text = tk.Text(frame, wrap=tk.WORD, height=10)
    config_text.pack(fill='both', expand=True)
    yaml_str = yaml.dump(doc['config'], default_flow_style=False)
    config_text.insert(tk.END, yaml_str)

    # --- Display Reward Function ---
    reward_function_text = tk.Text(frame, wrap=tk.WORD, height=5)
    reward_function_text.pack(fill='both', expand=True)
    reward_function_text.insert(tk.END, doc['results']['reward_function'])

    # --- Display Lesson Learned ---
    lesson_text = tk.Text(frame, wrap=tk.WORD, height=5)
    lesson_text.pack(fill='both', expand=True)
    lesson_text.insert(tk.END, doc['lessons'])

    # --- Display Plots ---
    fig, axes = plt.subplots(2, 4, figsize=(10, 8))
    plots = doc['results']['plots']

    plot_keys = [
        "reward_plot", "advanced_meters_plot", "avg_speed_plot",
        "abs_w_no_curves_avg_plot", "std_dev_v_plot", "std_dev_w_plot",
        "throttle_curves_plot", "throttle_no_curves_plot"
    ]

    for i, key in enumerate(plot_keys):
        try:
            row, col = divmod(i, 4)  # Convert to 2x4 grid
            img = decode_base64_image(plots.get(key, ""))
            axes[row, col].imshow(img)
            axes[row, col].set_title(key.replace('_', ' ').title())
            axes[row, col].axis('off')
        except Exception as outer_error:
            print(f"Failed to read TensorBoard logs: {outer_error}")
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Start Tkinter main loop
root.mainloop()
