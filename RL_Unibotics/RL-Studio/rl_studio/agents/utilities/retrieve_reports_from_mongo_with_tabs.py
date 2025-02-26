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

from rl_studio.agents.utilities.store_results_mongodb import lesson_learned

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['training_db']
collection = db['training_results']

# Query by date range
start_date = datetime(2025, 2, 1)
end_date = datetime(2025, 2, 26)

query = {"timestamp": {"$gte": start_date, "$lt": end_date}}
documents = collection.find(query)


# Function to decode base64 images and convert to a format for plotting
def decode_base64_image(base64_string):
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    return img


for doc in documents:
    config = doc['config']
    reward_function = doc['results']['reward_function']
    plots = doc['results']['plots']
    lesson_learned = doc['lessons']

    # Create the Tkinter window
    root = tk.Tk()
    root.title("Training Results")

    # Create Notebook (tabs)
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)

    # --- Tab 1: Plots ---
    frame1 = ttk.Frame(notebook)
    notebook.add(frame1, text='Plots')

    fig, axes = plt.subplots(2, 4, figsize=(10, 8))

    # Decode and plot each base64-encoded image
    reward_img = decode_base64_image(plots['reward_plot'])
    advanced_meters_img = decode_base64_image(plots['advanced_meters_plot'])
    avg_speed_img = decode_base64_image(plots['avg_speed_plot'])
    abs_w_no_curves_avg_img = decode_base64_image(plots.get('abs_w_no_curves_avg_plot'))
    std_dev_v_img = decode_base64_image(plots.get('std_dev_v_plot'))
    std_dev_w_img = decode_base64_image(plots.get('std_dev_w_plot'))
    throttle_curves_img = decode_base64_image(plots.get('throttle_curves_plot'))
    throttle_no_curves_img = decode_base64_image(plots.get('throttle_no_curves_plot'))

    # Display the images on the matplotlib subplots
    axes[0, 0].imshow(reward_img)
    axes[0, 0].set_title('Reward Plot')
    axes[0, 0].axis('off')  # Turn off axis

    axes[0, 1].imshow(advanced_meters_img)
    axes[0, 1].set_title('Advanced Meters Plot')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(abs_w_no_curves_avg_img)
    axes[0, 2].set_title('avg_w_deviation')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(throttle_curves_img)
    axes[0, 3].set_title('throttle_curves_plot')
    axes[0, 3].axis('off')

    axes[1, 0].imshow(avg_speed_img)
    axes[1, 0].set_title('Avg Speed Plot')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(std_dev_w_img)
    axes[1, 1].set_title('Std Dev w Plot')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(std_dev_v_img)
    axes[1, 2].set_title('Std Dev v Plot')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(throttle_no_curves_img)
    axes[1, 3].set_title('throttle_no_curves_plot')
    axes[1, 3].axis('off')


    # Display the plots in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=frame1)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- Tab 2: Reward Function ---
    frame2 = ttk.Frame(notebook)
    notebook.add(frame2, text='Reward Function')

    reward_function_text = tk.Text(frame2, wrap=tk.WORD)
    reward_function_text.pack(fill='both', expand=True)
    reward_function_text.insert(tk.END, reward_function)

    # --- Tab 3: YAML Config ---
    frame3 = ttk.Frame(notebook)
    notebook.add(frame3, text='YAML Config')

    config_text = tk.Text(frame3, wrap=tk.WORD)
    config_text.pack(fill='both', expand=True)

    # --- Tab 4: Reward Function ---
    frame4 = ttk.Frame(notebook)
    notebook.add(frame4, text='Lesson learned')

    reward_function_text = tk.Text(frame4, wrap=tk.WORD)
    reward_function_text.pack(fill='both', expand=True)
    reward_function_text.insert(tk.END, lesson_learned)


    # Format the YAML and insert into the text widget
    yaml_str = yaml.dump(config, default_flow_style=False)
    config_text.insert(tk.END, yaml_str)

    # Start the Tkinter main loop
    root.mainloop()
