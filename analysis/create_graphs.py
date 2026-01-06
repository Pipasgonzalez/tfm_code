from openai.types.beta.threads.run import Run
from collections import defaultdict
import numpy as np
import pandas as pd
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns
import matplotlib.pyplot as plt

def get_training_loss_components(logdir):
    if not os.path.isdir(logdir):
        raise ModuleNotFoundError()
    else:
        results = []
        loss_types = ["dice", "focal"]
        dice_dir = os.path.join(logdir, "loss_supervised_dice")
        focal_dir = os.path.join(logdir, "loss_supervised_focal")
        loss_dirs = [dice_dir, focal_dir]
        for idx, loss_dir in enumerate(loss_dirs):
            ea = EventAccumulator(loss_dir)
            ea.Reload()
            print(ea.Tags())
            events = ea.Scalars(f"loss/supervised")
            results.extend([{"type": loss_types[idx],"value": event.value, "step": step} for step,event in enumerate(events)])
        return pd.DataFrame(results)

def plot_loss_components(df_loss,ax):
    # 1. Separate the data by metric type
    df_dice = df_loss[df_loss['type'] == 'dice'].copy()
    df_focal = df_loss[df_loss['type'] == 'focal'].copy()

    # Define a smoothing factor (alpha=0.01 is light, alpha=0.1 is heavier)
    # Or use a window size (span=20 means roughly a 20-step average)
    SMOOTHING_SPAN = 33

    # 2. Apply Exponential Moving Average (EWMA)
    df_dice['smoothed_value'] = df_dice['value'].ewm(span=SMOOTHING_SPAN).mean()
    df_focal['smoothed_value'] = df_focal['value'].ewm(span=SMOOTHING_SPAN).mean()

    # 3. Combine the smoothed data back
    df_smoothed = pd.concat([df_dice, df_focal])

    # 4. Plot the smoothed data
    sns.lineplot(
        data=df_smoothed,
        x='step',
        y='smoothed_value',  # Plot the new smoothed column
        hue='type',
        linewidth=1,
        ax=ax
    )

    ax.set_title(f'Supervised Loss Components Trend (EWMA Span={SMOOTHING_SPAN})')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Smoothed Loss Value')
    ax.legend(title='Metric')

def get_dice_scores(logdir):
    if not os.path.isdir(logdir):
        raise ModuleNotFoundError()
    else:
        resolutions =  [3,4]
        classes = [0,1,2,3]
        CLASS_NAMES = ["Background", "dHGP", "Liver", "Tumor"]
        results_dict = defaultdict(lambda: defaultdict(list))
        for resolution in resolutions:
            try:
                results_dict[f"dice_res{resolution}"] = {
                        "Background":[],
                        "dHGP":[],
                        "Liver":[],
                        "Tumor":[]
                }
                for cls in classes:
                    res_dir = os.path.join(logdir, f"val_dsc_res{resolution}_Class{cls}")
                    if not os.path.isdir(res_dir):
                        print("Triggered with", str(res_dir))
                        break
                    ea = EventAccumulator(res_dir)
                    ea.Reload()
                    events = ea.Scalars(f"val/dsc_res{resolution}")
                    results_dict[f"dice_res{resolution}"][CLASS_NAMES[cls]] = events
            except Exception as e:
                raise RuntimeError(str(e))

        # Postprocess
        cleaned_dict = defaultdict(dict)
        for res_key  in results_dict.keys():
            dice_dict = results_dict[res_key]
            for cls in dice_dict.keys():
                cls_events = dice_dict[cls]
                cleaned_dict[res_key][cls] = [event.value for event in cls_events]
        return cleaned_dict

def flatten_dice_scores(dice_scores_dict):
    data = []
    for res_key, class_dict in dice_scores_dict.items():
        resolution = res_key.split('_')[1] 
        for class_name, scores_list in class_dict.items():
            for step, score in enumerate(scores_list):
                data.append({
                    'Step': (step + 1) * 1000,  # Assuming scores are logged every 250 steps (iter_num % 250 == 0)
                    'Value': score,
                    'Metric': 'Dice Score',
                    'Class': class_name,
                    'Resolution': resolution
                })
    return pd.DataFrame(data)

def plot_dice(df_dice,ax):
    sns.set_theme(style="whitegrid")
    sns.lineplot(
        data=df_dice,
        x='Step',
        y='Value',
        hue='Class',          # Differentiate lines by class (Background, dHGP, etc.)
        style='Resolution',   # Differentiate line styles by resolution (res4, res5, etc.)
        marker='o',           # Add markers for clarity
        dashes=False,
        ax=ax
    )

    ax.set_title('Dice Score Development Over Training Iterations (by Class and Resolution)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Dice Score')
    ax.legend(title='Metric')

def produce_and_save_overview_plot(experiment_name: str = "20251218_211433_split2_40001iterations_16bs",experiment_title:str = "Final Supervised Model - Split 2 Training", base_logdir: str = "D:/tfm_data/hpc_results", plots_dir: str = "./analysis/plots"):

    """
    Generates a two-panel plot overview for an experiment and saves it.
    """
    logdir = os.path.join(base_logdir, experiment_name)
    try:
        dice_scores = get_dice_scores(logdir)
        flattened_dice_scores = flatten_dice_scores(dice_scores)
        loss_components = get_training_loss_components(logdir)
    except Exception as e:
        print(f"Error loading data for {experiment_name}: {e}")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10)) # Increased width for two plots
    
    # 3a. Plot Loss Components
    # We call the function but pass the target axes (ax=ax1)
    plot_loss_components(loss_components, ax=ax1) 
    
    # 3b. Plot Dice Scores
    # We call the function but pass the target axes (ax=ax2)
    plot_dice(flattened_dice_scores, ax=ax2) 

    # --- 4. Saving the Figure ---
    
    # Create the output directory if it doesn't exist
    # output_dir = os.path.join(plots_dir, )
    os.makedirs(plots_dir, exist_ok=True)
    
    # Define the output file path
    file_path = os.path.join(plots_dir, f"{experiment_name}_overview.png")

    # Save the figure
    fig.suptitle(f"Experiment Overview: {experiment_title}", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    fig.savefig(file_path, dpi=300)
    plt.close(fig) # Close the figure to free memory
    
    print(f"Successfully saved plot to: {file_path}")

if __name__ == "__main__":
    produce_and_save_overview_plot()