import os
import collections
import wandb
import wandb.errors # wandb.history() returns a DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # For better y-axis formatting

def get_wandb_run_info(run_path: str):
    """
    Fetches comprehensive information for a given WandB run.

    Args:
        run_path (str): The path to the WandB run, in the format "entity/project/run_id".
                        Example: "my_team/image_classification/abcdef12"

    Returns:
        dict: A dictionary containing the run's information.
              Returns None if the run is not found or an error occurs.
    
    Raises:
        wandb.errors.CommError: If the run path is invalid or there's a communication issue.
        Exception: For other potential errors during API interaction.
    """
    print(f"Attempting to fetch run: {run_path}")
    
    run_info = {}

    try:
        # Initialize the WandB API.
        # Ensure you are logged in (wandb login) or WANDB_API_KEY is set.
        api = wandb.Api()

        # Fetch the run object
        run = api.run(path=run_path)

        # Basic run information
        run_info["id"] = run.id
        run_info["name"] = run.name
        run_info["project"] = run.project
        run_info["entity"] = run.entity
        run_info["state"] = run.state
        run_info["created_at"] = run.created_at
        run_info["url"] = run.url # Direct link to the run in WandB UI
        run_info["notes"] = run.notes
        run_info["tags"] = run.tags
        
        # Configuration (hyperparameters)
        # run.config can sometimes include non-serializable items if not careful during logging.
        # We'll try to make a clean dictionary.
        config_dict = {}
        for key, value in run.config.items():
            # Skip internal wandb keys
            if key.startswith("_"): 
                continue
            config_dict[key] = value
        run_info["config"] = config_dict

        # Summary metrics (final or best metrics)
        # run.summary._json_dict provides a clean dictionary of summary metrics
        run_info["summary_metrics"] = run.summary._json_dict

        # List of files associated with the run
        run_info["files"] = [f.name for f in run.files()]

        # Full history of logged metrics
        # This can be large. For very long runs, consider sampling or filtering.
        # run.history() returns a pandas DataFrame.
        history_df = run.history()
        run_info["history_df"] = history_df
            
        print(f"Successfully fetched information for run: {run.name} ({run.id})")
        return run_info

    except wandb.errors.CommError as e:
        print(f"Error: Could not find or access run '{run_path}'. Please check the path and your permissions.")
        print(f"WandB CommError: {e}")
    except AttributeError as e:
        # This can happen if the run object is not as expected, e.g. run not found leading to None.
        # The api.run() call itself might raise CommError for not found, but good to be safe.
        print(f"Error: Attribute error, possibly run '{run_path}' not found or API issue.")
        print(f"AttributeError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while fetching run '{run_path}': {e}")

def plot_history_metrics(history_df: pd.DataFrame, base_metric_prefixes_to_plot: list|None = None, run_name: str = "",  output_dir: str = "wandb_plots"):
    """
    Plots specified metrics from the history DataFrame.

    Args:
        history_df (pd.DataFrame): DataFrame containing the run history (from run.history()).
        metrics_to_plot (list, optional): A list of metric names (column names in history_df) 
                                          to plot. If None, attempts to plot all numeric columns.
        run_name (str, optional): Name of the run for plot titles.
    """
    if history_df.empty:
        print("History DataFrame is empty. Nothing to plot.")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory for plots: {output_dir}")
    
    numeric_cols = history_df.select_dtypes(include='number').columns.tolist()
    
    # --- Group per-class metrics ---
    # Key: base_plot_metric_name (e.g., "val/f1_score"), Value: {class_label: full_metric_column_name}
    grouped_per_class_metrics = collections.defaultdict(dict)
    individual_metrics = [] # Store full column names of individual metrics

    for col_name in numeric_cols:
        if col_name.startswith('_'): # Skip internal wandb metrics like _step, _runtime
            continue

        parts = col_name.split('/')
        if len(parts) == 2: # Expected format like "val/f1_score.akiec" or "train/accuracy"
            metric_type = parts[0] # e.g., "val", "train"
            name_and_class_candidate = parts[1] # e.g., "f1_score.akiec", "accuracy"

            sub_parts = name_and_class_candidate.split('.', 1) # Split only on the first dot
            if len(sub_parts) == 2: # Likely a per-class metric, e.g., "f1_score.akiec"
                metric_name = sub_parts[0] # e.g., "f1_score"
                class_label = sub_parts[1] # e.g., "akiec"
                base_plot_metric = f"{metric_type}/{metric_name}" # e.g., "val/f1_score"
                grouped_per_class_metrics[base_plot_metric][class_label] = col_name
            else: # Likely an overall metric, e.g., "train/accuracy" or "learning_rate" (if LR is not split by /)
                individual_metrics.append(col_name)
        else: # Not in "type/name..." format, treat as individual (e.g. "epoch", "learning_rate")
            individual_metrics.append(col_name)
    
    # --- Filter metrics based on base_metric_prefixes_to_plot ---
    final_per_class_groups_to_plot = {}
    final_individual_metrics_to_plot = []

    if base_metric_prefixes_to_plot is None or not base_metric_prefixes_to_plot: # Plot all found
        final_per_class_groups_to_plot = {k: v for k, v in grouped_per_class_metrics.items() if len(v) > 0} # Ensure group is not empty
        final_individual_metrics_to_plot = individual_metrics
    else:
        for prefix in base_metric_prefixes_to_plot:
            # Check per-class groups
            for base_metric, class_dict in grouped_per_class_metrics.items():
                if base_metric.startswith(prefix) and len(class_dict) > 0 :
                    final_per_class_groups_to_plot[base_metric] = class_dict
            # Check individual metrics
            for ind_metric in individual_metrics:
                if ind_metric.startswith(prefix):
                    final_individual_metrics_to_plot.append(ind_metric)
        # Remove duplicates from individual metrics if any
        final_individual_metrics_to_plot = sorted(list(set(final_individual_metrics_to_plot)))


    # Further refine: if a "grouped" metric only has one class after filtering, 
    # and wasn't specifically targeted by a full prefix match, move it to individual.
    # This avoids plotting a "group" of one unless explicitly asked for.
    refined_per_class_groups = {}
    for base_metric, class_dict in final_per_class_groups_to_plot.items():
        if len(class_dict) > 1:
            refined_per_class_groups[base_metric] = class_dict
        elif len(class_dict) == 1:
            # If it was explicitly requested by full base_metric name, keep it as a "group" of one
            if base_metric_prefixes_to_plot and base_metric in base_metric_prefixes_to_plot:
                 refined_per_class_groups[base_metric] = class_dict
            else: # Otherwise, treat its single member as an individual metric
                final_individual_metrics_to_plot.extend(class_dict.values())
    final_individual_metrics_to_plot = sorted(list(set(final_individual_metrics_to_plot))) # Re-sort and unique
    final_per_class_groups_to_plot = refined_per_class_groups


    # --- Plotting ---
    x_axis_col = '_step' if '_step' in history_df.columns else ('epoch' if 'epoch' in history_df.columns else None)
    if x_axis_col and history_df[x_axis_col].isnull().all(): # If x_axis_col exists but is all NaN
        print(f"Warning: X-axis column '{x_axis_col}' contains all NaN values. Plotting against index.")
        x_axis_col = None # Fallback to index

    plot_count = 0

    # Plot grouped per-class metrics
    for base_plot_metric, class_dict in final_per_class_groups_to_plot.items():
        fig, ax = plt.subplots(figsize=(14, 7))
        # Sanitize base_plot_metric for title
        title_base = base_plot_metric.replace('_', ' ').replace('/', ' / ').title()
        plot_title = f"{run_name}: {title_base} (Per-Class)" if run_name else f"{title_base} (Per-Class)"
        ax.set_title(plot_title)

        for class_label, full_metric_name in sorted(class_dict.items()):
            if full_metric_name not in history_df.columns:
                print(f"Warning: Column {full_metric_name} for class {class_label} not found. Skipping.")
                continue
            if history_df[full_metric_name].isnull().all():
                print(f"Info: Metric {full_metric_name} for class {class_label} is all NaN. Skipping plot for this line.")
                continue
            
            y_values = history_df[full_metric_name]
            if x_axis_col:
                ax.plot(history_df[x_axis_col], y_values, marker='.', linestyle='-', label=class_label.title())
            else:
                ax.plot(y_values, marker='.', linestyle='-', label=class_label.title())
        
        if x_axis_col:
            ax.set_xlabel(x_axis_col.replace('_', ' ').title())
        else:
            ax.set_xlabel("Step/Index")
            ax.set_ylabel(title_base)
        if len(class_dict) > 0 :
            ax.legend(title="Classes", loc="best") # Show legend even for one class if explicitly grouped
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(x, '.2e') if (abs(x) < 1e-3 or abs(x) > 1e3) and x != 0 else format(x, '.3f')))
        
        plt.tight_layout()
        safe_base_metric_name = "".join(c if c.isalnum() else "_" for c in base_plot_metric.replace('/', '_'))
        safe_run_name = "".join(c if c.isalnum() else "_" for c in run_name) if run_name else "run"
        plot_filename = os.path.join(output_dir, f"{safe_run_name}_{safe_base_metric_name}_per_class.png")
        
        try:
            plt.savefig(plot_filename)
            print(f"Saved plot: {plot_filename}")
            plot_count += 1
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")
        plt.close(fig)

    # Plot individual metrics
    for metric_name in final_individual_metrics_to_plot:
        if metric_name not in history_df.columns or not pd.api.types.is_numeric_dtype(history_df[metric_name]):
            continue
        if history_df[metric_name].isnull().all():
            print(f"Info: Metric {metric_name} is all NaN. Skipping plot.")
            continue
            
        fig, ax = plt.subplots(figsize=(12, 6))
        y_values = history_df[metric_name]
        if x_axis_col:
            ax.plot(history_df[x_axis_col], y_values, marker='.', linestyle='-')
            ax.set_xlabel(x_axis_col.replace('_', ' ').title())
        else:
            ax.plot(y_values, marker='.', linestyle='-')
            ax.set_xlabel("Step/Index")

        # Sanitize metric_name for title
        title_metric = metric_name.replace('_', ' ').replace('/', ' / ').title()
        plot_title = f"{run_name}: {title_metric}" if run_name else title_metric
        ax.set_title(plot_title)
        ax.set_ylabel(title_metric)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(x, '.2e') if (abs(x) < 1e-3 or abs(x) > 1e3) and x != 0 else format(x, '.3f')))
        
        plt.tight_layout()
        safe_metric_name = "".join(c if c.isalnum() else "_" for c in metric_name.replace('/', '_'))
        safe_run_name = "".join(c if c.isalnum() else "_" for c in run_name) if run_name else "run"
        plot_filename = os.path.join(output_dir, f"{safe_run_name}_{safe_metric_name}_individual.png")
        try:
            plt.savefig(plot_filename)
            print(f"Saved plot: {plot_filename}")
            plot_count += 1
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")
        plt.close(fig)

    if plot_count == 0:
        print("No metrics were plotted. Check metric names and prefixes, or if data is all NaN.")
    elif plot_count > 0:
         print(f"\nAll plots saved in the '{output_dir}' directory.")

if __name__ == "__main__":
    run_path = "bmnunes-universidade-federal-de-s-o-paulo-unifesp/ham10000-resnet/runs/wjfgb8jk" # One more public run example

    print(f"Attempting to fetch data for run: {run_path}\n")
    
    run_data = get_wandb_run_info(run_path)

    if run_data:
        print("\n--- Run Information ---")
        print(f"ID: {run_data.get('id')}")
        print(f"Name: {run_data.get('name')}")
        print(f"Project: {run_data.get('project')}")
        print(f"Entity: {run_data.get('entity')}")
        # ... (rest of the print statements for run info) ...
        print(f"URL: {run_data.get('url')}")
        
        print("\n--- Configuration ---")
        if run_data.get('config'):
            for key, val in run_data['config'].items():
                print(f"  {key}: {val}") 
        else:
            print("No configuration found.")

        print("\n--- Summary Metrics ---")
        if run_data.get('summary_metrics'):
            for key, val in run_data['summary_metrics'].items():
                print(f"  {key}: {val}")
        else:
            print("  No summary metrics found.")

        # ... (print statements for files, tags, notes) ...

        history_df = run_data.get("history_df") 
        
        if history_df is not None and not history_df.empty:
        #     print("\n--- History (first 5 entries from DataFrame) ---")
        #     print(history_df.head().to_string()) # .to_string() for better console output of DataFrame
            
        #     if len(history_df) > 11:
        #         print("\n--- Selected History Entries (from row 11 of DataFrame) ---")
        #         for i, (_, row) in enumerate(history_df.iloc[11:].iterrows()):
        #             epoch_val = row.get('epoch', f"Index_in_slice_{i}") 
        #             print(f"  Epoch {epoch_val}: {row.to_dict()}") # Print full row dict
        #             print("") 
        #     print(f"  Total history entries: {len(history_df)}")
        #     print("")
            
            print("\n--- Plotting History Metrics ---")
            
            # Define base metrics or prefixes for per-class plots.
            # Examples based on your sample: "val/f1_score", "val/auroc", "train/recall"
            # If you want all 'val' per-class metrics: ["val/"] (though the code handles full prefix better)
            # Or specific ones:
            metrics_for_plots = [
                "val/f1_score", 
                "val/auroc", 
                "val/recall",
                "val/precision",
                "val/accuracy",   # This will be an individual plot
                "val/loss",       # Individual
                # "train/f1_score",
                # "train/recall",
                # "train/precision",
                # "train/auroc",
                # "train/accuracy",
                # "train/loss",  
                "learning_rate"
            ]
            # To plot all discovered per-class groups and all individual metrics, pass None or []:
            # prefixes_for_plots = None
            
            plot_history_metrics(
                history_df,
                base_metric_prefixes_to_plot=metrics_for_plots, 
                run_name=run_data['name'],
                output_dir='wandb_plots'
            )

        else:
            print("\nNo history data found or history is empty.")
    else:
        print(f"\nFailed to retrieve information for run: {run_path}")


