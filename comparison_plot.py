import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to plot comparison between Benchmark and Experimental models
def plot_comparison(df_metrics):
    plt.style.use('ggplot')
    n_metrics = len(df_metrics['Metric'])
    ind = np.arange(n_metrics)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(ind - width/2, df_metrics['NYC-Benchmark'], width, label='7days-NYC-Benchmark', color='skyblue')
    bars2 = ax.bar(ind + width/2, df_metrics['TKY-Experimental'], width, label='7days-TKY-Experimental', color='salmon')

    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Evaluation Metrics')
    ax.set_xticks(ind)
    ax.set_xticklabels(df_metrics['Metric'])
    ax.legend()

    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()
    plt.savefig('./imgs/7days_foursquare_comparison_metrics.png')
    plt.show()

# Function to calculate percentage improvement and generate visualizations
def analyze_results():
    from results_data.foursquare_7days import data  # Import data from the second file
    df_metrics = pd.DataFrame(data)

    # Plot comparison 
    plot_comparison(df_metrics)

if __name__ == "__main__":
    analyze_results()
