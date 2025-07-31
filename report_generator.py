import os
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

score_descriptions = {
    'C-FID': 'Conditional Frechet Inception Distance - measures similarity of distributions.',
    'Skewness Difference (SD)': 'Measure of asymmetry between real and synthetic data.',
    'Kurtosis Difference (KD)': 'Measure of tail heaviness/ outliers between real and synthetic data.',
    'DTW': 'Dynamic Time Warping distance - compares data sequences in time series data.',
    'NNDR (5th pct)': 'Nearest Neighbor Distance Ratio (5th percentile) - highlight privacy risks if synthetic points too similar to real points.',
}

def generate_definitions_html(definitions: dict) -> str:
    lines = ["<ul>"]
    for metric, desc in definitions.items():
        lines.append(f"<li><b>{metric}</b>: {desc}</li>")
    lines.append("</ul>")
    return "\n".join(lines)

def normalize_metric(score, best, worst, higher_is_better=False):
    if higher_is_better:
        norm = (score - worst) / (best - worst)
    else:
        norm = (score - best) / (worst - best)
    return min(max(norm, 0), 1)

def generate_summary(scores, norm_scores):
    summary_lines = []
    for metric, actual_value in scores.items():
        norm_value = norm_scores[metric]
        status = 'Good' if norm_value <= 0.5 else 'Needs Work'
        line = f"{metric:30}: {actual_value:.3f} — {status}"
        summary_lines.append(line)
    return "<br>\n".join(summary_lines)

def plot_score_ranges(norm_scores, norm_best_case, norm_worst_case, output_path):
    metrics = list(norm_scores.keys())
    y_pos = np.arange(len(metrics))
    actual = [norm_scores[m] for m in metrics]
    best = [norm_best_case.get(m, 0) for m in metrics]
    worst = [norm_worst_case.get(m, 1) for m in metrics]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (b, w, a) in enumerate(zip(best, worst, actual)):
        left = min(w, b)
        right = max(w, b)
        ax.hlines(y=i, xmin=left, xmax=right, color='lightgray', linewidth=8)
        ax.plot(a, i, 'ro')
        ax.text(a, i + 0.2, f'{a:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized Score (0=Best, 1=Worst)")
    ax.set_title("Normalized Score Ranges with Normalised Value (red dot)")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def generate_report(scores: dict, worst_case: dict, best_case: dict,
                    barplot_path: str = 'outputs/summary_scores_barplot.png',
                    radarplot_path: str = 'outputs/summary_scores_radarplot.png',
                    html_report_path: str = 'outputs/report.html'):
    """
    Generate a summary report with results, plots, and embedded base64 images.
    """
    os.makedirs(os.path.dirname(barplot_path), exist_ok=True)
    os.makedirs(os.path.dirname(radarplot_path), exist_ok=True)
    os.makedirs(os.path.dirname(html_report_path), exist_ok=True)

    definitions_html = generate_definitions_html(score_descriptions)

    df_scores = pd.DataFrame.from_dict(scores, orient='index', columns=['Score'])
    df_scores.index.name = 'Metric'
    print("\nEvaluation Report (Mean Values):")
    print(df_scores)

    # Barplot
    plt.figure(figsize=(8, 4))
    sns.barplot(x='Score', y=df_scores.index, data=df_scores, palette='viridis')
    plt.title('Synthetic vs Real Data — Evaluation Metrics (Mean Values)')
    plt.xlabel('Absolute Score (Lower is Better), except for NNDR')
    plt.tight_layout()
    plt.savefig(barplot_path)
    plt.close()

    df_worst = pd.DataFrame.from_dict(worst_case, orient='index', columns=['Worst Case'])
    print("\nWorst Case Scores (for random noisy synthetic data):")
    print(df_worst)

    norm_scores = {
        'C-FID': normalize_metric(scores['C-FID'], best=0, worst=worst_case['C-FID']),
        'Skewness Difference (SD)': normalize_metric(scores['Skewness Difference (SD)'], best=0, worst=worst_case['Skewness Difference (SD)']),
        'Kurtosis Difference (KD)': normalize_metric(scores['Kurtosis Difference (KD)'], best=0, worst=worst_case['Kurtosis Difference (KD)']),
        'DTW': normalize_metric(scores['DTW'], best=0, worst=worst_case['DTW']),
        'NNDR (5th pct)': normalize_metric(scores['NNDR (5th pct)'], best=best_case['NNDR (5th pct)'], worst=0, higher_is_better=True)
    }

    labels = list(norm_scores.keys())
    values = list(norm_scores.values()) + [list(norm_scores.values())[0]]
    angles = [n / float(len(labels)) * 2 * np.pi for n in range(len(labels))] + [0]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], labels)
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, 'skyblue', alpha=0.4)
    plt.title("Normalized Metrics (0 = Best, 1 = Worst)", y=1.1)
    plt.tight_layout()
    plt.savefig(radarplot_path)
    plt.close()

    # Score range plot
    norm_best_case = {metric: 0 for metric in norm_scores.keys()}
    norm_worst_case = {metric: 1 for metric in norm_scores.keys()}
    rangeplot_path = os.path.join(os.path.dirname(barplot_path), 'score_ranges_plot.png')
    plot_score_ranges(norm_scores, norm_best_case, norm_worst_case, rangeplot_path)

    # Convert all images to base64
    barplot_base64 = img_to_base64(barplot_path)
    radarplot_base64 = img_to_base64(radarplot_path)
    rangeplot_base64 = img_to_base64(rangeplot_path)

    summary_html = generate_summary(scores, norm_scores)

    html_content = f"""
    <html>
    <head>
        <title>Synthetic Data Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            img {{ max-width: 600px; height: auto; display: block; margin-bottom: 20px; }}
            .summary {{ white-space: pre-line; font-family: monospace; background-color: #f9f9f9; padding: 10px; border-radius: 5px; }}
            .definitions {{ margin-bottom: 20px; font-size: 14px; }}
            ul {{ padding-left: 20px; }}
        </style>
    </head>
    <body>
        <h1>Synthetic Data Evaluation Report</h1>

        <h2>Metric Definitions</h2>
        <div class="definitions">
            {definitions_html}
        </div>

        <h2>Score Summary</h2>
        <div class="summary">{summary_html}</div>

        <h2>Barplot of Scores</h2>
        <img src="data:image/png;base64,{barplot_base64}" alt="Scores Barplot">

        <h2>Normalised Score Range</h2>
        <img src="data:image/png;base64,{rangeplot_base64}" alt="Score Ranges Plot">

        <h2>Radar Plot of Normalized Scores</h2>
        <img src="data:image/png;base64,{radarplot_base64}" alt="Radar Plot">
    </body>
    </html>
    """

    with open(html_report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\nReport saved to {html_report_path}")
