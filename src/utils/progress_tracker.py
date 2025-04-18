"""
Progress Tracker Utility for Dimensional Cascade

This module provides utilities to track progress, generate reports,
and compare results between different runs of the dimensional cascade.
It helps maintain documentation and visualize improvements over time.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import glob
import markdown
from matplotlib.ticker import MaxNLocator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProgressTracker:
    """Tracks progress and generates reports for dimensional cascade runs."""
    
    def __init__(self, base_dir: str = "progress_tracking"):
        """
        Initialize the progress tracker.
        
        Args:
            base_dir: Directory to store progress tracking data
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "runs"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
        
        # Load existing run history if available
        self.history_file = os.path.join(base_dir, "run_history.json")
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = []
    
    def add_run(self, 
                run_name: str, 
                results_dir: str, 
                description: str, 
                parameters: Dict[str, Any],
                metrics: Dict[str, Dict[str, float]]) -> str:
        """
        Add a new run to the progress tracker.
        
        Args:
            run_name: Name of the run
            results_dir: Directory containing the results
            description: Description of the run
            parameters: Parameters used for the run
            metrics: Dictionary of metrics from the run
            
        Returns:
            ID of the run
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_{run_name}"
        
        run_data = {
            "id": run_id,
            "name": run_name,
            "timestamp": timestamp,
            "description": description,
            "parameters": parameters,
            "metrics": metrics,
            "results_dir": results_dir
        }
        
        # Save run data
        run_dir = os.path.join(self.base_dir, "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        with open(os.path.join(run_dir, "run_data.json"), 'w') as f:
            json.dump(run_data, f, indent=2)
        
        # Add to history
        self.history.append({
            "id": run_id,
            "name": run_name,
            "timestamp": timestamp,
            "description": description
        })
        
        # Save updated history
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Added run {run_id} to progress tracker")
        return run_id
    
    def import_validation_results(self, 
                                  validation_dir: str, 
                                  run_name: Optional[str] = None,
                                  description: Optional[str] = None) -> str:
        """
        Import validation results from a validation directory.
        
        Args:
            validation_dir: Directory containing validation results
            run_name: Name to give the run (default: derived from directory)
            description: Description of the run (default: empty)
            
        Returns:
            ID of the imported run
        """
        if not os.path.exists(validation_dir):
            raise FileNotFoundError(f"Validation directory {validation_dir} does not exist")
        
        # Look for metrics.json
        metrics_file = os.path.join(validation_dir, "metrics.json")
        if not os.path.exists(metrics_file):
            raise FileNotFoundError(f"Metrics file not found in {validation_dir}")
        
        # Look for parameters.json
        params_file = os.path.join(validation_dir, "parameters.json")
        if not os.path.exists(params_file):
            logger.warning(f"Parameters file not found in {validation_dir}, using empty parameters")
            parameters = {}
        else:
            with open(params_file, 'r') as f:
                parameters = json.load(f)
        
        # Load metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Derive run name if not provided
        if run_name is None:
            run_name = os.path.basename(os.path.normpath(validation_dir))
        
        # Use empty description if not provided
        if description is None:
            description = ""
        
        # Add the run
        return self.add_run(
            run_name=run_name,
            results_dir=validation_dir,
            description=description,
            parameters=parameters,
            metrics=metrics
        )
    
    def get_run_data(self, run_id: str) -> Dict[str, Any]:
        """Get data for a specific run."""
        run_dir = os.path.join(self.base_dir, "runs", run_id)
        data_file = os.path.join(run_dir, "run_data.json")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Run data file for {run_id} not found")
        
        with open(data_file, 'r') as f:
            return json.load(f)
    
    def list_runs(self) -> List[Dict[str, str]]:
        """List all tracked runs."""
        return self.history
    
    def compare_runs(self, run_ids: List[str], 
                     metrics: Optional[List[str]] = None,
                     dimensions: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Compare metrics across multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare (default: all shared metrics)
            dimensions: List of dimensions to compare (default: all shared dimensions)
            
        Returns:
            DataFrame comparing the runs
        """
        if not run_ids:
            raise ValueError("No run IDs provided for comparison")
        
        # Collect run data
        run_data_list = []
        for run_id in run_ids:
            try:
                run_data_list.append(self.get_run_data(run_id))
            except FileNotFoundError:
                logger.warning(f"Run {run_id} not found, skipping")
        
        if not run_data_list:
            raise ValueError("No valid runs found for comparison")
        
        # Find common metrics and dimensions if not specified
        if metrics is None:
            # Get all metrics from first run
            metrics = list(run_data_list[0]["metrics"].keys())
            
            # Filter to metrics present in all runs
            for run_data in run_data_list[1:]:
                metrics = [m for m in metrics if m in run_data["metrics"]]
        
        # Initialize comparison data
        comparison_data = []
        
        # Process each run
        for run_data in run_data_list:
            run_metrics = run_data["metrics"]
            
            for metric in metrics:
                if metric in run_metrics:
                    metric_data = run_metrics[metric]
                    
                    # Handle dimensions
                    if dimensions is None:
                        dims = list(metric_data.keys())
                    else:
                        dims = [str(d) for d in dimensions if str(d) in metric_data]
                    
                    for dim in dims:
                        comparison_data.append({
                            "Run ID": run_data["id"],
                            "Run Name": run_data["name"],
                            "Metric": metric,
                            "Dimension": dim,
                            "Value": metric_data[dim]
                        })
        
        return pd.DataFrame(comparison_data)
    
    def plot_comparison(self, 
                        comparison_df: pd.DataFrame,
                        metric: str,
                        save_path: Optional[str] = None,
                        show: bool = True) -> plt.Figure:
        """
        Plot comparison of a specific metric across runs.
        
        Args:
            comparison_df: DataFrame from compare_runs
            metric: Metric to plot
            save_path: Path to save the plot (default: None, don't save)
            show: Whether to show the plot (default: True)
            
        Returns:
            Matplotlib figure
        """
        # Filter data for the specific metric
        df = comparison_df[comparison_df["Metric"] == metric].copy()
        
        if df.empty:
            raise ValueError(f"No data found for metric {metric}")
        
        # Convert dimension to numeric for proper sorting
        df["Dimension"] = pd.to_numeric(df["Dimension"])
        df = df.sort_values("Dimension")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each run as a line
        for run_name in df["Run Name"].unique():
            run_df = df[df["Run Name"] == run_name]
            ax.plot(run_df["Dimension"], run_df["Value"], marker='o', label=run_name)
        
        # Add labels and legend
        ax.set_xlabel("Dimension")
        ax.set_ylabel(metric)
        ax.set_title(f"Comparison of {metric} Across Runs")
        ax.legend()
        
        # Set x-axis to integer ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Grid for easier reading
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show if requested
        if show:
            plt.show()
        
        return fig
    
    def generate_progress_report(self, 
                                run_ids: Optional[List[str]] = None,
                                metrics: Optional[List[str]] = None,
                                dimensions: Optional[List[int]] = None,
                                report_name: str = "progress_report") -> str:
        """
        Generate a comprehensive progress report.
        
        Args:
            run_ids: List of run IDs to include (default: all runs)
            metrics: List of metrics to include (default: all metrics)
            dimensions: List of dimensions to include (default: all dimensions)
            report_name: Name for the report file
            
        Returns:
            Path to the generated report
        """
        # Get run IDs if not provided
        if run_ids is None:
            run_ids = [run["id"] for run in self.history]
        
        if not run_ids:
            raise ValueError("No runs available for progress report")
        
        # Generate comparison data
        comparison_df = self.compare_runs(run_ids, metrics, dimensions)
        
        # Create report directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.base_dir, "reports", f"{timestamp}_{report_name}")
        os.makedirs(report_dir, exist_ok=True)
        plots_dir = os.path.join(report_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Start markdown content
        md_content = f"# Dimensional Cascade Progress Report\n\n"
        md_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add run summary
        md_content += "## Runs Included\n\n"
        md_content += "| Run ID | Name | Timestamp | Description |\n"
        md_content += "|--------|------|-----------|-------------|\n"
        
        for run_id in run_ids:
            try:
                run_data = self.get_run_data(run_id)
                md_content += f"| {run_data['id']} | {run_data['name']} | {run_data['timestamp']} | {run_data['description']} |\n"
            except FileNotFoundError:
                continue
        
        md_content += "\n"
        
        # Get unique metrics from comparison data
        unique_metrics = comparison_df["Metric"].unique()
        
        # Generate plots for each metric
        md_content += "## Metrics Comparison\n\n"
        
        for metric in unique_metrics:
            plot_filename = f"{metric.replace(' ', '_').lower()}_comparison.png"
            plot_path = os.path.join(plots_dir, plot_filename)
            
            fig = self.plot_comparison(
                comparison_df=comparison_df,
                metric=metric,
                save_path=plot_path,
                show=False
            )
            plt.close(fig)
            
            md_content += f"### {metric}\n\n"
            md_content += f"![{metric} Comparison](plots/{plot_filename})\n\n"
            
            # Add table of values
            metric_df = comparison_df[comparison_df["Metric"] == metric].copy()
            metric_df["Dimension"] = pd.to_numeric(metric_df["Dimension"])
            metric_df = metric_df.sort_values(["Run Name", "Dimension"])
            
            md_content += "#### Values\n\n"
            md_content += "| Run | Dimension | Value |\n"
            md_content += "|-----|-----------|-------|\n"
            
            for _, row in metric_df.iterrows():
                md_content += f"| {row['Run Name']} | {row['Dimension']} | {row['Value']:.4f} |\n"
            
            md_content += "\n"
        
        # Add detailed parameters for each run
        md_content += "## Run Parameters\n\n"
        
        for run_id in run_ids:
            try:
                run_data = self.get_run_data(run_id)
                md_content += f"### {run_data['name']} ({run_data['id']})\n\n"
                
                params = run_data["parameters"]
                if params:
                    md_content += "| Parameter | Value |\n"
                    md_content += "|-----------|-------|\n"
                    
                    for param, value in params.items():
                        if isinstance(value, dict):
                            value_str = json.dumps(value)
                        else:
                            value_str = str(value)
                        md_content += f"| {param} | {value_str} |\n"
                else:
                    md_content += "No parameters recorded for this run.\n"
                
                md_content += "\n"
            except FileNotFoundError:
                continue
        
        # Write markdown to file
        md_path = os.path.join(report_dir, "progress_report.md")
        with open(md_path, 'w') as f:
            f.write(md_content)
        
        # Convert to HTML
        html_content = markdown.markdown(md_content, extensions=['tables'])
        html_path = os.path.join(report_dir, "progress_report.html")
        
        with open(html_path, 'w') as f:
            f.write(f"""
            <html>
            <head>
                <title>Dimensional Cascade Progress Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #3498db; margin-top: 30px; }}
                    h3 {{ color: #2980b9; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    img {{ max-width: 100%; height: auto; }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """)
        
        logger.info(f"Progress report generated at {report_dir}")
        return report_dir
    
    def load_validation_metrics(self, validation_dir: str) -> Dict[str, Dict[str, float]]:
        """
        Load metrics from a validation directory.
        
        Args:
            validation_dir: Path to validation directory
            
        Returns:
            Dictionary of metrics
        """
        metrics_file = os.path.join(validation_dir, "metrics.json")
        
        if not os.path.exists(metrics_file):
            raise FileNotFoundError(f"Metrics file not found in {validation_dir}")
        
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    def batch_import_validations(self, base_dir: str, pattern: str = "*") -> List[str]:
        """
        Import multiple validation results at once.
        
        Args:
            base_dir: Base directory containing validation folders
            pattern: Pattern to match validation folders
            
        Returns:
            List of imported run IDs
        """
        import_paths = glob.glob(os.path.join(base_dir, pattern))
        run_ids = []
        
        for path in import_paths:
            if os.path.isdir(path):
                try:
                    run_id = self.import_validation_results(path)
                    run_ids.append(run_id)
                    logger.info(f"Imported validation results from {path}")
                except Exception as e:
                    logger.error(f"Failed to import {path}: {str(e)}")
        
        return run_ids


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dimensional Cascade Progress Tracker")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Import validation results
    import_parser = subparsers.add_parser("import", help="Import validation results")
    import_parser.add_argument("validation_dir", type=str, help="Directory containing validation results")
    import_parser.add_argument("--name", type=str, help="Name for the run")
    import_parser.add_argument("--description", type=str, help="Description of the run")
    
    # Batch import validation results
    batch_parser = subparsers.add_parser("batch-import", help="Import multiple validation results")
    batch_parser.add_argument("base_dir", type=str, help="Base directory containing validation folders")
    batch_parser.add_argument("--pattern", type=str, default="*", help="Pattern to match validation folders")
    
    # List runs
    list_parser = subparsers.add_parser("list", help="List tracked runs")
    
    # Generate report
    report_parser = subparsers.add_parser("report", help="Generate progress report")
    report_parser.add_argument("--runs", type=str, nargs="+", help="Run IDs to include (default: all)")
    report_parser.add_argument("--metrics", type=str, nargs="+", help="Metrics to include (default: all)")
    report_parser.add_argument("--dimensions", type=int, nargs="+", help="Dimensions to include (default: all)")
    report_parser.add_argument("--name", type=str, default="progress_report", help="Name for the report")
    
    # Compare runs
    compare_parser = subparsers.add_parser("compare", help="Compare runs")
    compare_parser.add_argument("runs", type=str, nargs="+", help="Run IDs to compare")
    compare_parser.add_argument("--metrics", type=str, nargs="+", help="Metrics to compare (default: all shared)")
    compare_parser.add_argument("--dimensions", type=int, nargs="+", help="Dimensions to compare (default: all shared)")
    
    # Parse args
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = ProgressTracker()
    
    if args.command == "import":
        # Import validation results
        run_id = tracker.import_validation_results(
            args.validation_dir,
            run_name=args.name,
            description=args.description
        )
        print(f"Imported run {run_id}")
        
    elif args.command == "batch-import":
        # Batch import validation results
        run_ids = tracker.batch_import_validations(args.base_dir, args.pattern)
        print(f"Imported {len(run_ids)} runs")
        for run_id in run_ids:
            print(f"- {run_id}")
            
    elif args.command == "list":
        # List runs
        runs = tracker.list_runs()
        if runs:
            print(f"Found {len(runs)} tracked runs:")
            for run in runs:
                print(f"- {run['id']}: {run['name']} ({run['timestamp']})")
        else:
            print("No tracked runs found")
            
    elif args.command == "report":
        # Generate report
        report_dir = tracker.generate_progress_report(
            run_ids=args.runs,
            metrics=args.metrics,
            dimensions=args.dimensions,
            report_name=args.name
        )
        print(f"Report generated at {report_dir}")
        print(f"Markdown: {os.path.join(report_dir, 'progress_report.md')}")
        print(f"HTML: {os.path.join(report_dir, 'progress_report.html')}")
        
    elif args.command == "compare":
        # Compare runs
        comparison_df = tracker.compare_runs(
            args.runs,
            metrics=args.metrics,
            dimensions=args.dimensions
        )
        
        print(comparison_df)
        
        # Plot each metric
        for metric in comparison_df["Metric"].unique():
            tracker.plot_comparison(comparison_df, metric)
    
    else:
        parser.print_help() 