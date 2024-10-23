# Base
# -----------------------------------
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import datetime
import os

# Configuration
# -----------------------------------
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.4f}'.format

import evidently

from evidently.metrics import ColumnDriftMetric
from evidently.metrics import ColumnSummaryMetric
from evidently.metrics import DatasetDriftMetric
from evidently.metrics import DatasetMissingValuesMetric
from evidently.metrics import DatasetCorrelationsMetric
from evidently.metrics import DatasetSummaryMetric
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite
from evidently.ui.dashboards import CounterAgg
from evidently.ui.dashboards import DashboardPanelCounter
from evidently.ui.dashboards import DashboardPanelPlot
from evidently.ui.dashboards import PanelValue
from evidently.ui.dashboards import PlotType
from evidently.ui.dashboards import ReportFilter
from evidently.ui.remote import RemoteWorkspace
from evidently.ui.workspace import Workspace
from evidently.ui.workspace import WorkspaceBase
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset

# ------------------------------------- [README] ---------------------------------------
# 1º: Execute this file giving two paths: python evidently_drifting.py --refence-loc "dataframes\reference.csv" --current-loc "dataframes\current.csv" --name "space-separated column name"
# 2º: Execute: evidently ui --workspace ./evidently_workspace --port 8080
# 3º: Go to http://localhost:8080/
# --------------------------------------------------------------------------------------

def create_report(names, df):
    '''Generates a report on data drift using Evidently.

    Parameters:
    ----------
    names : list
        List of column names for which drift metrics will be created.
    df : pd.DataFrame
        The dataframe to be analyzed.

    Returns:
    -------
    Report
        An Evidently Report object containing the drift metrics and summaries.
    '''

    def create_metric(name):
        '''Creates drift and summary metrics for a given column based on its unique values.
        
        Parameters:
        ----------
        name : str
            The column name for which metrics are to be created.
        
        Returns:
        -------
        list
            A list of Evidently metric objects for the column.
        '''
        if len(df[name].unique()) > 2:
            met = [ColumnDriftMetric(column_name=name, stattest="wasserstein"),
                   ColumnSummaryMetric(column_name=name)]
        else:
            met = [ColumnDriftMetric(column_name=name, stattest="jensenshannon"),
                   ColumnSummaryMetric(column_name=name)]
        return met
    
    # Basic metrics for the dataset
    metrics = [DatasetDriftMetric(),
               DatasetMissingValuesMetric(),
               DatasetSummaryMetric(),
               DatasetCorrelationsMetric(),
               DataDriftPreset(), 
               TargetDriftPreset(),
               ColumnDriftMetric(column_name="MotorData.ActCurrent", stattest="wasserstein"),
               ColumnSummaryMetric(column_name="MotorData.ActCurrent"),
               ColumnDriftMetric(column_name="MotorData.ActPosition", stattest="wasserstein"),
               ColumnSummaryMetric(column_name="MotorData.ActPosition")]

    # Add metrics for each column in the provided names list
    for i in names:
        metrics += create_metric(i)
    
    # Create and run the data drift report
    data_drift_report = Report(
        metrics=metrics,
        timestamp=datetime.datetime.now())

    data_drift_report.run(reference_data=reference, current_data=current)
    
    return data_drift_report

def create_test_suite():
    '''Creates and runs a data drift test suite using Evidently.
    
    Returns:
    -------
    TestSuite
        An Evidently TestSuite object containing the results of the data drift tests.
    '''
    data_drift_test_suite = TestSuite(
        tests=[DataDriftTestPreset()],
        timestamp=datetime.datetime.now()
    )

    data_drift_test_suite.run(reference_data=reference, current_data=current)
    return data_drift_test_suite

def create_project(workspace: WorkspaceBase, names, df: pd.DataFrame):
    '''Creates a project in the Evidently workspace and adds panels for data drift analysis.
    
    Parameters:
    ----------
    workspace : WorkspaceBase
        The Evidently workspace where the project will be created.
    names : list
        List of column names for which panels will be added.
    df : pd.DataFrame
        The dataframe used for analysis.

    Returns:
    -------
    Project
        The created Evidently project.
    '''
    project = workspace.create_project(PROJECT_NAME)
    project.description = PROJECT_DESCRIPTION

    def add_column(column, project):
        '''Adds a panel to the project dashboard for a specific column's drift metric.
        
        Parameters:
        ----------
        column : str
            The column name for which a panel will be added.
        project : Project
            The Evidently project to which the panel will be added.
        
        Returns:
        -------
        Project
            The updated Evidently project with the added panel.
        '''
        if len(df[column].unique()) <= 2:
            project.dashboard.add_panel(
                DashboardPanelPlot(title=f"{column}: Jensen-Shannon Distance",
                                   filter=ReportFilter(metadata_values={}, 
                                                       tag_values=[]),
                                                       values=[
                                                           PanelValue(metric_id="ColumnDriftMetric",
                                                                      metric_args={"column_name.name": f"{column}"},
                                                                      field_path=ColumnDriftMetric.fields.drift_score,
                                                                      legend="Drift Score"),],
                                                       plot_type=PlotType.BAR, size=1))
        else:
            project.dashboard.add_panel(
                DashboardPanelPlot(title=f"{column}: Wasserstein Distance",
                                   filter=ReportFilter(metadata_values={}, 
                                                       tag_values=[]),
                                                       values=[
                                                           PanelValue(metric_id="ColumnDriftMetric",
                                                                      metric_args={"column_name.name": f"{column}"},
                                                                      field_path=ColumnDriftMetric.fields.drift_score,
                                                                      legend="Drift Score"),],
                                                       plot_type=PlotType.BAR, size=1))
        return project
            

    # Add basic panels to the dashboard
    project.dashboard.add_panel(
        DashboardPanelCounter(
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            agg=CounterAgg.NONE,
            title="PanelCounter",
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Model Calls",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="DatasetMissingValuesMetric",
                field_path=DatasetMissingValuesMetric.fields.current.number_of_rows,
                legend="count",
            ),
            text="count",
            agg=CounterAgg.SUM,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Share of Drifted Features",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="DatasetDriftMetric",
                field_path="share_of_drifted_columns",
                legend="share",
            ),
            text="share",
            agg=CounterAgg.LAST,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Dataset Quality",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(metric_id="DatasetDriftMetric", field_path="share_of_drifted_columns", legend="Drift Share"),
                PanelValue(
                    metric_id="DatasetMissingValuesMetric",
                    field_path=DatasetMissingValuesMetric.fields.current.share_of_missing_values,
                    legend="Missing Values Share",
                ),
            ],
            plot_type=PlotType.LINE,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="MotorData.ActCurrent: Wasserstein drift distance",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    metric_args={"column_name.name": "MotorData.ActCurrent"},
                    field_path=ColumnDriftMetric.fields.drift_score,
                    legend="Drift Score",
                ),
            ],
            plot_type=PlotType.BAR,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="MotorData.ActPosition: Wasserstein drift distance",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    metric_args={"column_name.name": "MotorData.ActPosition"},
                    field_path=ColumnDriftMetric.fields.drift_score,
                    legend="Drift Score",
                ),
            ],
            plot_type=PlotType.BAR,
            size=1,
        )
    )
    for i in names:
        add_column(i, project)

    project.save()

    return project

def create_evi_project(workspace: str, names: list, df: pd.DataFrame):
    '''Creates an Evidently project, generates a report, and adds it to the workspace.
    
    Parameters:
    ----------
    workspace : str
        The path to the Evidently workspace directory.
    names : list
        List of column names for which metrics and panels will be created.
    df : pd.DataFrame
        The dataframe used for analysis.
    '''
    ws = Workspace.create(workspace)
    project = create_project(ws, names, df)

    report = create_report(names, df)
    ws.add_report(project.id, report)

    test_suite = create_test_suite()
    ws.add_test_suite(project.id, test_suite)


# Argparse to handle command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--reference-loc')
parser.add_argument('--current-loc')
parser.add_argument('--names', '--list', nargs='+')

# Configuration variables
WORKSPACE = "evidently_workspace" # Workspace defines the folder where Evidently will log data to.
PROJECT_NAME = "KAIROS_WORKSPACE"
PROJECT_DESCRIPTION = "Kairos monitorization system"

if __name__ == '__main__':
    # Parse command-line arguments
    args, _ = parser.parse_known_args()
    print(f"Reference Location: {args.reference_loc}")
    print(f"Current Location: {args.current_loc}")
    print(f"Names: {args.names}")

    print(args)

    names = args.names if args.names is not None else []


    # Load the reference and current datasets from the specified locations
    if args.reference_loc is not None and args.current_loc is not None:
        print("HIIII")
        try:
            reference = pd.read_csv(args.reference_loc)
            current = pd.read_csv(args.current_loc)
            print("Reference and current data loaded successfully")
            print(f"Columns in current dataset: {current.columns}")
        except Exception as e:
            print(f"Error reading .csv files: {e}. Switching to default config.")
    else:
        print('Error: Both reference_loc and current_loc must be provided.')
    
    # Ensure names is not None or empty
    if names:
        print('column names loaded correctly')
        print(f'Columns to be analyzed: {names}')
    else:
        print('No column names provided or parsed.')

    # Create the Evidently project using the loaded data
    if 'current' in locals():  # Only proceed if 'current' was successfully loaded
        create_evi_project(WORKSPACE, names, current)
    else:
        print("Current data not loaded. Aborting Evidently project creation.")


