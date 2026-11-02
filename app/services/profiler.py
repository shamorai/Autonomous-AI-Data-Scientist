import polars as pl
import pandas as pd
from typing import Union
from pathlib import Path
from app.schemas.profile import ColumnProfile, DatasetProfile

class DataProfiler:
    """Service class responsible for analyzing CSV files and building a DatasetProfile object."""
    
    def __init__(self, file_path: Union[str, Path]):
        # Convert string file path to a standard pathlib Path object
        self.file_path = Path(file_path)
        
        # Load dataset into Polars dataframe for fast metadata/null scanning
        self.df_pl = pl.read_csv(self.file_path)
        
        # Load dataset into Pandas dataframe for standard statistical analysis
        self.df_pd = pd.read_csv(self.file_path)

    def profile(self) -> DatasetProfile:
        # Extract row count and column count from Polars dataframe tuple (rows, cols)
        row_count, col_count = self.df_pl.shape
        
        columns_profile: list[ColumnProfile] = []
        summary_notes: list[str] = []

        # Iterate through every column in the dataset
        for col_name in self.df_pl.columns:
            # Get column object from both Polars and Pandas DataFrames
            series_pl = self.df_pl[col_name]
            series_pd = self.df_pd[col_name]
            
            # Count nulls and calculate percentage missing
            null_cnt = series_pl.null_count()
            null_pct = round((null_cnt / row_count) * 100, 2)
            
            # Count distinct unique elements
            unique_cnt = series_pl.n_unique()
            
            # Check if data type is numeric (integer or float)
            is_num = series_pl.dtype.is_numeric()

            # Trigger automated warnings based on thresholds
            if null_pct > 20.0:
                summary_notes.append(f"Warning: Column '{col_name}' has {null_pct}% missing values.")
            if unique_cnt == row_count and not is_num:
                summary_notes.append(f"Notice: Column '{col_name}' appears to be a unique ID field.")

            # Calculate numerical summary statistics if column is numeric and not completely empty
            stats_dict = None
            if is_num and null_cnt < row_count:
                stats_dict = {
                    "min": float(series_pd.min()),
                    "max": float(series_pd.max()),
                    "mean": float(round(series_pd.mean(), 4)),
                    "median": float(round(series_pd.median(), 4)),
                    "std": float(round(series_pd.std(), 4)) if pd.notna(series_pd.std()) else 0.0
                }

            # Extract up to 3 non-null sample values for context preview
            samples = series_pd.dropna().head(3).tolist()

            # Instantiate and append ColumnProfile object to list
            columns_profile.append(
                ColumnProfile(
                    name=col_name,
                    dtype=str(series_pl.dtype),
                    null_count=null_cnt,
                    null_percentage=null_pct,
                    unique_count=unique_cnt,
                    is_numeric=is_num,
                    sample_values=samples,
                    stats=stats_dict
                )
            )

        # Compute Pearson Correlation Matrix only for numeric columns (if at least 2 numeric columns exist)
        corr_matrix = None
        numeric_df = self.df_pd.select_dtypes(include=['number'])
        if numeric_df.shape[1] > 1:
            # Compute correlation matrix, round to 3 decimal places, and fill NaN values with 0.0
            raw_corr = numeric_df.corr().round(3).fillna(0.0).to_dict()
            # Convert inner dictionary layout into standard Python nested dictionary
            corr_matrix = {col: dict(vals) for col, vals in raw_corr.items()}

        # Construct and return the overall structured dataset profile
        return DatasetProfile(
            filename=self.file_path.name,
            row_count=row_count,
            column_count=col_count,
            columns=columns_profile,
            correlations=corr_matrix,
            summary_notes=summary_notes
        )