import pandas as pd
from pathlib import Path
import re


def load_annotations(file_name: str):
    """
    Load the raw annotation file and keep only the relevant columns.

    """
    df_raw = pd.read_csv(file_name, delimiter='\t', header=None)
    df = df_raw.iloc[:, [0, 2, 4, 6, 8]].copy()
    df.columns = ['Tier_Name', 'Start_Time', 'End_Time', 'Duration', 'Annotation_Value']
    return df


def filter_by_scene(df: pd.DataFrame):
    """
    Keep only annotations that fall within 'Scene' tier time intervals.

    Args:
        df: DataFrame of annotations.

    Returns:
        Filtered DataFrame containing only annotations within scene intervals.
        If 'Scene' tier is not found, returns all annotations.
    """
    scene_records = df[df['Tier_Name'] == 'Scene']
    if scene_records.empty:
        print("Scene tier not found. Keeping all annotations.")
        return df

    # Collect start/end times for each scene
    scene_intervals = []
    for _, row in scene_records.iterrows():
        scene_intervals.append((row['Start_Time'], row['End_Time']))

    # Keep only annotations that occur fully inside any scene interval
    df_filtered = pd.DataFrame()
    for start_time, end_time in scene_intervals:
        df_fragment = df[
            (df['Start_Time'] >= start_time) &
            (df['End_Time'] <= end_time)
        ].copy()
        df_filtered = pd.concat([df_filtered, df_fragment])

    return df_filtered.copy()


def time_string_to_ms(time_str: str):
    """
    Convert a time string in the format 'HH:MM:SS.ms' to milliseconds.

    """
    if pd.isna(time_str):
        return None
    h, m, s_ms = time_str.split(':')
    s, ms = s_ms.split('.')
    return (
        int(h) * 3600000 +
        int(m) * 60000 +
        int(s) * 1000 +
        int(ms)
    )


def preprocess_times(df: pd.DataFrame):
    """
    Add millisecond and 100ms-based time columns
    """
    df = df.copy()
    # Convert times to milliseconds
    df['Start_Time_ms'] = df['Start_Time'].apply(time_string_to_ms)
    df['End_Time_ms'] = df['End_Time'].apply(time_string_to_ms)

    # Convert to 100ms
    df['Start'] = (df['Start_Time_ms'] // 100).astype('Int64')
    df['End'] = (df['End_Time_ms'] // 100).astype('Int64')

    # Remove rows with missing times
    df = df.dropna()

    # Clean string fields
    for c in ['Tier_Name', 'Annotation_Value']:
        df[c] = df[c].astype(str).str.strip()

    # Remove reference numbers in annotations (e.g., "14_mom spits...")
    df['Annotation_Value'] = df['Annotation_Value'].apply(
        lambda x: re.sub(r'^\d+_', '', x) if isinstance(x, str) else x
    )

    return df


def annotations_per_s(df: pd.DataFrame):
    """
    Convert annotations into per-100ms time slices.

    """
    rows = []
    max_t = int(df['End'].max())

    for t in range(0, max_t + 1):
        # Get annotations active at time t
        active_rows = df[(df['Start'] <= t) & (df['End'] > t)]
        if not active_rows.empty:
            for _, row in active_rows.iterrows():
                if row['Tier_Name'] not in ["Round", "Scene", "Comment"]:
                    rows.append({
                        'Time': t,
                        'Tier': row['Tier_Name'],
                        'Annotation': row['Annotation_Value']
                    })

    return pd.DataFrame(rows)


def export(df: pd.DataFrame, output_name: str):
    """
    Save the DataFrame to a CSV file.

    """
    df = df.dropna()
    df.to_csv(output_name, index=False)
    print(f"Per-second annotations saved to {Path(output_name).resolve()}")


def process_annotations(file_name: str, output_name: str = "annotations_per_t.csv"):

    df = load_annotations(file_name)
    df = filter_by_scene(df)
    df = preprocess_times(df)
    per_t_df = annotations_per_s(df)
    export(per_t_df, output_name)
