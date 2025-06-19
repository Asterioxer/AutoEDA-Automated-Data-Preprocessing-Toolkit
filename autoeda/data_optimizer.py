import pandas as pd
import logging


# Configure logging basic settings
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


# Two blank lines before function definition
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimizes the data types of columns in a Pandas DataFrame.

    Args:
        df: The input DataFrame.

    Returns:
        The DataFrame with optimized data types.
    """
    df_optimized = df.copy()

    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype

        if col_type == 'object':
            # Attempt to convert columns with specific keywords to datetime
            is_datetime_like = any(keyword in col.lower()
                                   for keyword in ['date', 'time', 'timestamp'])
            if is_datetime_like:
                try:
                    df_optimized[col] = pd.to_datetime(df_optimized[col])
                    log_msg = (f"Col '{col}' conv "  # Minimized first part
                               f"to datetime.")
                    logging.info(log_msg)
                except ValueError:
                    log_msg = (f"Could not convert col '{col}' to datetime. "
                               "Might not be valid date/time format.")
                    logging.warning(log_msg)
                except Exception as e:
                    log_msg = (f"ERR conv '{col}' "  # Minimized first part
                               f"to datetime: {e}")
                    logging.error(log_msg)
            # Convert to 'category' if low cardinality and still object type
            elif (df_optimized[col].nunique() < 50 and
                  df_optimized[col].dtype == 'object'):
                try:
                    df_optimized[col] = df_optimized[col].astype('category')
                    log_msg = f"Col '{col}' to category."
                    logging.info(log_msg)
                except Exception as e:
                    log_msg = (f"ERR conv '{col}' "
                               f"to category: {e}")
                    logging.error(log_msg)

        # Downcast float64 columns to float32
        elif col_type == 'float64':
            try:
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
                log_msg = f"Col '{col}' to float32."
                logging.info(log_msg)
            except Exception as e:
                log_msg = (f"ERR downcast '{col}' "
                           f"to float32: {e}")
                logging.error(log_msg)

        # Downcast int64 columns to the smallest possible integer subtype
        elif col_type == 'int64':
            try:
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
                log_msg = f"Col '{col}' to smallest int."
                logging.info(log_msg)
            except Exception as e:
                log_msg = (f"ERR downcast '{col}' "  # Minimized first part
                           f"to smaller int: {e}")
                logging.error(log_msg)

    return df_optimized


def optimize_csv(input_path: str, output_path: str) -> None:
    """
    Reads a CSV, optimizes its data types, and saves it to a new CSV.

    Args:
        input_path: Path to the input CSV file.
        output_path: Path to save the optimized CSV file.
    """
    try:
        df = pd.read_csv(input_path)
        log_msg = (f"Read CSV "  # Minimized first part
                   f"from '{input_path}'.")
        logging.info(log_msg)
    except FileNotFoundError:
        log_msg = f"CSV not found: '{input_path}'."
        logging.error(log_msg)
        return
    except Exception as e:
        log_msg = (f"ERR reading CSV "
                   f"'{input_path}': {e}")
        logging.error(log_msg)
        return

    original_memory_usage = df.memory_usage(deep=True).sum()
    mem_orig_mb = original_memory_usage / (1024 * 1024)
    logging.info(f"Original memory: {mem_orig_mb:.2f} MB")

    df_optimized = optimize_dtypes(df)

    optimized_memory_usage = df_optimized.memory_usage(deep=True).sum()
    mem_opt_mb = optimized_memory_usage / (1024 * 1024)
    logging.info(f"Optimized memory: {mem_opt_mb:.2f} MB")

    reduction_bytes = original_memory_usage - optimized_memory_usage
    reduction_percent = 0
    if original_memory_usage > 0:
        reduction_percent = (reduction_bytes / original_memory_usage) * 100
    log_msg_reduction = (
        f"Mem. reduced by {reduction_bytes} bytes ({reduction_percent:.2f}%)."
    )
    logging.info(log_msg_reduction)

    try:
        df_optimized.to_csv(output_path, index=False)
        log_msg = f"Saved optimized CSV to '{output_path}'."
        logging.info(log_msg)
    except Exception as e:
        log_msg = f"ERR saving CSV to '{output_path}': {e}"
        logging.error(log_msg)
