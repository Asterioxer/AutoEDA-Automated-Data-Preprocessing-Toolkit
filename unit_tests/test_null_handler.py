import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pandas.testing import assert_frame_equal

# Import functions from the module to be tested
from autoeda.null_handler import (
    drop_nulls,
    replace_with_fixed,
    replace_with_mean,
    replace_with_median,
    replace_with_mode,
    forward_fill,
    backward_fill,
    evaluate_methods,
    process_csv
)

# Basic test to ensure the file is set up correctly
def test_pytest_setup():
    assert True

# --------------*PYTEST FIXTURES*------------------

@pytest.fixture
def sample_dfs():
    """Provides a dictionary of sample DataFrames for testing."""
    data_numeric_nulls = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, 4, 5],
        'C': [1.0, 2.0, 3.0, 4.0, np.nan]
    }
    df_numeric_nulls = pd.DataFrame(data_numeric_nulls)

    data_categorical_nulls = {
        'X': ['apple', 'banana', np.nan, 'orange', 'banana'],
        'Y': [np.nan, 'cat', 'dog', 'cat', np.nan],
        'Z': ['one', 'two', 'three', 'four', 'five']
    }
    df_categorical_nulls = pd.DataFrame(data_categorical_nulls)

    data_mixed_nulls = {
        'NumCol': [1, np.nan, 3, 4, np.nan],
        'CatCol': ['a', 'b', np.nan, 'd', 'a'],
        'FullNumCol': [1.1, 2.2, 3.3, 4.4, 5.5],
        'FullCatCol': ['x', 'y', 'z', 'x', 'y']
    }
    df_mixed_nulls = pd.DataFrame(data_mixed_nulls)

    data_all_null_col = {
        'A': [1, 2, 3, 4, 5],
        'B': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'C': ['x', 'y', 'z', 'a', 'b']
    }
    df_all_null_col = pd.DataFrame(data_all_null_col)
    
    data_all_null_df = {
        'A': [np.nan, np.nan, np.nan],
        'B': [np.nan, np.nan, np.nan],
    }
    df_all_null_df = pd.DataFrame(data_all_null_df)

    df_empty = pd.DataFrame()

    data_no_nulls = {
        'P': [1, 2, 3, 4, 5],
        'Q': [1.1, 2.2, 3.3, 4.4, 5.5],
        'R': ['one', 'two', 'three', 'four', 'five']
    }
    df_no_nulls = pd.DataFrame(data_no_nulls)

    return {
        "numeric_nulls": df_numeric_nulls,
        "categorical_nulls": df_categorical_nulls,
        "mixed_nulls": df_mixed_nulls,
        "all_null_col": df_all_null_col,
        "all_null_df": df_all_null_df,
        "empty": df_empty,
        "no_nulls": df_no_nulls,
    }

# --------------*TESTS FOR drop_nulls()*------------------

def test_drop_nulls_numeric(sample_dfs):
    df_in = sample_dfs["numeric_nulls"].copy()
    df_out = drop_nulls(df_in)
    assert df_out.isnull().sum().sum() == 0
    assert df_out.shape[0] < df_in.shape[0] # Rows should be dropped
    assert_frame_equal(df_in.dropna(), df_out)

def test_drop_nulls_categorical(sample_dfs):
    df_in = sample_dfs["categorical_nulls"].copy()
    df_out = drop_nulls(df_in)
    assert df_out.isnull().sum().sum() == 0
    assert df_out.shape[0] < df_in.shape[0]
    assert_frame_equal(df_in.dropna(), df_out)

def test_drop_nulls_mixed(sample_dfs):
    df_in = sample_dfs["mixed_nulls"].copy()
    df_out = drop_nulls(df_in)
    assert df_out.isnull().sum().sum() == 0
    assert df_out.shape[0] < df_in.shape[0]
    assert_frame_equal(df_in.dropna(), df_out)

def test_drop_nulls_all_null_col(sample_dfs):
    df_in = sample_dfs["all_null_col"].copy() # This df has one col that is all nulls
    df_out = drop_nulls(df_in)
    # dropna() will drop rows that have *any* NaN. If a column is all NaN, all rows containing it are dropped.
    # If other columns define rows, then those rows are dropped.
    # If it's a single column of NaNs, the DataFrame becomes empty.
    # In this fixture, col 'B' is all NaN, so all rows will be dropped.
    assert df_out.empty
    assert df_out.shape[0] == 0
    assert df_out.shape[1] == df_in.shape[1] # Columns should remain, but 0 rows
    assert_frame_equal(df_in.dropna(), df_out)

def test_drop_nulls_all_null_df(sample_dfs):
    df_in = sample_dfs["all_null_df"].copy()
    df_out = drop_nulls(df_in)
    assert df_out.empty
    assert df_out.shape[0] == 0
    assert df_out.shape[1] == df_in.shape[1]
    assert_frame_equal(df_in.dropna(), df_out)

def test_drop_nulls_empty_df(sample_dfs):
    df_in = sample_dfs["empty"].copy()
    df_out = drop_nulls(df_in)
    assert df_out.empty
    assert_frame_equal(df_in, df_out) # Should be identical

def test_drop_nulls_no_nulls(sample_dfs):
    df_in = sample_dfs["no_nulls"].copy()
    original_shape = df_in.shape
    df_out = drop_nulls(df_in)
    assert df_out.isnull().sum().sum() == 0
    assert df_out.shape == original_shape # Shape should be preserved
    assert_frame_equal(df_in, df_out) # Data should be preserved

# --------------*TESTS FOR replace_with_fixed()*------------------

def test_replace_with_fixed_numeric_default(sample_dfs):
    df_in = sample_dfs["numeric_nulls"].copy()
    df_out = replace_with_fixed(df_in) # Default value is 0
    assert df_out.isnull().sum().sum() == 0
    assert df_out.shape == df_in.shape
    expected_df = df_in.fillna(0)
    assert_frame_equal(df_out, expected_df)

def test_replace_with_fixed_numeric_custom_value(sample_dfs):
    df_in = sample_dfs["numeric_nulls"].copy()
    custom_val = -99
    df_out = replace_with_fixed(df_in, value=custom_val)
    assert df_out.isnull().sum().sum() == 0
    assert df_out.shape == df_in.shape
    expected_df = df_in.fillna(custom_val)
    assert_frame_equal(df_out, expected_df)

def test_replace_with_fixed_categorical_default(sample_dfs):
    df_in = sample_dfs["categorical_nulls"].copy()
    df_out = replace_with_fixed(df_in) # Default value is 0
    assert df_out.isnull().sum().sum() == 0
    assert df_out.shape == df_in.shape
    expected_df = df_in.fillna(0) # Categorical cols will be filled with int 0
    assert_frame_equal(df_out, expected_df)

def test_replace_with_fixed_categorical_custom_value(sample_dfs):
    df_in = sample_dfs["categorical_nulls"].copy()
    custom_val = "MISSING"
    df_out = replace_with_fixed(df_in, value=custom_val)
    assert df_out.isnull().sum().sum() == 0
    assert df_out.shape == df_in.shape
    expected_df = df_in.fillna(custom_val)
    assert_frame_equal(df_out, expected_df)

def test_replace_with_fixed_mixed_default(sample_dfs):
    df_in = sample_dfs["mixed_nulls"].copy()
    df_out = replace_with_fixed(df_in)
    assert df_out.isnull().sum().sum() == 0
    assert df_out.shape == df_in.shape
    expected_df = df_in.fillna(0)
    assert_frame_equal(df_out, expected_df)

def test_replace_with_fixed_all_null_col(sample_dfs):
    df_in = sample_dfs["all_null_col"].copy()
    custom_val = -1
    df_out = replace_with_fixed(df_in, value=custom_val)
    assert df_out.isnull().sum().sum() == 0
    assert df_out.shape == df_in.shape
    assert (df_out['B'] == custom_val).all() # Check the all-null column
    expected_df = df_in.fillna(custom_val)
    assert_frame_equal(df_out, expected_df)

def test_replace_with_fixed_all_null_df(sample_dfs):
    df_in = sample_dfs["all_null_df"].copy()
    custom_val = "N/A"
    df_out = replace_with_fixed(df_in, value=custom_val)
    assert df_out.isnull().sum().sum() == 0
    assert df_out.shape == df_in.shape
    expected_df = df_in.fillna(custom_val)
    assert_frame_equal(df_out, expected_df)
    for col in df_out.columns:
        assert (df_out[col] == custom_val).all()


def test_replace_with_fixed_empty_df(sample_dfs):
    df_in = sample_dfs["empty"].copy()
    df_out = replace_with_fixed(df_in, value=123)
    assert df_out.empty
    assert df_out.shape == df_in.shape
    assert_frame_equal(df_in, df_out)

def test_replace_with_fixed_no_nulls(sample_dfs):
    df_in = sample_dfs["no_nulls"].copy()
    original_shape = df_in.shape
    df_out = replace_with_fixed(df_in, value=999)
    assert df_out.isnull().sum().sum() == 0
    assert df_out.shape == original_shape
    assert_frame_equal(df_in, df_out) # Data should be unchanged

# --------------*TESTS FOR replace_with_median()*------------------

def test_replace_with_median_numeric(sample_dfs):
    df_in = sample_dfs["numeric_nulls"].copy()
    df_out = replace_with_median(df_in)
    assert df_out.isnull().sum().sum() == 0 # All nulls in numeric cols should be filled
    assert df_out.shape == df_in.shape
    for col in df_in.select_dtypes(include=np.number).columns:
        median_val = df_in[col].median()
        expected_col = df_in[col].fillna(median_val)
        assert_frame_equal(df_out[[col]], expected_col.to_frame(), check_dtype=False)
    for col in df_in.select_dtypes(exclude=np.number).columns:
         assert_frame_equal(df_out[[col]], df_in[[col]])

def test_replace_with_median_mixed(sample_dfs):
    df_in = sample_dfs["mixed_nulls"].copy()
    df_out = replace_with_median(df_in)
    
    assert df_out['NumCol'].isnull().sum() == 0
    assert df_out['CatCol'].isnull().sum() == df_in['CatCol'].isnull().sum()
    assert df_out.shape == df_in.shape

    median_numcol = df_in['NumCol'].median()
    expected_numcol = df_in['NumCol'].fillna(median_numcol)
    assert_frame_equal(df_out[['NumCol']], expected_numcol.to_frame(), check_dtype=False)

    assert_frame_equal(df_out[['CatCol']], df_in[['CatCol']])
    assert_frame_equal(df_out[['FullNumCol']], df_in[['FullNumCol']])
    assert_frame_equal(df_out[['FullCatCol']], df_in[['FullCatCol']])

def test_replace_with_median_all_null_numeric_col(sample_dfs):
    df_in = sample_dfs["all_null_col"].copy()
    if df_in['B'].dtype == 'object':
         df_in['B'] = pd.to_numeric(df_in['B'], errors='coerce')

    df_out = replace_with_median(df_in)
    assert df_out.shape == df_in.shape
    # Median of an all-NaN column is NaN. fillna(NaN) doesn't change it.
    assert df_out['B'].isnull().all() 
    
    assert_frame_equal(df_out[['A']], df_in[['A']])
    assert_frame_equal(df_out[['C']], df_in[['C']])

def test_replace_with_median_all_null_df(sample_dfs):
    df_in = sample_dfs["all_null_df"].copy()
    df_out = replace_with_median(df_in)
    assert df_out.shape == df_in.shape
    assert_frame_equal(df_out, df_in) 
    assert df_out.isnull().all().all()

def test_replace_with_median_categorical_only(sample_dfs):
    df_in = sample_dfs["categorical_nulls"].copy()
    df_out = replace_with_median(df_in)
    assert df_out.shape == df_in.shape
    assert_frame_equal(df_out, df_in)

def test_replace_with_median_empty_df(sample_dfs):
    df_in = sample_dfs["empty"].copy()
    df_out = replace_with_median(df_in)
    assert df_out.empty
    assert df_out.shape == df_in.shape
    assert_frame_equal(df_in, df_out)

def test_replace_with_median_no_nulls(sample_dfs):
    df_in = sample_dfs["no_nulls"].copy()
    original_shape = df_in.shape
    df_out = replace_with_median(df_in)
    assert df_out.isnull().sum().sum() == 0
    assert df_out.shape == original_shape
    assert_frame_equal(df_in, df_out)

# --------------*TESTS FOR replace_with_mode()*------------------

def test_replace_with_mode_numeric(sample_dfs):
    df_in = sample_dfs["numeric_nulls"].copy()
    # Example: df_in['A']: [1, 2, np.nan, 4, 5], mode is ambiguous, pandas takes first of sorted unique if counts are same
    # For testing, ensure modes are clear or understand pandas behavior
    # A: mode could be any of [1,2,4,5] if they appear once. Let's ensure a clear mode for a column.
    # df_in has 5 rows from sample_dfs["numeric_nulls"]
    df_in_copy = df_in.copy() # Operate on a copy for adding test-specific columns
    df_in_copy['A_with_mode'] = [1, 1, np.nan, 1, np.nan] # Mode is 1. Length must match df_in_copy (5 rows)
    df_out = replace_with_mode(df_in_copy) 

    assert df_out['A_with_mode'].isnull().sum() == 0
    # Original NaNs at index 2 and 4. Expected: [1, 1, 1, 1, 1]
    # (df_out['A_with_mode'] == 1).sum() should be 5
    assert (df_out['A_with_mode'] == 1).sum() == 5
    
    # Test standard numeric_nulls columns, mode might be less predictable if multiple values have same highest frequency
    # For column 'A': [1, 2, np.nan, 4, 5]. All non-NaNs appear once. Pandas mode() returns them sorted: [1,2,4,5]. iloc[0] -> 1
    # For column 'B': [np.nan, 2, 3, 4, 5]. All non-NaNs appear once. Pandas mode() -> [2,3,4,5]. iloc[0] -> 2
    # For column 'C': [1.0, 2.0, 3.0, 4.0, np.nan]. All non-NaNs appear once. Pandas mode() -> [1,2,3,4]. iloc[0] -> 1.0
    df_numeric_original = sample_dfs["numeric_nulls"].copy()
    df_out_numeric = replace_with_mode(df_numeric_original.copy())
    assert df_out_numeric.isnull().sum().sum() == 0
    assert df_out_numeric.shape == df_numeric_original.shape
    
    expected_A = df_numeric_original['A'].fillna(df_numeric_original['A'].mode().iloc[0])
    pd.testing.assert_series_equal(df_out_numeric['A'], expected_A, check_dtype=False)
    
    expected_B = df_numeric_original['B'].fillna(df_numeric_original['B'].mode().iloc[0])
    pd.testing.assert_series_equal(df_out_numeric['B'], expected_B, check_dtype=False)

    expected_C = df_numeric_original['C'].fillna(df_numeric_original['C'].mode().iloc[0])
    pd.testing.assert_series_equal(df_out_numeric['C'], expected_C, check_dtype=False)


def test_replace_with_mode_categorical(sample_dfs):
    df_in = sample_dfs["categorical_nulls"].copy()
    # 'X': ['apple', 'banana', np.nan, 'orange', 'banana'] -> mode is 'banana'
    # 'Y': [np.nan, 'cat', 'dog', 'cat', np.nan] -> mode is 'cat'
    df_out = replace_with_mode(df_in.copy())

    assert df_out.isnull().sum().sum() == 0
    assert df_out.shape == df_in.shape

    expected_X = df_in['X'].fillna(df_in['X'].mode().iloc[0])
    pd.testing.assert_series_equal(df_out['X'], expected_X)
    
    expected_Y = df_in['Y'].fillna(df_in['Y'].mode().iloc[0])
    pd.testing.assert_series_equal(df_out['Y'], expected_Y)
    
    assert_frame_equal(df_out[['Z']], df_in[['Z']]) # Z has no nulls


def test_replace_with_mode_mixed(sample_dfs):
    df_in = sample_dfs["mixed_nulls"].copy()
    # NumCol: [1, np.nan, 3, 4, np.nan]. Modes [1,3,4]. fillna(1)
    # CatCol: ['a', 'b', np.nan, 'd', 'a']. Mode 'a'. fillna('a')
    df_out = replace_with_mode(df_in.copy())

    assert df_out.isnull().sum().sum() == 0
    assert df_out.shape == df_in.shape

    expected_NumCol = df_in['NumCol'].fillna(df_in['NumCol'].mode().iloc[0])
    pd.testing.assert_series_equal(df_out['NumCol'], expected_NumCol, check_dtype=False)

    expected_CatCol = df_in['CatCol'].fillna(df_in['CatCol'].mode().iloc[0])
    pd.testing.assert_series_equal(df_out['CatCol'], expected_CatCol)
    
    assert_frame_equal(df_out[['FullNumCol']], df_in[['FullNumCol']])
    assert_frame_equal(df_out[['FullCatCol']], df_in[['FullCatCol']])


def test_replace_with_mode_all_null_col(sample_dfs):
    df_in = sample_dfs["all_null_col"].copy() # Column 'B' is all np.nan
    df_out = replace_with_mode(df_in.copy())
    assert df_out.shape == df_in.shape
    
    # For column 'B' (all NaN):
    # If dtype is numeric (e.g. float because of NaN), mode is empty. Fills with 0.
    # If dtype is object, mode is empty. Fills with "Unknown".
    # Let's check the implementation's behavior.
    # The implementation uses `df[col].mode()`. If empty, it checks dtype.
    # `df_in['B']` is float64. `df_in['B'].mode()` is an empty Series.
    # So it should fill with 0.
    assert (df_out['B'] == 0).all()
    assert df_out['B'].isnull().sum() == 0
    
    # Check other columns are untouched
    assert_frame_equal(df_out[['A']], df_in[['A']])
    assert_frame_equal(df_out[['C']], df_in[['C']])

def test_replace_with_mode_all_null_col_object(sample_dfs):
    df_data = {'A': [1,2,3], 'B': [np.nan, np.nan, np.nan], 'C': ['x','y','z']}
    df_in = pd.DataFrame(df_data)
    df_in['B'] = df_in['B'].astype(object) # Force B to be object type
    
    df_out = replace_with_mode(df_in.copy())
    assert df_out.shape == df_in.shape
    # Column 'B' is all NaN but object type. Mode is empty. Should fill with "Unknown".
    assert (df_out['B'] == "Unknown").all()
    assert df_out['B'].isnull().sum() == 0


def test_replace_with_mode_all_null_df(sample_dfs):
    df_in = sample_dfs["all_null_df"].copy() # All columns are all np.nan (float)
    df_out = replace_with_mode(df_in.copy())
    assert df_out.shape == df_in.shape
    # Each column's mode will be empty, and dtype is float, so filled with 0.
    for col in df_out.columns:
        assert (df_out[col] == 0).all()
    assert df_out.isnull().sum().sum() == 0

def test_replace_with_mode_empty_df(sample_dfs):
    df_in = sample_dfs["empty"].copy()
    df_out = replace_with_mode(df_in.copy())
    assert df_out.empty
    assert df_out.shape == df_in.shape
    assert_frame_equal(df_in, df_out)

def test_replace_with_mode_no_nulls(sample_dfs):
    df_in = sample_dfs["no_nulls"].copy()
    original_shape = df_in.shape
    df_out = replace_with_mode(df_in.copy())
    assert df_out.isnull().sum().sum() == 0
    assert df_out.shape == original_shape
    assert_frame_equal(df_in, df_out) # Data should be unchanged

def test_replace_with_mode_multiple_modes(sample_dfs):
    # Pandas mode() returns all modes if they have same frequency, sorted.
    # The implementation uses .iloc[0] so it picks the first one.
    data = {'col1': [1, 1, 2, 2, np.nan, np.nan, 3]} # Modes are 1 and 2. .iloc[0] will pick 1.
    df_in = pd.DataFrame(data)
    df_out = replace_with_mode(df_in.copy())
    
    assert df_out['col1'].isnull().sum() == 0
    # Expected: NaNs are filled with 1 (the first mode)
    # Original: 1, 1, 2, 2, np.nan, np.nan, 3
    # Expected: 1, 1, 2, 2, 1, 1, 3
    expected_series = pd.Series([1, 1, 2, 2, 1, 1, 3], name='col1')
    pd.testing.assert_series_equal(df_out['col1'], expected_series, check_dtype=False)

# Add a test for a column with no non-NaN values, but not all NaN (e.g. mixed types that became object)
# This is covered by all_null_col tests where type is object or numeric.

# --------------*TESTS FOR forward_fill()*------------------

def test_forward_fill_basic(sample_dfs):
    df_in = sample_dfs["mixed_nulls"].copy()
    # NumCol: [1, np.nan, 3, 4, np.nan] -> [1, 1, 3, 4, 4]
    # CatCol: ['a', 'b', np.nan, 'd', 'a'] -> ['a', 'b', 'b', 'd', 'a']
    df_out = forward_fill(df_in)
    
    assert df_out.shape == df_in.shape
    expected_df = df_in.ffill()
    assert_frame_equal(df_out, expected_df)
    # Check that some nulls are filled, but not necessarily all (e.g. leading nulls)
    assert df_out['NumCol'].isnull().sum() < df_in['NumCol'].isnull().sum() or df_in['NumCol'].isnull().sum() == 0
    assert df_out['CatCol'].isnull().sum() < df_in['CatCol'].isnull().sum() or df_in['CatCol'].isnull().sum() == 0


def test_forward_fill_leading_nulls(sample_dfs):
    data = {'A': [np.nan, np.nan, 1, 2, np.nan, 3], 'B': [np.nan, 'x', 'y', np.nan, np.nan, 'z']}
    df_in = pd.DataFrame(data)
    # Expected A: [np.nan, np.nan, 1, 2, 2, 3]
    # Expected B: [np.nan, 'x', 'y', 'y', 'y', 'z']
    df_out = forward_fill(df_in)
    
    assert df_out.shape == df_in.shape
    expected_df = df_in.ffill()
    assert_frame_equal(df_out, expected_df)
    
    assert df_out['A'].isnull().sum() == 2 # Leading NaNs remain
    assert df_out['B'].isnull().sum() == 1 # Leading NaN remains

def test_forward_fill_all_null_col(sample_dfs):
    df_in = sample_dfs["all_null_col"].copy() # Column 'B' is all np.nan
    df_out = forward_fill(df_in)
    
    assert df_out.shape == df_in.shape
    expected_df = df_in.ffill() # ffill on all-NaN column doesn't change it
    assert_frame_equal(df_out, expected_df)
    assert df_out['B'].isnull().all() # Column B should remain all null

def test_forward_fill_all_null_df(sample_dfs):
    df_in = sample_dfs["all_null_df"].copy()
    df_out = forward_fill(df_in)
    assert df_out.shape == df_in.shape
    expected_df = df_in.ffill()
    assert_frame_equal(df_out, expected_df) # Should remain all NaNs
    assert df_out.isnull().all().all()

def test_forward_fill_empty_df(sample_dfs):
    df_in = sample_dfs["empty"].copy()
    df_out = forward_fill(df_in)
    assert df_out.empty
    assert_frame_equal(df_in, df_out)

def test_forward_fill_no_nulls(sample_dfs):
    df_in = sample_dfs["no_nulls"].copy()
    original_shape = df_in.shape
    df_out = forward_fill(df_in)
    assert df_out.isnull().sum().sum() == 0
    assert df_out.shape == original_shape
    assert_frame_equal(df_in, df_out) # Data should be unchanged

# --------------*TESTS FOR backward_fill()*------------------

def test_backward_fill_basic(sample_dfs):
    df_in = sample_dfs["mixed_nulls"].copy()
    # NumCol: [1, np.nan, 3, 4, np.nan] -> [1, 3, 3, 4, np.nan] (final nan remains if no subsequent value)
    # CatCol: ['a', 'b', np.nan, 'd', 'a'] -> ['a', 'b', 'd', 'd', 'a']
    df_out = backward_fill(df_in)
    
    assert df_out.shape == df_in.shape
    expected_df = df_in.bfill()
    assert_frame_equal(df_out, expected_df)
    # Check that some nulls are filled, but not necessarily all (e.g. trailing nulls)
    assert df_out['NumCol'].isnull().sum() < df_in['NumCol'].isnull().sum() or df_in['NumCol'].isnull().sum() == 0 or df_in['NumCol'].iloc[-1] is np.nan
    assert df_out['CatCol'].isnull().sum() < df_in['CatCol'].isnull().sum() or df_in['CatCol'].isnull().sum() == 0


def test_backward_fill_trailing_nulls(sample_dfs):
    data = {'A': [np.nan, 1, np.nan, 2, 3, np.nan, np.nan], 'B': ['x', np.nan, 'y', 'z', np.nan, np.nan, np.nan]}
    df_in = pd.DataFrame(data)
    # Expected A: [1, 1, 2, 2, 3, np.nan, np.nan]
    # Expected B: ['x', 'y', 'y', 'z', np.nan, np.nan, np.nan]
    df_out = backward_fill(df_in)
    
    assert df_out.shape == df_in.shape
    expected_df = df_in.bfill()
    assert_frame_equal(df_out, expected_df)
    
    assert df_out['A'].isnull().sum() == 2 # Trailing NaNs remain
    assert df_out['B'].isnull().sum() == 3 # Trailing NaNs remain

def test_backward_fill_all_null_col(sample_dfs):
    df_in = sample_dfs["all_null_col"].copy() # Column 'B' is all np.nan
    df_out = backward_fill(df_in)
    
    assert df_out.shape == df_in.shape
    expected_df = df_in.bfill() # bfill on all-NaN column doesn't change it
    assert_frame_equal(df_out, expected_df)
    assert df_out['B'].isnull().all() # Column B should remain all null

def test_backward_fill_all_null_df(sample_dfs):
    df_in = sample_dfs["all_null_df"].copy()
    df_out = backward_fill(df_in)
    assert df_out.shape == df_in.shape
    expected_df = df_in.bfill()
    assert_frame_equal(df_out, expected_df) # Should remain all NaNs
    assert df_out.isnull().all().all()

def test_backward_fill_empty_df(sample_dfs):
    df_in = sample_dfs["empty"].copy()
    df_out = backward_fill(df_in)
    assert df_out.empty
    assert_frame_equal(df_in, df_out)

def test_backward_fill_no_nulls(sample_dfs):
    df_in = sample_dfs["no_nulls"].copy()
    original_shape = df_in.shape
    df_out = backward_fill(df_in)
    assert df_out.isnull().sum().sum() == 0
    assert df_out.shape == original_shape
    assert_frame_equal(df_in, df_out) # Data should be unchanged

# --------------*TESTS FOR evaluate_methods()*------------------

@pytest.fixture
def evaluation_data():
    original_df = pd.DataFrame({
        'A': [1, np.nan, 3, np.nan, 5],
        'B': [np.nan, 'x', 'y', 'z', np.nan],
        'C': [10, 20, 30, 40, 50] # No nulls
    }) # Total 4 nulls, shape (5,3)

    # Method 1: Drops rows with nulls
    cleaned_drop = original_df.dropna() # Shape (1,3), 0 nulls, 4 nulls removed

    # Method 2: Fills nulls with a fixed value
    cleaned_fixed = original_df.fillna(0) # Shape (5,3), 0 nulls, 4 nulls removed
    
    # Method 3: Fills only some nulls, keeps shape
    cleaned_partial = original_df.copy()
    cleaned_partial['A'] = cleaned_partial['A'].fillna(original_df['A'].mean()) # 2 nulls removed in A
    # B still has 2 nulls. Total 2 nulls remaining. Shape (5,3)

    # Method 4: Drops a column that had nulls
    cleaned_drop_col = original_df.drop(columns=['A']) # Shape (5,2), 2 nulls remaining in B, 2 nulls removed from A
                                                    # Original nulls considered are from original_df (4)
                                                    # Nulls removed = 4 (original) - 2 (remaining in B within this new df) = 2
    
    # Method 5: All null df
    cleaned_all_null = pd.DataFrame({'A': [np.nan, np.nan], 'B': [np.nan, np.nan]}) # Shape (2,2), 4 nulls

    return {
        "original": original_df,
        "cleaned_versions": {
            "drop": cleaned_drop, # Score: (4/4)*0.5 + (1/5)*0.25 + (3/3)*0.25 = 0.5 + 0.05 + 0.25 = 0.8
            "fixed": cleaned_fixed, # Score: (4/4)*0.5 + (5/5)*0.25 + (3/3)*0.25 = 0.5 + 0.25 + 0.25 = 1.0 (BEST)
            "partial": cleaned_partial, # Score: (2/4)*0.5 + (5/5)*0.25 + (3/3)*0.25 = 0.25 + 0.25 + 0.25 = 0.75
            "drop_col": cleaned_drop_col, # Score: (2/4)*0.5 + (5/5)*0.25 + (2/3)*0.25 = 0.25 + 0.25 + 0.1666 = 0.6666
        },
         "cleaned_all_null_versions": { # original_df has 4 nulls
            "all_null_output": cleaned_all_null # Score: (0/4)*0.5 + (2/5)*0.25 + (2/3)*0.25 = 0 + 0.1 + 0.1666 = 0.2666
        }
    }

def test_evaluate_methods_selects_best(evaluation_data):
    log_lines = []
    best_method = evaluate_methods(
        evaluation_data["original"],
        evaluation_data["cleaned_versions"],
        log_lines
    )
    assert best_method == "fixed" # Based on manual score calculation
    assert len(log_lines) > 0 # Check logs are being populated
    assert "Best strategy selected: fixed" in log_lines[-1]


def test_evaluate_methods_empty_original_df():
    original_empty = pd.DataFrame()
    cleaned_versions = {
        "empty_too": pd.DataFrame(),
        "some_data": pd.DataFrame({'A': [1,2]}) # This case is a bit odd, but test it
    }
    log_lines = []
    # original_nulls = 0, original_shape = (0,0) -> division by zero for ratios if not handled
    # Score: (1.0)*0.5 + (0/0? -> 0)*0.25 + (0/0? -> 0)*0.25 = 0.5 if ratios are 0
    # Current code: row_ratio = df.shape[0] / original_shape[0] if original_shape[0] else 0 -> good
    
    best_method = evaluate_methods(original_empty, cleaned_versions, log_lines)
    
    # If original is empty, nulls_removed/original_nulls is 1.0 (1.0 * 0.5)
    # "empty_too": df.shape=(0,0) -> row_ratio=0, col_ratio=0. Score = 0.5
    # "some_data": df.shape=(2,1) -> row_ratio=0, col_ratio=0. Score = 0.5
    # If scores are tied, the first one encountered with that score might be chosen, or last one.
    # The loop updates `best_method` if `score > best_score`. So last one with max score.
    # In this case, "some_data" should be selected if it's processed last and has same score.
    # Let's ensure cleaned_versions is a dict, order can be an issue for Py <3.7
    # For the test, let's make one score clearly higher if original is empty
    
    cleaned_versions_for_empty = {
        "c1": pd.DataFrame(), # Score 0.5
        "c2": pd.DataFrame({'X':[]}) # Score 0.5 (shape (0,1), original (0,0)) -> col_ratio = 0
    }
    # If original is empty, original_nulls = 0. (nulls_removed / original_nulls if original_nulls > 0 else 1.0) -> 1.0
    # So, (1.0 * 0.5) is the first term.
    # c1: shape (0,0). original_shape (0,0). row_ratio = 0, col_ratio = 0. Score = 0.5
    # c2: shape (0,1). original_shape (0,0). row_ratio = 0, col_ratio = 0. Score = 0.5
    # It will pick the last one if scores are equal.
    
    # Let's test with a scenario where ratios make a difference
    original_df_small = pd.DataFrame({'A': [1]}) # 0 nulls, shape (1,1)
    cleaned_v = {
        "s1": pd.DataFrame({'A': [1]}), # 0 nulls rem. Score (1)*0.5 + (1/1)*0.25 + (1/1)*0.25 = 1.0
        "s2": pd.DataFrame({'A': [1,2]}) # 0 nulls rem. Score (1)*0.5 + (2/1)*0.25 + (1/1)*0.25 = 0.5 + 0.5 + 0.25 = 1.25
    }
    log_lines_2 = []
    best_method_2 = evaluate_methods(original_df_small, cleaned_v, log_lines_2)
    assert best_method_2 == "s2"


def test_evaluate_methods_all_null_cleaned(evaluation_data):
    log_lines = []
    # original_df has 4 nulls. cleaned_all_null_versions["all_null_output"] has 4 nulls.
    # nulls_removed = 4 - 4 = 0.
    # Score: (0/4)*0.5 + (2/5)*0.25 + (2/3)*0.25 = 0 + 0.1 + 0.1666... = 0.2666...
    best_method = evaluate_methods(
        evaluation_data["original"],
        evaluation_data["cleaned_all_null_versions"],
        log_lines
    )
    assert best_method == "all_null_output" # Only one method
    assert "all_null_output" in log_lines[-1]
    score_line = [line for line in log_lines if "Strategy score:" in line][0]
    assert "0.2667" in score_line # Check formatting and calculation


def test_evaluate_methods_no_nulls_original(sample_dfs):
    original_no_nulls = sample_dfs["no_nulls"].copy() # 0 nulls, shape (5,3)
    cleaned_versions = {
        "c1_same": original_no_nulls.copy(), # Score (1)*0.5 + (1)*0.25 + (1)*0.25 = 1.0
        "c2_rows_dropped": original_no_nulls.drop([0,1]) # Score (1)*0.5 + (3/5)*0.25 + (1)*0.25 = 0.5 + 0.15 + 0.25 = 0.9
    }
    log_lines = []
    best_method = evaluate_methods(original_no_nulls, cleaned_versions, log_lines)
    assert best_method == "c1_same"


def test_evaluate_methods_logging(evaluation_data):
    log_lines = []
    evaluate_methods(
        evaluation_data["original"],
        evaluation_data["cleaned_versions"],
        log_lines
    )
    # evaluate_methods adds 5 lines for each method, plus 1 summary line for the best method.
    assert len(log_lines) == (len(evaluation_data["cleaned_versions"]) * 5) + 1 
    assert "Method tried: drop" in "".join(log_lines)
    assert "Method tried: fixed" in "".join(log_lines)
    assert "Method tried: partial" in "".join(log_lines)
    assert "Method tried: drop_col" in "".join(log_lines)
    assert "Nulls removed:" in "".join(log_lines)
    assert "Remaining nulls:" in "".join(log_lines)
    assert "Shape after cleaning:" in "".join(log_lines)
    assert "Strategy score:" in "".join(log_lines)
    assert "✅ Best strategy selected:" in log_lines[-1]

# --------------*TESTS FOR process_csv() (Integration)*------------------

@pytest.fixture
def temp_csv_files():
    """Creates temporary input and output directories and file paths for process_csv."""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.csv")
        output_path = os.path.join(temp_dir, "output", "cleaned_output.csv") # Mirorring process_csv structure
        log_path = os.path.join(temp_dir, "output", "null_handling_log.txt")
        yield input_path, output_path, log_path # Log path is derived, but good to have for direct check

def test_process_csv_basic_scenario(temp_csv_files, sample_dfs):
    input_path, output_path, log_path = temp_csv_files
    df_input = sample_dfs["mixed_nulls"].copy() # Has numeric and categorical nulls
    df_input.to_csv(input_path, index=False)

    process_csv(input_path, output_path)

    assert os.path.exists(output_path)
    df_output = pd.read_csv(output_path)
    
    # The best method should have been chosen. We expect no nulls if "fixed" or similar was chosen.
    # Or fewer nulls if ffill/bfill was chosen and some leading/trailing nulls remained.
    # The key is that some processing happened and a valid CSV was output.
    # The exact best method can vary, so we check general properties.
    assert df_output.shape[0] <= df_input.shape[0] # Rows might be dropped
    assert df_output.shape[1] == df_input.shape[1] # Columns should ideally not be dropped by these methods unless all null
                                                 # (evaluate_methods prefers solutions that keep cols)
    
    # Check that the output CSV has fewer or equal nulls than input.
    # If the best method still leaves some nulls (e.g. ffill on data with leading nulls),
    # this check is still generally valid.
    # A perfect method would leave 0 nulls.
    assert df_output.isnull().sum().sum() < df_input.isnull().sum().sum() or df_input.isnull().sum().sum() == 0

    assert os.path.exists(log_path)
    with open(log_path, "r") as f:
        log_content = f.read()
    assert "Processing file:" in log_content
    assert "Best strategy selected:" in log_content
    # The "Cleaned CSV saved at:" message is a logging.info call, not part of the log_lines list saved to the file.
    # We can optionally check caplog.text if we want to assert that specific logging.info happened.
    # For this test, ensuring the primary log content (evaluation) is present is sufficient.

def test_process_csv_empty_input(temp_csv_files):
    input_path, output_path, log_path = temp_csv_files
    pd.DataFrame().to_csv(input_path, index=False)

    process_csv(input_path, output_path)

    # process_csv should log a warning and return, not create output or log file.
    assert not os.path.exists(output_path)
    assert not os.path.exists(log_path) 
    # To verify logging, we'd need to capture stderr/stdout or check a log file if process_csv wrote to one directly.
    # The current process_csv uses logging module, which might go to console.
    # For this test, checking for absence of output files is the main goal.

def test_process_csv_no_nulls_input(temp_csv_files, sample_dfs):
    input_path, output_path, log_path = temp_csv_files
    df_input = sample_dfs["no_nulls"].copy()
    df_input.to_csv(input_path, index=False)

    process_csv(input_path, output_path)

    assert os.path.exists(output_path)
    df_output = pd.read_csv(output_path)
    
    assert_frame_equal(df_output, df_input) # Should preserve data
    assert df_output.isnull().sum().sum() == 0

    assert os.path.exists(log_path)
    with open(log_path, "r") as f:
        log_content = f.read()
    assert "Total null values: 0" in log_content
    # The best strategy should still be identified, likely one that doesn't change the df.
    assert "Best strategy selected:" in log_content


def test_process_csv_all_null_column(temp_csv_files, sample_dfs):
    input_path, output_path, log_path = temp_csv_files
    df_input = sample_dfs["all_null_col"].copy() # Has column 'B' all null
    df_input.to_csv(input_path, index=False)

    process_csv(input_path, output_path)

    assert os.path.exists(output_path)
    df_output = pd.read_csv(output_path)

    # Depending on the chosen strategy, the all-null column might be filled or rows dropped.
    # We expect the number of nulls to decrease significantly or become zero.
    assert df_output.isnull().sum().sum() < df_input.isnull().sum().sum()
    
    # If 'B' was filled (e.g. by replace_with_mode or replace_with_fixed), it should have no nulls.
    # If drop_nulls was chosen, 'B' would still exist but df_output would be empty.
    # This makes asserting on df_output['B'] tricky without knowing the chosen method.
    # The general assertion above (nulls decreased) is safer.

    assert os.path.exists(log_path)
    with open(log_path, "r") as f:
        log_content = f.read()
    assert "Best strategy selected:" in log_content

def test_process_csv_input_not_found(temp_csv_files, caplog):
    input_path, output_path, log_path = temp_csv_files
    # Do not create input_path

    process_csv(input_path, output_path)

    assert not os.path.exists(output_path)
    assert not os.path.exists(log_path)
    assert f"Input file not found: {input_path}" in caplog.text # Check log message for error

def test_process_csv_output_path_creation(temp_csv_files):
    input_path, output_path, _ = temp_csv_files
    # Output path is like /tmp/random_dir/output/cleaned_output.csv
    # Ensure the 'output' subdir is created by process_csv
    
    # Create a simple CSV for input
    pd.DataFrame({'A': [1, np.nan]}).to_csv(input_path, index=False)
    
    # Modify output_path to be in a sub-sub-directory to test recursive creation
    deep_output_dir = os.path.join(os.path.dirname(output_path), "deeper", "sub")
    deep_output_file = os.path.join(deep_output_dir, "final.csv")
    
    # Ensure the specific path for the deep output file does not exist initially
    assert not os.path.exists(deep_output_dir)

    process_csv(input_path, deep_output_file)

    assert os.path.exists(deep_output_file)
    df_output = pd.read_csv(deep_output_file)
    assert not df_output['A'].isnull().any() # Should be filled

    deep_log_file = os.path.join(deep_output_dir, "null_handling_log.txt")
    assert os.path.exists(deep_log_file)


# --------------*TESTS FOR replace_with_mean()*------------------

def test_replace_with_mean_numeric(sample_dfs):
    df_in = sample_dfs["numeric_nulls"].copy()
    df_out = replace_with_mean(df_in)
    assert df_out.isnull().sum().sum() == 0 # All nulls in numeric cols should be filled
    assert df_out.shape == df_in.shape
    for col in df_in.select_dtypes(include=np.number).columns:
        mean_val = df_in[col].mean()
        # Create a Series of expected values for the column
        expected_col = df_in[col].fillna(mean_val)
        assert_frame_equal(df_out[[col]], expected_col.to_frame(), check_dtype=False) # Check dtype loosely due to potential float conversion
    # Non-numeric columns should remain unchanged if any
    for col in df_in.select_dtypes(exclude=np.number).columns:
         assert_frame_equal(df_out[[col]], df_in[[col]])


def test_replace_with_mean_mixed(sample_dfs):
    df_in = sample_dfs["mixed_nulls"].copy()
    df_out = replace_with_mean(df_in)
    
    # Nulls in numeric columns should be filled
    assert df_out['NumCol'].isnull().sum() == 0
    # Nulls in categorical columns should remain
    assert df_out['CatCol'].isnull().sum() == df_in['CatCol'].isnull().sum()
    assert df_out.shape == df_in.shape

    # Verify numeric column 'NumCol'
    mean_numcol = df_in['NumCol'].mean()
    expected_numcol = df_in['NumCol'].fillna(mean_numcol)
    assert_frame_equal(df_out[['NumCol']], expected_numcol.to_frame(), check_dtype=False)

    # Verify categorical column 'CatCol' remained as is (with its nulls)
    assert_frame_equal(df_out[['CatCol']], df_in[['CatCol']])
    
    # Verify full columns remained as is
    assert_frame_equal(df_out[['FullNumCol']], df_in[['FullNumCol']])
    assert_frame_equal(df_out[['FullCatCol']], df_in[['FullCatCol']])

def test_replace_with_mean_all_null_numeric_col(sample_dfs):
    # Column 'B' in 'all_null_col' is all np.nan (numeric by virtue of np.nan)
    df_in = sample_dfs["all_null_col"].copy()
    # Ensure 'B' is treated as numeric if possible, or handle if it becomes object
    if df_in['B'].dtype == 'object': # if it was inferred as object because all NaNs
         df_in['B'] = pd.to_numeric(df_in['B'], errors='coerce')

    df_out = replace_with_mean(df_in)
    assert df_out.shape == df_in.shape
    # The mean of an all-NaN column is NaN. Pandas fillna(NaN) doesn't change it.
    # The current implementation of replace_with_mean fills with Series.mean().
    # If Series.mean() is NaN, then fillna(NaN) doesn't change the NaNs.
    assert df_out['B'].isnull().all() # 'B' should remain all null if its mean is NaN
    
    # Other numeric columns (if any) should be filled if they had nulls
    # Column 'A' is numeric and full, should be unchanged
    assert_frame_equal(df_out[['A']], df_in[['A']])
    # Column 'C' is categorical and full, should be unchanged
    assert_frame_equal(df_out[['C']], df_in[['C']])

def test_replace_with_mean_all_null_df(sample_dfs):
    df_in = sample_dfs["all_null_df"].copy()
    df_out = replace_with_mean(df_in)
    assert df_out.shape == df_in.shape
    # All columns are numeric (due to np.nan) but all null. Means are NaN.
    # So, filling with NaN doesn't change anything.
    assert_frame_equal(df_out, df_in) # Should remain all NaNs
    assert df_out.isnull().all().all()

def test_replace_with_mean_categorical_only(sample_dfs):
    df_in = sample_dfs["categorical_nulls"].copy()
    df_out = replace_with_mean(df_in)
    assert df_out.shape == df_in.shape
    # No numeric columns to fill, so DataFrame should be identical
    assert_frame_equal(df_out, df_in)

def test_replace_with_mean_empty_df(sample_dfs):
    df_in = sample_dfs["empty"].copy()
    df_out = replace_with_mean(df_in)
    assert df_out.empty
    assert df_out.shape == df_in.shape
    assert_frame_equal(df_in, df_out)

def test_replace_with_mean_no_nulls(sample_dfs):
    df_in = sample_dfs["no_nulls"].copy()
    original_shape = df_in.shape
    df_out = replace_with_mean(df_in)
    assert df_out.isnull().sum().sum() == 0
    assert df_out.shape == original_shape
    assert_frame_equal(df_in, df_out) # Data should be unchanged
