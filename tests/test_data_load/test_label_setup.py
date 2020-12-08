from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler, LabelEncoder

from data_load.label_setup import merge_target_columns
from snp_pred.data_load import label_setup
from snp_pred.data_load.common_ops import ColumnOperation


@pytest.fixture()
def create_test_column_ops():
    def test_column_op_1(df, column_name, replace_dict):
        df = df.replace({column_name: replace_dict})
        return df

    def test_column_op_2(df, column_name, multiplier):
        df[column_name] = df[column_name] * multiplier
        return df

    def test_column_op_3(df, column_name, replace_with_col):
        df[column_name] = df[replace_with_col]
        return df

    replace_dict_args = {"replace_dict": {"Europe": "Iceland"}}
    multiplier_dict_arg = {"multiplier": 2}
    replace_column_dict_arg = {"replace_with_col": "ExtraCol3"}

    test_column_ops = {
        "Origin": [
            ColumnOperation(function=test_column_op_1, function_args=replace_dict_args),
            ColumnOperation(
                function=test_column_op_2, function_args=multiplier_dict_arg
            ),
        ],
        "OriginExtraColumnsAll": [
            ColumnOperation(
                function=test_column_op_1,
                function_args=replace_dict_args,
                extra_columns_deps=("ExtraCol1", "ExtraCol2"),
            ),
            ColumnOperation(
                function=test_column_op_3,
                function_args=replace_column_dict_arg,
                extra_columns_deps=("ExtraCol3",),
            ),
        ],
        "OriginExtraColumnsPartial1": [
            ColumnOperation(
                function=test_column_op_1,
                function_args=replace_dict_args,
                extra_columns_deps=("ExtraCol1", "ExtraCol2"),
            )
        ],
        "OriginExtraColumnsPartial2": [
            ColumnOperation(
                function=test_column_op_3,
                function_args=replace_column_dict_arg,
                extra_columns_deps=("ExtraCol3",),
            )
        ],
        "ExtraTarget": [
            ColumnOperation(
                function=test_column_op_1,
                function_args=replace_dict_args,
                extra_columns_deps=(),
                only_apply_if_target=True,
            )
        ],
    }

    return test_column_ops


@pytest.mark.parametrize(
    "create_test_cl_args",
    [
        {
            "custom_cl_args": {
                "label_parsing_chunk_size": None,
            }
        },
        {
            "custom_cl_args": {
                "label_parsing_chunk_size": 50,
            }
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_set_up_train_and_valid_labels(
    parse_test_cl_args, create_test_data, create_test_cl_args
):
    c = create_test_data
    cl_args = create_test_cl_args
    n_classes = len(c.target_classes)

    (
        train_labels_dict,
        valid_labels_dict,
    ) = label_setup.set_up_train_and_valid_tabular_data(
        cl_args=cl_args, custom_label_ops=None
    )

    assert len(train_labels_dict) + len(valid_labels_dict) == n_classes * c.n_per_class
    assert len(train_labels_dict) > len(valid_labels_dict)

    train_ids_set = set(train_labels_dict.keys())
    valid_ids_set = set(valid_labels_dict.keys())

    assert len(train_ids_set) == len(train_labels_dict)
    assert len(valid_ids_set) == len(valid_labels_dict)

    assert valid_ids_set.isdisjoint(train_ids_set)


@pytest.mark.parametrize("create_test_data", [{"task_type": "binary"}], indirect=True)
def test_label_df_parse_wrapper(
    parse_test_cl_args, create_test_data, create_test_cl_args
):
    c = create_test_data
    cl_args = create_test_cl_args
    test_target_column = cl_args.target_cat_columns[0]  # Origin

    df_labels = label_setup.label_df_parse_wrapper(cl_args)

    # since we're only testing binary case here
    n_total = c.n_per_class * 2

    assert df_labels.shape == (n_total, 1)
    assert set(df_labels[test_target_column].unique()) == {"Asia", "Europe"}


@pytest.mark.parametrize("create_test_data", [{"task_type": "binary"}], indirect=True)
def test_gather_ids_from_data_source(create_test_data):
    c = create_test_data

    test_path = c.scoped_tmp_path / "test_arrays"
    expected_num_samples = c.n_per_class * len(c.target_classes)

    test_ids = label_setup.gather_ids_from_data_source(data_source=test_path)

    assert len(test_ids) == expected_num_samples

    # check that ids are properly formatted, not paths
    assert not any(".npy" in i for i in test_ids)


@pytest.mark.parametrize("create_test_data", [{"task_type": "binary"}], indirect=True)
def test_get_array_path_iterator_file(create_test_data):
    c = create_test_data

    test_path = c.scoped_tmp_path / "test_arrays"
    test_label_file_path = c.scoped_tmp_path / "test_paths.txt"

    with open(test_label_file_path, "w") as test_label_file:
        for path in test_path.iterdir():
            test_label_file.write(str(path) + "\n")

    expected_num_samples = c.n_per_class * len(c.target_classes)
    text_file_iterator = label_setup.get_array_path_iterator(
        data_source=test_label_file_path
    )

    assert len([i for i in text_file_iterator]) == expected_num_samples


@pytest.mark.parametrize("create_test_data", [{"task_type": "binary"}], indirect=True)
def test_get_array_path_iterator_folder(create_test_data):
    c = create_test_data

    test_path = c.scoped_tmp_path / "test_arrays"

    expected_num_samples = c.n_per_class * len(c.target_classes)
    folder_iterator = label_setup.get_array_path_iterator(data_source=test_path)

    assert len([i for i in folder_iterator]) == expected_num_samples


@pytest.mark.parametrize("create_test_data", [{"task_type": "binary"}], indirect=True)
def test_get_array_path_iterator_fail(create_test_data):
    c = create_test_data

    with pytest.raises(FileNotFoundError):
        label_setup.get_array_path_iterator(data_source=Path("does/not/exist"))

    test_label_file_path = c.scoped_tmp_path / "test_paths_fail.txt"

    with open(test_label_file_path, "w") as test_label_file:
        for i in range(5):
            test_label_file.write("non/existent/path.npy" + "\n")

    with pytest.raises(FileNotFoundError):
        iterator = label_setup.get_array_path_iterator(data_source=test_label_file_path)
        _ = [i for i in iterator]


@pytest.mark.parametrize(
    "test_input_args,expected",
    [
        ({"target_cat_columns": ["Origin"]}, ["Origin"]),
        (
            {"target_cat_columns": ["Origin", "OriginExtraColumnsAll"]},
            ["Origin", "OriginExtraColumnsAll", "ExtraCol1", "ExtraCol2", "ExtraCol3"],
        ),
        (
            {
                "target_cat_columns": ["Origin"],
                "extra_con_columns": ["OriginExtraColumnsPartial1"],
            },
            ["Origin", "OriginExtraColumnsPartial1", "ExtraCol1", "ExtraCol2"],
        ),
        (
            {
                "target_con_columns": ["Origin"],
                "target_cat_columns": ["OriginExtraColumnsAll"],
                "extra_con_columns": ["OriginExtraColumnsPartial1"],
                "extra_cat_columns": ["OriginExtraColumnsPartial2"],
            },
            [
                "Origin",
                "OriginExtraColumnsAll",
                "OriginExtraColumnsPartial1",
                "OriginExtraColumnsPartial2",
                "ExtraCol1",
                "ExtraCol2",
                "ExtraCol3",
            ],
        ),
    ],
)
def test_get_all_label_columns_needed(
    test_input_args, expected, args_config, create_test_column_ops
):

    for key, columns in test_input_args.items():
        setattr(args_config, key, columns)

    all_cols = label_setup._get_label_columns_and_dtypes(
        cl_args=args_config, column_ops=create_test_column_ops
    )
    assert set(all_cols) == set(expected)


@pytest.mark.parametrize(
    "test_input_args,expected",
    [
        ({"target_cat_columns": ["Origin"]}, ["Origin"]),
        (
            {"target_cat_columns": ["Origin", "OriginExtraColumnsAll"]},
            ["Origin", "OriginExtraColumnsAll"],
        ),
        (
            {
                "target_cat_columns": ["Origin"],
                "extra_con_columns": ["OriginExtraColumnsPartial1"],
            },
            ["Origin", "OriginExtraColumnsPartial1"],
        ),
        (
            {
                "target_con_columns": ["Origin"],
                "target_cat_columns": ["OriginExtraColumnsAll"],
                "extra_con_columns": ["OriginExtraColumnsPartial1"],
                "extra_cat_columns": ["OriginExtraColumnsPartial2"],
            },
            [
                "Origin",
                "OriginExtraColumnsAll",
                "OriginExtraColumnsPartial1",
                "OriginExtraColumnsPartial2",
            ],
        ),
    ],
)
def test_get_label_columns_from_cl_args(
    test_input_args, expected, args_config, create_test_column_ops
):
    for key, columns in test_input_args.items():
        setattr(args_config, key, columns)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (["Origin"], []),
        (["OriginExtraColumnsAll"], ["ExtraCol1", "ExtraCol2", "ExtraCol3"]),
        (["OriginExtraColumnsPartial1", "Origin"], ["ExtraCol1", "ExtraCol2"]),
        (
            ["OriginExtraColumnsPartial1", "OriginExtraColumnsPartial2"],
            ["ExtraCol1", "ExtraCol2", "ExtraCol3"],
        ),
    ],
)
def test_get_extra_columns(test_input, expected, create_test_column_ops):
    test_column_ops = create_test_column_ops

    test_output = label_setup._get_extra_columns(test_input, test_column_ops)
    assert test_output == expected


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_load_label_df_one_target_no_extra_col(parse_test_cl_args, create_test_data):
    c = create_test_data

    label_fpath = c.scoped_tmp_path / "labels.csv"
    n_classes = len(c.target_classes)

    label_columns = ["Origin"]
    df_label = label_setup._load_label_df(
        label_fpath=label_fpath, columns=label_columns, custom_label_ops=None
    )

    assert df_label.shape[0] == c.n_per_class * n_classes
    assert df_label.index.name == "ID"
    assert [i for i in df_label.Origin.value_counts()] == [c.n_per_class] * n_classes


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_load_label_df_one_target_one_extra_col(parse_test_cl_args, create_test_data):
    c = create_test_data

    label_fpath = c.scoped_tmp_path / "labels.csv"

    label_columns = ["Origin", "OriginExtraCol"]

    df_label_extra = label_setup._load_label_df(
        label_fpath=label_fpath, columns=label_columns, custom_label_ops=None
    )

    assert df_label_extra.shape[1] == 2

    # OriginExtraCol is same as Origin by definition
    assert (df_label_extra["OriginExtraCol"] == df_label_extra["Origin"]).all()


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_load_label_df_missing_col_fail(parse_test_cl_args, create_test_data):
    c = create_test_data

    label_fpath = c.scoped_tmp_path / "labels.csv"

    label_columns = ["Origin", "NonExistentColumn"]

    with pytest.raises(ValueError):
        label_setup._load_label_df(
            label_fpath=label_fpath, columns=label_columns, custom_label_ops=None
        )


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_load_label_df_missing_col_pass(
    parse_test_cl_args, create_test_data, test_column_operations
):
    c = create_test_data

    label_fpath = c.scoped_tmp_path / "labels.csv"

    label_columns = ["Origin", "NonExistentColumn"]

    df_labels = label_setup._load_label_df(
        label_fpath=label_fpath,
        columns=label_columns,
        custom_label_ops=test_column_operations,
    )
    assert df_labels.shape[1] == 1


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_load_label_extra_target_extra_col(parse_test_cl_args, create_test_data):
    c = create_test_data

    label_fpath = c.scoped_tmp_path / "labels.csv"

    label_columns = ["Origin", "OriginExtraCol", "Height", "ExtraTarget"]
    df_label_multi_target = label_setup._load_label_df(
        label_fpath=label_fpath, columns=label_columns, custom_label_ops=None
    )

    assert df_label_multi_target.shape[1] == 4

    # Check that they're all the same, as defined
    part_1 = df_label_multi_target["Origin"]
    part_2 = df_label_multi_target["OriginExtraCol"]

    assert (part_1 == part_2).all()

    part_3 = df_label_multi_target["Height"]
    part_4 = df_label_multi_target["ExtraTarget"]
    assert ((part_3 - 50).astype(int) == part_4.astype(int)).all()


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_cast_label_df_dtypes(parse_test_cl_args, create_test_data):
    c = create_test_data

    label_fpath = c.scoped_tmp_path / "labels.csv"

    label_columns = ["Origin", "OriginExtraCol", "Height", "ExtraTarget"]
    df_label_multi_target = label_setup._load_label_df(
        label_fpath=label_fpath, columns=label_columns, custom_label_ops=None
    )

    df_label_multi_target["OriginExtraCol"] = 10

    df_label_multi_target = label_setup._cast_label_df_dtypes(
        df_labels=df_label_multi_target, extra_cat_columns=["OriginExtraCol"]
    )

    assert df_label_multi_target["OriginExtraCol"].dtype.name == "object"


@pytest.fixture
def test_column_operations():
    def _dummy_func(df, column_name):
        return df

    test_column_ops = (ColumnOperation(function=_dummy_func, function_args={}),)

    return test_column_ops


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_get_currently_available_columns_pass(
    parse_test_cl_args, create_test_data, test_column_operations
):
    c = create_test_data

    label_fpath = c.scoped_tmp_path / "labels.csv"
    label_columns = ["Origin", "NotExisting1", "NotExisting2"]

    available_columns = label_setup._get_currently_available_columns(
        label_fpath=label_fpath,
        requested_columns=label_columns,
        custom_label_ops=test_column_operations,
    )

    assert available_columns == ["Origin"]


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_get_currently_available_columns_fail(parse_test_cl_args, create_test_data):
    c = create_test_data

    label_fpath = c.scoped_tmp_path / "labels.csv"
    label_columns = ["Origin", "NotExisting1", "NotExisting2"]

    with pytest.raises(ValueError):
        label_setup._get_currently_available_columns(
            label_fpath=label_fpath,
            requested_columns=label_columns,
            custom_label_ops=None,
        )


@pytest.mark.parametrize("create_test_data", [{"task_type": "binary"}], indirect=True)
def test_parse_label_df_applied_1(create_test_data, create_test_column_ops):
    """
    Here we run column operations for 'Origin'. Hence we expect to apply:

        - test_column_op_1: Replace "Europe with Iceland"
        - test_column_op_2: Multiply the values by 2.

    As these are the column operations associated with "Origin" in
    create_test_column_ops.
    """
    c = create_test_data
    label_fpath = c.scoped_tmp_path / "labels.csv"

    test_column_ops = create_test_column_ops

    label_columns = ["Origin"]
    df_labels = label_setup._load_label_df(
        label_fpath=label_fpath, columns=label_columns, custom_label_ops=None
    )
    df_labels_parsed = label_setup._parse_label_df(
        df=df_labels, operations_dict=test_column_ops, label_columns=label_columns
    )

    assert set(df_labels_parsed.Origin.unique()) == {"Iceland" * 2, "Asia" * 2}


@pytest.mark.parametrize("create_test_data", [{"task_type": "binary"}], indirect=True)
def test_parse_label_df_applied_2(create_test_data, create_test_column_ops):
    """
    Here we run column operations for 'OriginExtraColumnsAll'. Hence we expect to apply:

        - test_column_op_1: Replace "Europe with Iceland"
        - test_column_op_3: Replace value of of 'OriginExtraColumnsAll' with the values
          in ExtraCol3.
    """

    c = create_test_data
    label_fpath = c.scoped_tmp_path / "labels.csv"

    test_column_ops = create_test_column_ops

    label_columns = ["Origin"]
    df_labels = label_setup._load_label_df(
        label_fpath=label_fpath, columns=label_columns, custom_label_ops=None
    )

    extra_cols = ("ExtraCol3",)
    for col in extra_cols:
        df_labels[col] = "Iceland"

    df_labels = df_labels.rename(columns={"Origin": "OriginExtraColumnsAll"})

    new_label_columns = ["OriginExtraColumnsAll"]
    df_labels_parsed = label_setup._parse_label_df(
        df=df_labels, operations_dict=test_column_ops, label_columns=new_label_columns
    )

    assert df_labels_parsed["OriginExtraColumnsAll"].unique().item() == "Iceland"


@patch("snp_pred.data_load.label_setup.logger.debug", autospec=True)
@pytest.mark.parametrize("create_test_data", [{"task_type": "binary"}], indirect=True)
def test_parse_label_df_not_applied(
    m_logger, create_test_data, create_test_column_ops, test_column_operations
):
    """
    Here we run column operations for 'Origin'. Hence we expect to apply:

        - test_column_op_1: Replace "Europe with Iceland"
        - test_column_op_2: Multiply the values by 2.

    As these are the column operations associated with "Origin" in
    create_test_column_ops.

    Additionally, we manually add two columns to the label df. For these, we don't
    expect them to change as no column operations should be performed on them.
    Firstly because one op should only run if it's a target, secondly because it's
    a random extra column.

    So in the logging, we expect only 'Applying func' to be called twice, for the
    'Origin' column.
    """

    def _check_mocked_logger_call_count():

        calls = []
        for call in m_logger.call_args_list:
            cur_call_first_arg = call[0][0]
            if cur_call_first_arg.startswith("Applying"):
                calls.append(cur_call_first_arg)

        assert len(calls) == 2

    c = create_test_data
    label_fpath = c.scoped_tmp_path / "labels.csv"

    test_column_ops = create_test_column_ops

    label_columns = ["Origin"]
    df_labels = label_setup._load_label_df(
        label_fpath=label_fpath,
        columns=label_columns,
        custom_label_ops=test_column_operations,
    )
    df_labels["OnlyApplyIfTarget"] = 1
    df_labels["SomeRandomCol"] = 1

    df_labels_parsed = label_setup._parse_label_df(
        df=df_labels, operations_dict=test_column_ops, label_columns=label_columns
    )

    assert set(df_labels_parsed.Origin.unique()) == {"Iceland" * 2, "Asia" * 2}
    _check_mocked_logger_call_count()
    assert df_labels_parsed["OnlyApplyIfTarget"].unique().item() == 1
    assert df_labels_parsed["SomeRandomCol"].unique().item() == 1


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_check_parsed_label_df_pass(parse_test_cl_args, create_test_data):
    c = create_test_data
    label_fpath = c.scoped_tmp_path / "labels.csv"

    label_columns = ["Origin", "ExtraTarget"]

    df_labels = label_setup._load_label_df(
        label_fpath=label_fpath, columns=label_columns, custom_label_ops=None
    )

    df_labels_checked = label_setup._check_parsed_label_df(
        df_labels=df_labels, supplied_label_columns=label_columns
    )
    assert df_labels is df_labels_checked


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_check_parsed_label_df_fail(
    parse_test_cl_args, create_test_data, test_column_operations
):
    c = create_test_data
    label_fpath = c.scoped_tmp_path / "labels.csv"

    label_columns = ["Origin", "ExtraTarget", "NotExisting"]

    df_labels = label_setup._load_label_df(
        label_fpath=label_fpath,
        columns=label_columns,
        custom_label_ops=test_column_operations,
    )

    with pytest.raises(ValueError):
        label_setup._check_parsed_label_df(
            df_labels=df_labels, supplied_label_columns=label_columns
        )


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_split_df(create_test_data, create_test_cl_args):
    cl_args = create_test_cl_args

    df_labels = label_setup.label_df_parse_wrapper(cl_args)

    for valid_fraction in (0.1, 0.5, 0.7):

        df_train, df_valid = label_setup._split_df(df_labels, valid_fraction)
        expected_train = df_labels.shape[0] * (1 - valid_fraction)
        expected_valid = df_labels.shape[0] * valid_fraction

        assert df_train.shape[0] == int(expected_train)
        assert df_valid.shape[0] == int(expected_valid)


@pytest.fixture
def get_test_nan_df():
    """
    >>> df
         A    B   C  D
    0  NaN  2.0 NaN  0
    1  3.0  4.0 NaN  1
    2  NaN  NaN NaN  5
    3  NaN  3.0 NaN  4
    """
    df = pd.DataFrame(
        [
            [np.nan, 2, np.nan, 0],
            [3, 4, np.nan, 1],
            [np.nan, np.nan, np.nan, 5],
            [np.nan, 3, np.nan, 4],
        ],
        columns=list("ABCD"),
    )

    return df


@pytest.fixture
def get_test_nan_args():
    cl_args = Namespace(
        **{
            "target_cat_columns": ["A"],
            "extra_cat_columns": ["B"],
            "target_con_columns": ["C"],
            "extra_con_columns": ["D"],
        }
    )

    return cl_args


def test_process_train_and_label_dfs(get_test_nan_df, get_test_nan_args):
    test_df = get_test_nan_df
    cl_args = get_test_nan_args

    train_df = test_df.fillna(5)
    valid_df = test_df

    train_df_filled, valid_df_filled = label_setup._process_train_and_label_dfs(
        cl_args=cl_args, df_labels_train=train_df, df_labels_valid=valid_df
    )

    assert set(train_df_filled["A"].unique()) == {5.0, 3.0}
    assert set(valid_df_filled["A"].unique()) == {"NA", 3}

    assert set(train_df_filled["B"].unique()) == {2.0, 3.0, 4.0, 5.0}
    assert set(valid_df_filled["B"].unique()) == {"NA", 2, 3, 4}

    assert (train_df_filled["C"] == 5.0).all()
    assert (valid_df_filled["C"] == 5.0).all()

    assert (train_df_filled["D"] == valid_df_filled["D"]).all()


def test_handle_missing_label_values_in_df(get_test_nan_df, get_test_nan_args):
    test_df = get_test_nan_df
    cl_args = get_test_nan_args

    test_df_filled = label_setup.handle_missing_label_values_in_df(
        df=test_df, cl_args=cl_args, con_manual_values={"C": 3.0}
    )
    assert set(test_df_filled["A"].unique()) == {"NA", 3}
    assert set(test_df_filled["B"].unique()) == {"NA", 2, 3, 4}
    assert (test_df_filled["C"] == 3.0).all()
    assert (test_df_filled["D"] == test_df["D"]).all()


def test_fill_categorical_nans(get_test_nan_df):
    test_df = get_test_nan_df
    test_df_filled = label_setup._fill_categorical_nans(
        df=test_df, column_names=["A", "B"]
    )

    assert set(test_df_filled["A"].unique()) == {"NA", 3}
    assert set(test_df_filled["B"].unique()) == {"NA", 2, 3, 4}
    assert test_df_filled["C"].isna().values.all()
    assert test_df_filled["D"].notna().values.all()


def test_get_con_manual_vals_dict(get_test_nan_df):
    test_df = get_test_nan_df

    means_dict = label_setup._get_con_manual_vals_dict(
        df=test_df, con_columns=["A", "B", "C", "D"]
    )
    assert means_dict["A"] == 3.0
    assert means_dict["B"] == 3.0
    assert np.isnan(means_dict["C"])
    assert means_dict["D"] == 2.5


def test_fill_continuous_nans(get_test_nan_df):
    test_df = get_test_nan_df
    manual_values = {"A": 1.0, "B": 2.0, "C": 3.0}
    test_df_filled = label_setup._fill_continuous_nans(
        df=test_df, column_names=["A", "B", "C"], con_means_dict=manual_values
    )

    assert test_df_filled["A"].loc[0] == 1.0
    assert test_df_filled["A"].loc[1] == 3.0
    assert test_df_filled["B"].loc[2] == 2.0
    assert (test_df_filled["C"] == 3.0).all()


def get_joblib_patch_target():
    return "snp_pred.data_load.label_setup.joblib"


@patch(get_joblib_patch_target(), autospec=True)
def test_save_target_transformer(patched_joblib):

    test_transformer = StandardScaler()
    test_transformer.fit([[1, 2, 3, 4, 5]])

    label_setup.save_label_transformer(
        run_folder=Path("/tmp/"),
        transformer_name="harry_du_bois",
        target_transformer_object=test_transformer,
    )
    assert patched_joblib.dump.call_count == 1

    _, m_kwargs = patched_joblib.dump.call_args
    # check that we have correct name, with target_transformers tagged on
    assert m_kwargs["filename"].name == "harry_du_bois.save"


def test_set_up_all_target_transformers(get_transformer_test_data):
    df_test_labels, test_target_columns_dict = get_transformer_test_data

    all_target_transformers = label_setup._get_fit_label_transformers(
        df_labels=df_test_labels, label_columns=test_target_columns_dict
    )

    height_transformer = all_target_transformers["Height"]
    assert isinstance(height_transformer, StandardScaler)

    origin_transformer = all_target_transformers["Origin"]
    assert isinstance(origin_transformer, LabelEncoder)


def test_fit_scaler_transformer_on_target_column(get_transformer_test_data):
    df_test_labels, test_target_columns_dict = get_transformer_test_data

    transformer = label_setup._get_transformer(column_type="con")

    height_transformer = label_setup._fit_transformer_on_label_column(
        column_series=df_test_labels["Height"], transformer=transformer
    )

    assert height_transformer.n_samples_seen_ == 3
    assert height_transformer.mean_ == 170
    assert height_transformer.transform([[170]]) == 0


def test_fit_labelencoder_transformer_on_target_column(get_transformer_test_data):
    df_test_labels, test_target_columns_dict = get_transformer_test_data

    transformer = label_setup._get_transformer(column_type="cat")

    origin_transformer = label_setup._fit_transformer_on_label_column(
        column_series=df_test_labels["Origin"], transformer=transformer
    )

    assert origin_transformer.transform(["Africa"]).item() == 0
    assert origin_transformer.transform(["Europe"]).item() == 2


def test_streamline_values_for_transformer():
    test_values = np.array([1, 2, 3, 4, 5])

    scaler_transformer = StandardScaler()
    streamlined_values_scaler = label_setup._streamline_values_for_transformers(
        transformer=scaler_transformer, values=test_values
    )
    assert streamlined_values_scaler.shape == (5, 1)

    encoder_transformer = LabelEncoder()
    streamlined_values_encoder = label_setup._streamline_values_for_transformers(
        transformer=encoder_transformer, values=test_values
    )
    assert streamlined_values_encoder.shape == (5,)


@pytest.mark.parametrize(
    "test_input_key,expected",
    [
        ("1", {"Origin_as_int": 1, "Scaled_height_int": -1}),  # asia
        ("2", {"Origin_as_int": 0, "Scaled_height_int": 1}),  # africa
        ("3", {"Origin_as_int": 2, "Scaled_height_int": 0}),  # europe
    ],
)
def test_transform_all_labels_in_sample_targets_only(
    test_input_key, expected, get_transformer_test_data
):
    df_test_labels, test_target_columns_dict = get_transformer_test_data

    target_transformers = label_setup._get_fit_label_transformers(
        df_labels=df_test_labels,
        label_columns=test_target_columns_dict,
    )

    transformed_df = label_setup.transform_label_df(
        df_labels=df_test_labels, label_transformers=target_transformers
    )

    transformed_sample_labels = transformed_df.loc[test_input_key].to_dict()

    assert transformed_sample_labels["Origin"] == expected["Origin_as_int"]
    assert int(transformed_sample_labels["Height"]) == expected["Scaled_height_int"]


@pytest.mark.parametrize(
    "test_input_key,expected",
    [
        ("1", {"Extra_con_int": -1}),  # asia
        ("2", {"Extra_con_int": 1}),  # africa
        ("3", {"Extra_con_int": 0}),  # europe
    ],
)
def test_transform_all_labels_in_sample_with_extra_con(
    test_input_key, expected, get_transformer_test_data
):
    df_test_labels, test_target_columns_dict = get_transformer_test_data

    df_test_labels["Extra_Con"] = np.nan
    df_test_labels.loc["1", "Extra_Con"] = 130
    df_test_labels.loc["2", "Extra_Con"] = 170
    df_test_labels.loc["3", "Extra_Con"] = 150

    test_target_columns_dict["con"].append("Extra_Con")
    label_transformers = label_setup._get_fit_label_transformers(
        df_labels=df_test_labels, label_columns=test_target_columns_dict
    )

    df_test_labels_transformed = label_setup.transform_label_df(
        df_labels=df_test_labels, label_transformers=label_transformers
    )

    transformed_sample_labels = df_test_labels_transformed.loc[test_input_key].to_dict()

    assert int(transformed_sample_labels["Extra_Con"]) == expected["Extra_con_int"]


@pytest.mark.parametrize(
    "test_input,expected",
    [  # test case 1
        (
            (["con_1", "con_2"], ["cat_1", "cat_2"]),
            {"con": ["con_1", "con_2"], "cat": ["cat_1", "cat_2"]},
        ),
        # test case 2
        ((["con_1", "con_2"], []), {"con": ["con_1", "con_2"], "cat": []}),
    ],
)
def test_merge_target_columns_pass(test_input, expected):
    test_output = merge_target_columns(*test_input)
    assert test_output == expected


def test_merge_target_columns_fail():
    with pytest.raises(ValueError):
        merge_target_columns([], [])
