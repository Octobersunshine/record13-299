import pandas as pd
from typing import Union, List, Dict, Any, Optional


class GroupAggregator:
    SUPPORTED_AGG_FUNCS = {
        'count', 'sum', 'mean', 'avg', 'max', 'min',
        'std', 'var', 'median', 'first', 'last'
    }

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self._df = df.copy()
        self._group_cols: Optional[List[str]] = None
        self._result: Optional[pd.DataFrame] = None

    @property
    def result(self) -> Optional[pd.DataFrame]:
        return self._result

    @staticmethod
    def _validate_agg_func(agg_func: str) -> str:
        func_lower = agg_func.lower()
        if func_lower == 'avg':
            return 'mean'
        if func_lower not in GroupAggregator.SUPPORTED_AGG_FUNCS:
            raise ValueError(
                f"Unsupported aggregation function: {agg_func}. "
                f"Supported: {sorted(GroupAggregator.SUPPORTED_AGG_FUNCS)}"
            )
        return func_lower

    @staticmethod
    def _build_agg_dict(
        agg_cols: Union[str, List[str]],
        agg_funcs: Union[str, List[str]]
    ) -> Dict[str, List[str]]:
        if isinstance(agg_cols, str):
            agg_cols = [agg_cols]
        if isinstance(agg_funcs, str):
            agg_funcs = [agg_funcs]

        validated_funcs = [GroupAggregator._validate_agg_func(f) for f in agg_funcs]

        agg_dict: Dict[str, List[str]] = {}
        for col in agg_cols:
            agg_dict[col] = validated_funcs

        return agg_dict

    def aggregate(
        self,
        group_cols: Union[str, List[str]],
        agg_cols: Union[str, List[str]],
        agg_funcs: Union[str, List[str]],
        sort: bool = True,
        as_index: bool = True,
        dropna: bool = False
    ) -> pd.DataFrame:
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        for col in group_cols:
            if col not in self._df.columns:
                raise KeyError(f"Group column '{col}' not found in DataFrame")

        if isinstance(agg_cols, str):
            agg_cols_list = [agg_cols]
        else:
            agg_cols_list = agg_cols

        for col in agg_cols_list:
            if col not in self._df.columns:
                raise KeyError(f"Aggregation column '{col}' not found in DataFrame")

        self._group_cols = group_cols

        agg_dict = self._build_agg_dict(agg_cols, agg_funcs)

        grouped = self._df.groupby(
            by=group_cols,
            sort=sort,
            as_index=as_index,
            dropna=dropna
        )

        result = grouped.agg(agg_dict)

        if as_index:
            result.columns = [
                f"{col}_{func}" for col, func in result.columns
            ]
            self._result = result.reset_index()
        else:
            new_cols = []
            agg_col_set = set(agg_dict.keys())
            for col in result.columns:
                if isinstance(col, tuple):
                    if col[0] in agg_col_set:
                        new_cols.append(f"{col[0]}_{col[1]}")
                    else:
                        new_cols.append(col[0] if col[0] else col[1])
                else:
                    new_cols.append(col)
            result.columns = new_cols
            self._result = result

        return self._result

    def count(self, group_cols: Union[str, List[str]], **kwargs) -> pd.DataFrame:
        agg_col = self._df.columns[0]
        return self.aggregate(group_cols, agg_col, 'count', **kwargs)

    def sum(self, group_cols: Union[str, List[str]], agg_cols: Union[str, List[str]], **kwargs) -> pd.DataFrame:
        return self.aggregate(group_cols, agg_cols, 'sum', **kwargs)

    def mean(self, group_cols: Union[str, List[str]], agg_cols: Union[str, List[str]], **kwargs) -> pd.DataFrame:
        return self.aggregate(group_cols, agg_cols, 'mean', **kwargs)

    def max(self, group_cols: Union[str, List[str]], agg_cols: Union[str, List[str]], **kwargs) -> pd.DataFrame:
        return self.aggregate(group_cols, agg_cols, 'max', **kwargs)

    def min(self, group_cols: Union[str, List[str]], agg_cols: Union[str, List[str]], **kwargs) -> pd.DataFrame:
        return self.aggregate(group_cols, agg_cols, 'min', **kwargs)

    def multi_aggregate(
        self,
        group_cols: Union[str, List[str]],
        agg_config: Dict[str, Union[str, List[str]]],
        dropna: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        for col in group_cols:
            if col not in self._df.columns:
                raise KeyError(f"Group column '{col}' not found in DataFrame")

        validated_config: Dict[str, List[str]] = {}
        for col, funcs in agg_config.items():
            if col not in self._df.columns:
                raise KeyError(f"Aggregation column '{col}' not found in DataFrame")
            if isinstance(funcs, str):
                funcs = [funcs]
            validated_config[col] = [self._validate_agg_func(f) for f in funcs]

        self._group_cols = group_cols

        grouped = self._df.groupby(by=group_cols, dropna=dropna, **kwargs)
        result = grouped.agg(validated_config)

        result.columns = [f"{col}_{func}" for col, func in result.columns]
        self._result = result.reset_index()
        return self._result

    def composite_group(
        self,
        group_cols: Union[str, List[str]],
        agg_cols: Union[str, List[str]],
        agg_funcs: Union[str, List[str]],
        sep: str = '_',
        dropna: bool = False,
        sort: bool = True
    ) -> pd.DataFrame:
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        if len(group_cols) < 2:
            raise ValueError(
                f"composite_group requires at least 2 group columns, got {len(group_cols)}. "
                f"Use aggregate() for single-column grouping."
            )

        for col in group_cols:
            if col not in self._df.columns:
                raise KeyError(f"Group column '{col}' not found in DataFrame")

        self._group_cols = group_cols

        composite_col_name = sep.join(group_cols)

        work_df = self._df.copy()
        work_df[composite_col_name] = work_df[group_cols].apply(
            lambda row: sep.join(str(v) if pd.notna(v) else 'NaN' for v in row),
            axis=1
        )

        agg_dict = self._build_agg_dict(agg_cols, agg_funcs)

        grouped = work_df.groupby(
            by=composite_col_name,
            sort=sort,
            dropna=dropna
        )

        result = grouped.agg(agg_dict)
        result.columns = [f"{col}_{func}" for col, func in result.columns]
        result = result.reset_index()

        self._result = result
        return self._result

    def pivot_aggregate(
        self,
        row_col: str,
        col_col: str,
        agg_col: str,
        agg_func: str = 'sum',
        fill_value: Any = None,
        dropna: bool = False
    ) -> pd.DataFrame:
        for name, col in [('row_col', row_col), ('col_col', col_col), ('agg_col', agg_col)]:
            if col not in self._df.columns:
                raise KeyError(f"{name} '{col}' not found in DataFrame")

        self._group_cols = [row_col, col_col]

        agg_func = self._validate_agg_func(agg_func)

        pivot_df = self._df.pivot_table(
            index=row_col,
            columns=col_col,
            values=agg_col,
            aggfunc=agg_func,
            fill_value=fill_value,
            dropna=dropna
        )

        pivot_df.columns = [f"{col_col}_{v}" if pd.notna(v) else f"{col_col}_NaN"
                            for v in pivot_df.columns]
        pivot_df = pivot_df.reset_index()
        pivot_df.columns.name = None

        self._result = pivot_df
        return self._result

    @property
    def group_info(self) -> Dict[str, Any]:
        if self._group_cols is None:
            return {"status": "no grouping performed yet"}

        info: Dict[str, Any] = {
            "group_columns": self._group_cols,
            "num_groups": None,
            "group_keys": None
        }

        if self._group_cols and all(col in self._df.columns for col in self._group_cols):
            grouped = self._df.groupby(by=self._group_cols, dropna=False)
            info["num_groups"] = grouped.ngroups
            info["group_keys"] = [
                list(key) if isinstance(key, tuple) else key
                for key in grouped.groups.keys()
            ]

        if self._result is not None:
            info["result_shape"] = list(self._result.shape)
            info["result_columns"] = list(self._result.columns)

        return info

    def to_csv(self, path: str, **kwargs) -> None:
        if self._result is None:
            raise ValueError("No aggregation result to save. Call aggregate() first.")
        self._result.to_csv(path, **kwargs)

    def to_dict(self, **kwargs) -> Dict[str, Any]:
        if self._result is None:
            raise ValueError("No aggregation result to convert. Call aggregate() first.")
        return self._result.to_dict(**kwargs)

    def __repr__(self) -> str:
        return f"GroupAggregator(shape={self._df.shape}, group_cols={self._group_cols})"
