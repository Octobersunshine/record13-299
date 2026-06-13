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
        dropna: bool = True
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

        grouped = self._df.groupby(by=group_cols, **kwargs)
        result = grouped.agg(validated_config)

        result.columns = [f"{col}_{func}" for col, func in result.columns]
        self._result = result.reset_index()
        return self._result

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
