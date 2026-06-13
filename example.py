import pandas as pd
import numpy as np
from group_aggregator import GroupAggregator


def create_sample_data():
    data = {
        '部门': ['技术部', '销售部', '技术部', '市场部', '销售部', '技术部', '市场部', '销售部', '技术部', '市场部'],
        '城市': ['北京', '上海', '北京', '深圳', '上海', '深圳', '北京', '上海', '深圳', '北京'],
        '员工姓名': ['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '吴十', '郑十一', '冯十二'],
        '年龄': [28, 32, 25, 30, 35, 29, 27, 31, 33, 26],
        '薪资': [15000, 18000, 14000, 16000, 20000, 17000, 15500, 19000, 17500, 14500],
        '绩效评分': [4.5, 4.8, 4.2, 4.6, 4.9, 4.3, 4.7, 4.4, 4.1, 4.6]
    }
    return pd.DataFrame(data)


def main():
    print("=" * 60)
    print("Pandas 分组聚合服务示例")
    print("=" * 60)

    df = create_sample_data()
    print("\n原始数据:")
    print(df.to_string(index=False))

    aggregator = GroupAggregator(df)
    print(f"\nAggregator 初始化: {aggregator}")

    print("\n" + "=" * 40)
    print("1. 按部门统计员工数量 (count)")
    print("=" * 40)
    result_count = aggregator.count(group_cols='部门')
    print(result_count)

    print("\n" + "=" * 40)
    print("2. 按部门统计薪资总和 (sum)")
    print("=" * 40)
    result_sum = aggregator.sum(group_cols='部门', agg_cols='薪资')
    print(result_sum)

    print("\n" + "=" * 40)
    print("3. 按部门计算平均薪资和平均年龄 (mean)")
    print("=" * 40)
    result_mean = aggregator.mean(group_cols='部门', agg_cols=['薪资', '年龄'])
    print(result_mean)

    print("\n" + "=" * 40)
    print("4. 按部门+城市分组，统计薪资的 max/min")
    print("=" * 40)
    result_max_min = aggregator.aggregate(
        group_cols=['部门', '城市'],
        agg_cols='薪资',
        agg_funcs=['max', 'min']
    )
    print(result_max_min)

    print("\n" + "=" * 40)
    print("5. 多维度聚合：不同列用不同统计函数")
    print("=" * 40)
    result_multi = aggregator.multi_aggregate(
        group_cols='部门',
        agg_config={
            '员工姓名': 'count',
            '薪资': ['sum', 'mean', 'max', 'min'],
            '绩效评分': ['mean', 'max']
        }
    )
    print(result_multi.to_string())

    print("\n" + "=" * 40)
    print("6. 不排序，保留分组列为普通列 (as_index=False)")
    print("=" * 40)
    result_no_index = aggregator.aggregate(
        group_cols='部门',
        agg_cols='薪资',
        agg_funcs=['mean', 'sum'],
        as_index=False,
        sort=False
    )
    print(result_no_index)

    print("\n" + "=" * 40)
    print("7. 将结果导出为字典")
    print("=" * 40)
    result_dict = aggregator.to_dict(orient='records')
    print(f"第一条记录: {result_dict[0]}")

    print("\n" + "=" * 40)
    print("8. 使用简写 avg (自动映射为 mean)")
    print("=" * 40)
    result_avg = aggregator.aggregate(
        group_cols='城市',
        agg_cols='薪资',
        agg_funcs='avg'
    )
    print(result_avg)

    print("\n" + "=" * 40)
    print("9. 链式调用 - 查看最终结果")
    print("=" * 40)
    print(f"最后一次聚合结果:")
    print(aggregator.result)

    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)


def demo_null_groups():
    print("\n" + "=" * 60)
    print("含空值分组聚合示例")
    print("=" * 60)

    data = {
        '部门': ['技术部', '销售部', None, '技术部', '市场部', None, '销售部', '技术部'],
        '城市': ['北京', '上海', '深圳', None, '北京', '上海', '上海', '北京'],
        '薪资': [15000, 18000, 12000, 14000, 16000, 13000, 20000, 17000],
        '绩效': [4.5, 4.8, 4.0, 4.2, 4.6, 3.9, 4.9, 4.3]
    }
    df_null = pd.DataFrame(data)
    print("\n含空值的数据:")
    print(df_null.to_string(index=False))

    agg = GroupAggregator(df_null)

    print("\n" + "-" * 40)
    print("1. dropna=False (默认): 空值作为独立分组")
    print("-" * 40)
    result_keep = agg.aggregate(
        group_cols='部门',
        agg_cols='薪资',
        agg_funcs=['count', 'sum', 'mean']
    )
    print(result_keep)

    print("\n" + "-" * 40)
    print("2. dropna=True: 丢弃含空值的分组行")
    print("-" * 40)
    result_drop = agg.aggregate(
        group_cols='部门',
        agg_cols='薪资',
        agg_funcs=['count', 'sum', 'mean'],
        dropna=True
    )
    print(result_drop)

    print("\n" + "-" * 40)
    print("3. 多列分组含空值 (dropna=False)")
    print("-" * 40)
    result_multi_null = agg.aggregate(
        group_cols=['部门', '城市'],
        agg_cols='薪资',
        agg_funcs=['count', 'mean']
    )
    print(result_multi_null)

    print("\n" + "-" * 40)
    print("4. multi_aggregate 含空值 (dropna=False)")
    print("-" * 40)
    result_multi_agg = agg.multi_aggregate(
        group_cols='部门',
        agg_config={
            '薪资': ['count', 'sum', 'mean'],
            '绩效': ['mean', 'max']
        }
    )
    print(result_multi_agg.to_string())

    print("\n" + "=" * 60)
    print("含空值分组示例完成！")
    print("=" * 60)


def demo_multi_column_group():
    print("\n" + "=" * 60)
    print("多列分组聚合示例")
    print("=" * 60)

    data = {
        '部门': ['技术部', '销售部', '技术部', '市场部', '销售部', '技术部',
                 '市场部', '销售部', '技术部', '市场部', '技术部', '销售部'],
        '城市': ['北京', '上海', '北京', '深圳', '上海', '深圳',
                 '北京', '广州', '深圳', '北京', '广州', '广州'],
        '级别': ['高级', '高级', '中级', '中级', '高级', '中级',
                 '高级', '中级', '高级', '中级', '中级', '高级'],
        '薪资': [25000, 22000, 15000, 16000, 28000, 17000,
                 20000, 18000, 24000, 15500, 16000, 26000],
        '绩效': [4.8, 4.6, 4.2, 4.5, 4.9, 4.3,
                 4.7, 4.1, 4.8, 4.0, 4.4, 4.7]
    }
    df = pd.DataFrame(data)
    print("\n原始数据:")
    print(df.to_string(index=False))

    agg = GroupAggregator(df)

    print("\n" + "-" * 40)
    print("1. 按部门+城市 两列分组 (aggregate)")
    print("-" * 40)
    result = agg.aggregate(
        group_cols=['部门', '城市'],
        agg_cols='薪资',
        agg_funcs=['count', 'sum', 'mean']
    )
    print(result)

    print("\n" + "-" * 40)
    print("2. 按部门+城市+级别 三列分组 (aggregate)")
    print("-" * 40)
    result_3col = agg.aggregate(
        group_cols=['部门', '城市', '级别'],
        agg_cols='薪资',
        agg_funcs=['count', 'mean']
    )
    print(result_3col.to_string())

    print("\n" + "-" * 40)
    print("3. 组合键分组 (composite_group): 部门+城市 -> 部门_城市")
    print("-" * 40)
    result_composite = agg.composite_group(
        group_cols=['部门', '城市'],
        agg_cols='薪资',
        agg_funcs=['count', 'sum', 'mean'],
        sep='_'
    )
    print(result_composite)

    print("\n" + "-" * 40)
    print("4. 组合键分组 - 自定义分隔符: 部门-城市")
    print("-" * 40)
    result_custom_sep = agg.composite_group(
        group_cols=['部门', '城市'],
        agg_cols='薪资',
        agg_funcs=['count', 'mean'],
        sep='-'
    )
    print(result_custom_sep)

    print("\n" + "-" * 40)
    print("5. 交叉透视表 (pivot_aggregate): 行=部门, 列=城市, 值=薪资sum")
    print("-" * 40)
    result_pivot = agg.pivot_aggregate(
        row_col='部门',
        col_col='城市',
        agg_col='薪资',
        agg_func='sum',
        fill_value=0
    )
    print(result_pivot)

    print("\n" + "-" * 40)
    print("6. 交叉透视表: 行=部门, 列=级别, 值=薪资mean")
    print("-" * 40)
    result_pivot2 = agg.pivot_aggregate(
        row_col='部门',
        col_col='级别',
        agg_col='薪资',
        agg_func='mean',
        fill_value=0
    )
    print(result_pivot2)

    print("\n" + "-" * 40)
    print("7. 多列分组 + multi_aggregate")
    print("-" * 40)
    result_multi = agg.multi_aggregate(
        group_cols=['部门', '级别'],
        agg_config={
            '薪资': ['count', 'sum', 'mean'],
            '绩效': ['mean', 'max']
        }
    )
    print(result_multi.to_string())

    print("\n" + "-" * 40)
    print("8. 查看 group_info 分组详情")
    print("-" * 40)
    info = agg.group_info
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("多列分组聚合示例完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
    demo_null_groups()
    demo_multi_column_group()
