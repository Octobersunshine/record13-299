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


if __name__ == '__main__':
    main()
