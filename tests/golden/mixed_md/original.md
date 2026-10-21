---
title: 用户数据流水线说明
lang: zh-CN
---

# Pipeline field notes

This article mixes English and Chinese on purpose: the prose below
alternates, and the code comments are queued for translation into English.

## 背景 Background

用户数据流水线每天凌晨两点开始同步。所有 region 的数据先进入临时队列，
任何一个 region 连续失败三次，整条流水线就会暂停，等待值班工程师确认。
原始日志保留三十天，汇总后的报表保留一年，磁盘紧张时优先清理旧日志。

## The ingestion script

```python
def summarize(users):
    # 遍历所有用户 / iterate all users
    total = 0
    # The per-region quota must not be exceeded
    for user in users:
        # 累加订单金额 (sum the order amounts)
        total += user.order_amount
    # Keep the report id stable across reruns
    return {"users": len(users), "total": total}
```

## 汇总任务

汇总任务在内存里维护一张哈希表，异常处理遵循"先记录再重试"的原则。
值班工程师会在早会前检查死信队列，并把失败的批次手工重放。

## The go exporter

```go
func Export(rows []Row) error {
    // 打开输出文件 / open the output file
    f, err := os.Create("report.csv")
    if err != nil {
        return err
    }
    // The header row is written once, before any data
    w := csv.NewWriter(f)
    // 逐行写入报表 (write the report row by row)
    for _, row := range rows {
        if err := w.Write(row); err != nil {
            return err
        }
    }
    w.Flush()
    return w.Error()
}
```

## Embedded widget

The docs site embeds the summary widget inline:

```html
<div id="summary-widget" lang="zh-CN">
  <p>本季度共处理订单 <b>42,000</b> 笔。</p>
</div>
```

## 附录：故意保留的排版缺陷

下方代码块的闭合围栏在一次失败的合并中丢失了，此后一直没有补上。
编辑器必须原样保留它：快速编辑是编辑器，不是更正器。

~~~
遗留问题：报表脚注的样式还没有迁移到新的主题。
遗留任务：把清理任务的日志级别调成 warning。
