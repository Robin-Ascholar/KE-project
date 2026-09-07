# PDF 正文清洗

本目录保存针对 Harry Potter 七本单栏、带文字层 PDF 编写的清洗脚本。脚本只读取原始 PDF，不会修改源文件。

## 输入与输出

```text
原始数据/*.pdf                  # 七本原始 PDF
cleanUP/txt/*.txt              # UTF-8 章节标题与正文
cleanUP/txt/*.report.json      # 清洗审计报告
```

当前七本书识别的标题数依次为 17、18、22、37、38、30、37，总计 199 章；第七本包含尾声 `NINETEEN YEARS LATER`。

## 运行

在项目根目录执行：

```bash
python cleanUP/clean_pdf.py

# 重新生成已有结果
python cleanUP/clean_pdf.py --overwrite

# 处理单本书或使用其他输出目录
python cleanUP/clean_pdf.py \
  --input "原始数据/Book 1 - The Philosopher's Stone.pdf" \
  --output cleanUP/sample
```

依赖 Python 3 和系统命令 `pdftotext`，不依赖额外 Python 包。Ubuntu 缺少命令时安装：

```bash
sudo apt-get install poppler-utils
```

默认拒绝覆盖已有结果；只有显式传入 `--overwrite` 才会替换输出。

## 清洗内容

脚本使用文字坐标、行高和重复位置识别版面元素：

- 跳过第一章前的封面、目录和出版信息
- 识别居中的章节标题
- 移除重复页眉、页脚和页码
- 移除页面底部带数字或符号标记的小字号脚注
- 跳过纯图片页，图片中的文字不会进入 TXT
- 按行间距恢复段落并连接正文折行
- 保留正文中的大写对白、引语、书信和复合词

报告记录源文件 SHA256、页数、章节起止页、删除行和删除原因，便于抽样核查。当前输出正文中未检出 `Page |` 页脚标记。

## 限制

规则针对仓库中的七本 PDF 版式，不是通用 PDF 清洗器。扫描版 PDF 需要先 OCR；独立图注、表格文本、无标记脚注和正文后的附录不保证全部自动删除。

发现残留时可以重复使用 `--drop-regex` 删除匹配整行，并在报告中查看 `custom_regex` 记录：

```bash
python cleanUP/clean_pdf.py --overwrite --drop-regex '正则表达式'
```

清洗器不自动纠正原书拼写。跨页段落根据句末标点推断，不能保证完全还原印刷版段落。
