# PDF 正文清洗

针对本项目 Book 中七本单栏、有文字层的 PDF 编写，原始 PDF 不会修改。

## 运行

在 KE-project 目录执行：

```bash
python cleanUP/clean_pdf.py
# 重新生成已有结果
python cleanUP/clean_pdf.py --overwrite
# 处理单本书或其他输入目录
python cleanUP/clean_pdf.py --input "Book/Book 1 - The Philosopher's Stone.pdf" --output cleanUP/sample
```

依赖 Python 3 和系统命令 pdftotext（服务器已安装），不依赖额外 Python 包。
缺少命令时安装：`sudo apt-get install poppler-utils`。

## 输出

- `cleanUP/txt/*.txt`：每本书一个 UTF-8 文本，章节标题与正文用空行分隔。
- `cleanUP/txt/*.report.json`：源文件 SHA256、页数、章节起止页、删除行及原因。
- 默认拒绝覆盖已有文件；只有显式传入 `--overwrite` 才会替换生成结果。

## 清洗规则与限制

使用文字坐标和行高识别居中章节标题、重复页眉页脚、页码，以及底部带数字或符号标记的小字号脚注；跳过首章前内容。图片不会被提取。按行间距恢复段落并连接折行，保留正文大写对白、引语和书信。

这是一套针对当前版本的版式规则，并非任意 PDF 通用识别器。没有文字层的扫描 PDF 需要先 OCR；夹在文字页中的纯图片页会被跳过。图片中的文字不会 OCR，但独立的图注、表格文本、无标记脚注以及正文后的附录并不保证自动删除。发现残留时可用重复的 `--drop-regex '正则表达式'` 删除匹配整行，并查看报告中 custom_regex 记录。

不自动纠正原书拼写；保留行尾连字符以免破坏 low-growing 等复合词，因此排版产生的单词断词仍可能需要后续校正。跨页段落通过句末标点推断，不能保证完全还原原书段落。清洗报告保留被删除原文，适合抽样审查。

当前七本书识别的标题数依次为 17、18、22、37、38、30、37；第七本包含尾声 NINETEEN YEARS LATER。输出正文中未检出 Page | 页脚标记。
