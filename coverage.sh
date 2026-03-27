#!/bin/bash
# 生成测试覆盖率报告
# 用法：./coverage.sh [html|lcov|term]

set -e

OUTPUT_FORMAT="${1:-html}"

echo "🦀 生成 god-gragh 测试覆盖率报告..."
echo "输出格式：$OUTPUT_FORMAT"

case "$OUTPUT_FORMAT" in
    html)
        echo "生成 HTML 报告..."
        cargo tarpaulin --all-features --out Html --output-dir coverage
        echo "✅ HTML 报告已生成：coverage/tarpaulin-report.html"
        echo "查看报告：xdg-open coverage/tarpaulin-report.html"
        ;;
    lcov)
        echo "生成 LCOV 报告..."
        cargo tarpaulin --all-features --out Lcov
        echo "✅ LCOV 报告已生成：target/tarpaulin/lcov.info"
        ;;
    term)
        echo "在终端显示覆盖率..."
        cargo tarpaulin --all-features --out Term
        ;;
    *)
        echo "❌ 未知格式：$OUTPUT_FORMAT"
        echo "支持的格式：html, lcov, term"
        exit 1
        ;;
esac

echo ""
echo "📊 覆盖率统计完成！"
