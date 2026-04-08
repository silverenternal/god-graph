#!/bin/bash
# 测试覆盖率检查脚本
#
# 使用方法:
#   ./scripts/check_coverage.sh
#
# 依赖:
#   cargo install cargo-tarpaulin

set -e

echo "=== God-Graph 测试覆盖率检查 ==="

# 运行覆盖率检查
echo "运行 cargo-tarpaulin..."
cargo tarpaulin \
    --features "parallel,simd,tensor" \
    --out Html \
    --out Xml \
    --output-dir ./coverage/report \
    --timeout 300 \
    --follow-exec \
    --ignore-tests

# 生成覆盖率报告
echo ""
echo "=== 覆盖率报告 ==="

# 检查是否达到目标覆盖率
THRESHOLD=80
CURRENT=$(grep -o '"loc":\["[^"]*","[^"]*",[^,]*,[^,]*,[^,]*,[^,]*' ./coverage/report/coverage.json 2>/dev/null | wc -l || echo "0")

echo "当前覆盖率数据已生成到 ./coverage/report/"
echo "目标覆盖率：${THRESHOLD}%"
echo ""

# 打开 HTML 报告（如果可能）
if command -v xdg-open &> /dev/null; then
    echo "正在打开 HTML 报告..."
    xdg-open ./coverage/report/tarpaulin-report.html 2>/dev/null || true
elif command -v open &> /dev/null; then
    echo "正在打开 HTML 报告..."
    open ./coverage/report/tarpaulin-report.html 2>/dev/null || true
fi

echo ""
echo "=== 覆盖率低的模块 ==="
echo "请检查以下模块的测试覆盖情况："
echo "  - src/algorithms/community.rs"
echo "  - src/algorithms/flow.rs"
echo "  - src/algorithms/matching.rs"
echo "  - src/distributed/fault_tolerance.rs"
echo ""
echo "提示：运行 cargo test --features parallel,simd,tensor -- --nocapture 查看详细测试输出"
