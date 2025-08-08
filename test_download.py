#!/usr/bin/env python3
"""快速测试：生成一个小的2025年格式数据样本"""

from download_2025_bitcoin_data import BitcoinDataDownloader

# 创建下载器
downloader = BitcoinDataDownloader(output_dir=".")

# 生成1000条测试数据
print("生成测试数据...")
transactions = downloader.create_synthetic_data(n_transactions=1000, date="2025-01-15")

# 保存为与2020年相同的格式
downloader.save_to_csv_and_zip(transactions, filename="2025-01_test")

print("\n✅ 测试完成！可以对比查看:")
print("   原始2020: unzip -p ../2020-10_00.zip 2020-10_00.csv | head -3")
print("   新生成2025: unzip -p 2025-01_test.zip 2025-01_test.csv | head -3")

