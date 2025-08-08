#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and process 2025 Bitcoin transaction data
Output format: timestamp,source_address,destination_address,satoshi
Compatible with 2020-10_00.csv format
"""

import requests
import pandas as pd
import time
import json
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

class BitcoinDataDownloader:
    """
    Download Bitcoin transaction data from multiple sources
    """
    
    def __init__(self, output_dir="bitcoin_data_2025"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def download_from_blockchain_info(self, start_date="2025-01-01", days=1, max_blocks=100):
        """
        方法1: 使用 Blockchain.info API
        
        Args:
            start_date: 起始日期 YYYY-MM-DD
            days: 下载天数
            max_blocks: 最大区块数（限制数据量）
        """
        print(f"📥 Method 1: Downloading from Blockchain.info API...")
        print(f"   Date range: {start_date} + {days} days")
        print(f"   Block limit: {max_blocks} blocks\n")
        
        # 1. 获取起始日期的区块高度
        # 2025年1月1日大约是区块 #875000 (估算)
        # 每个区块约10分钟，每天约144个区块
        estimated_block_height = 875000  # 需要根据实际日期调整
        
        transactions = []
        
        for block_num in range(estimated_block_height, estimated_block_height + max_blocks):
            try:
                print(f"   Fetching block {block_num}...", end='\r')
                
                # Blockchain.info API
                url = f"https://blockchain.info/block-height/{block_num}?format=json"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    block_data = response.json()
                    
                    # 处理区块中的所有交易
                    for block in block_data.get('blocks', []):
                        timestamp = block.get('time', 0)
                        
                        for tx in block.get('tx', []):
                            # 提取交易的输入和输出
                            inputs = tx.get('inputs', [])
                            outputs = tx.get('out', [])
                            
                            # 构建交易边
                            for inp in inputs:
                                source_addr = inp.get('prev_out', {}).get('addr', '')
                                if not source_addr:
                                    continue
                                    
                                for out in outputs:
                                    dest_addr = out.get('addr', '')
                                    value = out.get('value', 0)
                                    
                                    if dest_addr and value > 0:
                                        transactions.append({
                                            'timestamp': timestamp,
                                            'source_address': source_addr,
                                            'destination_address': dest_addr,
                                            'satoshi': value
                                        })
                    
                    time.sleep(0.5)  # 避免API限流
                    
                else:
                    print(f"\n   ⚠️  Block {block_num} failed: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"\n   ❌ Error at block {block_num}: {e}")
                continue
        
        print(f"\n✅ Downloaded {len(transactions)} transactions")
        return transactions
    
    def download_from_bigquery(self, start_date="2025-01-01", end_date="2025-01-31", limit=1000000):
        """
        方法2: 使用 Google BigQuery (需要GCP凭证)
        
        Args:
            start_date: 起始日期
            end_date: 结束日期
            limit: 最大交易数
        """
        print(f"📥 Method 2: Downloading from Google BigQuery...")
        print(f"   ⚠️  需要 Google Cloud 凭证和 BigQuery API 权限\n")
        
        try:
            from google.cloud import bigquery
            
            client = bigquery.Client()
            
            query = f"""
            SELECT 
                UNIX_SECONDS(block_timestamp) as timestamp,
                inputs.addresses[SAFE_OFFSET(0)] as source_address,
                outputs.addresses[SAFE_OFFSET(0)] as destination_address,
                outputs.value as satoshi
            FROM `bigquery-public-data.crypto_bitcoin.transactions`,
                UNNEST(inputs) as inputs,
                UNNEST(outputs) as outputs
            WHERE DATE(block_timestamp) BETWEEN '{start_date}' AND '{end_date}'
                AND inputs.addresses[SAFE_OFFSET(0)] IS NOT NULL
                AND outputs.addresses[SAFE_OFFSET(0)] IS NOT NULL
                AND outputs.value > 0
            LIMIT {limit}
            """
            
            print("   Running query...")
            df = client.query(query).to_dataframe()
            print(f"✅ Downloaded {len(df)} transactions")
            
            return df.to_dict('records')
            
        except ImportError:
            print("   ❌ 需要安装: pip install google-cloud-bigquery")
            return []
        except Exception as e:
            print(f"   ❌ BigQuery error: {e}")
            return []
    
    def download_from_coinmetrics(self, asset="btc", date="2025-01-01", api_key=None):
        """
        方法3: 使用 Coin Metrics API (需要API key)
        
        Args:
            asset: 资产代码 (btc)
            date: 日期 YYYY-MM-DD
            api_key: Coin Metrics API key
        """
        print(f"📥 Method 3: Downloading from Coin Metrics API...")
        print(f"   ⚠️  需要 Coin Metrics API key (community或pro版)\n")
        
        if not api_key:
            print("   ❌ 请提供 api_key 参数")
            return []
        
        # Coin Metrics API endpoint
        url = f"https://api.coinmetrics.io/v4/blockchain/{asset}/transactions"
        
        params = {
            "api_key": api_key,
            "start_time": f"{date}T00:00:00Z",
            "end_time": f"{date}T23:59:59Z",
            "page_size": 10000
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            # 处理返回的交易数据
            transactions = []
            # ... (需要根据API响应格式处理)
            
            return transactions
            
        except Exception as e:
            print(f"   ❌ Coin Metrics error: {e}")
            return []
    
    def create_synthetic_data(self, n_transactions=100000, date="2025-01-01"):
        """
        方法4: 生成合成数据用于测试
        注意：这不是真实数据，仅用于测试代码流程
        """
        print(f"🔧 Method 4: Generating synthetic test data...")
        print(f"   Transactions: {n_transactions}")
        print(f"   ⚠️  这是合成数据，不是真实比特币交易!\n")
        
        import random
        
        # 生成随机地址
        def random_address():
            chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
            return '1' + ''.join(random.choices(chars, k=33))
        
        # 生成一些"重要节点"地址（模拟交易所、矿池等）
        hub_addresses = [random_address() for _ in range(100)]
        all_addresses = hub_addresses + [random_address() for _ in range(1000)]
        
        base_timestamp = int(datetime.strptime(date, "%Y-%m-%d").timestamp())
        
        transactions = []
        for i in range(n_transactions):
            # 80%的交易涉及hub地址（模拟真实网络结构）
            if random.random() < 0.8:
                source = random.choice(hub_addresses if random.random() < 0.5 else all_addresses)
                dest = random.choice(hub_addresses if random.random() < 0.5 else all_addresses)
            else:
                source = random.choice(all_addresses)
                dest = random.choice(all_addresses)
            
            transactions.append({
                'timestamp': base_timestamp + random.randint(0, 86400),  # 一天内的随机时间
                'source_address': source,
                'destination_address': dest,
                'satoshi': random.randint(1000, 100000000)  # 0.00001 - 1 BTC
            })
        
        print(f"✅ Generated {len(transactions)} synthetic transactions")
        return transactions
    
    def save_to_csv_and_zip(self, transactions, filename="2025-01_00"):
        """
        保存为CSV并压缩为ZIP（与2020年数据格式一致）
        """
        if not transactions:
            print("❌ No transactions to save")
            return
        
        print(f"\n💾 Saving data...")
        
        # 转换为DataFrame
        df = pd.DataFrame(transactions)
        
        # 确保列顺序一致
        df = df[['timestamp', 'source_address', 'destination_address', 'satoshi']]
        
        # 保存CSV
        csv_path = self.output_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        print(f"   ✅ Saved CSV: {csv_path}")
        print(f"   📊 Rows: {len(df):,}")
        print(f"   💽 Size: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # 压缩为ZIP
        zip_path = self.output_dir / f"{filename}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(csv_path, arcname=f"{filename}.csv")
        
        print(f"   ✅ Saved ZIP: {zip_path}")
        print(f"   💽 Compressed size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # 显示数据预览
        print(f"\n📋 Data preview:")
        print(df.head(3).to_string(index=False))
        
        return csv_path, zip_path


def main():
    """
    主函数：下载2025年比特币数据
    """
    print("=" * 70)
    print("  Bitcoin 2025 Transaction Data Downloader")
    print("  Output format: timestamp,source_address,destination_address,satoshi")
    print("=" * 70)
    print()
    
    downloader = BitcoinDataDownloader()
    
    # 选择下载方法
    print("可用方法:")
    print("  1. Blockchain.info API (免费，有限流)")
    print("  2. Google BigQuery (需要GCP凭证)")
    print("  3. Coin Metrics API (需要API key)")
    print("  4. 生成合成测试数据 (仅用于测试)")
    print()
    
    method = input("选择方法 (1-4) [默认: 4]: ").strip() or "4"
    
    transactions = []
    
    if method == "1":
        # Blockchain.info API
        start_date = input("起始日期 (YYYY-MM-DD) [默认: 2025-01-01]: ").strip() or "2025-01-01"
        max_blocks = int(input("最大区块数 [默认: 100]: ").strip() or "100")
        transactions = downloader.download_from_blockchain_info(start_date=start_date, max_blocks=max_blocks)
        
    elif method == "2":
        # BigQuery
        start_date = input("起始日期 (YYYY-MM-DD) [默认: 2025-01-01]: ").strip() or "2025-01-01"
        end_date = input("结束日期 (YYYY-MM-DD) [默认: 2025-01-31]: ").strip() or "2025-01-31"
        limit = int(input("最大交易数 [默认: 1000000]: ").strip() or "1000000")
        transactions = downloader.download_from_bigquery(start_date, end_date, limit)
        
    elif method == "3":
        # Coin Metrics
        date = input("日期 (YYYY-MM-DD) [默认: 2025-01-01]: ").strip() or "2025-01-01"
        api_key = input("Coin Metrics API key: ").strip()
        transactions = downloader.download_from_coinmetrics(date=date, api_key=api_key)
        
    else:
        # 合成数据
        n_trans = int(input("生成交易数 [默认: 100000]: ").strip() or "100000")
        date = input("日期 (YYYY-MM-DD) [默认: 2025-01-01]: ").strip() or "2025-01-01"
        transactions = downloader.create_synthetic_data(n_transactions=n_trans, date=date)
    
    # 保存数据
    if transactions:
        filename = input("\n输出文件名 (不含扩展名) [默认: 2025-01_00]: ").strip() or "2025-01_00"
        downloader.save_to_csv_and_zip(transactions, filename)
        
        print("\n" + "=" * 70)
        print("✅ 完成！数据已保存，可以使用 bitcoin_rwfb_llm4tg.py 处理")
        print("=" * 70)
    else:
        print("\n❌ 没有下载到数据")


if __name__ == "__main__":
    main()
