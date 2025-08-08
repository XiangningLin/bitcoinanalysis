#!/usr/bin/env python3
"""
Download Ethereum transaction data using Etherscan API
Free tier: 5 calls/sec, 100,000 calls/day
"""
import requests
import pandas as pd
import time
from datetime import datetime
import json

class EtherscanDownloader:
    """Download Ethereum blockchain data via Etherscan API"""
    
    def __init__(self, api_key=None):
        """
        Initialize downloader
        
        Args:
            api_key: Etherscan API key (get free at https://etherscan.io/apis)
                    If None, uses public API (slower, rate limited)
        """
        self.api_key = api_key or "YourApiKeyToken"  # Replace with your key
        self.base_url = "https://api.etherscan.io/api"
        self.rate_limit = 0.2  # 5 calls/sec for free tier
        
    def get_block_number(self, timestamp=None):
        """Get block number by timestamp or latest"""
        if timestamp:
            params = {
                'module': 'block',
                'action': 'getblocknobytime',
                'timestamp': timestamp,
                'closest': 'before',
                'apikey': self.api_key
            }
        else:
            params = {
                'module': 'proxy',
                'action': 'eth_blockNumber',
                'apikey': self.api_key
            }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if timestamp:
                return int(data.get('result', 0))
            else:
                # Convert hex to int
                return int(data.get('result', '0x0'), 16)
        except Exception as e:
            print(f"Error getting block number: {e}")
            return None
    
    def get_block_transactions(self, block_number):
        """Get all transactions in a block"""
        params = {
            'module': 'proxy',
            'action': 'eth_getBlockByNumber',
            'tag': hex(block_number),
            'boolean': 'true',
            'apikey': self.api_key
        }
        
        try:
            time.sleep(self.rate_limit)
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if 'result' in data and data['result']:
                return data['result']
            return None
        except Exception as e:
            print(f"Error getting block {block_number}: {e}")
            return None
    
    def extract_tx_features(self, tx, block_data):
        """Extract features from a transaction"""
        try:
            return {
                'hash': tx.get('hash', ''),
                'block_number': int(tx.get('blockNumber', '0x0'), 16),
                'block_timestamp': int(block_data.get('timestamp', '0x0'), 16),
                'from': tx.get('from', ''),
                'to': tx.get('to', ''),
                'value_wei': int(tx.get('value', '0x0'), 16),
                'value_eth': int(tx.get('value', '0x0'), 16) / 1e18,
                'gas': int(tx.get('gas', '0x0'), 16),
                'gas_price_wei': int(tx.get('gasPrice', '0x0'), 16),
                'gas_price_gwei': int(tx.get('gasPrice', '0x0'), 16) / 1e9,
                'tx_fee_eth': (int(tx.get('gas', '0x0'), 16) * 
                              int(tx.get('gasPrice', '0x0'), 16)) / 1e18,
                'input_size': len(tx.get('input', '0x')) // 2,  # bytes
                'nonce': int(tx.get('nonce', '0x0'), 16),
                'transaction_index': int(tx.get('transactionIndex', '0x0'), 16),
                'is_contract': 1 if tx.get('to') is None else 0
            }
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def download_ethereum_data(self, start_block, num_blocks=1000, 
                              output_file='ethereum_2025.csv'):
        """
        Download Ethereum transactions
        
        Args:
            start_block: Starting block number
            num_blocks: Number of blocks to download
            output_file: Output CSV filename
        """
        print(f"Starting Ethereum data download...")
        print(f"Start block: {start_block}")
        print(f"Number of blocks: {num_blocks}")
        print(f"Rate limit: {1/self.rate_limit:.1f} calls/sec")
        
        all_transactions = []
        failed_blocks = []
        blocks_with_data = 0
        
        start_time = time.time()
        
        for i in range(num_blocks):
            block_number = start_block + i
            
            print(f"Block {block_number} ({i+1}/{num_blocks}) - "
                  f"{len(all_transactions)} txs", end='\r')
            
            block_data = self.get_block_transactions(block_number)
            
            if block_data is None:
                failed_blocks.append(block_number)
                continue
            
            transactions = block_data.get('transactions', [])
            
            if transactions:
                blocks_with_data += 1
                for tx in transactions:
                    features = self.extract_tx_features(tx, block_data)
                    if features:
                        all_transactions.append(features)
            
            # Save intermediate results every 100 blocks
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (num_blocks - i - 1) / rate
                
                print(f"\nProgress: {i+1}/{num_blocks} blocks "
                      f"({blocks_with_data} with data)")
                print(f"  Transactions: {len(all_transactions)}")
                print(f"  Rate: {rate:.1f} blocks/sec")
                print(f"  ETA: {eta/60:.1f} minutes")
                
                if all_transactions:
                    df_temp = pd.DataFrame(all_transactions)
                    temp_file = output_file.replace('.csv', f'_temp_{block_number}.csv')
                    df_temp.to_csv(temp_file, index=False)
        
        # Save final results
        if all_transactions:
            df = pd.DataFrame(all_transactions)
            df.to_csv(output_file, index=False)
            
            elapsed = time.time() - start_time
            
            print(f"\n\n{'='*60}")
            print(f"✓ Download Complete!")
            print(f"{'='*60}")
            print(f"Transactions: {len(df):,}")
            print(f"Blocks processed: {num_blocks}")
            print(f"Blocks with data: {blocks_with_data}")
            print(f"Failed blocks: {len(failed_blocks)}")
            print(f"Time elapsed: {elapsed/60:.1f} minutes")
            print(f"Output file: {output_file}")
            
            print(f"\n📊 Summary Statistics:")
            print(f"  Date range: {datetime.fromtimestamp(df['block_timestamp'].min())} "
                  f"to {datetime.fromtimestamp(df['block_timestamp'].max())}")
            print(f"  Block range: {df['block_number'].min()} to {df['block_number'].max()}")
            print(f"  Total ETH transferred: {df['value_eth'].sum():.2f} ETH")
            print(f"  Avg transaction value: {df['value_eth'].mean():.4f} ETH")
            print(f"  Avg gas price: {df['gas_price_gwei'].mean():.2f} Gwei")
            print(f"  Avg tx fee: {df['tx_fee_eth'].mean():.6f} ETH")
            print(f"  Contract creations: {df['is_contract'].sum()}")
            print(f"  Unique senders: {df['from'].nunique():,}")
            print(f"  Unique receivers: {df['to'].nunique():,}")
            
            return df
        else:
            print("\n✗ No transactions downloaded")
            return None

def main():
    """Main execution"""
    print("="*60)
    print("Ethereum Transaction Downloader (Etherscan API)")
    print("="*60)
    print()
    print("⚠️  Note: You need an Etherscan API key for optimal performance")
    print("   Get free key at: https://etherscan.io/apis")
    print()
    
    # Get API key
    api_key = input("Enter your Etherscan API key (or press Enter to skip): ").strip()
    if not api_key:
        print("⚠️  Using public API (slower, limited)")
        api_key = None
    
    downloader = EtherscanDownloader(api_key)
    
    # Get current block
    current_block = downloader.get_block_number()
    if current_block:
        print(f"\n📍 Current Ethereum block: {current_block:,}")
        
        # Suggest recent blocks
        suggested_start = current_block - 1000
        print(f"📅 Suggested start: {suggested_start:,} (last 1000 blocks)")
        print(f"   Estimated time: ~{1000 * 0.2 / 60:.1f} minutes")
    else:
        print("⚠️  Could not get current block. Using default.")
        suggested_start = 21000000  # Approximate Jan 2025
    
    # User input
    use_suggested = input(f"\nUse suggested start block {suggested_start:,}? (y/n): ").lower()
    
    if use_suggested == 'y':
        start_block = suggested_start
    else:
        start_block = int(input("Enter start block number: "))
    
    num_blocks = int(input("Enter number of blocks (default 1000): ") or "1000")
    output_file = input("Enter output filename (default 'ethereum_2025.csv'): ") or "ethereum_2025.csv"
    
    print(f"\n🚀 Starting download...")
    print(f"   Start block: {start_block:,}")
    print(f"   Number of blocks: {num_blocks}")
    print(f"   Output: {output_file}")
    print(f"   Estimated time: ~{num_blocks * 0.2 / 60:.1f} minutes")
    
    confirm = input("\nProceed? (y/n): ").lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Download
    df = downloader.download_ethereum_data(start_block, num_blocks, output_file)
    
    if df is not None:
        print("\n✅ Download complete!")
        print(f"\n📁 Output: {output_file}")
        print(f"📊 Transactions: {len(df):,}")
        print("\nYou can now run comparison analysis:")
        print(f"  python3 compare_btc_eth.py")
    else:
        print("\n❌ Download failed")

if __name__ == "__main__":
    main()

