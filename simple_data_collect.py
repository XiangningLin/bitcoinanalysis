#!/usr/bin/env python3
"""
Simple and Direct Data Collection for Paper Update
Gets REAL data from blockchain APIs
"""
import requests
import pandas as pd
import time
from datetime import datetime

print("="*70)
print("SIMPLE & DIRECT DATA COLLECTION")
print("="*70)

# ============================================================================
# Bitcoin Data Collection
# ============================================================================
print("\n📥 Collecting Bitcoin 2025 Data...")

def get_bitcoin_latest_block():
    """Get current Bitcoin block height"""
    try:
        response = requests.get("https://blockchain.info/q/getblockcount", timeout=10)
        return int(response.text.strip())
    except:
        return None

def collect_bitcoin_sample(num_blocks=50):
    """Collect Bitcoin transaction sample"""
    current_height = get_bitcoin_latest_block()
    
    if not current_height:
        print("❌ Could not get current block height")
        return None
    
    print(f"✓ Current block: {current_height:,}")
    print(f"✓ Collecting last {num_blocks} blocks...")
    
    start_block = current_height - num_blocks
    transactions = []
    
    for i in range(num_blocks):
        block_num = start_block + i
        print(f"  Block {block_num} ({i+1}/{num_blocks})", end='\r', flush=True)
        
        try:
            url = f"https://blockchain.info/rawblock/{block_num}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                block_data = response.json()
                
                for tx in block_data.get('tx', []):
                    # Extract transaction features
                    tx_hash = tx.get('hash', '')
                    size = tx.get('size', 0)
                    time_stamp = block_data.get('time', 0)
                    
                    # Count inputs/outputs
                    n_inputs = len(tx.get('inputs', []))
                    n_outputs = len(tx.get('out', []))
                    
                    # Calculate values
                    total_out = sum(out.get('value', 0) for out in tx.get('out', [])) / 1e8
                    
                    # Estimate fee (if coinbase tx, fee is 0)
                    is_coinbase = any(inp.get('prev_out') is None for inp in tx.get('inputs', []))
                    if is_coinbase:
                        fee = 0
                    else:
                        total_in = sum(
                            inp.get('prev_out', {}).get('value', 0) 
                            for inp in tx.get('inputs', [])
                            if inp.get('prev_out')
                        ) / 1e8
                        fee = total_in - total_out if total_in > 0 else 0
                    
                    transactions.append({
                        'hash': tx_hash,
                        'block_height': block_num,
                        'timestamp': time_stamp,
                        'size': size,
                        'inputs': n_inputs,
                        'outputs': n_outputs,
                        'total_out_btc': total_out,
                        'fee_btc': fee
                    })
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"\n  ⚠️  Block {block_num} error: {e}")
            continue
    
    print(f"\n✓ Collected {len(transactions)} transactions")
    return pd.DataFrame(transactions)

# Collect Bitcoin data
df_btc = collect_bitcoin_sample(50)  # 50 blocks for speed

if df_btc is not None and not df_btc.empty:
    # Save
    df_btc.to_csv('bitcoin_2025_real.csv', index=False)
    print(f"✓ Saved: bitcoin_2025_real.csv ({len(df_btc)} txs)")
    
    # Stats
    print(f"\nBitcoin 2025 Statistics:")
    print(f"  Transactions: {len(df_btc):,}")
    print(f"  Avg size: {df_btc['size'].mean():.1f} bytes")
    print(f"  Avg inputs: {df_btc['inputs'].mean():.2f}")
    print(f"  Avg outputs: {df_btc['outputs'].mean():.2f}")
    print(f"  Avg fee: {df_btc['fee_btc'].mean():.6f} BTC")
    print(f"  Total volume: {df_btc['total_out_btc'].sum():.2f} BTC")
else:
    print("❌ Bitcoin collection failed")

# ============================================================================
# Ethereum Data Collection
# ============================================================================
print("\n📥 Collecting Ethereum 2025 Data...")

import os
from dotenv import load_dotenv
load_dotenv()

ETHERSCAN_KEY = os.getenv('ETHERSCAN_API_KEY', '')

def collect_ethereum_sample(num_blocks=50):
    """Collect Ethereum transaction sample"""
    if not ETHERSCAN_KEY:
        print("⚠️  No Etherscan API key, skipping Ethereum")
        return None
    
    # Get latest block
    try:
        url = "https://api.etherscan.io/api"
        params = {
            'module': 'proxy',
            'action': 'eth_blockNumber',
            'apikey': ETHERSCAN_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        current_block = int(response.json()['result'], 16)
        
        print(f"✓ Current block: {current_block:,}")
        print(f"✓ Collecting last {num_blocks} blocks...")
        
    except Exception as e:
        print(f"❌ Could not get current block: {e}")
        return None
    
    start_block = current_block - num_blocks
    transactions = []
    
    for i in range(num_blocks):
        block_num = start_block + i
        print(f"  Block {block_num} ({i+1}/{num_blocks})", end='\r', flush=True)
        
        try:
            params = {
                'module': 'proxy',
                'action': 'eth_getBlockByNumber',
                'tag': hex(block_num),
                'boolean': 'true',
                'apikey': ETHERSCAN_KEY
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'result' in data and data['result']:
                block = data['result']
                
                for tx in block.get('transactions', []):
                    transactions.append({
                        'hash': tx.get('hash', ''),
                        'block_number': int(tx.get('blockNumber', '0x0'), 16),
                        'timestamp': int(block.get('timestamp', '0x0'), 16),
                        'from': tx.get('from', ''),
                        'to': tx.get('to', '') if tx.get('to') else 'CONTRACT_CREATION',
                        'value_eth': int(tx.get('value', '0x0'), 16) / 1e18,
                        'gas': int(tx.get('gas', '0x0'), 16),
                        'gas_price_gwei': int(tx.get('gasPrice', '0x0'), 16) / 1e9
                    })
            
            time.sleep(0.2)  # 5 req/sec limit
            
        except Exception as e:
            print(f"\n  ⚠️  Block {block_num} error: {e}")
            continue
    
    print(f"\n✓ Collected {len(transactions)} transactions")
    return pd.DataFrame(transactions)

# Collect Ethereum data
df_eth = collect_ethereum_sample(50)  # 50 blocks

if df_eth is not None and not df_eth.empty:
    # Save
    df_eth.to_csv('ethereum_2025_real.csv', index=False)
    print(f"✓ Saved: ethereum_2025_real.csv ({len(df_eth)} txs)")
    
    # Stats
    print(f"\nEthereum 2025 Statistics:")
    print(f"  Transactions: {len(df_eth):,}")
    print(f"  Avg gas: {df_eth['gas'].mean():.0f}")
    print(f"  Avg gas price: {df_eth['gas_price_gwei'].mean():.2f} Gwei")
    print(f"  Total volume: {df_eth['value_eth'].sum():.2f} ETH")
    print(f"  Contract creations: {(df_eth['to'] == 'CONTRACT_CREATION').sum()}")
else:
    print("❌ Ethereum collection failed (or skipped)")

# ============================================================================
# Quick Comparison
# ============================================================================
if df_btc is not None and df_eth is not None:
    print("\n" + "="*70)
    print("📊 QUICK COMPARISON: Bitcoin vs Ethereum (2025)")
    print("="*70)
    
    print(f"\nNetwork Activity:")
    print(f"  BTC: {len(df_btc):,} transactions in {df_btc['block_height'].nunique()} blocks")
    print(f"  ETH: {len(df_eth):,} transactions in {df_eth['block_number'].nunique()} blocks")
    
    print(f"\nAverage Transaction Complexity:")
    print(f"  BTC: {df_btc['inputs'].mean():.1f} inputs, {df_btc['outputs'].mean():.1f} outputs")
    print(f"  ETH: {df_eth['gas'].mean():.0f} gas per transaction")
    
    print(f"\nTransaction Fees:")
    print(f"  BTC: {df_btc['fee_btc'].mean():.6f} BTC")
    print(f"  ETH: {(df_eth['gas'] * df_eth['gas_price_gwei'] / 1e9).mean():.6f} ETH")

print("\n" + "="*70)
print("✅ DATA COLLECTION COMPLETE")
print("="*70)
print("\nGenerated files:")
if df_btc is not None:
    print(f"  ✓ bitcoin_2025_real.csv ({len(df_btc)} transactions)")
if df_eth is not None:
    print(f"  ✓ ethereum_2025_real.csv ({len(df_eth)} transactions)")
print("\nAll data is REAL from blockchain APIs (Oct 2025)")
print("="*70)

