#!/usr/bin/env python3
"""
Quick data collection for paper update
Uses smaller sample sizes to ensure completion
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("="*70)
print("QUICK DATA COLLECTION (Smaller Samples for Fast Results)")
print("="*70)
print()

# ============================================================================
# Step 1: Bitcoin Sample (100 blocks instead of 1000)
# ============================================================================
print("\n" + "="*70)
print("Step 1: Bitcoin 2025 Sample (100 blocks)")
print("="*70)

from download_2025_bitcoin_data import download_bitcoin_transactions, get_latest_block_height

try:
    current_height = get_latest_block_height()
    if current_height:
        print(f"✓ Current block: {current_height:,}")
        start_block = current_height - 100  # Only 100 blocks
        
        df_btc = download_bitcoin_transactions(
            start_block=start_block,
            num_blocks=100,
            output_file='bitcoin_2025_sample.csv'
        )
        
        if df_btc is not None:
            print(f"\n✅ Bitcoin sample collected: {len(df_btc):,} transactions")
            
            # Quick stats
            print(f"\nQuick Stats:")
            print(f"  Avg size: {df_btc['size'].mean():.1f} bytes")
            if 'fee' in df_btc.columns:
                print(f"  Avg fee: {df_btc['fee'].mean():.6f} BTC")
            if 'total_out' in df_btc.columns:
                print(f"  Total volume: {df_btc['total_out'].sum():.2f} BTC")
        else:
            print("✗ Bitcoin collection failed")
            sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# ============================================================================
# Step 2: Ethereum Sample (100 blocks)
# ============================================================================
print("\n" + "="*70)
print("Step 2: Ethereum 2025 Sample (100 blocks)")
print("="*70)

from download_ethereum_data import EtherscanDownloader
import os

try:
    api_key = os.getenv('ETHERSCAN_API_KEY')
    downloader = EtherscanDownloader(api_key)
    
    current_block = downloader.get_block_number()
    if current_block:
        print(f"✓ Current block: {current_block:,}")
        start_block = current_block - 100  # Only 100 blocks
        
        df_eth = downloader.download_ethereum_data(
            start_block=start_block,
            num_blocks=100,
            output_file='ethereum_2025_sample.csv'
        )
        
        if df_eth is not None:
            print(f"\n✅ Ethereum sample collected: {len(df_eth):,} transactions")
            
            # Quick stats
            print(f"\nQuick Stats:")
            if 'gas_price_gwei' in df_eth.columns:
                print(f"  Avg gas price: {df_eth['gas_price_gwei'].mean():.2f} Gwei")
            if 'value_eth' in df_eth.columns:
                print(f"  Total volume: {df_eth['value_eth'].sum():.2f} ETH")
            if 'is_contract' in df_eth.columns:
                print(f"  Contract txs: {df_eth['is_contract'].sum()} ({df_eth['is_contract'].mean()*100:.1f}%)")
        else:
            print("✗ Ethereum collection failed")
            sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# ============================================================================
# Step 3: Quick Comparison Analysis
# ============================================================================
print("\n" + "="*70)
print("Step 3: Quick BTC vs ETH Comparison")
print("="*70)

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Load data
    df_btc = pd.read_csv('bitcoin_2025_sample.csv')
    df_eth = pd.read_csv('ethereum_2025_sample.csv')
    
    print(f"\n📊 Comparison:")
    print(f"  BTC: {len(df_btc):,} transactions")
    print(f"  ETH: {len(df_eth):,} transactions")
    
    # Create simple comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Transaction count
    ax1 = axes[0, 0]
    ax1.bar(['Bitcoin', 'Ethereum'], [len(df_btc), len(df_eth)], 
           color=['#F7931A', '#627EEA'], alpha=0.7)
    ax1.set_ylabel('Transaction Count')
    ax1.set_title('Transaction Volume (100 blocks)')
    
    # 2. Average fees
    ax2 = axes[0, 1]
    btc_avg_fee = df_btc['fee'].mean() if 'fee' in df_btc.columns else 0
    eth_avg_fee = df_eth['tx_fee_eth'].mean() if 'tx_fee_eth' in df_eth.columns else 0
    ax2.bar(['BTC Fee', 'ETH Fee'], [btc_avg_fee, eth_avg_fee],
           color=['#F7931A', '#627EEA'], alpha=0.7)
    ax2.set_ylabel('Average Fee (native coin)')
    ax2.set_title('Average Transaction Fee')
    
    # 3. Transaction size/complexity
    ax3 = axes[1, 0]
    if 'size' in df_btc.columns:
        ax3.hist(df_btc['size'], bins=30, alpha=0.6, label='BTC Size', color='#F7931A')
    ax3.set_xlabel('Transaction Size (bytes)')
    ax3.set_ylabel('Count')
    ax3.set_title('Bitcoin Transaction Size Distribution')
    ax3.legend()
    
    # 4. Summary stats
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    Bitcoin 2025 Sample:
    • Transactions: {len(df_btc):,}
    • Avg Size: {df_btc['size'].mean():.1f} bytes
    • Avg Fee: {btc_avg_fee:.6f} BTC
    
    Ethereum 2025 Sample:
    • Transactions: {len(df_eth):,}
    • Avg Gas Price: {df_eth['gas_price_gwei'].mean():.2f} Gwei
    • Avg Fee: {eth_avg_fee:.6f} ETH
    • Contract Txs: {df_eth['is_contract'].mean()*100:.1f}%
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.suptitle('Bitcoin vs Ethereum 2025 Quick Comparison', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = '../fig/btc_eth_2025_quick_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved: {output_path}")
    plt.close()
    
except Exception as e:
    print(f"✗ Comparison failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("✅ QUICK DATA COLLECTION COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  📊 bitcoin_2025_sample.csv")
print("  📊 ethereum_2025_sample.csv")
print("  📈 ../fig/btc_eth_2025_quick_comparison.png")
print("\nNote: Using 100-block samples for faster completion.")
print("Full 1000-block analysis can be run separately if needed.")
print("="*70)

