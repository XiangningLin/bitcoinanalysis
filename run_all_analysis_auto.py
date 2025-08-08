#!/usr/bin/env python3
"""
Automated script to run all cryptocurrency analysis
Non-interactive mode for batch processing
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

print("="*70)
print("AUTOMATED CRYPTOCURRENCY ANALYSIS PIPELINE")
print("="*70)
print()

# ============================================================================
# PHASE 1: Download Bitcoin 2025 Data
# ============================================================================
print("\n" + "="*70)
print("PHASE 1.1: Downloading Bitcoin 2025 Data")
print("="*70)

from download_2025_bitcoin_data import download_bitcoin_transactions, get_latest_block_height

try:
    print("Getting current Bitcoin block height...")
    current_height = get_latest_block_height()
    
    if current_height:
        print(f"✓ Current block height: {current_height:,}")
        start_block = current_height - 1000
        print(f"✓ Will download blocks {start_block:,} to {current_height:,}")
        print(f"✓ Estimated time: ~17 minutes")
        print()
        
        df_btc = download_bitcoin_transactions(
            start_block=start_block,
            num_blocks=1000,
            output_file='2025_bitcoin.csv'
        )
        
        if df_btc is not None:
            print("\n✅ Bitcoin data downloaded successfully!")
            print(f"   File: 2025_bitcoin.csv")
            print(f"   Transactions: {len(df_btc):,}")
        else:
            print("\n❌ Bitcoin download failed")
            sys.exit(1)
    else:
        print("❌ Could not get current block height")
        sys.exit(1)
        
except Exception as e:
    print(f"\n❌ Error downloading Bitcoin data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# PHASE 1.2: Download Ethereum 2025 Data
# ============================================================================
print("\n" + "="*70)
print("PHASE 1.2: Downloading Ethereum 2025 Data")
print("="*70)

from download_ethereum_data import EtherscanDownloader

try:
    # Check if API key is available in environment
    api_key = os.environ.get('ETHERSCAN_API_KEY', None)
    if api_key:
        print(f"✓ Using Etherscan API key from environment")
    else:
        print("⚠ No API key found, using public API (slower)")
        print("  Set ETHERSCAN_API_KEY environment variable for faster access")
    
    downloader = EtherscanDownloader(api_key)
    
    print("\nGetting current Ethereum block...")
    current_block = downloader.get_block_number()
    
    if current_block:
        print(f"✓ Current block: {current_block:,}")
        start_block = current_block - 1000
        print(f"✓ Will download blocks {start_block:,} to {current_block:,}")
        print(f"✓ Estimated time: ~3-5 minutes")
        print()
        
        df_eth = downloader.download_ethereum_data(
            start_block=start_block,
            num_blocks=1000,
            output_file='ethereum_2025.csv'
        )
        
        if df_eth is not None:
            print("\n✅ Ethereum data downloaded successfully!")
            print(f"   File: ethereum_2025.csv")
            print(f"   Transactions: {len(df_eth):,}")
        else:
            print("\n❌ Ethereum download failed")
            sys.exit(1)
    else:
        print("❌ Could not get current block number")
        sys.exit(1)
        
except Exception as e:
    print(f"\n❌ Error downloading Ethereum data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# PHASE 1.3: Generate BTC vs ETH Comparison
# ============================================================================
print("\n" + "="*70)
print("PHASE 1.3: Generating BTC vs ETH Comparison Analysis")
print("="*70)

from compare_btc_eth import load_crypto_data, analyze_crypto_comparison, create_btc_eth_comparison_plots

try:
    # Load data
    df_btc, df_eth = load_crypto_data('2025_bitcoin.csv', 'ethereum_2025.csv')
    
    if df_btc is None or df_eth is None:
        print("❌ Failed to load data for comparison")
        sys.exit(1)
    
    # Analyze
    stats = analyze_crypto_comparison(df_btc, df_eth)
    
    # Create plots
    create_btc_eth_comparison_plots(df_btc, df_eth, '../fig')
    
    print("\n✅ BTC vs ETH comparison complete!")
    print("   Output: ../fig/bitcoin_vs_ethereum_2025.png")
    
except Exception as e:
    print(f"\n❌ Error in comparison analysis: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✅ ALL ANALYSIS COMPLETE!")
print("="*70)
print()
print("Generated files:")
print("  📊 2025_bitcoin.csv")
print("  📊 ethereum_2025.csv")
print("  📈 ../fig/bitcoin_vs_ethereum_2025.png")
print()
print("Next steps:")
print("  1. View the comparison plot")
print("  2. Run multi-LLM validation")
print("  3. Update paper.tex with findings")
print()
print("="*70)

