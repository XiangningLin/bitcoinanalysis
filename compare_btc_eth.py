#!/usr/bin/env python3
"""
Compare Bitcoin vs Ethereum transaction patterns
Multi-cryptocurrency analysis
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 14)
plt.rcParams['font.size'] = 10

def load_crypto_data(btc_file, eth_file):
    """Load Bitcoin and Ethereum datasets"""
    print("Loading data...")
    
    df_btc = None
    df_eth = None
    
    # Load Bitcoin
    try:
        df_btc = pd.read_csv(btc_file)
        print(f"✓ Loaded Bitcoin data: {len(df_btc):,} transactions")
    except Exception as e:
        print(f"✗ Error loading Bitcoin data: {e}")
    
    # Load Ethereum
    try:
        df_eth = pd.read_csv(eth_file)
        print(f"✓ Loaded Ethereum data: {len(df_eth):,} transactions")
    except Exception as e:
        print(f"✗ Error loading Ethereum data: {e}")
    
    return df_btc, df_eth

def analyze_crypto_comparison(df_btc, df_eth):
    """Analyze and compare cryptocurrency patterns"""
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS: Bitcoin vs Ethereum (2025)")
    print("="*70)
    
    if df_btc is None or df_eth is None:
        print("✗ Missing data for comparison")
        return None
    
    # Bitcoin statistics
    btc_stats = {
        'network': 'Bitcoin',
        'transactions': len(df_btc),
        'avg_size_bytes': df_btc['size'].mean() if 'size' in df_btc.columns else 0,
        'avg_inputs': df_btc.get('vin_sz', df_btc.get('inputs', pd.Series([0]))).mean(),
        'avg_outputs': df_btc.get('vout_sz', df_btc.get('outputs', pd.Series([0]))).mean(),
        'avg_fee_native': df_btc['fee'].mean() if 'fee' in df_btc.columns else 0,
        'total_value': df_btc['total_out'].sum() if 'total_out' in df_btc.columns else 0,
        'unique_addresses': (df_btc['input_addresses'].nunique() if 'input_addresses' in df_btc.columns else 0)
    }
    
    # Ethereum statistics  
    eth_stats = {
        'network': 'Ethereum',
        'transactions': len(df_eth),
        'avg_size_bytes': df_eth['input_size'].mean() if 'input_size' in df_eth.columns else 0,
        'avg_gas': df_eth['gas'].mean() if 'gas' in df_eth.columns else 0,
        'avg_gas_price_gwei': df_eth['gas_price_gwei'].mean() if 'gas_price_gwei' in df_eth.columns else 0,
        'avg_fee_native': df_eth['tx_fee_eth'].mean() if 'tx_fee_eth' in df_eth.columns else 0,
        'total_value': df_eth['value_eth'].sum() if 'value_eth' in df_eth.columns else 0,
        'contract_ratio': df_eth['is_contract'].mean() if 'is_contract' in df_eth.columns else 0,
        'unique_senders': df_eth['from'].nunique() if 'from' in df_eth.columns else 0,
        'unique_receivers': df_eth['to'].nunique() if 'to' in df_eth.columns else 0
    }
    
    # Print comparison
    print("\n📊 Network Comparison:")
    print("-" * 70)
    print(f"{'Metric':<30} {'Bitcoin':>18} {'Ethereum':>18}")
    print("-" * 70)
    print(f"{'Total Transactions':<30} {btc_stats['transactions']:>18,} {eth_stats['transactions']:>18,}")
    print(f"{'Avg Fee (native coin)':<30} {btc_stats['avg_fee_native']:>18.6f} {eth_stats['avg_fee_native']:>18.6f}")
    print(f"{'Total Value (native coin)':<30} {btc_stats['total_value']:>18,.2f} {eth_stats['total_value']:>18,.2f}")
    
    if 'gas_price_gwei' in df_eth.columns:
        print(f"{'Avg Gas Price (Gwei)':<30} {'N/A':>18} {eth_stats['avg_gas_price_gwei']:>18,.2f}")
    
    if 'is_contract' in df_eth.columns:
        print(f"{'Contract Interaction %':<30} {'N/A':>18} {eth_stats['contract_ratio']*100:>17.1f}%")
    
    print(f"{'Unique Addresses':<30} {btc_stats['unique_addresses']:>18,} {eth_stats['unique_senders']:>18,}")
    
    return btc_stats, eth_stats

def create_btc_eth_comparison_plots(df_btc, df_eth, output_dir='../fig'):
    """Create Bitcoin vs Ethereum comparison visualizations"""
    print("\n📈 Generating BTC vs ETH comparison plots...")
    
    Path(output_dir).mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    colors_btc = '#F7931A'  # Bitcoin orange
    colors_eth = '#627EEA'  # Ethereum blue
    
    # 1. Transaction count per block
    ax1 = fig.add_subplot(gs[0, 0])
    if 'block_height' in df_btc.columns and 'block_number' in df_eth.columns:
        btc_per_block = df_btc.groupby('block_height').size()
        eth_per_block = df_eth.groupby('block_number').size()
        
        ax1.hist([btc_per_block.values, eth_per_block.values], 
                bins=50, label=['Bitcoin', 'Ethereum'], 
                color=[colors_btc, colors_eth], alpha=0.7, density=True)
        ax1.set_xlabel('Transactions per Block')
        ax1.set_ylabel('Density')
        ax1.set_title('Block Utilization: BTC vs ETH')
        ax1.legend()
    
    # 2. Fee distribution (log scale)
    ax2 = fig.add_subplot(gs[0, 1])
    if 'fee' in df_btc.columns and 'tx_fee_eth' in df_eth.columns:
        btc_fees = df_btc[df_btc['fee'] > 0]['fee']
        eth_fees = df_eth[df_eth['tx_fee_eth'] > 0]['tx_fee_eth']
        
        # Filter outliers
        btc_fees = btc_fees[btc_fees < btc_fees.quantile(0.99)]
        eth_fees = eth_fees[eth_fees < eth_fees.quantile(0.99)]
        
        ax2.hist([np.log10(btc_fees + 1e-8), np.log10(eth_fees + 1e-8)],
                bins=50, label=['BTC fee (log)', 'ETH fee (log)'],
                color=[colors_btc, colors_eth], alpha=0.7, density=True)
        ax2.set_xlabel('log10(Transaction Fee in Native Coin)')
        ax2.set_ylabel('Density')
        ax2.set_title('Fee Distribution Comparison')
        ax2.legend()
    
    # 3. Value transferred distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if 'total_out' in df_btc.columns and 'value_eth' in df_eth.columns:
        btc_values = df_btc[df_btc['total_out'] > 0]['total_out']
        eth_values = df_eth[df_eth['value_eth'] > 0]['value_eth']
        
        btc_values = btc_values[btc_values < btc_values.quantile(0.95)]
        eth_values = eth_values[eth_values < eth_values.quantile(0.95)]
        
        ax3.hist([np.log10(btc_values + 1e-8), np.log10(eth_values + 1e-8)],
                bins=50, label=['BTC (log)', 'ETH (log)'],
                color=[colors_btc, colors_eth], alpha=0.7, density=True)
        ax3.set_xlabel('log10(Value Transferred)')
        ax3.set_ylabel('Density')
        ax3.set_title('Transfer Amount Distribution')
        ax3.legend()
    
    # 4. Transaction activity over time
    ax4 = fig.add_subplot(gs[1, :])
    if 'block_time' in df_btc.columns and 'block_timestamp' in df_eth.columns:
        df_btc['date'] = pd.to_datetime(df_btc['block_time'], unit='s')
        df_eth['date'] = pd.to_datetime(df_eth['block_timestamp'], unit='s')
        
        # Hourly aggregation
        btc_hourly = df_btc.groupby(df_btc['date'].dt.floor('H')).size()
        eth_hourly = df_eth.groupby(df_eth['date'].dt.floor('H')).size()
        
        ax4.plot(btc_hourly.index, btc_hourly.values, 
                label='Bitcoin', color=colors_btc, linewidth=2, alpha=0.8)
        ax4.plot(eth_hourly.index, eth_hourly.values,
                label='Ethereum', color=colors_eth, linewidth=2, alpha=0.8)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Transactions per Hour')
        ax4.set_title('Network Activity Timeline')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
    
    # 5. Transaction complexity (Bitcoin: inputs+outputs, Ethereum: gas)
    ax5 = fig.add_subplot(gs[2, 0])
    if 'vin_sz' in df_btc.columns and 'gas' in df_eth.columns:
        inputs = df_btc.get('vin_sz', df_btc.get('inputs', pd.Series([0])))
        outputs = df_btc.get('vout_sz', df_btc.get('outputs', pd.Series([0])))
        btc_complexity = inputs + outputs
        
        # Normalize ETH gas to similar scale
        eth_complexity = df_eth['gas'] / 21000  # Divide by base gas
        
        ax5.hist([btc_complexity, eth_complexity],
                bins=50, label=['BTC (I+O)', 'ETH (gas/21k)'],
                color=[colors_btc, colors_eth], alpha=0.7, density=True)
        ax5.set_xlabel('Transaction Complexity')
        ax5.set_ylabel('Density')
        ax5.set_title('Transaction Complexity Comparison')
        ax5.legend()
        ax5.set_xlim(0, 20)
    
    # 6. Contract interactions (Ethereum only)
    ax6 = fig.add_subplot(gs[2, 1])
    if 'is_contract' in df_eth.columns:
        contract_counts = df_eth['is_contract'].value_counts()
        labels = ['Regular TX', 'Contract Creation']
        ax6.pie(contract_counts.values, labels=labels, autopct='%1.1f%%',
               colors=[colors_eth, '#FFB84D'], startangle=90)
        ax6.set_title('Ethereum: Contract vs Regular Transactions')
    else:
        ax6.text(0.5, 0.5, 'No contract data available',
                ha='center', va='center', transform=ax6.transAxes)
        ax6.axis('off')
    
    # 7. Gas price over time (Ethereum)
    ax7 = fig.add_subplot(gs[2, 2])
    if 'gas_price_gwei' in df_eth.columns and 'date' in df_eth.columns:
        eth_gas_time = df_eth.groupby(df_eth['date'].dt.floor('H'))['gas_price_gwei'].mean()
        ax7.plot(eth_gas_time.index, eth_gas_time.values,
                color=colors_eth, linewidth=2, alpha=0.8)
        ax7.set_xlabel('Time')
        ax7.set_ylabel('Average Gas Price (Gwei)')
        ax7.set_title('Ethereum Gas Price Evolution')
        ax7.tick_params(axis='x', rotation=45)
    
    # 8. Network characteristics comparison (bar chart)
    ax8 = fig.add_subplot(gs[3, 0])
    if all(k in df_btc.columns for k in ['block_height']) and 'block_number' in df_eth.columns:
        btc_blocks = df_btc['block_height'].nunique()
        eth_blocks = df_eth['block_number'].nunique()
        btc_tx_per_block = len(df_btc) / btc_blocks if btc_blocks > 0 else 0
        eth_tx_per_block = len(df_eth) / eth_blocks if eth_blocks > 0 else 0
        
        metrics = ['Tx/Block', 'Blocks', 'Total Tx (k)']
        btc_values = [btc_tx_per_block, btc_blocks, len(df_btc)/1000]
        eth_values = [eth_tx_per_block, eth_blocks, len(df_eth)/1000]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax8.bar(x - width/2, btc_values, width, label='Bitcoin', color=colors_btc, alpha=0.8)
        ax8.bar(x + width/2, eth_values, width, label='Ethereum', color=colors_eth, alpha=0.8)
        ax8.set_ylabel('Count')
        ax8.set_title('Network Metrics Comparison')
        ax8.set_xticks(x)
        ax8.set_xticklabels(metrics)
        ax8.legend()
    
    # 9. Summary statistics table
    ax9 = fig.add_subplot(gs[3, 1:])
    ax9.axis('off')
    
    summary_data = [
        ['Metric', 'Bitcoin', 'Ethereum'],
        ['Total Transactions', f"{len(df_btc):,}", f"{len(df_eth):,}"],
        ['Avg Fee (native)', 
         f"{df_btc['fee'].mean():.6f}" if 'fee' in df_btc.columns else 'N/A',
         f"{df_eth['tx_fee_eth'].mean():.6f}" if 'tx_fee_eth' in df_eth.columns else 'N/A'],
        ['Total Value Transferred',
         f"{df_btc['total_out'].sum():.2f} BTC" if 'total_out' in df_btc.columns else 'N/A',
         f"{df_eth['value_eth'].sum():.2f} ETH" if 'value_eth' in df_eth.columns else 'N/A']
    ]
    
    if 'gas_price_gwei' in df_eth.columns:
        summary_data.append(['Avg Gas Price', 'N/A', 
                            f"{df_eth['gas_price_gwei'].mean():.2f} Gwei"])
    
    if 'is_contract' in df_eth.columns:
        summary_data.append(['Contract TX %', 'N/A',
                            f"{df_eth['is_contract'].mean()*100:.1f}%"])
    
    table = ax9.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Bitcoin vs Ethereum: 2025 Network Comparison',
                 fontsize=18, fontweight='bold', y=0.995)
    
    output_path = Path(output_dir) / 'bitcoin_vs_ethereum_2025.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {output_path}")
    
    plt.close()

def main():
    """Main execution"""
    print("="*70)
    print("Bitcoin vs Ethereum Comparison Analysis (2025)")
    print("="*70)
    
    # File paths
    btc_file = input("\nEnter Bitcoin data file (default: '2025-01_bitcoin.csv'): ") or "2025-01_bitcoin.csv"
    eth_file = input("Enter Ethereum data file (default: 'ethereum_2025.csv'): ") or "ethereum_2025.csv"
    
    # Load data
    df_btc, df_eth = load_crypto_data(btc_file, eth_file)
    
    if df_btc is None or df_eth is None:
        print("✗ Cannot proceed without both datasets")
        return
    
    # Analyze
    stats = analyze_crypto_comparison(df_btc, df_eth)
    
    # Create plots
    create_btc_eth_comparison_plots(df_btc, df_eth)
    
    print("\n✅ Analysis complete!")
    print("\n📊 Generated plot: ../fig/bitcoin_vs_ethereum_2025.png")

if __name__ == "__main__":
    main()

