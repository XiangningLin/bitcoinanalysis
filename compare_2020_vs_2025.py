#!/usr/bin/env python3
"""
Compare Bitcoin transaction patterns between 2020 and 2025
Generate visualization and analysis
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

def load_data(file_2020, file_2025):
    """Load both datasets"""
    print("Loading data...")
    
    try:
        df_2020 = pd.read_csv(file_2020)
        print(f"✓ Loaded 2020 data: {len(df_2020)} transactions")
    except Exception as e:
        print(f"✗ Error loading 2020 data: {e}")
        return None, None
    
    try:
        df_2025 = pd.read_csv(file_2025)
        print(f"✓ Loaded 2025 data: {len(df_2025)} transactions")
    except Exception as e:
        print(f"✗ Error loading 2025 data: {e}")
        return df_2020, None
    
    return df_2020, df_2025

def analyze_transaction_patterns(df_2020, df_2025):
    """Analyze and compare transaction patterns"""
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS: 2020 vs 2025")
    print("="*60)
    
    stats = {}
    
    for name, df in [("2020", df_2020), ("2025", df_2025)]:
        if df is not None:
            stats[name] = {
                'total_transactions': len(df),
                'avg_size': df['size'].mean() if 'size' in df.columns else 0,
                'avg_inputs': df['vin_sz'].mean() if 'vin_sz' in df.columns else (df['inputs'].mean() if 'inputs' in df.columns else 0),
                'avg_outputs': df['vout_sz'].mean() if 'vout_sz' in df.columns else (df['outputs'].mean() if 'outputs' in df.columns else 0),
                'avg_fee': df['fee'].mean() if 'fee' in df.columns else 0,
                'total_btc': df['total_out'].sum() if 'total_out' in df.columns else 0
            }
    
    # Print comparison
    print("\n📊 Key Metrics Comparison:")
    print("-" * 60)
    print(f"{'Metric':<25} {'2020':>15} {'2025':>15} {'Change':>15}")
    print("-" * 60)
    
    if '2020' in stats and '2025' in stats:
        for key in stats['2020'].keys():
            val_2020 = stats['2020'][key]
            val_2025 = stats['2025'][key]
            
            if val_2020 > 0:
                change = ((val_2025 - val_2020) / val_2020) * 100
                change_str = f"{change:+.1f}%"
            else:
                change_str = "N/A"
            
            print(f"{key:<25} {val_2020:>15.2f} {val_2025:>15.2f} {change_str:>15}")
    
    return stats

def create_comparison_plots(df_2020, df_2025, output_dir='../fig'):
    """Create comprehensive comparison visualizations"""
    print("\n📈 Generating comparison plots...")
    
    Path(output_dir).mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Transaction Size Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    if 'size' in df_2020.columns and df_2025 is not None and 'size' in df_2025.columns:
        bins = np.linspace(0, 5000, 50)
        ax1.hist(df_2020['size'], bins=bins, alpha=0.6, label='2020', color='#1f77b4', density=True)
        ax1.hist(df_2025['size'], bins=bins, alpha=0.6, label='2025', color='#ff7f0e', density=True)
        ax1.set_xlabel('Transaction Size (bytes)')
        ax1.set_ylabel('Density')
        ax1.set_title('Transaction Size Distribution')
        ax1.legend()
        ax1.set_xlim(0, 5000)
    
    # 2. Inputs vs Outputs
    ax2 = fig.add_subplot(gs[0, 1])
    if df_2025 is not None:
        inputs_col_2020 = 'vin_sz' if 'vin_sz' in df_2020.columns else 'inputs'
        outputs_col_2020 = 'vout_sz' if 'vout_sz' in df_2020.columns else 'outputs'
        inputs_col_2025 = 'vin_sz' if 'vin_sz' in df_2025.columns else 'inputs'
        outputs_col_2025 = 'vout_sz' if 'vout_sz' in df_2025.columns else 'outputs'
        
        sample_2020 = df_2020.sample(min(5000, len(df_2020)))
        sample_2025 = df_2025.sample(min(5000, len(df_2025)))
        
        ax2.scatter(sample_2020[inputs_col_2020], sample_2020[outputs_col_2020], 
                   alpha=0.3, s=10, label='2020', color='#1f77b4')
        ax2.scatter(sample_2025[inputs_col_2025], sample_2025[outputs_col_2025], 
                   alpha=0.3, s=10, label='2025', color='#ff7f0e')
        ax2.set_xlabel('Number of Inputs')
        ax2.set_ylabel('Number of Outputs')
        ax2.set_title('Inputs vs Outputs Pattern')
        ax2.legend()
        ax2.set_xlim(0, 20)
        ax2.set_ylim(0, 20)
    
    # 3. Fee Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if 'fee' in df_2020.columns and df_2025 is not None and 'fee' in df_2025.columns:
        # Filter outliers for better visualization
        fee_2020 = df_2020[df_2020['fee'] > 0]['fee']
        fee_2025 = df_2025[df_2025['fee'] > 0]['fee']
        
        fee_2020 = fee_2020[fee_2020 < fee_2020.quantile(0.99)]
        fee_2025 = fee_2025[fee_2025 < fee_2025.quantile(0.99)]
        
        ax3.violinplot([fee_2020, fee_2025], positions=[0, 1], showmeans=True)
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(['2020', '2025'])
        ax3.set_ylabel('Transaction Fee (BTC)')
        ax3.set_title('Fee Distribution Comparison')
    
    # 4. Average metrics over time (within each dataset)
    ax4 = fig.add_subplot(gs[1, 0])
    if 'block_time' in df_2020.columns:
        df_2020['date'] = pd.to_datetime(df_2020['block_time'], unit='s')
        daily_2020 = df_2020.groupby(df_2020['date'].dt.date).size()
        ax4.plot(daily_2020.index, daily_2020.values, label='2020', color='#1f77b4', linewidth=2)
        
        if df_2025 is not None and 'block_time' in df_2025.columns:
            df_2025['date'] = pd.to_datetime(df_2025['block_time'], unit='s')
            daily_2025 = df_2025.groupby(df_2025['date'].dt.date).size()
            ax4.plot(daily_2025.index, daily_2025.values, label='2025', color='#ff7f0e', linewidth=2)
        
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Transaction Count')
        ax4.set_title('Daily Transaction Volume')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
    
    # 5. Transaction complexity (inputs + outputs)
    ax5 = fig.add_subplot(gs[1, 1])
    if df_2025 is not None:
        inputs_2020 = df_2020[inputs_col_2020]
        outputs_2020 = df_2020[outputs_col_2020]
        complexity_2020 = inputs_2020 + outputs_2020
        
        inputs_2025 = df_2025[inputs_col_2025]
        outputs_2025 = df_2025[outputs_col_2025]
        complexity_2025 = inputs_2025 + outputs_2025
        
        bins = np.arange(0, 30, 1)
        ax5.hist(complexity_2020, bins=bins, alpha=0.6, label='2020', color='#1f77b4', density=True)
        ax5.hist(complexity_2025, bins=bins, alpha=0.6, label='2025', color='#ff7f0e', density=True)
        ax5.set_xlabel('Transaction Complexity (inputs + outputs)')
        ax5.set_ylabel('Density')
        ax5.set_title('Transaction Complexity Distribution')
        ax5.legend()
    
    # 6. BTC value transferred
    ax6 = fig.add_subplot(gs[1, 2])
    if 'total_out' in df_2020.columns and df_2025 is not None and 'total_out' in df_2025.columns:
        # Filter for reasonable visualization
        btc_2020 = df_2020[df_2020['total_out'] > 0]['total_out']
        btc_2025 = df_2025[df_2025['total_out'] > 0]['total_out']
        
        btc_2020 = btc_2020[btc_2020 < btc_2020.quantile(0.95)]
        btc_2025 = btc_2025[btc_2025 < btc_2025.quantile(0.95)]
        
        ax6.hist(np.log10(btc_2020 + 1e-8), bins=50, alpha=0.6, label='2020', color='#1f77b4', density=True)
        ax6.hist(np.log10(btc_2025 + 1e-8), bins=50, alpha=0.6, label='2025', color='#ff7f0e', density=True)
        ax6.set_xlabel('log10(BTC Transferred)')
        ax6.set_ylabel('Density')
        ax6.set_title('BTC Transfer Amount Distribution')
        ax6.legend()
    
    # 7. Network activity pattern (transactions per block)
    ax7 = fig.add_subplot(gs[2, 0])
    if 'block_height' in df_2020.columns:
        tx_per_block_2020 = df_2020.groupby('block_height').size()
        ax7.plot(tx_per_block_2020.values, alpha=0.7, label='2020', color='#1f77b4')
        
        if df_2025 is not None and 'block_height' in df_2025.columns:
            tx_per_block_2025 = df_2025.groupby('block_height').size()
            ax7.plot(tx_per_block_2025.values, alpha=0.7, label='2025', color='#ff7f0e')
        
        ax7.set_xlabel('Block Index')
        ax7.set_ylabel('Transactions per Block')
        ax7.set_title('Block Utilization Pattern')
        ax7.legend()
    
    # 8. Summary statistics table
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')
    
    if df_2025 is not None:
        summary_data = [
            ['Metric', '2020', '2025', 'Change'],
            ['Total Transactions', f"{len(df_2020):,}", f"{len(df_2025):,}", 
             f"{((len(df_2025)-len(df_2020))/len(df_2020)*100):+.1f}%"],
            ['Avg Size (bytes)', f"{df_2020['size'].mean():.1f}", 
             f"{df_2025['size'].mean():.1f}",
             f"{((df_2025['size'].mean()-df_2020['size'].mean())/df_2020['size'].mean()*100):+.1f}%"],
            ['Avg Inputs', f"{df_2020[inputs_col_2020].mean():.2f}", 
             f"{df_2025[inputs_col_2025].mean():.2f}",
             f"{((df_2025[inputs_col_2025].mean()-df_2020[inputs_col_2020].mean())/df_2020[inputs_col_2020].mean()*100):+.1f}%"],
            ['Avg Outputs', f"{df_2020[outputs_col_2020].mean():.2f}", 
             f"{df_2025[outputs_col_2025].mean():.2f}",
             f"{((df_2025[outputs_col_2025].mean()-df_2020[outputs_col_2020].mean())/df_2020[outputs_col_2020].mean()*100):+.1f}%"]
        ]
        
        if 'fee' in df_2020.columns and 'fee' in df_2025.columns:
            summary_data.append(['Avg Fee (BTC)', 
                                f"{df_2020['fee'].mean():.6f}", 
                                f"{df_2025['fee'].mean():.6f}",
                                f"{((df_2025['fee'].mean()-df_2020['fee'].mean())/df_2020['fee'].mean()*100):+.1f}%"])
        
        table = ax8.table(cellText=summary_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the header row
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Bitcoin Transaction Analysis: 2020 vs 2025 Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    output_path = Path(output_dir) / 'bitcoin_2020_vs_2025_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {output_path}")
    
    plt.close()

def main():
    """Main execution"""
    print("=" * 60)
    print("Bitcoin Transaction Comparison: 2020 vs 2025")
    print("=" * 60)
    
    # File paths
    file_2020 = input("\nEnter 2020 data file path (default: '../2020-10_00.csv'): ") or "../2020-10_00.csv"
    file_2025 = input("Enter 2025 data file path (default: '2025-01_bitcoin.csv'): ") or "2025-01_bitcoin.csv"
    
    # Load data
    df_2020, df_2025 = load_data(file_2020, file_2025)
    
    if df_2020 is None:
        print("✗ Cannot proceed without 2020 data")
        return
    
    # Analyze
    stats = analyze_transaction_patterns(df_2020, df_2025)
    
    # Create plots
    if df_2025 is not None:
        create_comparison_plots(df_2020, df_2025)
        print("\n✅ Analysis complete!")
    else:
        print("\n⚠ Could not complete comparison (missing 2025 data)")

if __name__ == "__main__":
    main()

