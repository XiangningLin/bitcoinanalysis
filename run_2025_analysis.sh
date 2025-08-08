#!/bin/bash
# One-click script to download 2025 Bitcoin data and compare with 2020

echo "======================================================================"
echo "  Bitcoin 2020 vs 2025 Comparison Analysis - Automated Pipeline"
echo "======================================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.7+${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python found:${NC} $(python3 --version)"

# Check dependencies
echo ""
echo "Checking dependencies..."
MISSING_DEPS=0

for package in requests pandas matplotlib seaborn numpy; do
    if ! python3 -c "import $package" 2>/dev/null; then
        echo -e "${RED}✗ Missing: $package${NC}"
        MISSING_DEPS=1
    else
        echo -e "${GREEN}✓ Found: $package${NC}"
    fi
done

if [ $MISSING_DEPS -eq 1 ]; then
    echo ""
    echo -e "${YELLOW}Installing missing dependencies...${NC}"
    pip3 install requests pandas matplotlib seaborn numpy
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Failed to install dependencies${NC}"
        exit 1
    fi
fi

echo ""
echo "======================================================================"
echo "  Step 1: Download 2025 Bitcoin Data"
echo "======================================================================"
echo ""

# Check if 2025 data already exists
if [ -f "2025-01_bitcoin.csv" ]; then
    echo -e "${YELLOW}⚠ Found existing 2025 data file: 2025-01_bitcoin.csv${NC}"
    read -p "Redownload data? This will take ~17 minutes (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting download..."
        python3 download_2025_bitcoin_data.py
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}✗ Download failed${NC}"
            exit 1
        fi
    else
        echo "Using existing data file."
    fi
else
    echo "Starting data download..."
    echo "This will take approximately 17 minutes for 1000 blocks..."
    echo ""
    
    # Run download script with default parameters
    python3 -c "
from download_2025_bitcoin_data import download_bitcoin_transactions, get_latest_block_height
import sys

print('Getting current block height...')
current_height = get_latest_block_height()
if current_height:
    print(f'Current blockchain height: {current_height}')
    start_block = current_height - 1000
    print(f'Downloading blocks {start_block} to {current_height}...')
    print('')
    
    df = download_bitcoin_transactions(start_block, 1000, '2025-01_bitcoin.csv')
    if df is not None:
        print('')
        print('✓ Download complete!')
        sys.exit(0)
    else:
        print('')
        print('✗ Download failed')
        sys.exit(1)
else:
    print('✗ Could not get current block height')
    sys.exit(1)
"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Download failed. Check your internet connection and try again.${NC}"
        exit 1
    fi
fi

echo ""
echo "======================================================================"
echo "  Step 2: Compare with 2020 Data"
echo "======================================================================"
echo ""

# Check if 2020 data exists
if [ ! -f "../2020-10_00.csv" ]; then
    echo -e "${RED}✗ 2020 data file not found: ../2020-10_00.csv${NC}"
    echo "Please ensure the 2020 data file exists before running comparison."
    exit 1
fi

echo "Running comparative analysis..."
python3 -c "
from compare_2020_vs_2025 import load_data, analyze_transaction_patterns, create_comparison_plots
import sys

# Load data
df_2020, df_2025 = load_data('../2020-10_00.csv', '2025-01_bitcoin.csv')

if df_2020 is None or df_2025 is None:
    print('✗ Failed to load data')
    sys.exit(1)

# Analyze
stats = analyze_transaction_patterns(df_2020, df_2025)

# Create plots
create_comparison_plots(df_2020, df_2025, '../fig')

print('')
print('✓ Analysis complete!')
sys.exit(0)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Analysis failed${NC}"
    exit 1
fi

echo ""
echo "======================================================================"
echo "  ✅ Analysis Complete!"
echo "======================================================================"
echo ""
echo "Results saved to:"
echo "  📊 Comparison plot: ../fig/bitcoin_2020_vs_2025_comparison.png"
echo "  📁 2025 data: 2025-01_bitcoin.csv"
echo ""
echo "You can view the comparison plot to see:"
echo "  • Transaction size distribution changes"
echo "  • Fee trends (2020 vs 2025)"
echo "  • Network complexity evolution"
echo "  • Transaction volume patterns"
echo "  • And more..."
echo ""
echo "To view the plot:"
echo "  open ../fig/bitcoin_2020_vs_2025_comparison.png"
echo ""
echo "======================================================================"

# Optionally open the plot
read -p "Open the comparison plot now? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open ../fig/bitcoin_2020_vs_2025_comparison.png
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        xdg-open ../fig/bitcoin_2020_vs_2025_comparison.png
    else
        echo "Please manually open: ../fig/bitcoin_2020_vs_2025_comparison.png"
    fi
fi

echo ""
echo "Thank you for using the Bitcoin analysis pipeline!"
echo "For more details, see README_2025_ANALYSIS.md"

