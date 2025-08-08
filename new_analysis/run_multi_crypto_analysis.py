#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Cryptocurrency Network Analysis
Tests framework generalizability across different blockchain architectures
"""

import os
import json
import pandas as pd
from typing import Dict, List, Any
from abc import ABC, abstractmethod
import argparse


class CryptoNetworkAnalyzer(ABC):
    """Abstract base class for cryptocurrency network analysis"""
    
    def __init__(self, crypto_name: str):
        self.crypto_name = crypto_name
        
    @abstractmethod
    def load_transaction_data(self, timeframe: str) -> pd.DataFrame:
        """Load transaction data for this cryptocurrency"""
        pass
    
    @abstractmethod
    def get_network_properties(self) -> Dict:
        """Get network-specific properties"""
        pass
    
    @abstractmethod  
    def adapt_sampling_for_network(self, method: str) -> Dict:
        """Adapt sampling method for this network's characteristics"""
        pass


class BitcoinAnalyzer(CryptoNetworkAnalyzer):
    """Bitcoin network analyzer (baseline)"""
    
    def load_transaction_data(self, timeframe: str) -> pd.DataFrame:
        print(f"📊 Loading Bitcoin data for {timeframe}")
        # Your existing Bitcoin data loading
        return pd.DataFrame()
    
    def get_network_properties(self) -> Dict:
        return {
            "type": "UTXO",
            "avg_confirmation_time": "10 minutes", 
            "block_size_limit": "1MB",
            "transaction_model": "UTXO",
            "scripting": "basic",
            "anonymity_level": "pseudonymous"
        }
    
    def adapt_sampling_for_network(self, method: str) -> Dict:
        # Bitcoin-specific adaptations (your current implementation)
        return {"method": method, "adaptations": []}


class EthereumAnalyzer(CryptoNetworkAnalyzer):
    """Ethereum network analyzer"""
    
    def load_transaction_data(self, timeframe: str) -> pd.DataFrame:
        print(f"📊 Loading Ethereum data for {timeframe}")
        # Ethereum has different data structure: account-based, smart contracts
        return pd.DataFrame()
    
    def get_network_properties(self) -> Dict:
        return {
            "type": "Account-based",
            "avg_confirmation_time": "13 seconds",
            "block_gas_limit": "variable",
            "transaction_model": "account-based", 
            "scripting": "turing-complete",
            "anonymity_level": "pseudonymous",
            "smart_contracts": True,
            "tokens": "ERC-20/721/1155"
        }
    
    def adapt_sampling_for_network(self, method: str) -> Dict:
        # Ethereum-specific adaptations
        adaptations = [
            "account_balance_weighting",  # Weight by ETH balance
            "contract_interaction_bonus", # Boost nodes with smart contract activity
            "token_transfer_consideration", # Consider ERC-20 transfers
            "gas_usage_weighting"  # Weight by gas consumption
        ]
        return {"method": method, "adaptations": adaptations}


class LitecoinAnalyzer(CryptoNetworkAnalyzer):
    """Litecoin network analyzer"""
    
    def load_transaction_data(self, timeframe: str) -> pd.DataFrame:
        print(f"📊 Loading Litecoin data for {timeframe}")
        # Similar to Bitcoin but with different parameters
        return pd.DataFrame()
    
    def get_network_properties(self) -> Dict:
        return {
            "type": "UTXO",
            "avg_confirmation_time": "2.5 minutes",
            "block_size_limit": "1MB", 
            "transaction_model": "UTXO",
            "scripting": "basic",
            "anonymity_level": "pseudonymous",
            "mining_algorithm": "Scrypt"
        }
    
    def adapt_sampling_for_network(self, method: str) -> Dict:
        # Litecoin-specific adaptations
        adaptations = [
            "faster_block_adjustment",  # Account for 2.5min blocks
            "scrypt_mining_consideration"  # Different mining dynamics
        ]
        return {"method": method, "adaptations": adaptations}


class MoneroAnalyzer(CryptoNetworkAnalyzer):
    """Monero network analyzer (privacy-focused)"""
    
    def load_transaction_data(self, timeframe: str) -> pd.DataFrame:
        print(f"📊 Loading Monero data for {timeframe}")
        print("⚠️ Note: Monero data is privacy-protected, limited analysis possible")
        return pd.DataFrame()
    
    def get_network_properties(self) -> Dict:
        return {
            "type": "UTXO-like",
            "avg_confirmation_time": "2 minutes",
            "transaction_model": "ring-signatures",
            "scripting": "limited",
            "anonymity_level": "private",
            "ring_size": "11+",
            "stealth_addresses": True
        }
    
    def adapt_sampling_for_network(self, method: str) -> Dict:
        # Monero-specific challenges
        adaptations = [
            "ring_signature_analysis",  # Analyze transaction rings
            "limited_graph_visibility", # Privacy limits graph construction  
            "timing_analysis_only",     # Focus on temporal patterns
            "decoy_handling"           # Account for decoy outputs
        ]
        return {"method": method, "adaptations": adaptations}


class MultiCryptoAnalyzer:
    """Orchestrator for multi-cryptocurrency analysis"""
    
    def __init__(self, output_dir: str = "outputs/multi_crypto"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize analyzers for different cryptocurrencies
        self.analyzers = {
            "bitcoin": BitcoinAnalyzer("Bitcoin"),
            "ethereum": EthereumAnalyzer("Ethereum"), 
            "litecoin": LitecoinAnalyzer("Litecoin"),
            "monero": MoneroAnalyzer("Monero")
        }
        
    def run_cross_crypto_analysis(self, cryptos: List[str] = None) -> Dict:
        """Run analysis across multiple cryptocurrencies"""
        
        if cryptos is None:
            cryptos = ["bitcoin", "ethereum", "litecoin"]  # Skip Monero for now
        
        results = {}
        
        for crypto in cryptos:
            if crypto in self.analyzers:
                print(f"\n🚀 Analyzing {crypto.upper()} network...")
                results[crypto] = self.analyze_crypto_network(crypto)
            else:
                print(f"❌ Analyzer for {crypto} not available")
        
        # Cross-network comparison
        comparison = self.compare_across_networks(results)
        
        return {
            "individual_results": results,
            "cross_network_comparison": comparison,
            "analysis_metadata": {
                "networks_analyzed": len(results),
                "frameworks_tested": 2,  # RWFB and CETraS
                "analysis_dimensions": 3  # Role, anomaly, decentralization
            }
        }
    
    def analyze_crypto_network(self, crypto_name: str) -> Dict:
        """Analyze specific cryptocurrency network"""
        analyzer = self.analyzers[crypto_name]
        
        # Get network properties
        properties = analyzer.get_network_properties()
        
        # Load data (simplified for demo)
        data = analyzer.load_transaction_data("2020-10")
        
        # Adapt sampling methods
        cetras_config = analyzer.adapt_sampling_for_network("CETraS")
        rwfb_config = analyzer.adapt_sampling_for_network("RWFB")
        
        # Run sampling (placeholder)
        cetras_sample = self.run_adapted_sampling(data, cetras_config)
        rwfb_sample = self.run_adapted_sampling(data, rwfb_config)
        
        # LLM analysis with network-specific prompts
        cetras_analysis = self.run_network_specific_llm_analysis(
            cetras_sample, crypto_name, "CETraS"
        )
        rwfb_analysis = self.run_network_specific_llm_analysis(
            rwfb_sample, crypto_name, "RWFB"
        )
        
        # Compute metrics
        sdm = self.compute_cross_network_sdm(cetras_analysis, rwfb_analysis)
        ci = self.compute_network_ci(cetras_analysis, rwfb_analysis, properties)
        
        return {
            "network": crypto_name,
            "properties": properties,
            "sampling_configs": {
                "cetras": cetras_config,
                "rwfb": rwfb_config
            },
            "analysis_results": {
                "cetras": cetras_analysis,
                "rwfb": rwfb_analysis
            },
            "metrics": {
                "sdm": sdm,
                "ci": ci
            }
        }
    
    def run_adapted_sampling(self, data: pd.DataFrame, config: Dict) -> List[Dict]:
        """Run sampling with network-specific adaptations"""
        print(f"🎯 Running {config['method']} with adaptations: {config['adaptations']}")
        # Placeholder - implement network-specific sampling logic
        return []
    
    def run_network_specific_llm_analysis(self, sample: List[Dict], 
                                        network: str, method: str) -> Dict:
        """Run LLM analysis with network-specific prompts"""
        
        # Network-specific prompt adaptations
        prompt_adaptations = self.get_network_prompt_adaptations(network)
        
        print(f"🤖 Running {network} LLM analysis for {method}")
        print(f"   Prompt adaptations: {prompt_adaptations}")
        
        # This would interface with your multi-agent framework
        # but use adapted prompts for each network type
        
        return {
            "method": method,
            "network": network,
            "prompt_adaptations": prompt_adaptations,
            "results": {}  # Placeholder
        }
    
    def get_network_prompt_adaptations(self, network: str) -> List[str]:
        """Get network-specific prompt adaptations"""
        
        adaptations = {
            "bitcoin": [
                "Focus on UTXO transaction patterns",
                "Consider mining pool behaviors", 
                "Analyze exchange hot/cold wallet patterns"
            ],
            "ethereum": [
                "Account for smart contract interactions",
                "Consider ERC-20 token transfers",
                "Analyze DeFi protocol participation",
                "Account for gas usage patterns"
            ],
            "litecoin": [
                "Consider faster block times",
                "Account for Scrypt mining dynamics",
                "Similar patterns to Bitcoin but faster"
            ],
            "monero": [
                "Focus on timing patterns only",
                "Cannot analyze specific transaction flows",
                "Emphasize privacy-preserving behaviors"
            ]
        }
        
        return adaptations.get(network, [])
    
    def compute_cross_network_sdm(self, cetras: Dict, rwfb: Dict) -> float:
        """Compute SDM adapted for cross-network comparison"""
        # Similar to your existing SDM but may need network-specific weights
        return 0.0  # Placeholder
    
    def compute_network_ci(self, cetras: Dict, rwfb: Dict, properties: Dict) -> Dict:
        """Compute CI accounting for network properties"""
        # Network properties might affect what "centralized" means
        return {"cetras_ci": 0.0, "rwfb_ci": 0.0}  # Placeholder
    
    def compare_across_networks(self, results: Dict) -> Dict:
        """Compare sampling bias across different networks"""
        
        print("\n📊 Cross-network comparison...")
        
        comparison = {
            "sdm_comparison": self.compare_sdm_across_networks(results),
            "ci_comparison": self.compare_ci_across_networks(results),
            "network_effect_analysis": self.analyze_network_effects(results),
            "framework_robustness": self.assess_framework_robustness(results)
        }
        
        return comparison
    
    def compare_sdm_across_networks(self, results: Dict) -> Dict:
        """Compare SDM values across networks"""
        sdm_values = {}
        for network, result in results.items():
            sdm_values[network] = result.get("metrics", {}).get("sdm", 0)
        
        return {
            "values": sdm_values,
            "ranking": sorted(sdm_values.items(), key=lambda x: x[1], reverse=True),
            "variance": self.calculate_variance(list(sdm_values.values()))
        }
    
    def compare_ci_across_networks(self, results: Dict) -> Dict:
        """Compare CI values across networks"""
        ci_differences = {}
        for network, result in results.items():
            ci_data = result.get("metrics", {}).get("ci", {})
            cetras_ci = ci_data.get("cetras_ci", 0)
            rwfb_ci = ci_data.get("rwfb_ci", 0)
            ci_differences[network] = abs(cetras_ci - rwfb_ci)
        
        return {
            "differences": ci_differences,
            "most_consistent": min(ci_differences.items(), key=lambda x: x[1])[0],
            "least_consistent": max(ci_differences.items(), key=lambda x: x[1])[0]
        }
    
    def analyze_network_effects(self, results: Dict) -> Dict:
        """Analyze how network properties affect sampling bias"""
        
        effects = {}
        
        for network, result in results.items():
            properties = result.get("properties", {})
            sdm = result.get("metrics", {}).get("sdm", 0)
            
            effects[network] = {
                "transaction_model": properties.get("transaction_model"),
                "anonymity_level": properties.get("anonymity_level"),
                "sdm": sdm,
                "predicted_effect": self.predict_network_effect(properties)
            }
        
        return effects
    
    def predict_network_effect(self, properties: Dict) -> str:
        """Predict how network properties might affect bias"""
        
        if properties.get("anonymity_level") == "private":
            return "high_bias_expected"  # Privacy limits sampling effectiveness
        elif properties.get("smart_contracts"):
            return "moderate_bias_expected"  # Complex interactions
        elif properties.get("transaction_model") == "UTXO":
            return "baseline_bias"  # Similar to Bitcoin
        else:
            return "unknown_effect"
    
    def assess_framework_robustness(self, results: Dict) -> Dict:
        """Assess how robust the framework is across networks"""
        
        successful_analyses = len([r for r in results.values() if "error" not in r])
        total_networks = len(results)
        
        sdm_consistency = self.calculate_sdm_consistency(results)
        
        return {
            "success_rate": successful_analyses / total_networks,
            "sdm_consistency": sdm_consistency,
            "adaptability_score": self.calculate_adaptability_score(results),
            "robustness_rating": self.get_robustness_rating(successful_analyses, total_networks)
        }
    
    def calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def calculate_sdm_consistency(self, results: Dict) -> float:
        """Calculate how consistent SDM values are across networks"""
        sdm_values = [r.get("metrics", {}).get("sdm", 0) for r in results.values()]
        if len(sdm_values) < 2:
            return 1.0
        variance = self.calculate_variance(sdm_values)
        return 1.0 / (1.0 + variance)  # Higher consistency = lower variance
    
    def calculate_adaptability_score(self, results: Dict) -> float:
        """Score how well framework adapts to different networks"""
        adaptation_scores = []
        
        for result in results.values():
            # Count number of successful adaptations
            cetras_adaptations = len(result.get("sampling_configs", {})
                                  .get("cetras", {}).get("adaptations", []))
            rwfb_adaptations = len(result.get("sampling_configs", {})
                                .get("rwfb", {}).get("adaptations", []))
            
            adaptation_scores.append((cetras_adaptations + rwfb_adaptations) / 10)
        
        return sum(adaptation_scores) / len(adaptation_scores) if adaptation_scores else 0
    
    def get_robustness_rating(self, successful: int, total: int) -> str:
        """Get qualitative robustness rating"""
        ratio = successful / total if total > 0 else 0
        
        if ratio >= 0.9:
            return "excellent"
        elif ratio >= 0.7:
            return "good" 
        elif ratio >= 0.5:
            return "moderate"
        else:
            return "poor"


def main():
    parser = argparse.ArgumentParser(description="Run multi-cryptocurrency analysis")
    parser.add_argument("--cryptos", nargs="+", 
                      default=["bitcoin", "ethereum", "litecoin"],
                      help="Cryptocurrencies to analyze")
    parser.add_argument("--output", default="outputs/multi_crypto",
                      help="Output directory")
    
    args = parser.parse_args()
    
    analyzer = MultiCryptoAnalyzer(args.output)
    results = analyzer.run_cross_crypto_analysis(args.cryptos)
    
    # Save results
    output_path = os.path.join(args.output, "multi_crypto_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Multi-crypto analysis complete!")
    print(f"📊 Results saved to {output_path}")
    
    # Print summary
    comparison = results["cross_network_comparison"]
    print(f"\n📈 Summary:")
    print(f"   Networks analyzed: {len(results['individual_results'])}")
    print(f"   Framework robustness: {comparison['framework_robustness']['robustness_rating']}")
    print(f"   Success rate: {comparison['framework_robustness']['success_rate']:.2%}")


if __name__ == "__main__":
    main()
