# RLN Benchmark

A benchmarking tool for Vac's RLN performance in a rollup environment.

## Overview

This benchmark measures three key components of RLN:
1. **Proof Generation**: Creating ZK proofs for transactions
2. **Proof Verification**: Verifying the generated proofs
3. **Blacklist Updates**: Maintaining the spam prevention list

## Configuration

The benchmark can be configured through constants in `src/main.rs`: