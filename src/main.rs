//! RLN Benchmarking
//! 
//! This benchmark simulates a rollup environment with configurable parameters
//! 
//! The benchmark measures three key components:
//! 1. Proof Generation: Creating ZK proofs for each transaction
//! 2. Proof Verification: Verifying the generated proofs
//! 3. Blacklist Updates: Maintaining the spam prevention list
//! 
//! Each transaction requires:
//! - Generating a Merkle proof
//! - Creating a ZK proof (most computationally intensive)
//! - Verifying the proof
//! - Updating the blacklist

use ark_groth16::{Proof, ProvingKey, VerifyingKey};
use ark_bn254::Bn254;
use rand::Rng;
use std::time::{Duration, Instant};
use std::collections::HashSet;
use color_eyre::Result;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rln::circuit::{Fr, TEST_TREE_HEIGHT, zkey_from_folder, vk_from_folder};
use rln::poseidon_tree::PoseidonTree;
use rln::pm_tree_adapter::PmTreeProof;
use rln::hashers::{hash_to_field, poseidon_hash};
use rln::protocol::{generate_proof, proof_values_from_witness, verify_proof, seeded_keygen, rln_witness_from_values, RLNProofValues};
use zerokit_utils::merkle_tree::merkle_tree::ZerokitMerkleTree;

const TPS: usize = 10;
const TEST_DURATION_SECS: u64 = 10;
const NUM_THREADS: usize = 8;
const TOTAL_TRANSACTIONS: usize = TPS * TEST_DURATION_SECS as usize;

fn print_section_header(title: &str) {
    println!("\n{}", "=".repeat(80));
    println!("  {}", title);
    println!("{}\n", "=".repeat(80));
}

fn benchmark_proof_generation(
    proving_key: &(ProvingKey<Bn254>, ark_relations::r1cs::ConstraintMatrices<Fr>),
    tasks: Vec<(Fr, PmTreeProof, Vec<u8>, Fr, Fr)>,
) -> Vec<(Proof<Bn254>, RLNProofValues, Duration)> {
    let task_count = tasks.len();
    let start_time = Instant::now();
    let progress_interval = task_count.max(10) / 10;
    let completed = Arc::new(AtomicUsize::new(0));
    let stdout_mutex = Arc::new(Mutex::new(()));

    println!("Starting parallel proof generation for {} transactions...", task_count);
    println!("Progress will be shown every {}% ({} transactions)\n", 10, progress_interval);

    let results = tasks
        .into_par_iter()
        .map(|(identity_secret, merkle_proof, signal, external_nullifier, message_id)| {
            let proof_start = Instant::now();
            
            let witness = rln_witness_from_values(
                identity_secret,
                &merkle_proof,
                hash_to_field(&signal),
                external_nullifier,
                Fr::from(100u64),
                message_id,
            ).unwrap();

            let proof = generate_proof(proving_key, &witness).unwrap();
            let proof_values = proof_values_from_witness(&witness).unwrap();
            let proof_time = proof_start.elapsed();

            // Progress reporting
            let current = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if current % progress_interval == 0 || current == task_count {
                let elapsed = start_time.elapsed();
                let tps = current as f64 / elapsed.as_secs_f64();
                let _lock = stdout_mutex.lock().unwrap();
                println!(
                    "Progress: {}/{} ({}%) - Elapsed: {:.2?} - Current TPS: {:.2}",
                    current,
                    task_count,
                    (current * 100) / task_count,
                    elapsed,
                    tps
                );
            }

            (proof, proof_values, proof_time)
        })
        .collect();

    println!("\nProof generation completed.");
    results
}

fn benchmark_proof_verification(
    verifying_key: &VerifyingKey<Bn254>,
    proofs: Vec<(Proof<Bn254>, RLNProofValues)>,
) -> Vec<Duration> {
    let proof_count = proofs.len();
    println!("Starting parallel proof verification for {} proofs...", proof_count);
    
    let start_time = Instant::now();
    let verify_times: Vec<Duration> = proofs.par_iter()
        .enumerate()
        .map(|(idx, (proof, proof_values))| {
            let verify_start = Instant::now();
            verify_proof(verifying_key, proof, proof_values).unwrap();
            let verify_time = verify_start.elapsed();
            
            if (idx + 1) % (proof_count / 10).max(1) == 0 {
                println!(
                    "Verified {}/{} proofs ({}%) - Current time: {:.2?}",
                    idx + 1,
                    proof_count,
                    ((idx + 1) * 100) / proof_count,
                    start_time.elapsed()
                );
            }
            verify_time
        })
        .collect();
    
    println!("Proof verification completed.\n");
    verify_times
}

fn benchmark_blacklist_update(blacklist: &mut HashSet<usize>, updates: Vec<usize>) -> Duration {
    let update_count = updates.len();
    println!("Starting blacklist updates for {} entries...", update_count);
    
    let start = Instant::now();
    for (idx, user_idx) in updates.into_iter().enumerate() {
        blacklist.insert(user_idx);
        
        if (idx + 1) % (update_count / 10).max(1) == 0 {
            println!(
                "Updated {}/{} entries ({}%) - Current time: {:.2?}",
                idx + 1,
                update_count,
                ((idx + 1) * 100) / update_count,
                start.elapsed()
            );
        }
    }
    
    let total_time = start.elapsed();
    println!("Blacklist updates completed.\n");
    total_time
}

fn main() -> Result<()> {
    print_section_header("RLN Benchmark Configuration");
    println!("Target TPS: {}", TPS);
    println!("Duration: {} seconds", TEST_DURATION_SECS);
    println!("Total Transactions: {}", TOTAL_TRANSACTIONS);
    println!("Parallel Threads: {}", NUM_THREADS);

    print_section_header("Initialization");
    println!("Loading proving and verifying keys...");
    let proving_key = zkey_from_folder();
    let verifying_key = vk_from_folder();
    println!("✓ Keys loaded successfully");

    let mut rng = rand::thread_rng();
    
    print_section_header("Test Data Preparation");
    println!("Building Merkle tree and generating {} transactions...", TOTAL_TRANSACTIONS);
    
    let mut tasks = Vec::with_capacity(TOTAL_TRANSACTIONS);
    let mut tree = PoseidonTree::new(TEST_TREE_HEIGHT, Fr::from(0), Default::default())?;

    // Precompute external nullifiers for each epoch
    let num_epochs = (TOTAL_TRANSACTIONS + TPS - 1) / TPS;
    let mut external_nullifiers = Vec::with_capacity(num_epochs);
    for epoch in 0..num_epochs {
        external_nullifiers.push(hash_to_field(format!("epoch-{}", epoch).as_bytes()));
    }

    println!("Building Merkle tree and generating tasks...");
    for i in 0..TOTAL_TRANSACTIONS {
        if i % 1000 == 0 {
            println!("Prepared {} tasks...", i);
        }
        
        let (identity_secret, id_commitment) = seeded_keygen(format!("user-{}", i).as_bytes());
        let rate_commitment = poseidon_hash(&[id_commitment, Fr::from(100u64)]);
        tree.set(i, rate_commitment)?;
        
        let signal: Vec<u8> = (0..32).map(|_| rng.gen()).collect();
        let epoch = i / TPS;
        let merkle_proof = tree.proof(i)?;
        
        tasks.push((
            identity_secret,
            merkle_proof,
            signal,
            external_nullifiers[epoch],
            Fr::from((i % 99 + 1) as u64),
        ));
    }
    println!("Task preparation complete.");

    print_section_header("Proof Generation Benchmark");
    let proof_gen_start = Instant::now();
    let proofs_with_times = benchmark_proof_generation(&proving_key, tasks);
    let total_gen_time = proof_gen_start.elapsed();
    
    let avg_gen_time: Duration = proofs_with_times.iter()
        .map(|(_, _, time)| *time)
        .sum::<Duration>() / TOTAL_TRANSACTIONS as u32;
    
    print_section_header("Proof Verification Benchmark");
    let proofs: Vec<(Proof<Bn254>, RLNProofValues)> = proofs_with_times.into_iter()
        .map(|(proof, values, _)| (proof, values))
        .collect();
    
    let verify_start = Instant::now();
    let verify_times = benchmark_proof_verification(&verifying_key, proofs);
    let total_verify_time = verify_start.elapsed();
    
    let avg_verify_time: Duration = verify_times.iter().sum::<Duration>() / TOTAL_TRANSACTIONS as u32;
    
    print_section_header("Blacklist Update Benchmark");
    let mut blacklist = HashSet::new();
    let updates: Vec<usize> = (0..TOTAL_TRANSACTIONS).collect();
    
    let blacklist_time = benchmark_blacklist_update(&mut blacklist, updates);
    
    print_section_header("Final Benchmark Results");
    println!("\nProof Generation Performance:");
    println!("├─ Total time: {:?}", total_gen_time);
    println!("├─ Average per proof: {:?}", avg_gen_time);
    println!("├─ Effective TPS: {:.2}", TOTAL_TRANSACTIONS as f64 / total_gen_time.as_secs_f64());

    println!("\nProof Verification Performance:");
    println!("├─ Total time: {:?}", total_verify_time);
    println!("├─ Average per verification: {:?}", avg_verify_time);
    println!("├─ Effective TPS: {:.2}", TOTAL_TRANSACTIONS as f64 / total_verify_time.as_secs_f64());

    println!("\nBlacklist Update Performance:");
    println!("├─ Total time: {:?}", blacklist_time);
    println!("├─ Average per update: {:?}", blacklist_time / TOTAL_TRANSACTIONS as u32);
    println!("├─ Updates per second: {:.2}", TOTAL_TRANSACTIONS as f64 / blacklist_time.as_secs_f64());
    println!("└─ Analysis: Negligible overhead");

    Ok(())
}