use ark_serialize::CanonicalSerialize;
use ark_groth16::{Proof, ProvingKey};
use ark_bn254::{Bn254, Fr};
use ark_relations::r1cs::ConstraintMatrices;
use rand::Rng;
use std::time::{Duration, Instant};
use std::collections::HashSet;
use std::fs::File;
use color_eyre::{Result, Report};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::env;
use num_cpus;
use lazy_static::lazy_static;
use sysinfo::{System, SystemExt, CpuExt};
use csv::Writer;
use std::path::Path;

// RLN components
use rln::poseidon_tree::PoseidonTree;
use rln::pm_tree_adapter::PmTreeProof;
use rln::hashers::{hash_to_field, poseidon_hash};
use rln::protocol::{generate_proof, proof_values_from_witness, verify_proof, seeded_keygen, rln_witness_from_values, RLNProofValues};
use zerokit_utils::merkle_tree::merkle_tree::ZerokitMerkleTree;

// Configuration
const TPS: usize = 1000;
const TEST_DURATION_SECS: u64 = 100;
const WARMUP_PERCENT: f64 = 0.1;
const RESOURCE_MONITOR_INTERVAL: Duration = Duration::from_secs(5);
const BENCH_TREE_HEIGHT: usize = 22;

const ZKEY_HEIGHT_10: &[u8] = include_bytes!("../resources/tree_height_10/rln_final.zkey");
const GRAPH_HEIGHT_10: &[u8] = include_bytes!("../resources/tree_height_10/graph.bin");

lazy_static! {
    static ref NUM_THREADS: usize = {
        env::var("RLN_THREADS")
            .map(|t| t.parse().unwrap_or_else(|_| get_optimal_threads()))
            .unwrap_or_else(|_| get_optimal_threads())
    };
    static ref OUTPUT_PATH: String = {
        env::var("BENCH_OUTPUT").unwrap_or_else(|_| "benchmark_results.csv".into())
    };
}

fn get_optimal_threads() -> usize {
    std::cmp::max(1, (num_cpus::get() * 4) / 5)
}

#[derive(Debug)]
struct PhaseMetrics {
    total_time: Duration,
    avg_time: Duration,
    p50: Duration,
    p90: Duration,
    p99: Duration,
    max_mem_mb: f64,
    avg_cpu: f64,
    tps: f64,
    proof_sizes: Option<Vec<usize>>,
}

struct ResourceMonitor {
    system: System,
    max_memory: f64,
    cpu_readings: Vec<f64>,
}

impl ResourceMonitor {
    fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            system: System::new_all(),
            max_memory: 0.0,
            cpu_readings: Vec::new(),
        }))
    }

    fn sample(monitor: &Arc<Mutex<Self>>) {
        let mut guard = monitor.lock().unwrap();
        guard.system.refresh_all();
        
        let mem_mb = guard.system.used_memory() as f64 / 1024.0 / 1024.0;
        if mem_mb > guard.max_memory {
            guard.max_memory = mem_mb;
        }
        
        let cpu_usage = guard.system.global_cpu_info().cpu_usage() as f64;
        guard.cpu_readings.push(cpu_usage);
    }

    fn finalize(monitor: Arc<Mutex<Self>>) -> (f64, f64) {
        let guard = monitor.lock().unwrap();
        let avg_cpu = guard.cpu_readings.iter().sum::<f64>() / guard.cpu_readings.len() as f64;
        (guard.max_memory, avg_cpu)
    }
}

fn benchmark_proof_generation(
    proving_key: &(ProvingKey<Bn254>, ConstraintMatrices<Fr>),
    tasks: &[(Fr, PmTreeProof, Vec<u8>, Fr, Fr)],
    monitor: Arc<Mutex<ResourceMonitor>>,
) -> Result<(Vec<(Proof<Bn254>, RLNProofValues)>, PhaseMetrics)> {
    println!("\n🚀 Starting {} transaction benchmark", tasks.len());
    println!("🔧 Configuration:");
    println!("   - Threads: {}", *NUM_THREADS);
    println!("   - Proof System: Groth16");
    println!("   - Circuit: RLN");
    println!("   - Tree Depth: {}", BENCH_TREE_HEIGHT);

    let total_tx = tasks.len();
    let progress_interval = (total_tx / 10).max(1);
    let mut metrics = PhaseMetrics {
        total_time: Duration::ZERO,
        avg_time: Duration::ZERO,
        p50: Duration::ZERO,
        p90: Duration::ZERO,
        p99: Duration::ZERO,
        max_mem_mb: 0.0,
        avg_cpu: 0.0,
        tps: 0.0,
        proof_sizes: Some(Vec::with_capacity(total_tx)),
    };
    
    let start_time = Instant::now();
    let progress = Arc::new((AtomicUsize::new(0), Mutex::new(start_time)));
    
    let results = tasks
        .into_par_iter()
        .map(|(identity_secret, merkle_proof, signal, external_nullifier, message_id)| {
            let (counter, last_print) = &*progress;
            let idx = counter.fetch_add(1, Ordering::Relaxed);
            if idx % progress_interval == 0 {
                let completed = idx + 1;
                let elapsed = start_time.elapsed().as_secs_f64();
                let tps = completed as f64 / elapsed;
                println!(
                    "   █ Progress: {}/{} ({:.0}%) | Elapsed: {:.1}s | Instant TPS: {:.1}",
                    completed,
                    total_tx,
                    (completed as f64 / total_tx as f64) * 100.0,
                    elapsed,
                    tps
                );
            }
            // Witness creation
            let witness_start = Instant::now();
            let witness = rln_witness_from_values(
                *identity_secret,
                merkle_proof,
                hash_to_field(signal),
                *external_nullifier,
                Fr::from(100u64),
                *message_id,
            )?;

            // Proof generation
            let proof_start = Instant::now();
            let proof = generate_proof(proving_key, &witness, GRAPH_HEIGHT_10)?;
            let proof_time = proof_start.elapsed();

            // Proof size
            let mut serialized = Vec::new();
            proof.serialize_compressed(&mut serialized)?;
            let proof_size = serialized.len();

            if idx % 100 == 0 {
                let mut last = last_print.lock().unwrap();
                if last.elapsed() >= RESOURCE_MONITOR_INTERVAL {
                    ResourceMonitor::sample(&monitor);
                    *last = Instant::now();
                }
            }

            Ok((proof, proof_values_from_witness(&witness)?, witness_start.elapsed(), proof_time, proof_size))
        })
        .collect::<Result<Vec<_>>>()?;

    let (proofs, times, _witness_times, proof_sizes) = results.into_iter().fold(
        (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
        |mut acc, (proof, values, wt, pt, size)| {
            acc.0.push((proof, values));
            acc.1.push(wt + pt);
            acc.2.push(wt);
            acc.3.push(size);
            acc
        },
    );

    metrics.total_time = start_time.elapsed();
    metrics.avg_time = metrics.total_time / total_tx as u32;
    metrics.p50 = calculate_percentile(&times, 50.0);
    metrics.p90 = calculate_percentile(&times, 90.0);
    metrics.p99 = calculate_percentile(&times, 99.0);
    metrics.tps = total_tx as f64 / metrics.total_time.as_secs_f64();
    metrics.proof_sizes.as_mut().unwrap().extend(proof_sizes);

    let (max_mem, avg_cpu) = ResourceMonitor::finalize(monitor);
    metrics.max_mem_mb = max_mem;
    metrics.avg_cpu = avg_cpu;

    println!(
        "\n✅ Completed {} transactions in {:.2}s (avg {:.2} TPS)",
        total_tx,
        metrics.total_time.as_secs_f64(),
        metrics.tps
    );
    println!("{}", generate_performance_report(&metrics, "Generation"));

    Ok((proofs, metrics))
}

fn calculate_percentile(times: &[Duration], percentile: f64) -> Duration {
    let mut sorted = times.to_vec();
    sorted.sort();
    let idx = (sorted.len() as f64 * percentile / 100.0).ceil() as usize - 1;
    sorted[idx.min(sorted.len() - 1)]
}

fn write_metrics(writer: &mut csv::Writer<File>, phase: &str, metrics: &PhaseMetrics) -> Result<()> {
    writer.write_record(&[
        phase,
        &format!("{:.2}s", metrics.total_time.as_secs_f64()),
        &format!("{:.2}ms", metrics.avg_time.as_secs_f64() * 1000.0),
        &format!("{:.2}ms", metrics.p50.as_secs_f64() * 1000.0),
        &format!("{:.2}ms", metrics.p90.as_secs_f64() * 1000.0),
        &format!("{:.2}ms", metrics.p99.as_secs_f64() * 1000.0),
        &format!("{:.2}", metrics.tps),
        &format!("{:.2}MB", metrics.max_mem_mb),
        &format!("{:.1}%", metrics.avg_cpu),
        &metrics.proof_sizes.as_ref().map_or(0, |v| v.len()).to_string(),
    ])?;
    Ok(())
}

fn generate_performance_report(metrics: &PhaseMetrics, phase: &str) -> String {
    let per_tx_ms = metrics.avg_time.as_secs_f64() * 1000.0;
    let proof_size_avg = metrics.proof_sizes.as_ref().map_or(0.0, |v| {
        v.iter().sum::<usize>() as f64 / v.len() as f64
    });

    format!(
        "{} Performance Report:
        - Throughput: {:.2} tx/s (max theoretical)
        - Latency: {:.2} ms per transaction (avg)
        - Peak Memory Usage: {:.2} MB
        - CPU Utilization: {:.1}% (avg)
        - Proof Size: {:.2} KB (avg)
        - Estimated 1000 tx Time: {:.2} seconds
        - Reliability: p99 latency {:.2} ms",
        phase,
        metrics.tps,
        per_tx_ms,
        metrics.max_mem_mb,
        metrics.avg_cpu,
        proof_size_avg / 1024.0,
        1000.0 / metrics.tps,
        metrics.p99.as_secs_f64() * 1000.0
    )
}

// Helper function to load tree height 10 zkey from bytes
fn zkey_from_raw_height_10() -> Result<(ProvingKey<Bn254>, ConstraintMatrices<Fr>)> {
    use std::io::Cursor;
    
    let mut reader = Cursor::new(ZKEY_HEIGHT_10);
    ark_circom::read_zkey(&mut reader).map_err(|e| Report::msg(format!("Failed to read zkey: {}", e)))
}

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;
    let total_tx = TPS * TEST_DURATION_SECS as usize;
    
    let mut csv_writer = Writer::from_path(Path::new(&*OUTPUT_PATH))?;
    csv_writer.write_record(&[
        "Phase", "TotalTime", "AvgTime", "P50", "P90", "P99", "TPS", "MaxMemMB", "AvgCPU", "Proofs"
    ])?;

    rayon::ThreadPoolBuilder::new()
        .num_threads(*NUM_THREADS)
        .build_global()?;

    // Load tree height 10 resources
    let pk_and_matrices = zkey_from_raw_height_10()?;
    let verifying_key = pk_and_matrices.0.vk.clone();

    // Use tree height 10 for the Merkle tree
    let mut tree = PoseidonTree::new(BENCH_TREE_HEIGHT, Fr::from(0), Default::default())?;
    let mut tasks = Vec::with_capacity(total_tx);
    let mut rng = rand::thread_rng();

    for i in 0..total_tx {
        if i % 100 == 0 {
            tree.update_next(poseidon_hash(&[Fr::from(i as u64), Fr::from(100u64)]))?;
        }

        let (id_secret, id_commit) = seeded_keygen(format!("user-{}", i).as_bytes());
        let rate_commit = poseidon_hash(&[id_commit, Fr::from(100u64)]);
        tree.set(i, rate_commit)?;

        tasks.push((
            id_secret,
            tree.proof(i)?,
            (0..32).map(|_| rng.gen()).collect(),
            hash_to_field(format!("epoch-{}", i / TPS).as_bytes()),
            Fr::from((i % 99 + 1) as u64),
        ));
    }

    // Split tasks into warmup and main
    let warmup_tx = (total_tx as f64 * WARMUP_PERCENT) as usize;
    let (warmup_tasks, main_tasks) = tasks.split_at(warmup_tx);

    // Warm-up phase
    let warmup_mon = ResourceMonitor::new();
    let (_, warmup_metrics) = benchmark_proof_generation(
        &pk_and_matrices,
        &warmup_tasks,
        warmup_mon.clone(),
    )?;
    write_metrics(&mut csv_writer, "Warmup", &warmup_metrics)?;

    // Main benchmark
    let main_mon = ResourceMonitor::new();
    let (proofs, gen_metrics) = benchmark_proof_generation(
        &pk_and_matrices,
        &main_tasks,
        main_mon.clone(),
    )?;
    write_metrics(&mut csv_writer, "Generation", &gen_metrics)?;

    // Verification phase
    let verify_start = Instant::now();
    let verify_times = proofs.par_iter()
        .map(|(proof, values)| {
            let start = Instant::now();
            verify_proof(&verifying_key, proof, values)?;
            Ok(start.elapsed())
        })
        .collect::<Result<Vec<_>>>()?;
    
    let (max_mem, avg_cpu) = ResourceMonitor::finalize(main_mon.clone());
    let verify_metrics = PhaseMetrics {
        total_time: verify_start.elapsed(),
        avg_time: verify_times.iter().sum::<Duration>() / total_tx as u32,
        p50: calculate_percentile(&verify_times, 50.0),
        p90: calculate_percentile(&verify_times, 90.0),
        p99: calculate_percentile(&verify_times, 99.0),
        max_mem_mb: max_mem,
        avg_cpu,
        tps: total_tx as f64 / verify_start.elapsed().as_secs_f64(),
        proof_sizes: None,
    };
    write_metrics(&mut csv_writer, "Verification", &verify_metrics)?;

    // Blacklist benchmark
    let blacklist_start = Instant::now();
    let mut blacklist = HashSet::new();
    (0..total_tx).for_each(|i| { blacklist.insert(i); });
    
    let blacklist_metrics = PhaseMetrics {
        total_time: blacklist_start.elapsed(),
        avg_time: blacklist_start.elapsed() / total_tx as u32,
        p50: blacklist_start.elapsed() / total_tx as u32,
        p90: blacklist_start.elapsed() / total_tx as u32,
        p99: blacklist_start.elapsed() / total_tx as u32,
        max_mem_mb: max_mem,
        avg_cpu,
        tps: total_tx as f64 / blacklist_start.elapsed().as_secs_f64(),
        proof_sizes: None,
    };
    write_metrics(&mut csv_writer, "Blacklist", &blacklist_metrics)?;
    let final_report = format!(
        "\n📈 Final Performance Summary:
        Total System Throughput: {:.2} tx/s
        Verification Efficiency: {:.2} ms/verify
        Resource Utilization:
          - Peak Memory: {:.2} MB
          - Average CPU: {:.1}%
        Capacity Planning:
          - 1,000 tx would take: {:.2} seconds
          - 10,000 tx would take: {:.2} minutes
          - Max Daily Capacity: {:.2} million tx",
        gen_metrics.tps,
        verify_metrics.avg_time.as_secs_f64() * 1000.0,
        gen_metrics.max_mem_mb,
        gen_metrics.avg_cpu,
        1000.0 / gen_metrics.tps,
        (10000.0 / gen_metrics.tps) / 60.0,
        (gen_metrics.tps * 86400.0) / 1_000_000.0
    );

    println!("{}", final_report);
    std::fs::write("performance_summary.txt", final_report)?;
    println!("\n📄 Benchmark report saved to {}", *OUTPUT_PATH);
    Ok(())
}