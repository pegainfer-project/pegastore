//! NUMA / GPU topology from sysfs. Drives the distance function and thread
//! pinning for first-touch allocation.

use std::collections::HashMap;
use std::fs;

use pegastore::Device;

#[derive(Clone, Debug)]
pub struct Topology {
    /// NUMA nodes that actually have CPUs and memory, ascending.
    pub numa_nodes: Vec<u16>,
    pub cpus_by_numa: HashMap<u16, Vec<usize>>,
    /// GPU index → NUMA node (absent if unknown).
    pub gpu_numa: HashMap<u16, u16>,
}

impl Topology {
    /// `gpus` = (index, lowercase pci bus id like `0008:06:00.0`).
    pub fn detect(gpus: &[(u16, String)]) -> Self {
        let mut cpus_by_numa = HashMap::new();
        if let Ok(rd) = fs::read_dir("/sys/devices/system/node") {
            for e in rd.flatten() {
                let name = e.file_name();
                let name = name.to_string_lossy();
                let Some(id) = name.strip_prefix("node").and_then(|s| s.parse::<u16>().ok()) else {
                    continue;
                };
                let cpulist = fs::read_to_string(e.path().join("cpulist")).unwrap_or_default();
                let cpus = parse_cpulist(cpulist.trim());
                if cpus.is_empty() {
                    continue; // memory-only / GPU HBM pseudo-nodes
                }
                cpus_by_numa.insert(id, cpus);
            }
        }
        if cpus_by_numa.is_empty() {
            cpus_by_numa.insert(0, (0..num_cpus()).collect());
        }
        let mut numa_nodes: Vec<u16> = cpus_by_numa.keys().copied().collect();
        numa_nodes.sort_unstable();

        let mut gpu_numa = HashMap::new();
        for (idx, bus) in gpus {
            if let Some(n) = pci_numa(bus) {
                // Some platforms report -1; only accept nodes we know.
                if numa_nodes.contains(&n) {
                    gpu_numa.insert(*idx, n);
                }
            }
        }
        Self {
            numa_nodes,
            cpus_by_numa,
            gpu_numa,
        }
    }

    pub fn numa_of_gpu(&self, gpu: u16) -> Option<u16> {
        self.gpu_numa.get(&gpu).copied()
    }

    /// Cheap ordinal cost between two devices on this node. Only the order
    /// matters:
    /// same device (0) < NVLink peer (1) < local socket DRAM (2)
    /// < unknown affinity / DRAM↔DRAM (3) < cross-socket DRAM (4).
    pub fn distance(&self, a: Device, b: Device) -> u32 {
        if a == b {
            return 0;
        }
        match (a, b) {
            (Device::Gpu { .. }, Device::Gpu { .. }) => 1,
            (Device::Cpu { numa }, Device::Gpu { index }) | (Device::Gpu { index }, Device::Cpu { numa }) => {
                match self.numa_of_gpu(index) {
                    Some(n) if n == numa => 2,
                    Some(_) => 4,
                    None => 3,
                }
            }
            (Device::Cpu { .. }, Device::Cpu { .. }) => 3,
        }
    }

    pub fn describe(&self) -> String {
        let mut s = format!("numa nodes: {:?}", self.numa_nodes);
        let mut gpus: Vec<_> = self.gpu_numa.iter().collect();
        gpus.sort();
        for (g, n) in gpus {
            s.push_str(&format!("; gpu{g}→numa{n}"));
        }
        s
    }
}

fn pci_numa(bus: &str) -> Option<u16> {
    let candidates = [bus.to_string(), normalize_domain(bus)];
    for c in candidates {
        let p = format!("/sys/bus/pci/devices/{c}/numa_node");
        if let Ok(s) = fs::read_to_string(&p)
            && let Ok(v) = s.trim().parse::<i32>()
            && v >= 0
        {
            return Some(v as u16);
        }
    }
    None
}

/// `00000008:06:00.0` → `0008:06:00.0` (sysfs uses a 4-digit domain).
fn normalize_domain(bus: &str) -> String {
    match bus.split_once(':') {
        Some((dom, rest)) if dom.len() > 4 => format!("{}:{rest}", &dom[dom.len() - 4..]),
        _ => bus.to_string(),
    }
}

fn parse_cpulist(s: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in s.split(',').filter(|p| !p.is_empty()) {
        if let Some((a, b)) = part.split_once('-') {
            if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                out.extend(a..=b);
            }
        } else if let Ok(v) = part.parse::<usize>() {
            out.push(v);
        }
    }
    out
}

fn num_cpus() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Pin the calling thread to the CPUs of `numa`.
pub fn pin_to_numa(topo: &Topology, numa: u16) -> bool {
    let Some(cpus) = topo.cpus_by_numa.get(&numa) else {
        return false;
    };
    // SAFETY: cpu_set_t is plain data; sched_setaffinity(0) targets this thread.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for &c in cpus {
            if c < libc::CPU_SETSIZE as usize {
                libc::CPU_SET(c, &mut set);
            }
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
    }
}
