//! Bring a new model family into the engine.
//!
//!   lb-addmodel probe    --gguf FILE [--json]   what the file states, and what is still unknown
//!   lb-addmodel scaffold --gguf FILE [--out DIR] write a config.rs to finish by hand
use anyhow::Result;
use clap::{Parser, Subcommand};
use lb_addmodel::probe::{probe, Origin};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Probe {
        #[arg(long)]
        gguf: PathBuf,
        #[arg(long)]
        json: bool,
    },
    Scaffold {
        #[arg(long)]
        gguf: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    match Args::parse().cmd {
        Cmd::Probe { gguf, json } => {
            let p = probe(&gguf)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&p)?);
                return Ok(());
            }
            println!("arch: {}   moe layout: {:?}", p.arch, p.moe_layout);
            println!(
                "shared expert: {}   router bias: {}   q/k norm: {}   MTP: {}",
                p.has_shared_expert, p.has_router_bias, p.has_qk_norm, p.has_mtp
            );
            let (mut ok, mut def, mut miss) = (0, 0, 0);
            println!("\n{:<22} {:<12} {}", "field", "value", "origin");
            println!("{}", "-".repeat(78));
            for f in &p.fields {
                let v = f.value.clone().unwrap_or_else(|| "-".into());
                let o = match &f.origin {
                    Origin::Key(k) => {
                        ok += 1;
                        k.clone()
                    }
                    Origin::Default(d) => {
                        def += 1;
                        format!("DEFAULT {d} (absent)")
                    }
                    Origin::Missing => {
                        miss += 1;
                        "** MISSING - required **".into()
                    }
                };
                println!("{:<22} {:<12} {}", f.name, v, o);
            }
            println!("\n{ok} resolved from the file, {def} defaulted, {miss} missing");
            println!("\ntensor groups ({}):", p.tensor_groups.len());
            for g in &p.tensor_groups {
                println!("  {:<44} x{:<4} {:?} {}", g.pattern, g.count, g.dims, g.ty);
            }
            if !p.unclaimed_keys.is_empty() {
                println!("\nunmodelled `{}.*` keys:", p.arch);
                for k in &p.unclaimed_keys {
                    println!("  {k} = {}", p.arch_keys.get(k).cloned().unwrap_or_default());
                }
            }
            for n in &p.notes {
                println!("\nnote: {n}");
            }
        }
        Cmd::Scaffold { gguf, out } => {
            let p = probe(&gguf)?;
            let src = lb_addmodel::scaffold::config_rs(&p);
            match out {
                Some(d) => {
                    std::fs::create_dir_all(&d)?;
                    let f = d.join("config.rs");
                    std::fs::write(&f, &src)?;
                    println!("wrote {}", f.display());
                    let miss: Vec<&str> = p.fields.iter().filter(|f| f.origin == Origin::Missing).map(|f| f.name).collect();
                    if !miss.is_empty() {
                        println!("{} todo!() left for: {}", miss.len(), miss.join(", "));
                    }
                }
                None => print!("{src}"),
            }
        }
    }
    Ok(())
}
