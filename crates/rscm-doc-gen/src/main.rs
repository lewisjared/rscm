//! RSCM Documentation Generator
//!
//! A CLI tool that extracts component metadata from Rust source files
//! and generates JSON documentation files for use by MkDocs.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p rscm-doc-gen -- \
//!   --crates crates/rscm-components,crates/rscm-two-layer \
//!   --output docs/component_metadata/
//! ```

mod extractor;
mod parser;
mod schema;
mod stub_generator;

use clap::Parser;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::schema::ComponentDocMetadata;

/// RSCM component documentation generator
#[derive(Parser, Debug)]
#[command(name = "rscm-doc-gen")]
#[command(about = "Generate JSON documentation metadata from RSCM component source files")]
struct Args {
    /// Comma-separated list of crate paths to scan
    #[arg(short, long, value_delimiter = ',')]
    crates: Vec<PathBuf>,

    /// Output directory for JSON files
    #[arg(short, long)]
    output: PathBuf,

    /// Generate Python type stubs (.pyi files)
    #[arg(long)]
    generate_stubs: bool,

    /// Output directory for Python type stubs
    #[arg(long)]
    stubs_output: Option<PathBuf>,

    /// Print parsed components (verbose mode)
    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    // Create output directory if it doesn't exist
    if let Err(e) = fs::create_dir_all(&args.output) {
        eprintln!("Failed to create output directory: {}", e);
        std::process::exit(1);
    }

    let mut all_components: Vec<ComponentDocMetadata> = Vec::new();

    // Process each crate
    for crate_path in &args.crates {
        if !crate_path.exists() {
            eprintln!(
                "Warning: Crate path does not exist: {}",
                crate_path.display()
            );
            continue;
        }

        let src_path = crate_path.join("src");
        if !src_path.exists() {
            eprintln!("Warning: No src directory in {}", crate_path.display());
            continue;
        }

        if args.verbose {
            println!("Scanning {}", crate_path.display());
        }

        // Walk through all .rs files
        for entry in WalkDir::new(&src_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.extension().map(|e| e == "rs").unwrap_or(false) {
                let components = parser::parse_file(path);

                for mut component in components {
                    // Make the source file path relative to workspace root
                    component.source_file = make_relative_path(path);

                    if args.verbose {
                        println!(
                            "  Found component: {} ({} inputs, {} outputs, {} states)",
                            component.name,
                            component.inputs.len(),
                            component.outputs.len(),
                            component.states.len()
                        );
                        if !component.tags.is_empty() {
                            println!("    Tags: {:?}", component.tags);
                        }
                        if let Some(ref cat) = component.category {
                            println!("    Category: {}", cat);
                        }
                    }

                    all_components.push(component);
                }
            }
        }
    }

    // Write JSON files
    for component in &all_components {
        let filename = format!("{}.json", component.name.to_lowercase());
        let output_path = args.output.join(&filename);

        let json = serde_json::to_string_pretty(component).expect("Failed to serialize JSON");
        let json_with_newline = format!("{}\n", json);

        if let Err(e) = fs::write(&output_path, json_with_newline) {
            eprintln!("Failed to write {}: {}", output_path.display(), e);
        } else if args.verbose {
            println!("Wrote {}", output_path.display());
        }
    }

    println!(
        "Generated {} component metadata files in {}",
        all_components.len(),
        args.output.display()
    );

    // Generate Python type stubs if requested
    if args.generate_stubs {
        let stubs_output = args
            .stubs_output
            .unwrap_or_else(|| PathBuf::from("python/rscm/_lib"));

        if let Err(e) = fs::create_dir_all(&stubs_output) {
            eprintln!("Failed to create stubs output directory: {}", e);
            std::process::exit(1);
        }

        match stub_generator::generate_stubs(&all_components, &stubs_output) {
            Ok(_) => {
                println!(
                    "Generated {} Python stub files in {}",
                    all_components.len(),
                    stubs_output.display()
                );
            }
            Err(e) => {
                eprintln!("Failed to generate Python stubs: {}", e);
                std::process::exit(1);
            }
        }
    }
}

/// Convert an absolute path to a workspace-relative path
fn make_relative_path(path: &Path) -> String {
    let path_str = path.to_string_lossy();

    // Try to find "crates/" in the path and use that as the relative root
    if let Some(idx) = path_str.find("crates/") {
        return path_str[idx..].to_string();
    }

    // Otherwise return the full path
    path_str.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_relative_path() {
        let path =
            Path::new("/Users/jared/code/rust/rscm/crates/rscm-components/src/carbon_cycle.rs");
        let relative = make_relative_path(path);
        assert_eq!(relative, "crates/rscm-components/src/carbon_cycle.rs");
    }
}
