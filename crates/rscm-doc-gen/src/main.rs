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

    // Parse every file first, so a component can find its parameters struct in another file
    let mut files = Vec::new();
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
        let crate_name = crate_path
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_default();

        for entry in WalkDir::new(&src_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.extension().map(|e| e == "rs").unwrap_or(false) {
                if let Some(syntax) = parser::read_file(path) {
                    files.push((crate_name.clone(), path.to_path_buf(), syntax));
                }
            }
        }
    }

    let mut index = parser::StructIndex::new();
    for (crate_name, _, syntax) in &files {
        parser::index_structs(crate_name, syntax, &mut index);
    }

    let mut all_components: Vec<ComponentDocMetadata> = Vec::new();
    for (crate_name, path, syntax) in &files {
        for mut component in parser::parse_file(crate_name, path, syntax, &index) {
            // Make the source file path relative to workspace root
            component.source_file = make_relative_path(path);

            if args.verbose {
                println!(
                    "  Found component: {} ({} inputs, {} outputs, {} states, {} parameters)",
                    component.name,
                    component.inputs.len(),
                    component.outputs.len(),
                    component.states.len(),
                    component.parameters.len()
                );
            }

            all_components.push(component);
        }
    }

    // Write JSON files
    for component in &all_components {
        let filename = format!("{}.json", component.slug);
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
