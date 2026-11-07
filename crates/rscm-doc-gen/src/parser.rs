//! Rust source file parser for extracting component metadata

use crate::extractor::{extract_description, extract_doc_comments};
use crate::schema::{ComponentDocMetadata, DeclaredVariable, ParameterMetadata};
use std::fs;
use std::path::Path;
use syn::{Attribute, Fields, Item, ItemStruct, Meta, MetaList, Type};

/// Parse a Rust source file and extract component metadata
pub fn parse_file(path: &Path) -> Vec<ComponentDocMetadata> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to read {}: {}", path.display(), e);
            return Vec::new();
        }
    };

    let syntax = match syn::parse_file(&content) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to parse {}: {}", path.display(), e);
            return Vec::new();
        }
    };

    // First pass: collect all structs by name for parameter struct lookup
    let mut all_structs: std::collections::HashMap<String, &ItemStruct> =
        std::collections::HashMap::new();
    for item in &syntax.items {
        if let Item::Struct(item_struct) = item {
            all_structs.insert(item_struct.ident.to_string(), item_struct);
        }
    }

    let mut components = Vec::new();

    // Extract module-level documentation
    let module_docs = extract_doc_comments(&syntax.attrs);

    // Second pass: parse components and resolve their parameter structs
    for item in &syntax.items {
        if let Item::Struct(item_struct) = item {
            if let Some(metadata) =
                parse_component_struct(item_struct, path, &module_docs, &all_structs)
            {
                components.push(metadata);
            }
        }
    }

    components
}

/// Check if a struct has the ComponentIO derive macro
fn has_component_io_derive(attrs: &[Attribute]) -> bool {
    for attr in attrs {
        if attr.path().is_ident("derive") {
            if let Meta::List(MetaList { tokens, .. }) = &attr.meta {
                let tokens_str = tokens.to_string();
                if tokens_str.contains("ComponentIO") {
                    return true;
                }
            }
        }
    }
    false
}

/// Parse a struct that uses ComponentIO derive macro
fn parse_component_struct(
    item: &ItemStruct,
    source_path: &Path,
    module_docs: &str,
    all_structs: &std::collections::HashMap<String, &ItemStruct>,
) -> Option<ComponentDocMetadata> {
    // Only process structs with ComponentIO derive
    if !has_component_io_derive(&item.attrs) {
        return None;
    }

    let struct_name = item.ident.to_string();
    let struct_docs = extract_doc_comments(&item.attrs);

    // The page heading names the component, so the summary lines that title each doc block repeat it.
    let module_body = drop_summary_line(module_docs);
    let struct_body = if module_docs.is_empty() {
        struct_docs.as_str()
    } else {
        drop_summary_line(&struct_docs)
    };
    // Both introductions come before any section, so the struct's does not read as part of the last one.
    let (module_intro, module_sections) = split_intro(module_body);
    let (struct_intro, struct_sections) = split_intro(struct_body);
    let combined_docs = [module_intro, struct_intro, module_sections, struct_sections]
        .iter()
        .map(|docs| docs.trim())
        .filter(|docs| !docs.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");

    let mut metadata = ComponentDocMetadata {
        name: struct_name.clone(),
        module_path: infer_module_path(source_path),
        description: extract_description(&combined_docs),
        source_file: source_path.to_string_lossy().to_string(),
        ..Default::default()
    };

    // Parse struct-level attributes
    for attr in &item.attrs {
        if attr.path().is_ident("component") {
            parse_component_attr(attr, &mut metadata);
        } else if attr.path().is_ident("inputs") {
            if let Ok(vars) = parse_io_attr(attr) {
                metadata.inputs = vars;
            }
        } else if attr.path().is_ident("outputs") {
            if let Ok(vars) = parse_io_attr(attr) {
                metadata.outputs = vars;
            }
        } else if attr.path().is_ident("states") {
            if let Ok(vars) = parse_io_attr(attr) {
                metadata.states = vars;
            }
        }
    }

    // Parse struct fields and resolve parameter structs
    if let Fields::Named(fields) = &item.fields {
        for field in &fields.named {
            let field_type = type_to_string(&field.ty);

            // Check if this field's type is a *Parameters struct we can expand
            if let Some(params_struct) = all_structs.get(&field_type) {
                // Extract fields from the parameters struct
                if let Fields::Named(param_fields) = &params_struct.fields {
                    for param_field in &param_fields.named {
                        if let Some(param) = parse_parameter_field(param_field) {
                            metadata.parameters.push(param);
                        }
                    }
                }
            }
        }
    }

    Some(metadata)
}

/// Drop a doc block's summary line when it is a one-line paragraph followed by more text
fn drop_summary_line(docs: &str) -> &str {
    match docs.split_once("\n\n") {
        Some((summary, rest)) if !summary.contains('\n') => rest.trim_start(),
        _ => docs,
    }
}

/// Split docs into the text before the first heading and the sections from it on
fn split_intro(docs: &str) -> (&str, &str) {
    let mut in_fence = false;
    let mut offset = 0;
    for line in docs.split_inclusive('\n') {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
        } else if !in_fence && line.starts_with('#') {
            return docs.split_at(offset);
        }
        offset += line.len();
    }
    (docs, "")
}

/// Parse #[component(tags = [...], category = "...")] attribute
fn parse_component_attr(attr: &Attribute, metadata: &mut ComponentDocMetadata) {
    if let Meta::List(MetaList { tokens, .. }) = &attr.meta {
        let tokens_str = tokens.to_string();

        // Parse tags = ["tag1", "tag2"]
        if let Some(tags_start) = tokens_str.find("tags = [") {
            let tags_part = &tokens_str[tags_start + 8..];
            if let Some(tags_end) = tags_part.find(']') {
                let tags_content = &tags_part[..tags_end];
                metadata.tags = tags_content
                    .split(',')
                    .map(|s| s.trim().trim_matches('"').to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
        }

        // Parse category = "..."
        if let Some(cat_start) = tokens_str.find("category = ") {
            let cat_part = &tokens_str[cat_start + 11..];
            let cat_part = cat_part.trim_start_matches('"');
            if let Some(cat_end) = cat_part.find('"') {
                metadata.category = Some(cat_part[..cat_end].to_string());
            }
        }
    }
}

/// Parse #[inputs(...)], #[outputs(...)], or #[states(...)] attribute
fn parse_io_attr(attr: &Attribute) -> Result<Vec<DeclaredVariable>, ()> {
    let mut variables = Vec::new();

    if let Meta::List(MetaList { tokens, .. }) = &attr.meta {
        // Parse the tokens as a string and extract field definitions
        // Format: field_name { name = "...", unit = "...", grid = "...", timeseries_type = "..." }
        let tokens_str = tokens.to_string();

        let re = regex::Regex::new(
            r#"(\w+)\s*\{\s*name\s*=\s*"([^"]+)"(?:\s*,\s*unit\s*=\s*"([^"]*)")?(?:\s*,\s*grid\s*=\s*"([^"]*)")?(?:\s*,\s*timeseries_type\s*=\s*"[^"]*")?\s*\}"#,
        )
        .expect("Invalid regex");

        for cap in re.captures_iter(&tokens_str) {
            let rust_name = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            let var_name = cap.get(2).map(|m| m.as_str()).unwrap_or("");
            let unit = cap.get(3).map(|m| m.as_str()).unwrap_or("");
            let grid = cap.get(4).map(|m| m.as_str()).unwrap_or("scalar");

            variables.push(DeclaredVariable {
                rust_name: rust_name.to_string(),
                name: var_name.to_string(),
                unit: unit.to_string(),
                grid: grid.to_string(),
                description: String::new(),
            });
        }
    }

    Ok(variables)
}

/// Parse a struct field as a parameter
fn parse_parameter_field(field: &syn::Field) -> Option<ParameterMetadata> {
    let name = field.ident.as_ref()?.to_string();
    let param_type = type_to_string(&field.ty);
    let docs = extract_doc_comments(&field.attrs);

    Some(ParameterMetadata {
        name,
        param_type,
        unit: String::new(),
        description: docs,
        default: None,
    })
}

/// Convert a syn::Type to a string representation
fn type_to_string(ty: &Type) -> String {
    quote::quote!(#ty).to_string().replace(" ", "")
}

/// Infer the module path from the source file path
fn infer_module_path(path: &Path) -> String {
    // Convert path like "crates/rscm-components/src/components/carbon_cycle.rs"
    // to "rscm_components::components::carbon_cycle"
    let path_str = path.to_string_lossy();

    // Find "crates/" in the path and start from there
    let path_str = if let Some(idx) = path_str.find("crates/") {
        &path_str[idx..]
    } else {
        &path_str
    };

    // Remove common prefixes and suffixes
    let path_str = path_str
        .replace("crates/", "")
        .replace("/src/", "::")
        .replace(".rs", "")
        .replace('/', "::")
        .replace('-', "_");

    // Remove mod suffix if the file is named mod.rs
    if path_str.ends_with("::mod") {
        path_str[..path_str.len() - 5].to_string()
    } else {
        path_str
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_module_path() {
        let path = Path::new("crates/rscm-components/src/components/carbon_cycle.rs");
        assert_eq!(
            infer_module_path(path),
            "rscm_components::components::carbon_cycle"
        );
    }

    #[test]
    fn test_drop_summary_line() {
        assert_eq!(
            drop_summary_line("CH4 Chemistry Component\n\nSimulates methane."),
            "Simulates methane."
        );
        assert_eq!(
            drop_summary_line("Two lines\nof summary\n\nBody"),
            "Two lines\nof summary\n\nBody"
        );
        assert_eq!(drop_summary_line("Only a summary"), "Only a summary");
    }

    #[test]
    fn test_split_intro() {
        let docs = "Intro\n\n```rust\n# hidden\n```\n\n# Section\n\nBody";
        assert_eq!(
            split_intro(docs),
            ("Intro\n\n```rust\n# hidden\n```\n\n", "# Section\n\nBody")
        );
        assert_eq!(split_intro("No sections"), ("No sections", ""));
    }

    #[test]
    fn test_parse_io_attr_tokens() {
        // This tests the regex-based parsing
        let tokens_str = r#"emissions { name = "Emissions|CO2", unit = "GtC/yr" }"#;
        let re = regex::Regex::new(
            r#"(\w+)\s*\{\s*name\s*=\s*"([^"]+)"(?:\s*,\s*unit\s*=\s*"([^"]*)")?(?:\s*,\s*grid\s*=\s*"([^"]*)")?(?:\s*,\s*timeseries_type\s*=\s*"[^"]*")?\s*\}"#,
        )
        .unwrap();

        let cap = re.captures(tokens_str).expect("Should match");
        assert_eq!(cap.get(1).unwrap().as_str(), "emissions");
        assert_eq!(cap.get(2).unwrap().as_str(), "Emissions|CO2");
        assert_eq!(cap.get(3).unwrap().as_str(), "GtC/yr");

        let tokens_str =
            r#"lifetime { name = "Lifetime|CH4", unit = "yr", timeseries_type = "average" }"#;
        let cap = re.captures(tokens_str).expect("Should match");
        assert_eq!(cap.get(2).unwrap().as_str(), "Lifetime|CH4");
        assert_eq!(cap.get(3).unwrap().as_str(), "yr");
    }
}
