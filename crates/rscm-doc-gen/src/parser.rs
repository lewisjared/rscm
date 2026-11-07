//! Rust source file parser for extracting component metadata

use crate::extractor::{extract_description, extract_doc_comments};
use crate::schema::{ComponentDocMetadata, DeclaredVariable, ParameterMetadata};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use syn::parse::{ParseStream, Parser};
use syn::punctuated::Punctuated;
use syn::{
    braced, Attribute, Fields, Ident, Item, ItemStruct, LitStr, Meta, MetaList, Token, Type,
};

/// Parameters structs of every scanned file, keyed by crate so same-named structs don't collide
pub type StructIndex<'a> = HashMap<(String, String), &'a ItemStruct>;

/// Read and parse a Rust source file
pub fn read_file(path: &Path) -> Result<syn::File, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    syn::parse_file(&content).map_err(|e| format!("Failed to parse {}: {}", path.display(), e))
}

/// Add the parameters structs of a parsed file to the index
pub fn index_structs<'a>(crate_name: &str, syntax: &'a syn::File, index: &mut StructIndex<'a>) {
    for item in &syntax.items {
        if let Item::Struct(item_struct) = item {
            let name = item_struct.ident.to_string();
            if name.ends_with("Parameters") {
                index.insert((crate_name.to_string(), name), item_struct);
            }
        }
    }
}

/// Extract component metadata from a parsed file
pub fn parse_file(
    crate_name: &str,
    path: &Path,
    syntax: &syn::File,
    index: &StructIndex,
) -> Result<Vec<ComponentDocMetadata>, String> {
    let mut components = Vec::new();

    // Extract module-level documentation
    let module_docs = extract_doc_comments(&syntax.attrs);

    // Second pass: parse components and resolve their parameter structs
    for item in &syntax.items {
        if let Item::Struct(item_struct) = item {
            if let Some(metadata) =
                parse_component_struct(item_struct, crate_name, path, &module_docs, index)?
            {
                components.push(metadata);
            }
        }
    }

    Ok(components)
}

/// Check whether a struct derives the named macro
fn has_derive(attrs: &[Attribute], derive: &str) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident("derive")
            && matches!(&attr.meta, Meta::List(MetaList { tokens, .. })
                if tokens.to_string().split(|c: char| !c.is_alphanumeric() && c != '_')
                    .any(|token| token == derive))
    })
}

/// Parse a struct that uses ComponentIO derive macro
fn parse_component_struct(
    item: &ItemStruct,
    crate_name: &str,
    source_path: &Path,
    module_docs: &str,
    index: &StructIndex,
) -> Result<Option<ComponentDocMetadata>, String> {
    let declares_io = has_derive(&item.attrs, "ComponentIO");
    if !declares_io && !has_derive(&item.attrs, "ComponentMetadata") {
        return Ok(None);
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
    let join_docs = |docs: &[&str]| {
        docs.iter()
            .map(|docs| docs.trim())
            .filter(|docs| !docs.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n")
    };
    let mut combined_docs =
        join_docs(&[module_intro, struct_intro, module_sections, struct_sections]);
    // A component documented by a title alone keeps it rather than having no description
    if combined_docs.is_empty() {
        combined_docs = join_docs(&[module_docs, &struct_docs]);
    }

    let reference_model = match crate_name {
        "rscm-magicc" => Some("MAGICC7"),
        "rscm-fair" => Some("FaIR"),
        _ => None,
    };
    // Ported models reuse component names (both have a GHGForcing), so their pages take a prefix
    let slug = match reference_model {
        Some(_) => format!(
            "{}-{}",
            crate_name.trim_start_matches("rscm-"),
            struct_name.to_lowercase()
        ),
        None => struct_name.to_lowercase(),
    };
    let mut metadata = ComponentDocMetadata {
        name: struct_name.clone(),
        slug,
        reference_model: reference_model.map(str::to_string),
        variables_from_parameters: !declares_io,
        module_path: infer_module_path(source_path),
        description: extract_description(&combined_docs),
        source_file: source_path.to_string_lossy().to_string(),
        ..Default::default()
    };

    // Parse struct-level attributes
    for attr in &item.attrs {
        let target = if attr.path().is_ident("component") {
            parse_component_attr(attr, &mut metadata);
            continue;
        } else if attr.path().is_ident("inputs") {
            &mut metadata.inputs
        } else if attr.path().is_ident("outputs") {
            &mut metadata.outputs
        } else if attr.path().is_ident("states") {
            &mut metadata.states
        } else {
            continue;
        };
        *target = parse_io_attr(attr)
            .map_err(|e| format!("Failed to parse the IO of {struct_name}: {e}"))?;
    }

    // Parse struct fields and resolve parameter structs
    if let Fields::Named(fields) = &item.fields {
        for field in &fields.named {
            let field_type = type_to_string(&field.ty);

            // A *Parameters field in the same crate expands into the parameter table
            if let Some(params_struct) = index.get(&(crate_name.to_string(), field_type)) {
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

    Ok(Some(metadata))
}

/// Drop a doc block's summary line when its first paragraph is one line
fn drop_summary_line(docs: &str) -> &str {
    match docs.split_once("\n\n") {
        Some((summary, rest)) if !summary.contains('\n') => rest.trim_start(),
        None if !docs.contains('\n') => "",
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
///
/// Each entry is `field { name = "...", unit = "...", grid = "...", description = "..." }`.
fn parse_io_attr(attr: &Attribute) -> syn::Result<Vec<DeclaredVariable>> {
    let entry = |input: ParseStream| -> syn::Result<DeclaredVariable> {
        let rust_name: Ident = input.parse()?;
        let content;
        braced!(content in input);
        let pairs =
            Punctuated::<(Ident, LitStr), Token![,]>::parse_terminated_with(&content, |pair| {
                let key: Ident = pair.parse()?;
                pair.parse::<Token![=]>()?;
                Ok((key, pair.parse()?))
            })?;
        let mut variable = DeclaredVariable {
            rust_name: rust_name.to_string(),
            name: String::new(),
            unit: String::new(),
            grid: "scalar".to_string(),
            description: String::new(),
        };
        for (key, value) in pairs {
            match key.to_string().as_str() {
                "name" => variable.name = value.value(),
                "unit" => variable.unit = value.value(),
                "grid" => variable.grid = value.value(),
                "description" => variable.description = value.value(),
                _ => {}
            }
        }
        Ok(variable)
    };
    let list = |input: ParseStream| {
        Punctuated::<DeclaredVariable, Token![,]>::parse_terminated_with(input, entry)
    };
    Ok(list
        .parse2(attr.meta.require_list()?.tokens.clone())?
        .into_iter()
        .collect())
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
        assert_eq!(drop_summary_line("Only a summary"), "");
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
    fn test_parse_io_attr() {
        let attr: Attribute = syn::parse_quote! {
            #[inputs(
                emissions { name = "Emissions|CO2", unit = "GtC/yr" },
                lifetime { name = "Lifetime|CH4", unit = "yr", timeseries_type = "average",
                    description = "Total \"effective\" lifetime" },
                heat { name = "Heat Uptake", unit = "W/m^2", grid = "four_box" },
            )]
        };
        let vars = parse_io_attr(&attr).unwrap();
        assert_eq!(vars.len(), 3);
        assert_eq!(vars[0].rust_name, "emissions");
        assert_eq!(vars[0].name, "Emissions|CO2");
        assert_eq!(vars[0].unit, "GtC/yr");
        assert_eq!(vars[0].grid, "scalar");
        assert_eq!(vars[1].description, "Total \"effective\" lifetime");
        assert_eq!(vars[2].grid, "four_box");
    }

    #[test]
    fn test_has_derive() {
        let item: ItemStruct = syn::parse_quote! {
            #[derive(Debug, rscm_core::ComponentMetadata)]
            struct Foo;
        };
        assert!(has_derive(&item.attrs, "ComponentMetadata"));
        assert!(!has_derive(&item.attrs, "ComponentIO"));
    }
}
