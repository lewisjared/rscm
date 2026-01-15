//! Procedural macros for RSCM component development
//!
//! This crate provides derive macros that generate type-safe input/output structs
//! for RSCM components, eliminating stringly-typed APIs.
//!
//! # Overview
//!
//! The `ComponentIO` derive macro generates:
//! - A typed `Inputs` struct from fields marked with `#[input]`
//! - A typed `Outputs` struct from fields marked with `#[output]`
//! - Automatic `definitions()` implementation for the Component trait
//!
//! # Example
//!
//! ```ignore
//! use rscm_macros::ComponentIO;
//!
//! #[derive(ComponentIO)]
//! #[component(name = "CO2ERF")]
//! pub struct CO2ERFComponent {
//!     #[input(name = "Atmospheric Concentration|CO2", unit = "ppm")]
//!     concentration_co2: f64,
//!
//!     #[output(name = "Effective Radiative Forcing|CO2", unit = "W / m^2")]
//!     erf_co2: f64,
//!
//!     // Parameters (not marked as input/output)
//!     pub erf_2xco2: f64,
//!     pub conc_pi: f64,
//! }
//! ```
//!
//! This generates:
//! - `CO2ERFComponentInputs` with a `concentration_co2: TimeseriesWindow<'a>` field
//! - `CO2ERFComponentOutputs` with an `erf_co2: f64` field
//! - Implementation of `definitions()` returning the appropriate `RequirementDefinition`s

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Attribute, Data, DeriveInput, Fields, Ident, LitStr};

/// Metadata for an input field
struct InputField {
    rust_name: Ident,
    variable_name: String,
    unit: String,
    grid_type: String,
}

/// Metadata for an output field
struct OutputField {
    rust_name: Ident,
    variable_name: String,
    unit: String,
    grid_type: String,
}

/// Parse a #[input(...)] or #[output(...)] attribute using syn 2.0 API
fn parse_io_attribute(attr: &Attribute, rust_name: &Ident) -> Option<(String, String, String)> {
    let mut name = None;
    let mut unit = None;
    let mut grid = String::from("Scalar");

    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("name") {
            let value: LitStr = meta.value()?.parse()?;
            name = Some(value.value());
        } else if meta.path.is_ident("unit") {
            let value: LitStr = meta.value()?.parse()?;
            unit = Some(value.value());
        } else if meta.path.is_ident("grid") {
            let value: LitStr = meta.value()?.parse()?;
            grid = value.value();
        }
        Ok(())
    })
    .ok()?;

    // Use rust field name if name not specified
    let name = name.unwrap_or_else(|| rust_name.to_string());
    let unit = unit.unwrap_or_default();

    Some((name, unit, grid))
}

/// Extract input/output fields from the struct
fn extract_io_fields(fields: &Fields) -> (Vec<InputField>, Vec<OutputField>) {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    if let Fields::Named(named) = fields {
        for field in &named.named {
            let rust_name = field
                .ident
                .clone()
                .expect("Named fields should have idents");

            for attr in &field.attrs {
                if attr.path().is_ident("input") {
                    if let Some((name, unit, grid)) = parse_io_attribute(attr, &rust_name) {
                        inputs.push(InputField {
                            rust_name: rust_name.clone(),
                            variable_name: name,
                            unit,
                            grid_type: grid,
                        });
                    }
                } else if attr.path().is_ident("output") {
                    if let Some((name, unit, grid)) = parse_io_attribute(attr, &rust_name) {
                        outputs.push(OutputField {
                            rust_name: rust_name.clone(),
                            variable_name: name,
                            unit,
                            grid_type: grid,
                        });
                    }
                }
            }
        }
    }

    (inputs, outputs)
}

/// Generate the grid type token
fn grid_type_token(grid: &str) -> TokenStream2 {
    match grid {
        "FourBox" => quote! { GridType::FourBox },
        "Hemispheric" => quote! { GridType::Hemispheric },
        _ => quote! { GridType::Scalar },
    }
}

/// Generate the input window type based on grid
fn input_window_type(grid: &str) -> TokenStream2 {
    match grid {
        "FourBox" => quote! { GridTimeseriesWindow<'a, FourBoxGrid> },
        "Hemispheric" => quote! { GridTimeseriesWindow<'a, HemisphericGrid> },
        _ => quote! { TimeseriesWindow<'a> },
    }
}

/// Generate the output type based on grid
fn output_type(grid: &str) -> TokenStream2 {
    match grid {
        "FourBox" => quote! { FourBoxSlice },
        "Hemispheric" => quote! { HemisphericSlice },
        _ => quote! { FloatValue },
    }
}

/// Derive macro for generating typed component I/O structs
///
/// # Attributes
///
/// ## Field attributes
/// - `#[input(name = "...", unit = "...", grid = "...")]` - Mark as input variable
/// - `#[output(name = "...", unit = "...", grid = "...")]` - Mark as output variable
///
/// Where `grid` can be: "Scalar" (default), "FourBox", or "Hemispheric"
///
/// # Generated Types
///
/// For a struct `Foo`, this macro generates:
/// - `FooInputs<'a>` - Input struct with `TimeseriesWindow` or `GridTimeseriesWindow` fields
/// - `FooOutputs` - Output struct with typed fields
#[proc_macro_derive(ComponentIO, attributes(input, output))]
pub fn derive_component_io(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = &input.ident;
    let inputs_name = format_ident!("{}Inputs", struct_name);
    let outputs_name = format_ident!("{}Outputs", struct_name);

    let fields = match &input.data {
        Data::Struct(data) => &data.fields,
        _ => panic!("ComponentIO can only be derived for structs"),
    };

    let (input_fields, output_fields) = extract_io_fields(fields);

    // Generate Inputs struct fields
    let input_struct_fields: Vec<TokenStream2> = input_fields
        .iter()
        .map(|f| {
            let name = &f.rust_name;
            let ty = input_window_type(&f.grid_type);
            quote! { pub #name: #ty }
        })
        .collect();

    // Generate Outputs struct fields
    let output_struct_fields: Vec<TokenStream2> = output_fields
        .iter()
        .map(|f| {
            let name = &f.rust_name;
            let ty = output_type(&f.grid_type);
            quote! { pub #name: #ty }
        })
        .collect();

    // Generate definitions() items
    let input_definitions: Vec<TokenStream2> = input_fields
        .iter()
        .map(|f| {
            let name = &f.variable_name;
            let unit = &f.unit;
            let grid = grid_type_token(&f.grid_type);
            quote! {
                RequirementDefinition::with_grid(#name, #unit, RequirementType::Input, #grid)
            }
        })
        .collect();

    let output_definitions: Vec<TokenStream2> = output_fields
        .iter()
        .map(|f| {
            let name = &f.variable_name;
            let unit = &f.unit;
            let grid = grid_type_token(&f.grid_type);
            quote! {
                RequirementDefinition::with_grid(#name, #unit, RequirementType::Output, #grid)
            }
        })
        .collect();

    // Generate Into<OutputState> conversion for each output field
    let output_conversions: Vec<TokenStream2> = output_fields
        .iter()
        .map(|f| {
            let name = &f.rust_name;
            let var_name = &f.variable_name;
            match f.grid_type.as_str() {
                "FourBox" => quote! {
                    map.insert(#var_name.to_string(), self.#name.to_vec());
                },
                "Hemispheric" => quote! {
                    map.insert(#var_name.to_string(), self.#name.to_vec());
                },
                _ => quote! {
                    map.insert(#var_name.to_string(), self.#name);
                },
            }
        })
        .collect();

    // Generate the expanded code
    let expanded = quote! {
        /// Generated input struct for #struct_name
        #[derive(Debug)]
        pub struct #inputs_name<'a> {
            #(#input_struct_fields,)*
        }

        /// Generated output struct for #struct_name
        #[derive(Debug, Default)]
        pub struct #outputs_name {
            #(#output_struct_fields,)*
        }

        impl #struct_name {
            /// Returns the variable definitions for this component
            pub fn generated_definitions() -> Vec<RequirementDefinition> {
                vec![
                    #(#input_definitions,)*
                    #(#output_definitions,)*
                ]
            }
        }

        impl From<#outputs_name> for OutputState {
            fn from(outputs: #outputs_name) -> Self {
                let mut map = std::collections::HashMap::new();
                #(#output_conversions)*
                map
            }
        }
    };

    TokenStream::from(expanded)
}
