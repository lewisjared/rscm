//! Procedural macros for RSCM component development
//!
//! This crate provides derive macros that generate type-safe input/output structs
//! for RSCM components, eliminating stringly-typed APIs.
//!
//! # Overview
//!
//! The `ComponentIO` derive macro generates:
//! - A typed `Inputs` struct from `#[inputs(...)]` attribute
//! - A typed `Outputs` struct from `#[outputs(...)]` and `#[states(...)]` attributes
//! - Automatic `definitions()` implementation for the Component trait
//!
//! # Example
//!
//! ```ignore
//! use rscm_macros::ComponentIO;
//!
//! #[derive(ComponentIO)]
//! #[inputs(
//!     concentration_co2 { name = "Atmospheric Concentration|CO2", unit = "ppm" },
//! )]
//! #[outputs(
//!     erf_co2 { name = "Effective Radiative Forcing|CO2", unit = "W / m^2" },
//! )]
//! pub struct CO2ERFComponent {
//!     // Only actual parameters - no phantom fields needed
//!     pub erf_2xco2: f64,
//!     pub conc_pi: f64,
//! }
//! ```
//!
//! This generates:
//! - `CO2ERFComponentInputs` with a `concentration_co2: TimeseriesWindow<'a>` field
//! - `CO2ERFComponentOutputs` with an `erf_co2: f64` field
//! - Implementation of `generated_definitions()` returning the appropriate `RequirementDefinition`s
//!
//! # Compile-time Safety
//!
//! The generated structs provide compile-time validation of field access:
//!
//! ## Invalid Input Field Access
//!
//! The generated structs provide compile-time validation. For example, accessing
//! a field that doesn't exist in the inputs will fail at compile time:
//!
//! ```ignore
//! // ERROR: no field `temperature` on type `TestComponentInputs`
//! let temp = inputs.temperature.current();
//! ```
//!
//! ## Invalid Output Field
//!
//! Similarly, trying to set an output field that doesn't exist fails:
//!
//! ```ignore
//! // ERROR: no field `uptake` on type `TestComponentOutputs`
//! TestComponentOutputs {
//!     concentration: 280.0,
//!     uptake: 5.0,  // This field doesn't exist
//! }
//! ```
//!
//! ## Missing Required Output Field
//!
//! All output fields must be provided:
//!
//! ```ignore
//! // ERROR: missing field `uptake` in initializer of `TestComponentOutputs`
//! TestComponentOutputs {
//!     concentration: 280.0,
//!     // missing: uptake
//! }
//! ```
//!
//! ## Wrong Type for Grid Output
//!
//! Grid outputs must use the correct slice type:
//!
//! ```ignore
//! // ERROR: expected `FourBoxSlice`, found `f64`
//! TestComponentOutputs {
//!     heat_flux: 5.0,  // Should be FourBoxSlice::uniform(5.0)
//! }
//! ```

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::{
    braced, parse::Parse, parse::ParseStream, parse_macro_input, punctuated::Punctuated, Attribute,
    Data, DeriveInput, Ident, LitStr, Token,
};

/// A single I/O field declaration like: `emissions_co2 { name = "...", unit = "...", grid = "..." }`
struct IoFieldDecl {
    rust_name: Ident,
    variable_name: String,
    unit: String,
    grid_type: String,
}

impl Parse for IoFieldDecl {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let rust_name: Ident = input.parse()?;

        let content;
        braced!(content in input);

        let mut variable_name = None;
        let mut unit = None;
        let mut grid_type = String::from("Scalar");

        while !content.is_empty() {
            let key: Ident = content.parse()?;
            content.parse::<Token![=]>()?;
            let value: LitStr = content.parse()?;

            match key.to_string().as_str() {
                "name" => variable_name = Some(value.value()),
                "unit" => unit = Some(value.value()),
                "grid" => grid_type = value.value(),
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown attribute key: {}", other),
                    ))
                }
            }

            // Consume optional trailing comma
            if content.peek(Token![,]) {
                content.parse::<Token![,]>()?;
            }
        }

        let variable_name = variable_name
            .ok_or_else(|| syn::Error::new(rust_name.span(), "missing `name` attribute"))?;
        let unit = unit.unwrap_or_default();

        Ok(IoFieldDecl {
            rust_name,
            variable_name,
            unit,
            grid_type,
        })
    }
}

/// List of I/O field declarations: `field1 { ... }, field2 { ... }`
struct IoFieldList {
    fields: Punctuated<IoFieldDecl, Token![,]>,
}

impl Parse for IoFieldList {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Parse directly - tokens already exclude outer parentheses
        let fields = Punctuated::parse_terminated(input)?;
        Ok(IoFieldList { fields })
    }
}

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

/// Metadata for a state field (appears in both inputs and outputs)
struct StateField {
    rust_name: Ident,
    variable_name: String,
    unit: String,
    grid_type: String,
}

/// Component-level attributes (tags, category, and parameters)
#[derive(Default)]
struct ComponentAttrs {
    tags: Vec<String>,
    category: Option<String>,
    parameters_struct: Option<String>,
}

/// Parse `#[component(tags = ["tag1", "tag2"], category = "Category")]` attribute
fn parse_component_attr(attr: &Attribute) -> syn::Result<ComponentAttrs> {
    let mut attrs = ComponentAttrs::default();

    let meta_list = attr.meta.require_list()?;
    let tokens = meta_list.tokens.clone();

    if tokens.is_empty() {
        return Ok(attrs);
    }

    // Parse key = value pairs
    let parser = |input: ParseStream| {
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "tags" => {
                    // Parse array of strings: ["tag1", "tag2"]
                    let content;
                    syn::bracketed!(content in input);
                    let tags: Punctuated<LitStr, Token![,]> =
                        Punctuated::parse_terminated(&content)?;
                    attrs.tags = tags.iter().map(|s| s.value()).collect();
                }
                "category" => {
                    let value: LitStr = input.parse()?;
                    attrs.category = Some(value.value());
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown component attribute: {}", other),
                    ))
                }
            }

            // Consume optional trailing comma
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(())
    };

    syn::parse2(tokens)
        .and_then(|_: proc_macro2::TokenStream| {
            syn::parse2::<proc_macro2::TokenStream>(meta_list.tokens.clone()).ok();
            Ok(())
        })
        .ok();

    // Actually parse the tokens
    let _ = syn::parse::Parser::parse2(parser, meta_list.tokens.clone())?;

    Ok(attrs)
}

/// Parse `#[parameters(StructName)]` attribute
fn parse_parameters_attr(attr: &Attribute) -> syn::Result<String> {
    let meta_list = attr.meta.require_list()?;
    let struct_name: Ident = syn::parse2(meta_list.tokens.clone())?;
    Ok(struct_name.to_string())
}

/// Extract component attributes from struct-level attributes
fn extract_component_attrs(attrs: &[Attribute]) -> syn::Result<ComponentAttrs> {
    let mut result = ComponentAttrs::default();

    for attr in attrs {
        if attr.path().is_ident("component") {
            result = parse_component_attr(attr)?;
        } else if attr.path().is_ident("parameters") {
            result.parameters_struct = Some(parse_parameters_attr(attr)?);
        }
    }

    Ok(result)
}

/// Parse a struct-level attribute like `#[inputs(...)]` or `#[outputs(...)]`
fn parse_io_list_attribute(attr: &Attribute) -> syn::Result<Vec<IoFieldDecl>> {
    let list: IoFieldList = syn::parse2(attr.meta.require_list()?.tokens.clone())?;
    Ok(list.fields.into_iter().collect())
}

/// Extract inputs, outputs, and states from struct-level attributes
fn extract_io_from_attrs(
    attrs: &[Attribute],
) -> syn::Result<(Vec<InputField>, Vec<OutputField>, Vec<StateField>)> {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut states = Vec::new();

    for attr in attrs {
        if attr.path().is_ident("inputs") {
            for decl in parse_io_list_attribute(attr)? {
                inputs.push(InputField {
                    rust_name: decl.rust_name,
                    variable_name: decl.variable_name,
                    unit: decl.unit,
                    grid_type: decl.grid_type,
                });
            }
        } else if attr.path().is_ident("outputs") {
            for decl in parse_io_list_attribute(attr)? {
                outputs.push(OutputField {
                    rust_name: decl.rust_name,
                    variable_name: decl.variable_name,
                    unit: decl.unit,
                    grid_type: decl.grid_type,
                });
            }
        } else if attr.path().is_ident("states") {
            for decl in parse_io_list_attribute(attr)? {
                states.push(StateField {
                    rust_name: decl.rust_name,
                    variable_name: decl.variable_name,
                    unit: decl.unit,
                    grid_type: decl.grid_type,
                });
            }
        }
    }

    Ok((inputs, outputs, states))
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
/// ## Struct-level attributes
/// - `#[inputs(field { name = "...", unit = "...", grid = "..." }, ...)]` - Declare input variables
/// - `#[outputs(field { name = "...", unit = "...", grid = "..." }, ...)]` - Declare output variables
/// - `#[states(field { name = "...", unit = "...", grid = "..." }, ...)]` - Declare state variables
///   (states appear in both Inputs and Outputs structs)
/// - `#[component(tags = ["tag1", "tag2"], category = "Category Name")]` - Metadata for documentation
/// - `#[parameters(ParameterStructName)]` - Associate a parameter struct for metadata generation
///
/// Where `grid` can be: "Scalar" (default), "FourBox", or "Hemispheric"
///
/// # Generated Types
///
/// For a struct `Foo`, this macro generates:
/// - `FooInputs<'a>` - Input struct with `TimeseriesWindow` or `GridTimeseriesWindow` fields
/// - `FooOutputs` - Output struct with typed fields
/// - `Foo::generated_definitions()` - Returns `Vec<RequirementDefinition>` for the Component trait
/// - `Foo::component_metadata()` - Returns `ComponentMetadata` for documentation generation
#[proc_macro_derive(
    ComponentIO,
    attributes(inputs, outputs, states, component, parameters)
)]
pub fn derive_component_io(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = &input.ident;
    let struct_name_str = struct_name.to_string();
    let inputs_name = format_ident!("{}Inputs", struct_name);
    let outputs_name = format_ident!("{}Outputs", struct_name);

    // Verify it's a struct
    match &input.data {
        Data::Struct(_) => {}
        _ => {
            return syn::Error::new(
                Span::call_site(),
                "ComponentIO can only be derived for structs",
            )
            .to_compile_error()
            .into()
        }
    };

    // Extract I/O declarations from struct-level attributes
    let (input_fields, output_fields, state_fields) = match extract_io_from_attrs(&input.attrs) {
        Ok(result) => result,
        Err(e) => return e.to_compile_error().into(),
    };

    // Extract component attributes (tags, category)
    let component_attrs = match extract_component_attrs(&input.attrs) {
        Ok(attrs) => attrs,
        Err(e) => return e.to_compile_error().into(),
    };

    // Generate Inputs struct fields (inputs + states)
    let input_struct_fields: Vec<TokenStream2> = input_fields
        .iter()
        .map(|f| {
            let name = &f.rust_name;
            let ty = input_window_type(&f.grid_type);
            quote! { pub #name: #ty }
        })
        .chain(state_fields.iter().map(|f| {
            let name = &f.rust_name;
            let ty = input_window_type(&f.grid_type);
            quote! { pub #name: #ty }
        }))
        .collect();

    // Generate Outputs struct fields (outputs + states)
    let output_struct_fields: Vec<TokenStream2> = output_fields
        .iter()
        .map(|f| {
            let name = &f.rust_name;
            let ty = output_type(&f.grid_type);
            quote! { pub #name: #ty }
        })
        .chain(state_fields.iter().map(|f| {
            let name = &f.rust_name;
            let ty = output_type(&f.grid_type);
            quote! { pub #name: #ty }
        }))
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

    let state_definitions: Vec<TokenStream2> = state_fields
        .iter()
        .map(|f| {
            let name = &f.variable_name;
            let unit = &f.unit;
            let grid = grid_type_token(&f.grid_type);
            quote! {
                RequirementDefinition::with_grid(#name, #unit, RequirementType::State, #grid)
            }
        })
        .collect();

    // Generate Into<OutputState> conversion for each output field
    // Use StateValue directly since users must import it
    let output_conversions: Vec<TokenStream2> = output_fields
        .iter()
        .map(|f| {
            let name = &f.rust_name;
            let var_name = &f.variable_name;
            match f.grid_type.as_str() {
                "FourBox" => quote! {
                    map.insert(
                        #var_name.to_string(),
                        StateValue::FourBox(outputs.#name),
                    );
                },
                "Hemispheric" => quote! {
                    map.insert(
                        #var_name.to_string(),
                        StateValue::Hemispheric(outputs.#name),
                    );
                },
                _ => quote! {
                    map.insert(
                        #var_name.to_string(),
                        StateValue::Scalar(outputs.#name),
                    );
                },
            }
        })
        .chain(state_fields.iter().map(|f| {
            let name = &f.rust_name;
            let var_name = &f.variable_name;
            match f.grid_type.as_str() {
                "FourBox" => quote! {
                    map.insert(
                        #var_name.to_string(),
                        StateValue::FourBox(outputs.#name),
                    );
                },
                "Hemispheric" => quote! {
                    map.insert(
                        #var_name.to_string(),
                        StateValue::Hemispheric(outputs.#name),
                    );
                },
                _ => quote! {
                    map.insert(
                        #var_name.to_string(),
                        StateValue::Scalar(outputs.#name),
                    );
                },
            }
        }))
        .collect();

    // Generate code to construct Inputs from InputState
    let input_field_constructions: Vec<TokenStream2> = input_fields
        .iter()
        .map(|f| {
            let name = &f.rust_name;
            let var_name = &f.variable_name;
            match f.grid_type.as_str() {
                "FourBox" => quote! {
                    #name: input_state.get_four_box_window(#var_name)
                },
                "Hemispheric" => quote! {
                    #name: input_state.get_hemispheric_window(#var_name)
                },
                _ => quote! {
                    #name: input_state.get_scalar_window(#var_name)
                },
            }
        })
        .chain(state_fields.iter().map(|f| {
            let name = &f.rust_name;
            let var_name = &f.variable_name;
            match f.grid_type.as_str() {
                "FourBox" => quote! {
                    #name: input_state.get_four_box_window(#var_name)
                },
                "Hemispheric" => quote! {
                    #name: input_state.get_hemispheric_window(#var_name)
                },
                _ => quote! {
                    #name: input_state.get_scalar_window(#var_name)
                },
            }
        }))
        .collect();

    // Generate component metadata
    let tags = &component_attrs.tags;
    let category_token = match &component_attrs.category {
        Some(cat) => quote! { Some(#cat.to_string()) },
        None => quote! { None },
    };

    // Generate VariableMetadata for inputs (using fully qualified path for cross-crate compatibility)
    let input_metadata: Vec<TokenStream2> = input_fields
        .iter()
        .map(|f| {
            let rust_name = f.rust_name.to_string();
            let variable_name = &f.variable_name;
            let unit = &f.unit;
            let grid = grid_type_token(&f.grid_type);
            quote! {
                ::rscm_core::component::VariableMetadata {
                    rust_name: #rust_name.to_string(),
                    variable_name: #variable_name.to_string(),
                    unit: #unit.to_string(),
                    grid: #grid,
                    description: String::new(),
                }
            }
        })
        .collect();

    // Generate VariableMetadata for outputs (using fully qualified path for cross-crate compatibility)
    let output_metadata: Vec<TokenStream2> = output_fields
        .iter()
        .map(|f| {
            let rust_name = f.rust_name.to_string();
            let variable_name = &f.variable_name;
            let unit = &f.unit;
            let grid = grid_type_token(&f.grid_type);
            quote! {
                ::rscm_core::component::VariableMetadata {
                    rust_name: #rust_name.to_string(),
                    variable_name: #variable_name.to_string(),
                    unit: #unit.to_string(),
                    grid: #grid,
                    description: String::new(),
                }
            }
        })
        .collect();

    // Generate VariableMetadata for states (using fully qualified path for cross-crate compatibility)
    let state_metadata: Vec<TokenStream2> = state_fields
        .iter()
        .map(|f| {
            let rust_name = f.rust_name.to_string();
            let variable_name = &f.variable_name;
            let unit = &f.unit;
            let grid = grid_type_token(&f.grid_type);
            quote! {
                ::rscm_core::component::VariableMetadata {
                    rust_name: #rust_name.to_string(),
                    variable_name: #variable_name.to_string(),
                    unit: #unit.to_string(),
                    grid: #grid,
                    description: String::new(),
                }
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

        impl<'a> #inputs_name<'a> {
            /// Construct typed inputs from an InputState
            ///
            /// This extracts the appropriate TimeseriesWindow for each input field
            /// from the provided input state.
            ///
            /// # Panics
            ///
            /// Panics if any required variable is missing from the input state
            /// or if a variable has the wrong grid type.
            pub fn from_input_state(input_state: &'a InputState<'_>) -> Self {
                Self {
                    #(#input_field_constructions,)*
                }
            }
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
                    #(#state_definitions,)*
                ]
            }

            /// Returns metadata about this component for documentation generation
            ///
            /// This includes the component's name, tags, category, I/O definitions, and parameters.
            pub fn component_metadata() -> ::rscm_core::component::ComponentMetadata {
                ::rscm_core::component::ComponentMetadata {
                    name: #struct_name_str.to_string(),
                    tags: vec![#(#tags.to_string()),*],
                    category: #category_token,
                    inputs: vec![#(#input_metadata),*],
                    outputs: vec![#(#output_metadata),*],
                    states: vec![#(#state_metadata),*],
                    parameters: vec![],
                }
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
