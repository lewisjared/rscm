//! The `params` attribute, which keeps NaN and infinities in the serde form of a struct or enum.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    parse_quote, punctuated::Punctuated, Attribute, Error, Field, Fields, GenericArgument, Item,
    PathArguments, Token, Type,
};

pub(crate) fn expand(mut item: Item) -> syn::Result<TokenStream2> {
    match &mut item {
        Item::Struct(item) => {
            check_derives(&item.attrs, &item.ident)?;
            let container_default = serde_keys(&item.attrs)?.contains(&"default".to_string());
            annotate(&mut item.fields, container_default)?;
        }
        Item::Enum(item) => {
            check_derives(&item.attrs, &item.ident)?;
            for variant in &mut item.variants {
                annotate(&mut variant.fields, false)?;
            }
        }
        other => {
            return Err(Error::new_spanned(
                other,
                "params applies to a struct or an enum",
            ))
        }
    }
    Ok(quote!(#item))
}

/// Fails unless a serde derive follows, because a derive above this attribute would not see its changes.
fn check_derives(attrs: &[Attribute], ident: &syn::Ident) -> syn::Result<()> {
    let mut found = false;
    for attr in attrs.iter().filter(|a| a.path().is_ident("derive")) {
        let paths = attr.parse_args_with(Punctuated::<syn::Path, Token![,]>::parse_terminated)?;
        found |= paths.iter().any(|p| {
            p.segments
                .last()
                .is_some_and(|s| s.ident == "Serialize" || s.ident == "Deserialize")
        });
    }
    if found {
        Ok(())
    } else {
        Err(Error::new_spanned(
            ident,
            "params must sit above a #[derive(Serialize)] or #[derive(Deserialize)]",
        ))
    }
}

fn annotate(fields: &mut Fields, container_default: bool) -> syn::Result<()> {
    for field in fields.iter_mut() {
        annotate_field(field, container_default)?;
    }
    Ok(())
}

fn annotate_field(field: &mut Field, container_default: bool) -> syn::Result<()> {
    if !holds_floats(&field.ty) {
        return Ok(());
    }
    let keys = serde_keys(&field.attrs)?;
    let has = |key: &str| keys.iter().any(|k| k == key);
    if has("with") || has("serialize_with") || has("deserialize_with") || has("skip") {
        return Ok(());
    }
    field
        .attrs
        .push(parse_quote!(#[serde(with = "rscm_core::non_finite")]));
    // A field with `with` loses serde's implicit None for a missing Option.
    if outer_ident(&field.ty).is_some_and(|i| i == "Option")
        && !has("default")
        && !container_default
    {
        field.attrs.push(parse_quote!(#[serde(default)]));
    }
    Ok(())
}

/// The keys of every `#[serde(...)]` attribute, such as `default` or `rename`.
fn serde_keys(attrs: &[Attribute]) -> syn::Result<Vec<String>> {
    let mut keys = Vec::new();
    for attr in attrs.iter().filter(|a| a.path().is_ident("serde")) {
        attr.parse_nested_meta(|meta| {
            if let Some(ident) = meta.path.get_ident() {
                keys.push(ident.to_string());
            }
            if meta.input.peek(Token![=]) {
                meta.value()?.parse::<syn::Expr>()?;
            } else if meta.input.peek(syn::token::Paren) {
                let content;
                syn::parenthesized!(content in meta.input);
                content.parse::<TokenStream2>()?;
            }
            Ok(())
        })?;
    }
    Ok(keys)
}

fn outer_ident(ty: &Type) -> Option<&syn::Ident> {
    match ty {
        Type::Path(path) if path.qself.is_none() => path.path.segments.last().map(|s| &s.ident),
        _ => None,
    }
}

fn type_arguments(ty: &Type) -> Vec<&Type> {
    let Type::Path(path) = ty else {
        return Vec::new();
    };
    let Some(PathArguments::AngleBracketed(args)) = path.path.segments.last().map(|s| &s.arguments)
    else {
        return Vec::new();
    };
    args.args
        .iter()
        .filter_map(|arg| match arg {
            GenericArgument::Type(ty) => Some(ty),
            _ => None,
        })
        .collect()
}

/// Whether `ty` is a float, or a container `rscm_core::non_finite` supports holding floats.
fn holds_floats(ty: &Type) -> bool {
    match ty {
        Type::Array(array) => holds_floats(&array.elem),
        Type::Group(group) => holds_floats(&group.elem),
        Type::Paren(paren) => holds_floats(&paren.elem),
        Type::Path(_) => {
            let Some(ident) = outer_ident(ty) else {
                return false;
            };
            let args = type_arguments(ty);
            match ident.to_string().as_str() {
                "f64" | "f32" | "FloatValue" => args.is_empty(),
                "Option" | "Vec" | "VecDeque" | "Box" => args.len() == 1 && holds_floats(args[0]),
                "HashMap" | "BTreeMap" => args.len() == 2 && holds_floats(args[1]),
                "Array" | "Array1" | "Array2" => args.first().is_some_and(|t| holds_floats(t)),
                _ => false,
            }
        }
        _ => false,
    }
}
