//! Checks that the parameter TypedDicts in `python/rscm/_lib/magicc.pyi` match the Rust parameter structs.
//!
//! The TypedDicts exist only in the stub, so stubtest cannot see them.
//! Each struct is serialised and its keys compared with the TypedDict of the same name,
//! recursing into nested objects whose annotation names another TypedDict.

use std::collections::{BTreeMap, BTreeSet};

use rscm_magicc::parameters::*;
use serde::Serialize;
use serde_json::{json, Value};

type Stubs = BTreeMap<String, BTreeMap<String, String>>;

/// TypedDict name to its fields and their annotations.
fn parse_stub() -> Stubs {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../python/rscm/_lib/magicc.pyi"
    );
    let text = std::fs::read_to_string(path).expect("read magicc.pyi");
    let mut stubs = Stubs::new();
    let mut current: Option<String> = None;
    let mut in_docstring = false;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("class ") {
            current = rest
                .contains("TypedDict")
                .then(|| rest.split('(').next().unwrap().to_string());
            if let Some(name) = &current {
                stubs.insert(name.clone(), BTreeMap::new());
            }
            continue;
        }
        let Some(name) = &current else { continue };
        if !line.starts_with("    ") && !line.trim().is_empty() {
            current = None;
            continue;
        }
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("\"\"\"") {
            if !rest.contains("\"\"\"") {
                in_docstring = !in_docstring;
            }
            continue;
        }
        if in_docstring || line.starts_with("     ") {
            continue;
        }
        if let Some((field, annotation)) = trimmed.split_once(": ") {
            if field
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            {
                stubs
                    .get_mut(name)
                    .unwrap()
                    .insert(field.to_string(), annotation.to_string());
            }
        }
    }
    stubs
}

fn to_json<T: Serialize>(value: &T) -> Value {
    serde_json::to_value(value).expect("serialise parameters")
}

/// Compares `value` with the TypedDict `name`, then recurses into nested TypedDicts.
fn check(
    stubs: &Stubs,
    name: &str,
    value: &Value,
    seen: &mut BTreeSet<String>,
    errors: &mut Vec<String>,
) {
    let Some(fields) = stubs.get(name) else {
        errors.push(format!("{name}: no TypedDict in magicc.pyi"));
        return;
    };
    let Value::Object(object) = value else {
        errors.push(format!("{name}: serialised to {value}, not an object"));
        return;
    };
    seen.insert(name.to_string());

    let rust: BTreeSet<&str> = object.keys().map(String::as_str).collect();
    let stub: BTreeSet<&str> = fields.keys().map(String::as_str).collect();
    let missing: Vec<_> = rust.difference(&stub).collect();
    let extra: Vec<_> = stub.difference(&rust).collect();
    if !missing.is_empty() {
        errors.push(format!("{name}: fields missing from the stub: {missing:?}"));
    }
    if !extra.is_empty() {
        errors.push(format!("{name}: stub fields not in Rust: {extra:?}"));
    }

    for (field, annotation) in fields {
        let Some(child) = object.get(field) else {
            continue;
        };
        let nested = stubs
            .keys()
            .find(|candidate| annotation_names(annotation, candidate));
        let Some(nested) = nested else { continue };
        match child {
            Value::Object(_) => check(stubs, nested, child, seen, errors),
            Value::Array(items) => {
                for item in items.iter().filter(|item| item.is_object()) {
                    check(stubs, nested, item, seen, errors);
                }
            }
            _ => {}
        }
    }
}

/// Whether `annotation` refers to the TypedDict `name` as a whole word.
fn annotation_names(annotation: &str, name: &str) -> bool {
    annotation
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .any(|word| word == name)
}

#[test]
fn magicc_stub_typeddicts_match_rust_parameters() {
    let stubs = parse_stub();

    let mut cases: Vec<(&str, Value)> = vec![
        (
            "AerosolDirectParameters",
            to_json(&AerosolDirectParameters::default()),
        ),
        (
            "AerosolIndirectParameters",
            to_json(&AerosolIndirectParameters::default()),
        ),
        (
            "CH4ChemistryParameters",
            to_json(&CH4ChemistryParameters::default()),
        ),
        (
            "CH4OxidationParameters",
            to_json(&CH4OxidationParameters::default()),
        ),
        (
            "ClimateUDEBParameters",
            to_json(&ClimateUDEBParameters::default()),
        ),
        (
            "CO2BudgetParameters",
            to_json(&CO2BudgetParameters::default()),
        ),
        (
            "ForcingAggregationParameters",
            to_json(&ForcingAggregationParameters::default()),
        ),
        (
            "GHGForcingParameters",
            to_json(&GHGForcingParameters::default()),
        ),
        (
            "HalocarbonChemistryParameters",
            to_json(&HalocarbonChemistryParameters::default()),
        ),
        (
            "N2OChemistryParameters",
            to_json(&N2OChemistryParameters::default()),
        ),
        (
            "NitrateCloudForcingParameters",
            to_json(&NitrateCloudForcingParameters::default()),
        ),
        (
            "NitrogenParameters",
            to_json(&NitrogenParameters::default()),
        ),
        (
            "OceanCarbonParameters",
            to_json(&OceanCarbonParameters::default()),
        ),
        (
            "OzoneForcingParameters",
            to_json(&OzoneForcingParameters::default()),
        ),
        (
            "PermafrostParameters",
            to_json(&PermafrostParameters::default()),
        ),
        (
            "SeaLevelRiseParameters",
            to_json(&SeaLevelRiseParameters::default()),
        ),
        (
            "TerrestrialCarbonParameters",
            to_json(&TerrestrialCarbonParameters::default()),
        ),
        ("ThermalExpansion", to_json(&ThermalExpansion::default())),
    ];
    // Nested records that are None or empty in every default.
    let pattern = [1.0, 1.0, 1.0, 1.0];
    let subannual: SubannualForcing =
        serde_json::from_value(json!({"start": 1750.0, "values": []})).unwrap();
    let agent: ForcingAgent =
        serde_json::from_value(json!({"variable": "x", "category": "natural"})).unwrap();
    let climate: EfficacyClimate = serde_json::from_value(json!({
        "ecs": 3.0, "erf_2xco2": 3.7, "land_ocean_heat_exchange": 1.0,
        "north_south_heat_exchange": 1.0, "land_ocean_warming_ratio": 1.0,
        "amplify_ocean_to_land": 1.0, "co2_regional_pattern": pattern,
        "include_unset_lamcalc_error": false,
    }))
    .unwrap();
    let harmonisation: Harmonisation =
        serde_json::from_value(json!({"target": 1.0, "reference_year": 2019.0})).unwrap();
    let fixation: PrognosticFixation = serde_json::from_value(json!({
        "maximum_fixation_rate": 1.0, "half_saturation_npp": 1.0,
    }))
    .unwrap();
    let deposition: DepositionFromEmissions = serde_json::from_value(json!({
        "land_fraction": 0.5, "include_nh3_counted_twice": false,
    }))
    .unwrap();
    let ozone: TroposphericOzoneRegions = serde_json::from_value(json!({
        "regional_pattern": pattern, "nh_land_fraction": 0.4, "sh_land_fraction": 0.2,
    }))
    .unwrap();
    let fgas: FGasRegions = serde_json::from_value(json!({
        "regional_pattern": pattern, "lower_lifetime": 1.0, "upper_lifetime": 2.0,
        "nh_land_fraction": 0.4, "sh_land_fraction": 0.2, "last_split_year": null,
    }))
    .unwrap();
    cases.extend([
        ("SubannualForcing", to_json(&subannual)),
        ("ForcingAgent", to_json(&agent)),
        ("EfficacyClimate", to_json(&climate)),
        ("Harmonisation", to_json(&harmonisation)),
        ("PrognosticFixation", to_json(&fixation)),
        ("DepositionFromEmissions", to_json(&deposition)),
        ("TroposphericOzoneRegions", to_json(&ozone)),
        ("FGasRegions", to_json(&fgas)),
    ]);

    let mut seen = BTreeSet::new();
    let mut errors = Vec::new();
    for (name, value) in &cases {
        check(&stubs, name, value, &mut seen, &mut errors);
    }

    let unchecked: Vec<_> = stubs.keys().filter(|name| !seen.contains(*name)).collect();
    if !unchecked.is_empty() {
        errors.push(format!(
            "TypedDicts not reached from any case, add one to this test: {unchecked:?}"
        ));
    }

    assert!(
        errors.is_empty(),
        "magicc.pyi is out of date:\n{}",
        errors.join("\n")
    );
}
