//! ComponentIO declares names as literals, so these tests keep them equal to `rscm_core::names`.

use rscm_core::names::*;

fn declares(definitions: Vec<rscm_core::component::VariableDefinition>, expected: &[&str]) {
    let names: Vec<String> = definitions.into_iter().map(|d| d.name).collect();
    for name in expected {
        assert!(
            names.iter().any(|n| n == name),
            "{name} missing from {names:?}"
        );
    }
}

#[test]
fn two_layer_names() {
    declares(
        crate::component::TwoLayer::generated_definitions(),
        &[
            ERF,
            SURFACE_AIR_TEMPERATURE_CHANGE,
            &ocean_layer_temperature(2),
        ],
    );
}
