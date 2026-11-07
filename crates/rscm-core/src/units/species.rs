//! Species that a mass unit can count, such as the `CH4` in `Mt CH4/yr`.
//!
//! A species token follows a mass unit and names what the mass counts.
//! Species group into families.
//! Units of one family convert by molecular mass ratio, so `Gt C` converts to `Gt CO2` by 44/12.
//! A species outside every family converts only to itself.
//!
//! # Families
//!
//! | Family | Species | Basis |
//! |--------|---------|-------|
//! | carbon | `C`, `CO2` | `C` |
//! | sulfur | `S`, `SO2` | `S` |
//! | nitrogen oxides | `N`, `NO2`, `NOx` (counted as NO2) | `N` |
//! | nitrous oxide | `N2ON` (the nitrogen in N2O), `N2O` | `N2ON` |
//!
//! The molecular masses are the nominal integer masses (C 12, O 16, N 14, S 32).
//! These are the ratios openscm-units uses, so conversions match the MAGICC7 input tooling.
//!
//! # Synonyms
//!
//! `NMVOC` is `VOC` and `HFC4310` is `HFC4310mee`.
//! A synonym parses to its canonical species.

/// A species that a mass unit counts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Species {
    /// Canonical token, as written in a normalised unit.
    pub name: &'static str,
    /// Family the species converts within.
    /// A species outside every family is its own family.
    pub family: &'static str,
    /// Mass of the family basis per unit mass of this species.
    pub basis_per_mass: f64,
}

struct Family {
    name: &'static str,
    /// Members and their molecular masses in g/mol, the basis first.
    members: &'static [(&'static str, f64)],
}

static FAMILIES: &[Family] = &[
    Family {
        name: "carbon",
        members: &[("C", 12.0), ("CO2", 44.0)],
    },
    Family {
        name: "sulfur",
        members: &[("S", 32.0), ("SO2", 64.0)],
    },
    Family {
        name: "nitrogen oxides",
        members: &[("N", 14.0), ("NO2", 46.0), ("NOx", 46.0)],
    },
    Family {
        name: "nitrous oxide",
        members: &[("N2ON", 28.0), ("N2O", 44.0)],
    },
];

/// Species that convert only to themselves: reactive gases, aerosols and the gcages halocarbons.
static STANDALONE: &[&str] = &[
    "CH4",
    "CO",
    "NH3",
    "BC",
    "OC",
    "VOC",
    "CFC11",
    "CFC12",
    "CFC113",
    "CFC114",
    "CFC115",
    "HCFC22",
    "HCFC141b",
    "HCFC142b",
    "HFC23",
    "HFC32",
    "HFC125",
    "HFC134a",
    "HFC143a",
    "HFC152a",
    "HFC227ea",
    "HFC236fa",
    "HFC245fa",
    "HFC365mfc",
    "HFC4310mee",
    "Halon1202",
    "Halon1211",
    "Halon1301",
    "Halon2402",
    "CCl4",
    "CHCl3",
    "CH2Cl2",
    "CH3Br",
    "CH3Cl",
    "CH3CCl3",
    "CF4",
    "C2F6",
    "C3F8",
    "C4F10",
    "C5F12",
    "C6F14",
    "C7F16",
    "C8F18",
    "cC4F8",
    "NF3",
    "SF6",
    "SO2F2",
];

static SYNONYMS: &[(&str, &str)] = &[("NMVOC", "VOC"), ("HFC4310", "HFC4310mee")];

/// Species written fused to their mass unit, as in `GtC` and `MtCO2`, longest first.
pub(crate) static FUSED: &[&str] = &["CO2", "C"];

/// Looks up a species token, resolving synonyms.
#[must_use]
pub fn lookup(token: &str) -> Option<Species> {
    let token = SYNONYMS
        .iter()
        .find(|(synonym, _)| *synonym == token)
        .map_or(token, |(_, canonical)| canonical);

    for family in FAMILIES {
        let basis_mass = family.members[0].1;
        if let Some((name, mass)) = family.members.iter().find(|(name, _)| *name == token) {
            return Some(Species {
                name,
                family: family.name,
                basis_per_mass: basis_mass / mass,
            });
        }
    }

    STANDALONE
        .iter()
        .find(|name| **name == token)
        .map(|name| Species {
            name,
            family: name,
            basis_per_mass: 1.0,
        })
}

/// Every accepted species token, synonyms included.
pub fn tokens() -> impl Iterator<Item = &'static str> {
    FAMILIES
        .iter()
        .flat_map(|family| family.members.iter().map(|(name, _)| *name))
        .chain(STANDALONE.iter().copied())
        .chain(SYNONYMS.iter().map(|(synonym, _)| *synonym))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::registry::UNIT_REGISTRY;

    fn ratio(from: &str, to: &str) -> f64 {
        let from = lookup(from).unwrap();
        let to = lookup(to).unwrap();
        assert_eq!(from.family, to.family);
        from.basis_per_mass / to.basis_per_mass
    }

    #[test]
    fn test_family_ratios() {
        assert!((ratio("C", "CO2") - 44.0 / 12.0).abs() < 1e-12);
        assert!((ratio("S", "SO2") - 64.0 / 32.0).abs() < 1e-12);
        assert!((ratio("N", "NO2") - 46.0 / 14.0).abs() < 1e-12);
        assert!((ratio("N", "NOx") - 46.0 / 14.0).abs() < 1e-12);
        assert!((ratio("NOx", "NO2") - 1.0).abs() < 1e-12);
        assert!((ratio("N2ON", "N2O") - 44.0 / 28.0).abs() < 1e-12);
    }

    #[test]
    fn test_synonyms_resolve_to_canonical() {
        assert_eq!(lookup("NMVOC").unwrap().name, "VOC");
        assert_eq!(lookup("HFC4310").unwrap().name, "HFC4310mee");
    }

    #[test]
    fn test_standalone_species_are_their_own_family() {
        let ch4 = lookup("CH4").unwrap();
        assert_eq!(ch4.family, "CH4");
        assert!((ch4.basis_per_mass - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_unknown_species() {
        assert!(lookup("XYZ").is_none());
        assert!(lookup("co2").is_none());
        assert!(lookup("SOx").is_none());
    }

    #[test]
    fn test_species_tokens_are_not_units() {
        for token in tokens() {
            assert!(
                UNIT_REGISTRY.lookup(token).is_none(),
                "{token} is also a unit"
            );
        }
    }

    #[test]
    fn test_tokens_are_unique() {
        let mut all: Vec<_> = tokens().collect();
        let count = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), count);
    }
}
