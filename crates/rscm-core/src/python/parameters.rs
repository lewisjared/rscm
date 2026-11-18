//! Convert a Python dict of component parameters, naming every bad key in the error.

use crate::errors::close_matches;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyMapping;
use serde::de::{self, DeserializeOwned, DeserializeSeed, IntoDeserializer, MapAccess, Visitor};
use std::cell::RefCell;
use std::fmt;

/// Deserialise the parameters of `component` from a Python mapping.
///
/// The error names the component, lists every missing field with its expected type,
/// suggests close matches for unknown keys and names the field holding a bad value.
pub fn depythonize_parameters<T: DeserializeOwned>(
    component: &str,
    parameters: &Bound<'_, PyAny>,
) -> PyResult<T> {
    let mut depythonizer = pythonize::Depythonizer::from_object(parameters);
    let error = match serde_path_to_error::deserialize::<_, T>(&mut depythonizer) {
        Ok(value) => return Ok(value),
        Err(error) => error,
    };
    let Ok(mapping) = parameters.cast::<PyMapping>() else {
        return Err(PyValueError::new_err(format!(
            "{component} parameters must be a dict, not {}",
            parameters.get_type().name()?
        )));
    };
    let keys: Vec<String> = mapping
        .keys()?
        .iter()
        .map(|key| key.str().map(|s| s.to_string()))
        .collect::<PyResult<_>>()?;
    let Some(fields) = probe_fields::<T>() else {
        return Err(PyValueError::new_err(format!(
            "Invalid {component} parameters: {error}"
        )));
    };
    let names: Vec<&str> = fields.iter().map(|(name, _)| *name).collect();

    let mut problems = Vec::new();
    if error.inner().to_string().starts_with("unknown field") {
        for key in keys.iter().filter(|k| !names.contains(&k.as_str())) {
            let hint = match close_matches(key, &names).first() {
                Some(close) => format!(", did you mean '{close}'?"),
                None => String::new(),
            };
            problems.push(format!("unknown parameter '{key}'{hint}"));
        }
    }
    let missing: Vec<String> = fields
        .iter()
        .filter(|(name, _)| !keys.iter().any(|k| k == name) && is_required::<T>(name))
        .map(|(name, kind)| format!("{name} ({kind})"))
        .collect();
    if !missing.is_empty() {
        problems.push(format!("missing {}", missing.join(", ")));
    }
    if problems.is_empty() {
        let path = error.path().to_string();
        let kind = fields
            .iter()
            .find(|(name, _)| *name == path)
            .map(|(_, k)| k);
        let value = mapping.get_item(&path).ok();
        problems.push(match (kind, value) {
            (Some(kind), Some(value)) => format!(
                "parameter '{path}' must be {kind}, not {}",
                value.get_type().name()?
            ),
            _ => format!("parameter '{path}': {}", error.inner()),
        });
    }
    Err(PyValueError::new_err(format!(
        "Invalid {component} parameters: {}",
        problems.join("; ")
    )))
}

/// Whether deserialising `T` fails without `field`.
fn is_required<T: DeserializeOwned>(field: &str) -> bool {
    let mut deserializer = Probe {
        skip: Some(field),
        fields: RefCell::new(Vec::new()),
    };
    match T::deserialize(&mut deserializer) {
        Ok(_) => false,
        Err(ProbeError(message)) => {
            message == format!("missing field `{field}`") || !message.starts_with("missing field")
        }
    }
}

/// The fields of struct `T` with their expected types, or None if `T` is not a struct.
fn probe_fields<T: DeserializeOwned>() -> Option<Vec<(&'static str, String)>> {
    let mut deserializer = Probe {
        skip: None,
        fields: RefCell::new(Vec::new()),
    };
    let _ = T::deserialize(&mut deserializer);
    let fields = deserializer.fields.into_inner();
    (!fields.is_empty()).then_some(fields)
}

#[derive(Debug)]
struct ProbeError(String);

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ProbeError {}

impl de::Error for ProbeError {
    fn custom<M: fmt::Display>(message: M) -> Self {
        ProbeError(message.to_string())
    }
}

/// A deserializer that records a struct's fields and feeds each a placeholder of its type.
struct Probe<'s> {
    skip: Option<&'s str>,
    fields: RefCell<Vec<(&'static str, String)>>,
}

impl<'de> de::Deserializer<'de> for &mut Probe<'_> {
    type Error = ProbeError;

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, ProbeError> {
        *self.fields.borrow_mut() = fields.iter().map(|f| (*f, "value".to_string())).collect();
        visitor.visit_map(ProbeMap {
            probe: self,
            fields,
            index: 0,
        })
    }

    fn deserialize_any<V: Visitor<'de>>(self, _visitor: V) -> Result<V::Value, ProbeError> {
        Err(de::Error::custom("not a struct"))
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map enum identifier ignored_any
    }
}

struct ProbeMap<'a, 's> {
    probe: &'a Probe<'s>,
    fields: &'static [&'static str],
    index: usize,
}

impl<'de> MapAccess<'de> for ProbeMap<'_, '_> {
    type Error = ProbeError;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, ProbeError> {
        while self.index < self.fields.len() && Some(self.fields[self.index]) == self.probe.skip {
            self.index += 1;
        }
        let Some(field) = self.fields.get(self.index) else {
            return Ok(None);
        };
        seed.deserialize(field.into_deserializer()).map(Some)
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(
        &mut self,
        seed: V,
    ) -> Result<V::Value, ProbeError> {
        let index = self.index;
        self.index += 1;
        let kind = RefCell::new("value");
        let value = seed.deserialize(Placeholder { kind: &kind });
        self.probe.fields.borrow_mut()[index].1 = kind.into_inner().to_string();
        value
    }
}

/// Records the type a field asks for and hands back an empty value of it.
struct Placeholder<'k> {
    kind: &'k RefCell<&'static str>,
}

impl Placeholder<'_> {
    fn record(&self, kind: &'static str) {
        *self.kind.borrow_mut() = kind;
    }
}

macro_rules! placeholder {
    ($($method:ident => $kind:literal, |$visitor:ident| $value:expr;)*) => {
        $(
            fn $method<V: Visitor<'de>>(self, $visitor: V) -> Result<V::Value, ProbeError> {
                self.record($kind);
                $value
            }
        )*
    };
}

impl<'de> de::Deserializer<'de> for Placeholder<'_> {
    type Error = ProbeError;

    placeholder! {
        deserialize_bool => "bool", |v| v.visit_bool(false);
        deserialize_i8 => "int", |v| v.visit_i64(0);
        deserialize_i16 => "int", |v| v.visit_i64(0);
        deserialize_i32 => "int", |v| v.visit_i64(0);
        deserialize_i64 => "int", |v| v.visit_i64(0);
        deserialize_u8 => "int", |v| v.visit_u64(0);
        deserialize_u16 => "int", |v| v.visit_u64(0);
        deserialize_u32 => "int", |v| v.visit_u64(0);
        deserialize_u64 => "int", |v| v.visit_u64(0);
        deserialize_f32 => "float", |v| v.visit_f64(0.0);
        deserialize_f64 => "float", |v| v.visit_f64(0.0);
        deserialize_str => "str", |v| v.visit_str("");
        deserialize_string => "str", |v| v.visit_str("");
        deserialize_option => "optional", |v| v.visit_none();
        deserialize_seq => "list", |v| v.visit_seq(de::value::SeqDeserializer::new(std::iter::empty::<f64>()));
        deserialize_map => "dict", |v| v.visit_map(de::value::MapDeserializer::new(std::iter::empty::<(f64, f64)>()));
        deserialize_any => "value", |_v| Err(de::Error::custom("no placeholder"));
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        _visitor: V,
    ) -> Result<V::Value, ProbeError> {
        self.record("dict");
        Err(de::Error::custom("no placeholder"))
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        _visitor: V,
    ) -> Result<V::Value, ProbeError> {
        self.record("str");
        Err(de::Error::custom("no placeholder"))
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, ProbeError> {
        visitor.visit_newtype_struct(self)
    }

    serde::forward_to_deserialize_any! {
        i128 u128 char bytes byte_buf unit unit_struct tuple tuple_struct identifier ignored_any
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    #[allow(dead_code)]
    struct Parameters {
        rate: f64,
        layers: Vec<f64>,
        name: String,
        #[serde(default)]
        scale: f64,
        limit: Option<f64>,
    }

    #[test]
    fn probe_lists_fields_with_types() {
        assert_eq!(
            probe_fields::<Parameters>().unwrap(),
            [
                ("rate", "float".to_string()),
                ("layers", "list".to_string()),
                ("name", "str".to_string()),
                ("scale", "float".to_string()),
                ("limit", "optional".to_string()),
            ]
        );
    }

    #[test]
    fn defaults_and_options_are_not_required() {
        assert!(is_required::<Parameters>("rate"));
        assert!(is_required::<Parameters>("layers"));
        assert!(!is_required::<Parameters>("scale"));
        assert!(!is_required::<Parameters>("limit"));
    }
}
