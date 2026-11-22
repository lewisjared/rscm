//! Serde encoding of floats that keeps NaN and infinities, which JSON numbers cannot hold.
//!
//! A finite float is written as a JSON number and a non-finite one as the string `"NaN"`, `"inf"` or `"-inf"`.
//! Reading accepts either form.
//! Use it on a field as `#[serde(with = "rscm_core::non_finite")]`,
//! or let [`params`](crate::params) add it to every float field of a struct.
//!
//! It covers `f64` and `f32` alone or inside `Option`, `Vec`, `VecDeque`, `Box`, fixed arrays,
//! the values of `HashMap` and `BTreeMap`, and `ndarray` arrays, nested in any order.
//! `None` stays `null`, so `Some(NaN)` is written `"NaN"`.

use ndarray::{Array, Dimension};
use serde::de::{self, DeserializeOwned, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq, SerializeTuple};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;

/// A float or a container of floats with the non-finite encoding.
pub trait NonFinite: Sized {
    /// Write the value, with each non-finite float as a string.
    fn serialize_non_finite<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error>;
    /// Read the value, accepting a number or a non-finite string for each float.
    fn deserialize_non_finite<'de, D: Deserializer<'de>>(deserializer: D)
        -> Result<Self, D::Error>;
}

/// Serialise `value` with the non-finite encoding, for `#[serde(with)]`.
pub fn serialize<T: NonFinite, S: Serializer>(value: &T, serializer: S) -> Result<S::Ok, S::Error> {
    value.serialize_non_finite(serializer)
}

/// Deserialise a value with the non-finite encoding, for `#[serde(with)]`.
pub fn deserialize<'de, T: NonFinite, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<T, D::Error> {
    T::deserialize_non_finite(deserializer)
}

/// Borrows a value to serialise it with the non-finite encoding.
pub struct Encode<'a, T>(pub &'a T);

impl<T: NonFinite> Serialize for Encode<'_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize_non_finite(serializer)
    }
}

/// Holds a value deserialised with the non-finite encoding.
pub struct Decode<T>(pub T);

impl<'de, T: NonFinite> Deserialize<'de> for Decode<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        T::deserialize_non_finite(deserializer).map(Decode)
    }
}

/// What a float field expects, which names it in errors.
pub const EXPECTING: &str = "a number or one of \"NaN\", \"inf\" and \"-inf\"";

const NAN: &str = "NaN";
const INFINITY: &str = "inf";
const NEG_INFINITY: &str = "-inf";

fn non_finite_name(value: f64) -> &'static str {
    if value.is_nan() {
        NAN
    } else if value > 0.0 {
        INFINITY
    } else {
        NEG_INFINITY
    }
}

/// Reads a number, a non-finite string, or a null as NaN when `null_is_nan`.
struct FloatVisitor {
    null_is_nan: bool,
}

impl Visitor<'_> for FloatVisitor {
    type Value = f64;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(EXPECTING)
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<f64, E> {
        Ok(value)
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<f64, E> {
        Ok(value as f64)
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<f64, E> {
        Ok(value as f64)
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<f64, E> {
        match value {
            NAN => Ok(f64::NAN),
            INFINITY => Ok(f64::INFINITY),
            NEG_INFINITY => Ok(f64::NEG_INFINITY),
            _ => Err(E::invalid_value(de::Unexpected::Str(value), &self)),
        }
    }

    fn visit_unit<E: de::Error>(self) -> Result<f64, E> {
        if self.null_is_nan {
            Ok(f64::NAN)
        } else {
            Err(E::invalid_type(de::Unexpected::Unit, &self))
        }
    }
}

/// A float that also reads null as NaN.
pub(crate) struct NullAsNan(pub(crate) f64);

impl<'de> Deserialize<'de> for NullAsNan {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer
            .deserialize_any(FloatVisitor { null_is_nan: true })
            .map(NullAsNan)
    }
}

impl NonFinite for f64 {
    fn serialize_non_finite<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.is_finite() {
            serializer.serialize_f64(*self)
        } else {
            serializer.serialize_str(non_finite_name(*self))
        }
    }

    fn deserialize_non_finite<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        deserializer.deserialize_any(FloatVisitor { null_is_nan: false })
    }
}

impl NonFinite for f32 {
    fn serialize_non_finite<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.is_finite() {
            serializer.serialize_f32(*self)
        } else {
            serializer.serialize_str(non_finite_name(f64::from(*self)))
        }
    }

    fn deserialize_non_finite<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        f64::deserialize_non_finite(deserializer).map(|value| value as f32)
    }
}

impl<T: NonFinite> NonFinite for Option<T> {
    fn serialize_non_finite<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Some(value) => serializer.serialize_some(&Encode(value)),
            None => serializer.serialize_none(),
        }
    }

    fn deserialize_non_finite<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        Ok(Option::<Decode<T>>::deserialize(deserializer)?.map(|value| value.0))
    }
}

impl<T: NonFinite> NonFinite for Box<T> {
    fn serialize_non_finite<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (**self).serialize_non_finite(serializer)
    }

    fn deserialize_non_finite<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        T::deserialize_non_finite(deserializer).map(Box::new)
    }
}

fn serialize_seq<'a, T: NonFinite + 'a, S: Serializer>(
    values: impl ExactSizeIterator<Item = &'a T>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let mut seq = serializer.serialize_seq(Some(values.len()))?;
    for value in values {
        seq.serialize_element(&Encode(value))?;
    }
    seq.end()
}

impl<T: NonFinite> NonFinite for Vec<T> {
    fn serialize_non_finite<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serialize_seq(self.iter(), serializer)
    }

    fn deserialize_non_finite<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        let values = Vec::<Decode<T>>::deserialize(deserializer)?;
        Ok(values.into_iter().map(|value| value.0).collect())
    }
}

impl<T: NonFinite> NonFinite for VecDeque<T> {
    fn serialize_non_finite<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serialize_seq(self.iter(), serializer)
    }

    fn deserialize_non_finite<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        Ok(Vec::<T>::deserialize_non_finite(deserializer)?.into())
    }
}

struct ArrayVisitor<T, const N: usize>(PhantomData<T>);

impl<'de, T: NonFinite, const N: usize> Visitor<'de> for ArrayVisitor<T, N> {
    type Value = [T; N];

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "an array of length {N}")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<[T; N], A::Error> {
        let mut values = Vec::with_capacity(N);
        while let Some(Decode(value)) = seq.next_element::<Decode<T>>()? {
            values.push(value);
        }
        let found = values.len();
        values
            .try_into()
            .map_err(|_| de::Error::invalid_length(found, &self))
    }
}

impl<T: NonFinite, const N: usize> NonFinite for [T; N] {
    fn serialize_non_finite<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(N)?;
        for value in self {
            tuple.serialize_element(&Encode(value))?;
        }
        tuple.end()
    }

    fn deserialize_non_finite<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        deserializer.deserialize_tuple(N, ArrayVisitor(PhantomData))
    }
}

fn serialize_map<'a, K: Serialize + 'a, V: NonFinite + 'a, S: Serializer>(
    entries: impl ExactSizeIterator<Item = (&'a K, &'a V)>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let mut map = serializer.serialize_map(Some(entries.len()))?;
    for (key, value) in entries {
        map.serialize_entry(key, &Encode(value))?;
    }
    map.end()
}

impl<K, V> NonFinite for HashMap<K, V>
where
    K: Serialize + DeserializeOwned + Eq + Hash,
    V: NonFinite,
{
    fn serialize_non_finite<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serialize_map(self.iter(), serializer)
    }

    fn deserialize_non_finite<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        let map = HashMap::<K, Decode<V>>::deserialize(deserializer)?;
        Ok(map.into_iter().map(|(key, value)| (key, value.0)).collect())
    }
}

impl<K, V> NonFinite for BTreeMap<K, V>
where
    K: Serialize + DeserializeOwned + Ord,
    V: NonFinite,
{
    fn serialize_non_finite<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serialize_map(self.iter(), serializer)
    }

    fn deserialize_non_finite<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        let map = BTreeMap::<K, Decode<V>>::deserialize(deserializer)?;
        Ok(map.into_iter().map(|(key, value)| (key, value.0)).collect())
    }
}

/// An ndarray array in ndarray's own layout, version, shape and elements in row-major order.
#[derive(Serialize, Deserialize)]
struct Shaped<D, T> {
    v: u8,
    dim: D,
    data: T,
}

impl<T, D> NonFinite for Array<T, D>
where
    T: NonFinite + Clone,
    D: Dimension + Serialize + DeserializeOwned,
{
    fn serialize_non_finite<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let data: Vec<Encode<T>> = self.iter().map(Encode).collect();
        Shaped {
            v: 1,
            dim: self.raw_dim(),
            data,
        }
        .serialize(serializer)
    }

    fn deserialize_non_finite<'de, D2: Deserializer<'de>>(
        deserializer: D2,
    ) -> Result<Self, D2::Error> {
        let shaped = Shaped::<D, Vec<Decode<T>>>::deserialize(deserializer)?;
        let data = shaped.data.into_iter().map(|value| value.0).collect();
        Array::from_shape_vec(shaped.dim, data).map_err(de::Error::custom)
    }
}

/// Checks that components keep non-finite floats through serde, for each crate's tests.
#[doc(hidden)]
pub mod testing {
    use crate::component::{Component, ComponentState};
    use serde::de::DeserializeOwned;
    use serde_json::Value;
    use std::collections::BTreeSet;

    const NON_FINITE: [&str; 3] = ["NaN", "inf", "-inf"];

    /// The component types rscm-core registers.
    /// `PythonComponent` keeps its state as a string from Python, which the Python tests cover.
    pub const CORE_COMPONENT_TYPES: [&str; 3] =
        ["AggregateComponent", "PythonComponent", "TestComponent"];

    /// Panics unless `components` holds one of each registered type not in `upstream`,
    /// and each keeps its non-finite floats.
    pub fn assert_components_round_trip(components: &[Box<dyn Component>], upstream: &[&str]) {
        let covered: BTreeSet<_> = components
            .iter()
            .map(|c| c.typetag_name().to_string())
            .collect();
        let expected: BTreeSet<_> = registered_component_types()
            .into_iter()
            .filter(|name| !upstream.contains(&name.as_str()))
            .collect();
        assert_eq!(
            covered, expected,
            "every component type needs an instance to check"
        );
        for component in components {
            assert_round_trips(component.as_ref());
        }
    }

    /// The typetag name of every component type linked into this binary.
    pub fn registered_component_types() -> Vec<String> {
        let unknown = serde_json::json!({"type": "\u{0}"});
        let err = serde_json::from_value::<Box<dyn Component>>(unknown)
            .expect_err("no component type is named NUL")
            .to_string();
        let (_, expected) = err
            .split_once("expected")
            .unwrap_or_else(|| panic!("unexpected error listing component types: {err}"));
        expected
            .split('`')
            .skip(1)
            .step_by(2)
            .map(str::to_string)
            .collect()
    }

    /// Panics unless `component` and its initial state keep NaN and infinities in every float field.
    ///
    /// Each float is replaced by each non-finite value, all at once and, if the component rejects that,
    /// one at a time, then read back and written again.
    pub fn assert_round_trips(component: &dyn Component) {
        let name = component.typetag_name();
        assert_floats_round_trip::<Box<dyn Component>>(
            name,
            serde_json::to_value(component).unwrap(),
        );
        let state = component.create_initial_state();
        assert_floats_round_trip::<Box<dyn ComponentState>>(
            &format!("{name} state"),
            serde_json::to_value(&state).unwrap(),
        );
    }

    fn assert_floats_round_trip<T: serde::Serialize + DeserializeOwned>(what: &str, value: Value) {
        let paths = float_paths(&value, &mut Vec::new());
        let replace = |paths: &[Vec<String>], replacement: &str| {
            let mut replaced = value.clone();
            for path in paths {
                *pointer(&mut replaced, path) = Value::from(replacement);
            }
            replaced
        };
        for replacement in NON_FINITE {
            if read_back::<T>(what, &replace(&paths, replacement)).is_ok() {
                continue;
            }
            for path in &paths {
                let what = format!("{what} at /{}", path.join("/"));
                if let Err(Some(message)) =
                    read_back::<T>(&what, &replace(std::slice::from_ref(path), replacement))
                {
                    panic!("{what} cannot read a non-finite float: {message}");
                }
            }
        }
    }

    /// Reads `value` back and checks it is written unchanged.
    ///
    /// Fails with the message if a non-finite string was unreadable, or with none if construction rejected it.
    fn read_back<T: serde::Serialize + DeserializeOwned>(
        what: &str,
        value: &Value,
    ) -> Result<(), Option<String>> {
        let read = serde_json::from_value::<T>(value.clone()).map_err(|err| {
            let message = err.to_string();
            NON_FINITE
                .iter()
                .any(|s| message.contains(&format!("invalid type: string \"{s}\"")))
                .then_some(message)
        })?;
        let written = serde_json::to_value(&read).unwrap();
        assert_eq!(&written, value, "{what} did not keep its non-finite floats");
        Ok(())
    }

    fn float_paths(value: &Value, path: &mut Vec<String>) -> Vec<Vec<String>> {
        let mut found = Vec::new();
        let mut descend = |key: String, child: &Value, path: &mut Vec<String>| {
            path.push(key);
            found.extend(float_paths(child, path));
            path.pop();
        };
        match value {
            Value::Number(n) if n.is_f64() => return vec![path.clone()],
            Value::Array(items) => {
                for (i, item) in items.iter().enumerate() {
                    descend(i.to_string(), item, path);
                }
            }
            Value::Object(map) => {
                for (key, item) in map {
                    descend(key.clone(), item, path);
                }
            }
            _ => {}
        }
        found
    }

    fn pointer<'a>(value: &'a mut Value, path: &[String]) -> &'a mut Value {
        path.iter().fold(value, |value, key| match value {
            Value::Array(items) => &mut items[key.parse::<usize>().unwrap()],
            Value::Object(map) => map.get_mut(key).unwrap(),
            _ => unreachable!("a float path only passes through arrays and objects"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use serde_json::json;

    fn round_trip<T: NonFinite>(value: &T) -> (serde_json::Value, T) {
        let json = serde_json::to_value(Encode(value)).unwrap();
        let text = serde_json::to_string(&json).unwrap();
        let read = serde_json::from_str::<Decode<T>>(&text).unwrap().0;
        (json, read)
    }

    #[test]
    fn non_finite_floats_are_strings() {
        let (json, read) = round_trip(&vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.5]);
        assert_eq!(json, json!(["NaN", "inf", "-inf", 1.5]));
        assert!(read[0].is_nan());
        assert_eq!(read[1..], [f64::INFINITY, f64::NEG_INFINITY, 1.5]);
    }

    #[test]
    fn finite_floats_keep_every_bit() {
        let values = [0.1 + 0.2, 1e-310, -0.0, f64::MAX, 280.0];
        let (_, read) = round_trip(&values);
        for (value, read) in values.iter().zip(read) {
            assert_eq!(value.to_bits(), read.to_bits());
        }
    }

    #[test]
    fn option_keeps_null_for_none() {
        let (json, read) = round_trip(&vec![None, Some(f64::NAN), Some(2.0)]);
        assert_eq!(json, json!([null, "NaN", 2.0]));
        assert_eq!(read[0], None);
        assert!(read[1].unwrap().is_nan());
        assert_eq!(read[2], Some(2.0));
    }

    #[test]
    fn containers_round_trip() {
        let nested = vec![vec![f64::INFINITY], vec![]];
        assert_eq!(round_trip(&nested).1, nested);

        let map = HashMap::from([("a".to_string(), f64::NEG_INFINITY)]);
        assert_eq!(round_trip(&map).0, json!({"a": "-inf"}));

        let tree = BTreeMap::from([("b".to_string(), [1.0, f64::INFINITY])]);
        assert_eq!(round_trip(&tree).1, tree);

        let deque = VecDeque::from([f64::INFINITY, 3.0]);
        assert_eq!(round_trip(&deque).1, deque);

        let grid = array![[1.0, f64::INFINITY], [f64::NEG_INFINITY, 4.0]];
        let (json, read) = round_trip(&grid);
        assert_eq!(
            json,
            json!({"v": 1, "dim": [2, 2], "data": [1.0, "inf", "-inf", 4.0]})
        );
        assert_eq!(read, grid);
    }

    fn two() -> f64 {
        2.0
    }

    #[crate::params]
    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct Annotated {
        value: f64,
        #[serde(default = "two")]
        defaulted: f64,
        optional: Option<f64>,
        nested: Vec<[f64; 2]>,
        name: String,
    }

    #[crate::params]
    #[derive(Debug, Default, Serialize, Deserialize, PartialEq)]
    #[serde(default)]
    struct ContainerDefault {
        optional: Option<f64>,
        value: f64,
    }

    #[crate::params]
    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    enum Variants {
        Tuple(f64),
        Named { value: Option<f64> },
    }

    #[test]
    fn params_annotates_float_fields() {
        let annotated: Annotated = serde_json::from_value(json!({
            "value": "inf", "nested": [[1.0, "-inf"]], "name": "x",
        }))
        .unwrap();
        assert_eq!(
            annotated,
            Annotated {
                value: f64::INFINITY,
                defaulted: 2.0,
                optional: None,
                nested: vec![[1.0, f64::NEG_INFINITY]],
                name: "x".into(),
            }
        );
        let json = serde_json::to_value(&annotated).unwrap();
        assert_eq!(json["value"], "inf");
        assert_eq!(json["optional"], json!(null));
        assert!(serde_json::from_value::<Annotated>(json!({
            "value": 1.0, "nested": [], "name": "x", "extra": 1,
        }))
        .is_err());

        let read: ContainerDefault = serde_json::from_value(json!({"value": "-inf"})).unwrap();
        assert_eq!(
            read,
            ContainerDefault {
                optional: None,
                value: f64::NEG_INFINITY
            }
        );

        for variant in [
            Variants::Tuple(f64::INFINITY),
            Variants::Named {
                value: Some(f64::INFINITY),
            },
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            assert!(json.contains("\"inf\""), "{json}");
            assert_eq!(serde_json::from_str::<Variants>(&json).unwrap(), variant);
        }
    }

    #[test]
    fn reads_integers_and_rejects_null() {
        assert_eq!(serde_json::from_str::<Decode<f64>>("3").unwrap().0, 3.0);
        assert!(serde_json::from_str::<Decode<f64>>("null").is_err());
        assert!(serde_json::from_str::<Decode<f64>>("\"nan\"").is_err());
        assert!(serde_json::from_str::<Decode<[f64; 2]>>("[1.0]").is_err());
    }
}
