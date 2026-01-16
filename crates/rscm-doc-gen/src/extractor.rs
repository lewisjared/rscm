//! Doc comment and attribute extraction utilities

use regex::Regex;
use syn::{Attribute, Expr, ExprLit, Lit, Meta};

/// Extract doc comments from attributes
///
/// Combines all `/// ...` or `#[doc = "..."]` attributes into a single string.
pub fn extract_doc_comments(attrs: &[Attribute]) -> String {
    let mut docs = Vec::new();

    for attr in attrs {
        if attr.path().is_ident("doc") {
            if let Meta::NameValue(meta) = &attr.meta {
                if let Expr::Lit(ExprLit {
                    lit: Lit::Str(lit_str),
                    ..
                }) = &meta.value
                {
                    docs.push(lit_str.value());
                }
            }
        }
    }

    // Join with newlines, then clean up leading whitespace
    let combined = docs.join("\n");
    clean_doc_string(&combined)
}

/// Clean up a doc string by removing leading whitespace consistently
fn clean_doc_string(doc: &str) -> String {
    let lines: Vec<&str> = doc.lines().collect();
    if lines.is_empty() {
        return String::new();
    }

    // Find minimum leading whitespace (ignoring empty lines)
    let min_indent = lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min()
        .unwrap_or(0);

    // Remove that much whitespace from each line
    lines
        .iter()
        .map(|line| {
            if line.len() >= min_indent {
                &line[min_indent..]
            } else {
                line.trim()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Extract LaTeX equations from a doc string
///
/// Looks for content between `$$` markers.
pub fn extract_equations(doc: &str) -> String {
    let re = Regex::new(r"\$\$[\s\S]*?\$\$").expect("Invalid regex");
    let equations: Vec<&str> = re.find_iter(doc).map(|m| m.as_str()).collect();
    equations.join("\n\n")
}

/// Extract description (doc string without equations)
pub fn extract_description(doc: &str) -> String {
    let re = Regex::new(r"\$\$[\s\S]*?\$\$").expect("Invalid regex");
    let without_equations = re.replace_all(doc, "");
    without_equations.trim().to_string()
}

/// Parse a unit annotation from doc comments
///
/// Looks for patterns like:
/// - `/// unit: K`
/// - `unit = "K"`
pub fn extract_unit_from_docs(doc: &str) -> Option<String> {
    // Look for "unit: <value>" or "unit = <value>" pattern
    let re = Regex::new(r#"(?i)unit\s*[:=]\s*"?([^"\n]+)"?"#).expect("Invalid regex");
    re.captures(doc)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_equations() {
        let doc = r#"
This is a component description.

$$ \frac{dC}{dt} = E - \frac{C - C_0}{\tau} $$

Some more text.

$$ T = T_0 + \Delta T $$

Final text.
"#;
        let equations = extract_equations(doc);
        assert!(equations.contains(r"$$ \frac{dC}{dt}"));
        assert!(equations.contains(r"$$ T = T_0"));
    }

    #[test]
    fn test_extract_description() {
        let doc = r#"
This is a component description.

$$ \frac{dC}{dt} = E $$

Some more text.
"#;
        let desc = extract_description(doc);
        assert!(desc.contains("This is a component description"));
        assert!(desc.contains("Some more text"));
        assert!(!desc.contains("$$"));
    }

    #[test]
    fn test_extract_unit_from_docs() {
        assert_eq!(extract_unit_from_docs("/// unit: K"), Some("K".to_string()));
        assert_eq!(
            extract_unit_from_docs(r#"unit = "W / m^2""#),
            Some("W / m^2".to_string())
        );
        assert_eq!(extract_unit_from_docs("no unit here"), None);
    }

    #[test]
    fn test_clean_doc_string() {
        let doc = " This is line one\n This is line two";
        let cleaned = clean_doc_string(doc);
        assert_eq!(cleaned, "This is line one\nThis is line two");
    }
}
