//! Doc comment and attribute extraction utilities

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

/// Extract description from doc string
///
/// Returns the full doc string including any LaTeX equations.
/// Equations use `$$...$$` delimiters and are rendered by KaTeX.
pub fn extract_description(doc: &str) -> String {
    doc.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_description_preserves_equations() {
        let doc = r#"
This is a component description.

$$ \frac{dC}{dt} = E - \frac{C - C_0}{\tau} $$

Some more text.
"#;
        let desc = extract_description(doc);
        assert!(desc.contains("This is a component description"));
        assert!(desc.contains("Some more text"));
        // Equations should be preserved
        assert!(desc.contains(r"$$ \frac{dC}{dt}"));
    }

    #[test]
    fn test_clean_doc_string() {
        let doc = " This is line one\n This is line two";
        let cleaned = clean_doc_string(doc);
        assert_eq!(cleaned, "This is line one\nThis is line two");
    }
}
