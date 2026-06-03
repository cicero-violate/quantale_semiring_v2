//! Small structured console formatter for runtime status lines.

pub fn info(scope: &str, message: &str, fields: &[(&str, String)]) {
    println!("{}", line(scope, "INFO", message, fields));
}

pub fn warn(scope: &str, message: &str, fields: &[(&str, String)]) {
    eprintln!("{}", line(scope, "WARN", message, fields));
}

pub fn error(scope: &str, message: &str, fields: &[(&str, String)]) {
    eprintln!("{}", line(scope, "ERROR", message, fields));
}

fn line(scope: &str, level: &str, message: &str, fields: &[(&str, String)]) -> String {
    let mut out = format!("[{scope}] [{level}] {message}");
    for (key, value) in fields {
        out.push(' ');
        out.push_str(key);
        out.push('=');
        out.push_str(&format_value(value));
    }
    out
}

fn format_value(value: &str) -> String {
    if value.is_empty()
        || value
            .chars()
            .any(|ch| ch.is_whitespace() || matches!(ch, '"' | '[' | ']' | '(' | ')' | ','))
    {
        serde_json::to_string(value).unwrap_or_else(|_| format!("{value:?}"))
    } else {
        value.to_string()
    }
}
