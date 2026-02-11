use std::fs;
use std::path::{Path, PathBuf};

/// Minimal `.env` loader for local development.
///
/// - Looks for `.env` in `cwd` and then walks up parent directories.
/// - Only sets vars that are not already present in the process environment.
pub fn load() {
    let Some(dotenv_path) = find_dotenv(std::env::current_dir().ok()) else {
        return;
    };

    let Ok(contents) = fs::read_to_string(&dotenv_path) else {
        return;
    };

    for (idx, raw_line) in contents.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let Some((k, v)) = line.split_once('=') else {
            eprintln!(
                "Warning: ignoring malformed .env line {} in {}",
                idx + 1,
                dotenv_path.display()
            );
            continue;
        };

        let key = k.trim();
        if key.is_empty() || std::env::var_os(key).is_some() {
            continue;
        }

        let mut value = v.trim().to_string();
        // KEY="value" / KEY='value'
        if (value.starts_with('"') && value.ends_with('"'))
            || (value.starts_with('\'') && value.ends_with('\''))
        {
            value = value[1..value.len().saturating_sub(1)].to_string();
        }

        unsafe {
            std::env::set_var(key, value);
        }
    }
}

fn find_dotenv(mut dir: Option<PathBuf>) -> Option<PathBuf> {
    while let Some(d) = dir {
        let candidate = d.join(".env");
        if candidate.is_file() {
            return Some(candidate);
        }
        dir = d.parent().map(Path::to_path_buf);
    }
    None
}
