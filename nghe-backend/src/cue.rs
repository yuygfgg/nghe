use std::borrow::Cow;

use cue_rw::{CUEFile, FileID};
use encoding_rs as enc;
use nghe_api::common::format::Trait as _;
use num_rational::Rational32;
use typed_path::{Utf8TypedPath, Utf8TypedPathBuf};

use crate::{Error, file};

mod flac_split;

/// Marker directory suffix for "virtual files" derived from a `.cue` sheet.
///
/// Example virtual track path:
/// `Album.cue.__nghe_cue__/01.flac`
pub const VIRTUAL_DIR_SUFFIX: &str = ".__nghe_cue__";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CueTrackSpan {
    pub start_seconds: f64,
    pub duration_seconds: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct CueSheet {
    cue:  CUEFile,
    date: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualCueTrack {
    /// Absolute path to the `.cue` file on the configured filesystem.
    pub cue_path: Utf8TypedPathBuf,
    /// Virtual `NN` number (1-based) used by `build_virtual_track_relative_path`.
    pub track_number: u16,
}

impl VirtualCueTrack {
    pub fn parse_virtual_track_path(path: Utf8TypedPath<'_>) -> Option<Self> {
        let track_file_name = path.file_name()?;
        let track_number = parse_track_file_number(track_file_name)?;

        let virtual_dir = path.parent()?;
        let virtual_dir_name = virtual_dir.file_name()?;
        let cue_file_name = virtual_dir_name.strip_suffix(VIRTUAL_DIR_SUFFIX)?;
        if !cue_file_name.to_ascii_lowercase().ends_with(".cue") {
            return None;
        }

        let cue_parent = virtual_dir.parent()?;
        let cue_path = cue_parent.join(cue_file_name);
        Some(Self { cue_path, track_number })
    }
}

fn parse_track_file_number(file_name: &str) -> Option<u16> {
    // Expect something like "01.flac" (but tolerate arbitrary extensions).
    let (stem, _) = file_name.split_once('.')?;
    stem.parse().ok()
}

fn non_empty(s: &str) -> Option<&str> {
    let s = s.trim();
    if s.is_empty() { None } else { Some(s) }
}

fn cue_timestamp_seconds(ts: cue_rw::CUETimeStamp) -> f64 {
    let seconds = Rational32::from(ts);
    f64::from(*seconds.numer()) / f64::from(*seconds.denom())
}

fn strip_surrounding_quotes(s: &str) -> &str {
    let s = s.trim();
    s.strip_prefix('"').and_then(|s| s.strip_suffix('"')).unwrap_or(s)
}

fn split_word_rest(s: &str) -> (&str, &str) {
    let s = s.trim_start();
    let mut split_at = None;
    for (idx, ch) in s.char_indices() {
        if ch.is_whitespace() {
            split_at = Some(idx);
            break;
        }
    }

    match split_at {
        Some(idx) => (&s[..idx], s[idx..].trim()),
        None => (s, ""),
    }
}

fn parse_rem_date(comments: &[String]) -> Option<String> {
    comments.iter().find_map(|line| {
        // Be tolerant of different casing / spacing.
        let line = line.trim();
        if line.is_empty() {
            return None;
        }

        // `cue-rw` stores the remainder of `REM ...` lines (without the leading `REM`),
        // so `REM DATE 2001` becomes `DATE 2001` in `CUEFile.comments`.
        //
        // Still, accept both formats in case another parser changes this.
        let (tag, rest) = split_word_rest(line);
        if tag.eq_ignore_ascii_case("DATE") {
            return non_empty(strip_surrounding_quotes(rest)).map(str::to_owned);
        }

        if tag.eq_ignore_ascii_case("REM") {
            let (subtag, subrest) = split_word_rest(rest);
            if subtag.eq_ignore_ascii_case("DATE") {
                return non_empty(strip_surrounding_quotes(subrest)).map(str::to_owned);
            }
        }

        // Also tolerate `DATE=...` / `REM DATE=...` formats.
        let upper = line.to_ascii_uppercase();
        if let Some(rest) = upper.strip_prefix("DATE=") {
            let rest = line.get(line.len() - rest.len()..).unwrap_or("");
            return non_empty(strip_surrounding_quotes(rest)).map(str::to_owned);
        }
        if let Some(rest) = upper.strip_prefix("REM DATE=") {
            let rest = line.get(line.len() - rest.len()..).unwrap_or("");
            return non_empty(strip_surrounding_quotes(rest)).map(str::to_owned);
        }

        None
    })
}

pub fn decode_cue(bytes: &[u8]) -> Result<Cow<'_, str>, Error> {
    let encodings: [&'static enc::Encoding; 6] =
        [enc::UTF_8, enc::GBK, enc::GB18030, enc::SHIFT_JIS, enc::BIG5, enc::UTF_16LE];

    for encoding in encodings {
        let (s, _, had_errors) = encoding.decode(bytes);
        if !had_errors {
            return Ok(s);
        }
    }

    // Fall back to UTF-8 lossy as a last resort, so we can still attempt to parse timings.
    Ok(String::from_utf8_lossy(bytes))
}

impl CueSheet {
    pub fn parse(bytes: &[u8]) -> Result<Self, Error> {
        let s = decode_cue(bytes)?;
        Self::parse_str(&s)
    }

    pub fn parse_str(input: &str) -> Result<Self, Error> {
        // Be tolerant of surrounding whitespace; some cue sheets have leading/trailing blank lines.
        let cue = CUEFile::try_from(input.trim())?;
        let date = parse_rem_date(&cue.comments);
        Ok(Self { cue, date })
    }

    pub fn cue(&self) -> &CUEFile {
        &self.cue
    }

    pub fn title(&self) -> Option<&str> {
        non_empty(&self.cue.title)
    }

    pub fn performer(&self) -> Option<&str> {
        non_empty(&self.cue.performer)
    }

    pub fn catalog(&self) -> Option<&str> {
        self.cue.catalog.as_deref().and_then(non_empty)
    }

    pub fn date(&self) -> Option<&str> {
        self.date.as_deref().and_then(non_empty)
    }

    pub fn is_single_file(&self) -> bool {
        self.cue.files.len() == 1
    }

    pub fn file_name_matches(&self, audio_file_name: &str) -> bool {
        let Some(file) = self.cue.files.first() else { return false };
        // CUE's FILE may include a relative path. Only compare the basename.
        let cue_file_basename = file.rsplit(['/', '\\']).next().unwrap_or(file.as_str());
        cue_file_basename.eq_ignore_ascii_case(audio_file_name)
    }

    pub fn resolve_track_span(&self, track_number: u16) -> Option<CueTrackSpan> {
        let file_id = FileID::from(0usize);

        // Map virtual `NN` to the Nth track (in-file) with `INDEX 01`.
        let mut tracks: Vec<(u16, f64)> = vec![];
        let mut number: u16 = 0;
        for (_fid, t) in self.cue.tracks.iter().filter(|(fid, _)| *fid == file_id) {
            number = number.saturating_add(1);
            let Some((_, ts)) = t.indices.iter().find(|(id, _)| *id == 1) else {
                continue;
            };
            tracks.push((number, cue_timestamp_seconds(*ts)));
        }

        tracks.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let idx = tracks.iter().position(|(n, _)| *n == track_number)?;
        let start_seconds = tracks[idx].1;
        let duration_seconds = tracks.get(idx + 1).map(|(_, next)| next - start_seconds);
        Some(CueTrackSpan { start_seconds, duration_seconds })
    }

    pub fn resolve_audio_file_path<'a>(
        &self,
        cue_path: Utf8TypedPath<'a>,
    ) -> Option<Utf8TypedPathBuf> {
        let file = self.cue.files.first()?;
        let parent = cue_path.parent()?;
        // Be tolerant of Windows-style separators inside the CUE sheet.
        let name = file.replace('\\', "/");
        Some(parent.join(&name))
    }
}

pub(crate) async fn virtual_flac_track_response(
    filesystem: &crate::filesystem::Impl<'_>,
    config: &crate::config::Transcode,
    cache_key: &crate::file::Property<crate::file::audio::Format>,
    virtual_track: &VirtualCueTrack,
    range: Option<axum_extra::headers::Range>,
) -> Result<crate::http::binary::Response, Error> {
    let prepared =
        flac_split::prepare_virtual_flac_track(filesystem, config, cache_key, virtual_track).await?;
    flac_split::response_from_prepared_track(prepared, range).await
}

/// Construct a stable "virtual relative path" for a track of a CUE sheet.
pub fn build_virtual_track_relative_path(
    cue_relative_path: Utf8TypedPath<'_>,
    track_number: u16,
    format: file::audio::Format,
) -> Result<Utf8TypedPathBuf, Error> {
    let parent = cue_relative_path
        .parent()
        .ok_or_else(|| crate::error::Kind::MissingPathParent(cue_relative_path.to_path_buf()))?;
    let cue_file_name = cue_relative_path
        .file_name()
        .ok_or_else(|| crate::error::Kind::MissingPathExtension(cue_relative_path.to_path_buf()))?;
    let virtual_dir = parent.join(format!("{cue_file_name}{VIRTUAL_DIR_SUFFIX}"));

    // Keep filenames sortable and stable.
    let track_name =
        if track_number < 100 { format!("{track_number:02}") } else { track_number.to_string() };
    Ok(virtual_dir.join(format!("{track_name}.{}", format.extension())))
}

#[cfg(test)]
mod tests {
    use nghe_api::common::filesystem;

    use super::*;
    use crate::filesystem as fs;

    #[test]
    fn test_parse_cue_basic() {
        let cue = r#"
PERFORMER "Artist"
TITLE "Album"
REM DATE 2000-12-31
FILE "Album.flac" WAVE
  TRACK 01 AUDIO
    TITLE "One"
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    TITLE "Two"
    PERFORMER "Feat"
    INDEX 01 00:10:00
"#;
        let parsed = CueSheet::parse_str(cue).unwrap();
        assert_eq!(parsed.performer(), Some("Artist"));
        assert_eq!(parsed.title(), Some("Album"));
        assert_eq!(parsed.date(), Some("2000-12-31"));
        assert!(parsed.is_single_file());
        assert!(parsed.file_name_matches("Album.flac"));

        let span1 = parsed.resolve_track_span(1).unwrap();
        assert!((span1.start_seconds - 0.0).abs() < 1e-9);
        assert!(span1.duration_seconds.unwrap() > 9.0);

        let span2 = parsed.resolve_track_span(2).unwrap();
        assert!(span2.start_seconds > 9.0);
        assert!(span2.duration_seconds.is_none());
    }

    #[test]
    fn test_virtual_path_roundtrip() {
        let builder = fs::path::Builder(filesystem::Type::Local);
        let cue_rel = builder.from_str(&"a/b/Album.cue");
        let rel = build_virtual_track_relative_path(cue_rel, 1, file::audio::Format::Flac).unwrap();

        let prefix = builder.from_str(&"/music");
        let abs = prefix.join(rel);

        let parsed = VirtualCueTrack::parse_virtual_track_path(abs.to_path()).unwrap();
        assert_eq!(parsed.track_number, 1);
        assert_eq!(parsed.cue_path.as_str(), "/music/a/b/Album.cue");
    }

    #[test]
    fn test_virtual_path_root_relative() {
        let builder = fs::path::Builder(filesystem::Type::Local);
        let cue_rel = builder.from_str(&"Album.cue");
        let rel = build_virtual_track_relative_path(cue_rel, 1, file::audio::Format::Flac).unwrap();
        assert!(rel.as_str().ends_with("/01.flac"));
        assert!(rel.as_str().contains(VIRTUAL_DIR_SUFFIX));
    }
}
