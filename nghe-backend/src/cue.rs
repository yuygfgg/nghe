use std::borrow::Cow;

use encoding_rs as enc;
use nghe_api::common::format::Trait as _;
use typed_path::{Utf8TypedPath, Utf8TypedPathBuf};

use crate::{Error, file};

/// Marker directory suffix for "virtual files" derived from a `.cue` sheet.
///
/// Example virtual track path:
/// `Album.cue.__nghe_cue__/01.flac`
pub const VIRTUAL_DIR_SUFFIX: &str = ".__nghe_cue__";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CueTime {
    pub minutes: u32,
    pub seconds: u32,
    pub frames: u32,
}

impl CueTime {
    /// CUE time uses 75 frames per second.
    pub fn as_seconds_f64(self) -> f64 {
        (self.minutes as f64) * 60.0 + (self.seconds as f64) + (self.frames as f64) / 75.0
    }

    fn parse(input: &str) -> Option<Self> {
        let mut it = input.split(':');
        let minutes: u32 = it.next()?.trim().parse().ok()?;
        let seconds: u32 = it.next()?.trim().parse().ok()?;
        let frames: u32 = it.next()?.trim().parse().ok()?;
        Some(Self { minutes, seconds, frames })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CueTrack {
    pub number: u16,
    pub title: Option<String>,
    pub artist: Option<String>,
    pub isrc: Option<String>,
    pub index01: Option<CueTime>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CueFile {
    pub name: String,
    pub kind: Option<String>,
    pub tracks: Vec<CueTrack>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CueSheet {
    pub title: Option<String>,
    pub performer: Option<String>,
    pub catalog: Option<String>,
    pub date: Option<String>,
    pub files: Vec<CueFile>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CueTrackSpan {
    pub start_seconds: f64,
    pub duration_seconds: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualCueTrack {
    /// Absolute path to the `.cue` file on the configured filesystem.
    pub cue_path: Utf8TypedPathBuf,
    /// CUE `TRACK NN` number.
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

fn first_token_and_rest(line: &str) -> (&str, &str) {
    let line = line.trim();
    if line.is_empty() {
        return ("", "");
    }

    let split_at =
        line.char_indices().find(|(_, c)| c.is_whitespace()).map(|(i, _)| i).unwrap_or(line.len());
    let (head, tail) = line.split_at(split_at);
    (head, tail.trim())
}

fn parse_quoted_or_rest(input: &str) -> (String, &str) {
    let input = input.trim();
    if let Some(rest) = input.strip_prefix('"')
        && let Some(end) = rest.find('"')
    {
        let (value, rest) = rest.split_at(end);
        (value.to_owned(), rest[1..].trim())
    } else {
        (input.to_owned(), "")
    }
}

fn parse_file_directive(rest: &str) -> (String, Option<String>) {
    let (name, rest) = parse_quoted_or_rest(rest);
    let kind = first_token_and_rest(rest).0;
    let kind = if kind.is_empty() { None } else { Some(kind.to_owned()) };
    (name, kind)
}

fn parse_track_directive(rest: &str) -> Option<u16> {
    let (num, _rest) = first_token_and_rest(rest);
    num.parse().ok()
}

fn parse_index_directive(rest: &str) -> Option<(u8, CueTime)> {
    let (id_str, rest) = first_token_and_rest(rest);
    let (time_str, _rest) = first_token_and_rest(rest);
    let id: u8 = id_str.parse().ok()?;
    let time = CueTime::parse(time_str)?;
    Some((id, time))
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
        let mut sheet =
            Self { title: None, performer: None, catalog: None, date: None, files: vec![] };

        let mut current_file: Option<usize> = None;
        let mut current_track: Option<usize> = None;

        for raw in input.lines() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }

            let (cmd, rest) = first_token_and_rest(line);
            if cmd.is_empty() {
                continue;
            }

            if cmd.eq_ignore_ascii_case("REM") {
                let (rem_key, rem_rest) = first_token_and_rest(rest);
                if rem_key.eq_ignore_ascii_case("DATE") && sheet.date.is_none() {
                    let (value, _rest) = parse_quoted_or_rest(rem_rest);
                    if !value.is_empty() {
                        sheet.date = Some(value);
                    }
                }
                continue;
            }

            if cmd.eq_ignore_ascii_case("FILE") {
                let (name, kind) = parse_file_directive(rest);
                sheet.files.push(CueFile { name, kind, tracks: vec![] });
                current_file = Some(sheet.files.len() - 1);
                current_track = None;
                continue;
            }

            if cmd.eq_ignore_ascii_case("TRACK") {
                let Some(file_id) = current_file else { continue };
                let Some(number) = parse_track_directive(rest) else { continue };
                sheet.files[file_id].tracks.push(CueTrack {
                    number,
                    title: None,
                    artist: None,
                    isrc: None,
                    index01: None,
                });
                current_track = Some(sheet.files[file_id].tracks.len() - 1);
                continue;
            }

            // The rest are key/value directives with track-level overrides.
            let target_track = current_file.zip(current_track).and_then(|(file_id, track_id)| {
                sheet.files.get_mut(file_id)?.tracks.get_mut(track_id)
            });

            if cmd.eq_ignore_ascii_case("TITLE") {
                let (value, _rest) = parse_quoted_or_rest(rest);
                if let Some(track) = target_track {
                    track.title = Some(value);
                } else {
                    sheet.title = Some(value);
                }
                continue;
            }

            if cmd.eq_ignore_ascii_case("PERFORMER") {
                let (value, _rest) = parse_quoted_or_rest(rest);
                if let Some(track) = target_track {
                    track.artist = Some(value);
                } else {
                    sheet.performer = Some(value);
                }
                continue;
            }

            if cmd.eq_ignore_ascii_case("CATALOG") {
                let (value, _rest) = parse_quoted_or_rest(rest);
                sheet.catalog = Some(value);
                continue;
            }

            if cmd.eq_ignore_ascii_case("ISRC") {
                let (value, _rest) = parse_quoted_or_rest(rest);
                if let Some(track) = target_track {
                    track.isrc = Some(value);
                }
                continue;
            }

            if cmd.eq_ignore_ascii_case("INDEX") {
                let Some(track) = target_track else { continue };
                let Some((id, time)) = parse_index_directive(rest) else { continue };
                if id == 1 {
                    track.index01 = Some(time);
                }
                continue;
            }
        }

        Ok(sheet)
    }

    pub fn is_single_file(&self) -> bool {
        self.files.len() == 1
    }

    pub fn file_name_matches(&self, audio_file_name: &str) -> bool {
        let Some(file) = self.files.first() else { return false };
        // CUE's FILE may include a relative path. Only compare the basename.
        let cue_file_basename = file.name.rsplit(['/', '\\']).next().unwrap_or(file.name.as_str());
        cue_file_basename.eq_ignore_ascii_case(audio_file_name)
    }

    pub fn resolve_track_span(&self, track_number: u16) -> Option<CueTrackSpan> {
        let file = self.files.first()?;

        let mut tracks = file
            .tracks
            .iter()
            .filter_map(|t| Some((t.number, t.index01?.as_seconds_f64())))
            .collect::<Vec<_>>();
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
        let file = self.files.first()?;
        let parent = cue_path.parent()?;
        // Be tolerant of Windows-style separators inside the CUE sheet.
        let name = file.name.replace('\\', "/");
        Some(parent.join(&name))
    }
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
        assert_eq!(parsed.performer.as_deref(), Some("Artist"));
        assert_eq!(parsed.title.as_deref(), Some("Album"));
        assert_eq!(parsed.date.as_deref(), Some("2000-12-31"));
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
