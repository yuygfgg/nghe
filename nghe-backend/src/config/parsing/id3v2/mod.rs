pub mod frame;

use educe::Educe;
use lofty::id3;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum DelimitersInput {
    One(String),
    Many(Vec<String>),
}

fn default_id3v23_delimiters() -> Vec<String> {
    // Common separators seen in ID3v2.3 tags "in the wild".
    vec![
        "/".into(),
        "／".into(),
        "&".into(),
        "＆".into(),
        " x ".into(),
        ";".into(),
        "；".into(),
        ",".into(),
        "，".into(),
        "×".into(),
        "　".into(),
        "、".into(),
    ]
}

fn deserialize_delimiters<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let input = DelimitersInput::deserialize(deserializer)?;
    Ok(match input {
        DelimitersInput::One(s) => vec![s],
        DelimitersInput::Many(v) => v,
    })
}

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Common {
    pub name: frame::Id,
    pub date: Option<frame::Id>,
    pub release_date: Option<frame::Id>,
    pub original_release_date: Option<frame::Id>,
    pub mbz_id: frame::Id,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artist {
    pub name: frame::Id,
    pub mbz_id: frame::Id,
}

#[derive(Debug, Clone, Serialize, Deserialize, Educe)]
#[educe(Default)]
pub struct Artists {
    #[educe(Default(expression = Artist::default_song()))]
    pub song: Artist,
    #[educe(Default(expression = Artist::default_album()))]
    pub album: Artist,
}

#[derive(Debug, Clone, Serialize, Deserialize, Educe)]
#[educe(Default)]
pub struct TrackDisc {
    #[educe(Default(expression = "TEXT:TRCK".parse().unwrap()))]
    pub track_position: frame::Id,
    #[educe(Default(expression = "TEXT:TPOS".parse().unwrap()))]
    pub disc_position: frame::Id,
}

#[derive(Debug, Clone, Serialize, Deserialize, Educe)]
#[educe(Default)]
pub struct Id3v2 {
    #[educe(Default(expression = Common::default_song()))]
    pub song: Common,
    #[educe(Default(expression = Common::default_album()))]
    pub album: Common,
    pub artists: Artists,
    pub track_disc: TrackDisc,
    #[educe(Default(expression = "TEXT:TLAN".parse().unwrap()))]
    pub languages: frame::Id,
    #[educe(Default(expression = "TEXT:TCON".parse().unwrap()))]
    pub genres: frame::Id,
    #[educe(Default(expression = "TXXX:compilation".parse().unwrap()))]
    pub compilation: frame::Id,
    #[educe(Default(expression = default_id3v23_delimiters()))]
    #[serde(deserialize_with = "deserialize_delimiters")]
    pub separator: Vec<String>,
}

impl Common {
    fn default_song() -> Self {
        Self {
            name: "TEXT:TIT2".parse().unwrap(),
            date: None,
            release_date: None,
            original_release_date: None,
            mbz_id: "TXXX:MusicBrainz Release Track Id".parse().unwrap(),
        }
    }

    fn default_album() -> Self {
        Self {
            name: "TEXT:TALB".parse().unwrap(),
            date: Some("TIME:TDRC".parse().unwrap()),
            release_date: Some("TIME:TDRL".parse().unwrap()),
            original_release_date: Some("TIME:TDOR".parse().unwrap()),
            mbz_id: "TXXX:MusicBrainz Album Id".parse().unwrap(),
        }
    }
}

impl Artist {
    fn default_song() -> Self {
        Self {
            name: "TEXT:TPE1".parse().unwrap(),
            mbz_id: "TXXX:MusicBrainz Artist Id".parse().unwrap(),
        }
    }

    fn default_album() -> Self {
        Self {
            name: "TEXT:TPE2".parse().unwrap(),
            mbz_id: "TXXX:MusicBrainz Album Artist Id".parse().unwrap(),
        }
    }
}

impl Id3v2 {
    pub const SYNC_LYRIC_FRAME_ID: id3::v2::FrameId<'static> =
        id3::v2::FrameId::Valid(std::borrow::Cow::Borrowed("SYLT"));
}

#[cfg(test)]
#[coverage(off)]
mod test {
    use super::*;

    impl Id3v2 {
        pub fn test() -> Self {
            Self {
                song: Common {
                    date: Some("TXXX:TRCS".parse().unwrap()),
                    release_date: Some("TXXX:TSRL".parse().unwrap()),
                    original_release_date: Some("TXXX:TSOR".parse().unwrap()),
                    ..Common::default_song()
                },
                ..Self::default()
            }
        }
    }
}
