use std::borrow::Cow;
use std::num::NonZeroU32;
use std::sync::Arc;

use diesel::{
    ExpressionMethods, NullableExpressionMethods, OptionalExtension, QueryDsl, SelectableHelper,
};
use diesel_async::RunQueryDsl;
use lofty::config::ParseOptions;
use loole::Receiver;
use nghe_api::scan;
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use tracing::{Instrument, instrument};
use typed_path::Utf8TypedPath;
use uuid::Uuid;
use xxhash_rust::xxh3::xxh3_64;

use crate::cue;
use crate::database::Database;
use crate::file::{self, File, audio, image, lyric};
use crate::filesystem::{self, Entry, Filesystem, Trait, entry};
use crate::integration::Informant;
use crate::orm::{albums, music_folders, songs};
use crate::{Error, config, error};

#[derive(Debug, Clone)]
pub struct Config {
    pub lofty: ParseOptions,
    pub scan: config::filesystem::Scan,
    pub parsing: config::Parsing,
    pub index: config::Index,
    pub cover_art: config::CoverArt,
}

#[derive(Clone)]
pub struct Scanner<'db, 'fs, 'mf> {
    pub database: Cow<'db, Database>,
    pub filesystem: filesystem::Impl<'fs>,
    pub config: Config,
    pub informant: Informant,
    pub music_folder: music_folders::MusicFolder<'mf>,
    pub full: scan::start::Full,
}

impl<'db, 'fs, 'mf> Scanner<'db, 'fs, 'mf> {
    #[coverage(off)]
    pub async fn new(
        database: &'db Database,
        filesystem: &'fs Filesystem,
        config: Config,
        informant: Informant,
        request: scan::start::Request,
    ) -> Result<Self, Error> {
        Self::new_orm(
            database,
            filesystem,
            config,
            informant,
            music_folders::MusicFolder::query(database, request.music_folder_id).await?,
            request.full,
        )
    }

    pub fn new_orm(
        database: &'db Database,
        filesystem: &'fs Filesystem,
        config: Config,
        informant: Informant,
        music_folder: music_folders::MusicFolder<'mf>,
        full: scan::start::Full,
    ) -> Result<Self, Error> {
        let filesystem = filesystem.to_impl(music_folder.data.ty.into())?;
        Ok(Self {
            database: Cow::Borrowed(database),
            filesystem,
            config,
            informant,
            music_folder,
            full,
        })
    }

    pub fn into_owned(self) -> Scanner<'static, 'static, 'static> {
        Scanner {
            database: Cow::Owned(self.database.into_owned()),
            filesystem: self.filesystem.into_owned(),
            music_folder: self.music_folder.into_owned(),
            ..self
        }
    }

    fn path(&self) -> Utf8TypedPath<'_> {
        self.filesystem.path().from_str(&self.music_folder.data.path)
    }

    fn relative_path<'entry>(&self, entry: &'entry Entry) -> Result<Utf8TypedPath<'entry>, Error> {
        entry.path.strip_prefix(&self.music_folder.data.path).map_err(Error::from)
    }

    fn init(&self) -> (JoinHandle<Result<(), Error>>, Arc<Semaphore>, Receiver<Entry>) {
        let config = self.config.scan;
        let (tx, rx) = crate::sync::channel(config.channel_size);
        let filesystem = self.filesystem.clone().into_owned();
        let sender = entry::Sender { tx, minimum_size: config.minimum_size };
        let prefix = self.path().to_path_buf();
        (
            tokio::spawn(async move { filesystem.scan_folder(sender, prefix.to_path()).await }),
            Arc::new(Semaphore::const_new(config.pool_size)),
            rx,
        )
    }

    #[cfg_attr(not(coverage_nightly), instrument(skip_all, ret(level = "trace")))]
    async fn set_scanned_at(
        &self,
        entry: &Entry,
        started_at: time::OffsetDateTime,
    ) -> Result<Option<songs::IdTime>, Error> {
        let song_time = songs::table
            .inner_join(albums::table)
            .filter(albums::music_folder_id.eq(self.music_folder.id))
            .filter(
                songs::relative_path
                    .eq(entry.relative_path(&self.music_folder.data.path)?.as_str()),
            )
            .select(songs::IdTime::as_select())
            .get_result(&mut self.database.get().await?)
            .await
            .optional()?;

        // Only update `scanned_at` if it is sooner than `started_at`.
        // Else, it means that the current path is being scanned by another process or it is already
        // scanned.
        if let Some(song_time) = song_time
            && song_time.time.scanned_at < started_at
        {
            diesel::update(songs::table)
                .filter(songs::id.eq(song_time.id))
                .set(songs::scanned_at.eq(crate::time::now().await))
                .execute(&mut self.database.get().await?)
                .await?;
        }
        Ok(song_time)
    }

    #[cfg_attr(not(coverage_nightly), instrument(skip_all, ret(level = "trace")))]
    async fn set_scanned_at_relative_path(
        &self,
        relative_path: &str,
        started_at: time::OffsetDateTime,
    ) -> Result<Option<songs::IdTime>, Error> {
        let song_time = songs::table
            .inner_join(albums::table)
            .filter(albums::music_folder_id.eq(self.music_folder.id))
            .filter(songs::relative_path.eq(relative_path))
            .select(songs::IdTime::as_select())
            .get_result(&mut self.database.get().await?)
            .await
            .optional()?;

        if let Some(song_time) = song_time
            && song_time.time.scanned_at < started_at
        {
            diesel::update(songs::table)
                .filter(songs::id.eq(song_time.id))
                .set(songs::scanned_at.eq(crate::time::now().await))
                .execute(&mut self.database.get().await?)
                .await?;
        }

        Ok(song_time)
    }

    #[cfg_attr(not(coverage_nightly), instrument(skip_all, ret(level = "trace")))]
    async fn query_hash_size(
        &self,
        property: &file::Property<audio::Format>,
    ) -> Result<Option<songs::IdPath>, Error> {
        songs::table
            .inner_join(albums::table)
            .filter(albums::music_folder_id.eq(self.music_folder.id))
            .filter(songs::file_hash.eq(property.hash.cast_signed()))
            .filter(songs::file_size.eq(property.size.get().cast_signed()))
            .select(songs::IdPath::as_select())
            .get_result(&mut self.database.get().await?)
            .await
            .optional()
            .map_err(Error::from)
    }

    async fn update_dir_image(
        &self,
        song_id: Uuid,
        dir_image_id: Option<Uuid>,
    ) -> Result<(), Error> {
        diesel::update(albums::table)
            .filter(albums::id.nullable().eq(
                songs::table.filter(songs::id.eq(song_id)).select(songs::album_id).single_value(),
            ))
            .set(albums::cover_art_id.eq(dir_image_id))
            .execute(&mut self.database.get().await?)
            .await?;
        Ok(())
    }

    async fn update_external_lyric(
        &self,
        started_at: impl Into<Option<time::OffsetDateTime>>,
        song_id: Uuid,
        song_path: Utf8TypedPath<'_>,
    ) -> Result<(), Error> {
        lyric::Lyric::scan(
            &self.database,
            &self.filesystem,
            self.full.external_lyric,
            song_id,
            song_path,
        )
        .await?;
        if let Some(started_at) = started_at.into() {
            lyric::Lyric::cleanup_one_external(&self.database, started_at, song_id).await?;
        }
        Ok(())
    }

    async fn update_external(
        &self,
        started_at: time::OffsetDateTime,
        song_id: Uuid,
        song_path: Utf8TypedPath<'_>,
        dir_image_id: Option<Uuid>,
    ) -> Result<(), Error> {
        // We also need to set album cover_art_id and external lyrics since it might be
        // added or removed after the previous scan.
        self.update_dir_image(song_id, dir_image_id).await?;
        self.update_external_lyric(started_at, song_id, song_path).await?;
        Ok(())
    }

    async fn upsert_cue_flac(
        &self,
        started_at: time::OffsetDateTime,
        dir_image_id: Option<Uuid>,
        flac_relative_path: Utf8TypedPath<'_>,
        embedded: bool,
        cue_hash: u64,
        cue_sheet: &cue::CueSheet,
        base_information: &audio::Information<'_>,
    ) -> Result<Uuid, Error> {
        let database = &self.database;

        let Some(audio_file_name) = flac_relative_path.file_name() else {
            return error::Kind::MissingPathExtension(flac_relative_path.to_path_buf()).into();
        };

        if !cue_sheet.is_single_file()
            || (!embedded && !cue_sheet.file_name_matches(audio_file_name))
        {
            return error::Kind::NotFound.into();
        }

        let cue_source_relative_path = if embedded {
            flac_relative_path.to_path_buf()
        } else {
            flac_relative_path.to_path_buf().with_extension("cue")
        };

        let base_album = &base_information.metadata.album;
        let album_name = cue_sheet.title().unwrap_or(base_album.name.as_ref()).to_owned();
        let album_date =
            cue_sheet.date().and_then(|s| s.parse::<audio::Date>().ok()).unwrap_or(base_album.date);

        let album_artist_fallback = base_information
            .metadata
            .artists
            .album()
            .first()
            .map(|a| a.name.as_ref())
            .unwrap_or("Unknown Artist");
        let album_artist_name = cue_sheet.performer().unwrap_or(album_artist_fallback).to_owned();

        // Upsert album once; track songs will reference it.
        let album_id = audio::Album {
            name: Cow::Owned(album_name),
            date: album_date,
            release_date: base_album.release_date,
            original_release_date: base_album.original_release_date,
            mbz_id: base_album.mbz_id,
        }
        .upsert(
            database,
            albums::Foreign { music_folder_id: self.music_folder.id, cover_art_id: dir_image_id },
        )
        .await?;

        // Extract embedded cover art once, then reuse its id for every virtual track song.
        let embedded_cover_art_id =
            base_information.upsert_cover_art(database, self.config.cover_art.dir.as_ref()).await?;

        let base_languages = base_information.metadata.song.languages.clone();
        let base_disc = base_information.metadata.song.track_disc.disc;
        let base_genres: Vec<String> = base_information
            .metadata
            .genres
            .value
            .iter()
            .map(|g| g.value.clone().into_owned())
            .collect();

        let total_duration_s = f32::from(base_information.property.duration) as f64;
        let base_file_hash = base_information.file.hash;
        let base_file_size = base_information.file.size.get();
        let base_property = base_information.property;

        let file_id = cue_rw::FileID::from(0usize);
        let cue_track_ids = cue_sheet
            .cue()
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, (fid, _))| *fid == file_id)
            .map(|(track_id, _)| track_id)
            .collect::<Vec<_>>();
        let track_total: Option<u16> = cue_track_ids.len().try_into().ok();

        let mut first_song_id: Option<Uuid> = None;

        for (i, &track_id) in cue_track_ids.iter().enumerate() {
            let Ok(track_number) = u16::try_from(i + 1) else { break };

            let Some(span) = cue_sheet.resolve_track_span(track_number) else { continue };
            let start_seconds = span.start_seconds;
            if start_seconds >= total_duration_s {
                continue;
            }

            let duration_seconds =
                span.duration_seconds.unwrap_or(total_duration_s - start_seconds).max(0.0);

            if duration_seconds == 0.0 {
                continue;
            }

            let track_relative_path = cue::build_virtual_track_relative_path(
                cue_source_relative_path.to_path(),
                track_number,
                audio::Format::Flac,
            )?;
            let track_relative_path = track_relative_path.to_string();

            // Keep each virtual track unique within a music folder to avoid `file_hash+file_size`
            // conflicts (and to keep cache keys stable).
            let hash = {
                // Include both the audio file hash and cue hash so cache keys are invalidated when
                // either file changes.
                let mut buf = Vec::with_capacity(16 + track_relative_path.len());
                buf.extend_from_slice(&base_file_hash.to_le_bytes());
                buf.extend_from_slice(&cue_hash.to_le_bytes());
                buf.extend_from_slice(track_relative_path.as_bytes());
                xxh3_64(&buf)
            };

            let estimated_size = if total_duration_s > 0.0 {
                let ratio = (duration_seconds / total_duration_s).clamp(0.0, 1.0);
                ((base_file_size as f64) * ratio).round().clamp(1.0, f64::from(u32::MAX)) as u32
            } else {
                base_file_size
            };

            let track = &cue_sheet.cue().tracks[track_id].1;

            let title = match track.title.trim() {
                "" => format!("Track {track_number:02}"),
                t => t.to_owned(),
            };
            let song_artist_name = track
                .performer
                .as_deref()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or(&album_artist_name)
                .to_owned();

            let artists = audio::Artists::new(
                [audio::Artist { name: Cow::Owned(song_artist_name), mbz_id: None }],
                [audio::Artist { name: Cow::Owned(album_artist_name.clone()), mbz_id: None }],
                false,
            )?;

            let information = audio::Information {
                metadata: audio::Metadata {
                    song: audio::Song {
                        main: audio::NameDateMbz {
                            name: Cow::Owned(title),
                            date: audio::Date::default(),
                            release_date: audio::Date::default(),
                            original_release_date: audio::Date::default(),
                            mbz_id: None,
                        },
                        track_disc: audio::TrackDisc {
                            track: audio::position::Position {
                                number: Some(track_number),
                                total: track_total,
                            },
                            disc: base_disc,
                        },
                        languages: base_languages.clone(),
                    },
                    // Album is already upserted; keep the fields consistent anyway.
                    album: audio::Album {
                        name: Cow::Borrowed(""),
                        date: audio::Date::default(),
                        release_date: audio::Date::default(),
                        original_release_date: audio::Date::default(),
                        mbz_id: None,
                    },
                    artists,
                    genres: base_genres.iter().cloned().collect(),
                    lyrics: vec![],
                    image: None,
                },
                property: audio::Property {
                    duration: time::Duration::seconds_f64(duration_seconds).into(),
                    ..base_property
                },
                file: file::Property {
                    hash,
                    size: NonZeroU32::new(estimated_size)
                        .ok_or_else(|| error::Kind::EmptyFileEncountered)?,
                    format: audio::Format::Flac,
                },
            };

            let song_time =
                self.set_scanned_at_relative_path(&track_relative_path, started_at).await?;
            let old_song_id = song_time.map(|t| t.id);

            let song_id = information
                .upsert_song(
                    database,
                    songs::Foreign { album_id, cover_art_id: embedded_cover_art_id },
                    track_relative_path,
                    old_song_id,
                )
                .await?;
            information
                .upsert_artists(database, &self.config.index.ignore_prefixes, song_id)
                .await?;
            information.upsert_genres(database, song_id).await?;
            information.upsert_lyrics(database, song_id).await?;
            audio::Information::cleanup_one(database, started_at, song_id).await?;

            first_song_id.get_or_insert(song_id);
        }

        let Some(first_song_id) = first_song_id else { return error::Kind::NotFound.into() };

        // Keep the album folder cover art in sync.
        self.update_dir_image(first_song_id, dir_image_id).await?;

        // NOTE: external lyric scanning is intentionally skipped for CUE-derived virtual tracks.
        // If needed later, we can map `.lrc` discovery to CUE `TRACK` titles or virtual filenames.

        Ok(first_song_id)
    }

    #[cfg_attr(
        not(coverage_nightly),
        instrument(
            skip_all,
            fields(path = %entry.path, last_modified = ?entry.last_modified),
            ret(level = "debug"),
            err(Debug)
        )
    )]
    async fn one(&self, entry: &Entry, started_at: time::OffsetDateTime) -> Result<Uuid, Error> {
        let database = &self.database;

        enum CueSource {
            Sidecar { cue_hash: u64, cue_sheet: cue::CueSheet },
            Embedded { cue_hash: u64, cue_sheet: cue::CueSheet },
        }

        let cue: Option<CueSource> = if entry.format == audio::Format::Flac {
            let mut cue: Option<CueSource> = None;

            // 1) Sidecar `.cue` file next to the FLAC (preferred).
            let cue_path = entry.path.clone().with_extension("cue");
            if self.filesystem.exists(cue_path.to_path()).await? {
                cue = match self.filesystem.read(cue_path.to_path()).await {
                    Ok(bytes) => {
                        let cue_hash = xxh3_64(&bytes);
                        match cue::CueSheet::parse(&bytes) {
                            Ok(sheet) => {
                                let audio_file_name = entry.path.file_name().unwrap_or_default();
                                let file_id = cue_rw::FileID::from(0usize);
                                let has_track_index = sheet.cue().tracks.iter().any(|(fid, t)| {
                                    *fid == file_id && t.indices.iter().any(|(id, _)| *id == 1)
                                });
                                if has_track_index
                                    && sheet.is_single_file()
                                    && sheet.file_name_matches(audio_file_name)
                                {
                                    Some(CueSource::Sidecar { cue_hash, cue_sheet: sheet })
                                } else {
                                    None
                                }
                            }
                            Err(error) => {
                                tracing::debug!(?error, path = %cue_path, "invalid cue file");
                                None
                            }
                        }
                    }
                    Err(error) => {
                        tracing::debug!(?error, path = %cue_path, "failed to read cue file");
                        None
                    }
                };
            }

            // 2) Embedded cue sheet in FLAC Vorbis comments.
            if cue.is_none() {
                // For local filesystem we can do this with a small metadata-only read. For S3 we
                // currently skip this to avoid downloading the entire audio object during scans.
                cue = match &self.filesystem {
                    filesystem::Impl::Local(_) => {
                        let platform_path = crate::filesystem::local::Filesystem::to_platform(
                            entry.path.to_path(),
                        )?;
                        let file = std::fs::File::open(platform_path.as_str())?;
                        let mut reader = std::io::BufReader::new(file);
                        match crate::flac::extract_embedded_cuesheet_from_reader(&mut reader) {
                            Ok(Some(cue_str)) => {
                                let cue_hash = xxh3_64(cue_str.as_bytes());
                                match cue::CueSheet::parse_str(&cue_str) {
                                    Ok(sheet) => {
                                        let file_id = cue_rw::FileID::from(0usize);
                                        let has_track_index =
                                            sheet.cue().tracks.iter().any(|(fid, t)| {
                                                *fid == file_id
                                                    && t.indices.iter().any(|(id, _)| *id == 1)
                                            });
                                        if has_track_index && sheet.is_single_file() {
                                            Some(CueSource::Embedded { cue_hash, cue_sheet: sheet })
                                        } else {
                                            None
                                        }
                                    }
                                    Err(error) => {
                                        tracing::debug!(
                                            ?error,
                                            path = %entry.path,
                                            "invalid embedded cuesheet"
                                        );
                                        None
                                    }
                                }
                            }
                            Ok(None) => None,
                            Err(error) => {
                                tracing::debug!(
                                    ?error,
                                    path = %entry.path,
                                    "failed to parse embedded cuesheet"
                                );
                                None
                            }
                        }
                    }
                    filesystem::Impl::S3(_) => None,
                };
            }

            cue
        } else {
            None
        };

        // Query the database to see if we have any song within this music folder that has the same
        // relative path. If yes, update its scanned at to the current time.
        //
        // Doing this helps us avoiding working on the same file at the same time (which is mostly
        // the case for multiple scans).
        let song_id = if cue.is_none() {
            if let Some(song_time) = self.set_scanned_at(entry, started_at).await? {
                if started_at < song_time.time.scanned_at
                    || (!self.full.file
                        && entry
                            .last_modified
                            .is_some_and(|last_modified| last_modified < song_time.time.updated_at))
                {
                    // If `started_at` is sooner than its database's `scanned_at` or its filesystem's
                    // last modified is sooner than its database's `updated_at`, it means that we have
                    // the latest data or this file is being scanned by another process, we can return
                    // the function.
                    //
                    // Since the old `scanned_at` is returned, there is a case when the file is scanned
                    // in the previous scan but not in the current scan, thus `scanned_at` is sooner
                    // than `started_at`. We want to skip this file as well (unless in full mode) hence
                    // we have to check for its `last_modified` along with `scanned_at`.
                    return Ok(song_time.id);
                }
                Some(song_time.id)
            } else {
                None
            }
        } else {
            None
        };

        let absolute_path = entry.path.to_path();
        let file = File::new(entry.format, self.filesystem.read(absolute_path).await?)?;
        let dir_image_id = image::Image::scan(
            &self.database,
            &self.filesystem,
            &self.config.cover_art,
            self.full.dir_image,
            entry
                .path
                .parent()
                .ok_or_else(|| error::Kind::MissingPathParent(entry.path.clone()))?,
        )
        .await?;
        tracing::trace!(?dir_image_id);

        let relative_path = self.relative_path(entry)?;
        let relative_path_str = relative_path.as_str();
        let song_id = if cue.is_none()
            && let Some(song_path) = self.query_hash_size(&file.property).await?
        {
            if started_at < song_path.time.updated_at {
                // We will check if `song_path.updated_at` is later than `started_at`, since this
                // file has the same hash and size with that entry in the database, we can terminate
                // this function regardless of full mode as another file with the same data is
                // processed in the current scan.
                //
                // `song_id` can be None if there are more than two duplicated files in the same
                // music folder.

                self.update_external(started_at, song_path.id, absolute_path, dir_image_id).await?;
                tracing::debug!("already scanned");
                return Ok(song_path.id);
            } else if let Some(song_id) = song_id {
                // `DatabaseCorruption` can happen if all the below conditions hold:
                //  - There is a file on the filesystem that has the same hash and size as those of
                //    one entry in the database (`hash_size` constraint) but not the same relative
                //    (P_fs and P_db) path. Could be the result of a duplication or renaming
                //    operation.
                //  - The file with P_fs is scanned first and update the relative path in the
                //    database to P_fs (thread 1).
                //  - The file with P_db is scanned before the relative path is updated to P_fs
                //    therefore it still returns an entry (thread 2).
                //  - However, `query_hash_size` operation of thread 2 takes place after the update
                //    of relative path by thread 1, thus causing the `DatabaseCorruption` error as
                //    `relative_path != song_path.relative_path`.
                //
                // We prevent this error by checking the `song_path.updated_at` as above so we can
                // skip checking `song_path.relative_path` after being updated.
                if song_id == song_path.id && relative_path_str == song_path.relative_path {
                    if self.full.file {
                        // If file full scan is enabled, we return the song id so it can be
                        // re-scanned later.
                        Some(song_path.id)
                    } else {
                        // Everything is the same but the song's last modified for some reason,
                        // update its updated at and return the function.
                        diesel::update(songs::table)
                            .filter(songs::id.eq(song_id))
                            .set(songs::updated_at.eq(crate::time::now().await))
                            .execute(&mut database.get().await?)
                            .await?;

                        self.update_external(started_at, song_path.id, absolute_path, dir_image_id)
                            .await?;
                        tracing::debug!("stale last_modified");
                        return Ok(song_path.id);
                    }
                } else {
                    // Since `song_id` is queried only by music folder and relative path and there
                    // is a constraint `songs_album_id_file_hash_file_size_key`,
                    // other cases should be unreachable.
                    return error::Kind::DatabaseCorruptionDetected.into();
                }
            } else if self.full.file {
                // If file full scan is enabled, we return the song id so it can be
                // re-scanned later. Here `song_id` is None.
                Some(song_path.id)
            } else {
                // We have one entry that is in the same music folder, same hash and size but
                // different relative path (since song_id is None). We only need to update the
                // relative path, set scanned at and return the function.
                diesel::update(songs::table)
                    .filter(songs::id.eq(song_path.id))
                    .set((
                        songs::relative_path.eq(relative_path_str),
                        songs::scanned_at.eq(crate::time::now().await),
                    ))
                    .execute(&mut database.get().await?)
                    .await?;

                self.update_external(started_at, song_path.id, absolute_path, dir_image_id).await?;
                tracing::warn!(
                    old = %song_path.relative_path, new = %relative_path_str, "renamed duplication"
                );
                return Ok(song_path.id);
            }
        } else {
            song_id
        };

        let audio = file.audio(self.config.lofty)?;
        let information = audio.extract(&self.config.parsing)?;
        tracing::trace!(?information);

        if let Some(cue) = cue {
            let (embedded, cue_hash, cue_sheet) = match cue {
                CueSource::Sidecar { cue_hash, cue_sheet } => (false, cue_hash, cue_sheet),
                CueSource::Embedded { cue_hash, cue_sheet } => (true, cue_hash, cue_sheet),
            };
            return self
                .upsert_cue_flac(
                    started_at,
                    dir_image_id,
                    relative_path,
                    embedded,
                    cue_hash,
                    &cue_sheet,
                    &information,
                )
                .await;
        }

        let song_id = information
            .upsert(
                database,
                &self.config,
                albums::Foreign {
                    music_folder_id: self.music_folder.id,
                    cover_art_id: dir_image_id,
                },
                relative_path_str,
                song_id,
            )
            .await?;
        self.update_external_lyric(None, song_id, absolute_path).await?;
        audio::Information::cleanup_one(database, started_at, song_id).await?;

        Ok(song_id)
    }

    #[cfg_attr(not(coverage_nightly), instrument(skip_all, fields(started_at), err(Debug)))]
    pub async fn run(&self) -> Result<(), Error> {
        let span = tracing::Span::current();
        let started_at = crate::time::now().await;
        span.record("started_at", tracing::field::display(&started_at));
        tracing::info!(music_folder = ?self.music_folder);

        let (scan_handle, permit, rx) = self.init();
        let mut join_set = tokio::task::JoinSet::new();

        while let Ok(entry) = rx.recv_async().await {
            let permit = permit.clone().acquire_owned().await?;
            let scan = self.clone().into_owned();
            join_set.spawn(
                async move {
                    let _guard = permit;
                    scan.one(&entry, started_at).await
                }
                .instrument(span.clone()),
            );
        }

        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(Ok(_song_id)) => {}
                Ok(Err(_error)) => {
                    // Keep scanning other files even if this one fails to parse.
                }
                Err(error) => {
                    // A join error means the task panicked or was cancelled.
                    tracing::error!(?error, "scan worker task failed");
                }
            }
        }
        scan_handle.await??;

        audio::Information::cleanup(&self.database, started_at).await?;

        self.database.upsert_config(&self.config.index).await?;
        self.informant
            .search_and_upsert_artists(
                &self.database,
                &self.config.cover_art,
                self.full.information,
            )
            .await?;

        let latency: std::time::Duration =
            (time::OffsetDateTime::now_utc() - started_at).try_into()?;
        tracing::info!(took = ?latency);
        Ok(())
    }
}

#[cfg(test)]
#[coverage(off)]
mod tests {
    use diesel::{ExpressionMethods, QueryDsl};
    use diesel_async::RunQueryDsl;
    use fake::{Fake, Faker};
    use nghe_api::common::filesystem as api_filesystem;
    use nghe_api::scan;
    use rstest::rstest;

    use crate::file::audio;
    use crate::orm::{albums, songs};
    use crate::test::{Mock, mock};

    #[rstest]
    #[tokio::test]
    async fn test_simple_scan(#[future(awt)] mock: Mock, #[values(0, 10, 50)] n_song: usize) {
        let mut music_folder = mock.music_folder(0).await;
        music_folder.add_audio_filesystem::<&str>().n_song(n_song).call().await;

        let database_audio = music_folder.query_filesystem().await;
        assert_eq!(database_audio, music_folder.filesystem);
    }

    #[rstest]
    #[tokio::test]
    async fn test_full_scan(#[future(awt)] mock: Mock, #[values(true, false)] full: bool) {
        let mut music_folder = mock.music_folder(0).await;
        music_folder.add_audio_filesystem::<&str>().call().await;

        let song_id = music_folder.song_id_filesystem(0).await;
        let filesystem_audio = music_folder.filesystem[0].clone();
        // Don't modify lyric because we won't rescan it even in full mode (it will be rescanned
        // only in full lyric mode).
        music_folder
            .add_audio()
            .album(filesystem_audio.information.metadata.album)
            .file_property(filesystem_audio.information.file)
            .external_lyric(None)
            .relative_path(filesystem_audio.relative_path)
            .song_id(song_id)
            .call()
            .await;
        music_folder
            .scan(scan::start::Full { file: full, ..Default::default() })
            .run()
            .await
            .unwrap();

        let database_audio = music_folder.query_filesystem().await;
        if full {
            assert_eq!(database_audio, music_folder.filesystem);
        } else {
            // Could not compare information that uses more than one table.
            assert_eq!(
                database_audio[0].information.metadata.song,
                music_folder.database[0].information.metadata.song
            );
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_multiple_scan(#[future(awt)] mock: Mock) {
        let mut music_folder = mock.music_folder(0).await;
        music_folder.add_audio_filesystem::<&str>().n_song(20).scan(false).call().await;

        let mut join_set = tokio::task::JoinSet::new();
        for _ in 0..5 {
            let scanner = music_folder.scan(scan::start::Full::default()).into_owned();
            join_set.spawn(async move { scanner.run().await.unwrap() });
        }
        join_set.join_all().await;

        let database_audio = music_folder.query_filesystem().await;
        assert_eq!(database_audio, music_folder.filesystem);
    }

    #[rstest]
    #[tokio::test]
    async fn test_scan_cue_flac_virtual_tracks(
        #[future(awt)]
        #[with(1, 0)]
        mock: Mock,
        #[values(api_filesystem::Type::Local, api_filesystem::Type::S3)] ty: api_filesystem::Type,
    ) {
        use crate::cue;
        use crate::test::filesystem::Trait as _;

        mock.add_music_folder().ty(ty).call().await;
        let mut music_folder = mock.music_folder(0).await;
        music_folder
            .add_audio_filesystem::<&str>()
            .path("a/Album.flac")
            .format(audio::Format::Flac)
            .scan(false)
            .recompute_dir_image(false)
            .call()
            .await;

        let cue_data = r#"
PERFORMER "Cue Artist"
TITLE "Cue Album"
REM DATE 2001
FILE "Album.flac" WAVE
  TRACK 01 AUDIO
    TITLE "One"
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    TITLE "Two"
    INDEX 01 00:10:00
"#;
        let cue_path = music_folder.absolutize("a/Album.cue");
        music_folder.to_impl().write(cue_path.to_path(), cue_data.as_bytes()).await;

        music_folder.scan(scan::start::Full::default()).run().await.unwrap();

        let cue_relative_path = music_folder.path_str(&"a/Album.cue");
        let expected_1 =
            cue::build_virtual_track_relative_path(cue_relative_path, 1, audio::Format::Flac)
                .unwrap()
                .to_string();
        let expected_2 =
            cue::build_virtual_track_relative_path(cue_relative_path, 2, audio::Format::Flac)
                .unwrap()
                .to_string();

        let mut conn = mock.get().await;
        let mut rows: Vec<(String, String)> = albums::table
            .inner_join(songs::table)
            .filter(albums::music_folder_id.eq(music_folder.id()))
            .select((songs::relative_path, songs::title))
            .order_by(songs::relative_path.asc())
            .get_results(&mut conn)
            .await
            .unwrap();

        rows.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(rows, vec![(expected_1, "One".to_owned()), (expected_2, "Two".to_owned()),]);
        assert!(
            rows.iter().all(|(path, _)| path != "a/Album.flac"),
            "physical flac must not be inserted as a song when paired with a CUE"
        );
    }

    mod filesystem {
        use super::*;

        #[rstest]
        #[tokio::test]
        async fn test_overwrite(
            #[future(awt)] mock: Mock,
            #[values(true, false)] same_album: bool,
            #[values(true, false)] same_external_lyric: bool,
        ) {
            // Test a constraint with `album_id` and `relative_path`.
            let mut music_folder = mock.music_folder(0).await;
            let album: audio::Album = Faker.fake();

            music_folder.add_audio_filesystem().album(album.clone()).path("test").call().await;
            let database_audio = music_folder.query_filesystem().await;
            assert_eq!(database_audio, music_folder.filesystem);

            music_folder
                .add_audio_filesystem()
                .maybe_album(if same_album { Some(album) } else { None })
                .maybe_external_lyric(if same_external_lyric {
                    Some(database_audio[0].external_lyric.clone())
                } else {
                    None
                })
                .path("test")
                .format(database_audio[0].information.file.format)
                .call()
                .await;

            let mut database_audio = music_folder.query_filesystem().await;
            assert_eq!(database_audio.len(), 1);
            assert_eq!(music_folder.filesystem.len(), 1);

            let database_audio = database_audio.shift_remove_index(0).unwrap().1;
            let filesystem_audio = music_folder.filesystem.shift_remove_index(0).unwrap().1;

            let (database_audio, filesystem_audio) = if same_external_lyric {
                (database_audio, filesystem_audio)
            } else {
                (
                    database_audio.with_external_lyric(None),
                    filesystem_audio.with_external_lyric(None),
                )
            };

            let (database_audio, filesystem_audio) = if same_album {
                (database_audio.with_dir_image(None), filesystem_audio.with_dir_image(None))
            } else {
                (database_audio, filesystem_audio)
            };

            assert_eq!(database_audio, filesystem_audio);
        }

        #[rstest]
        #[tokio::test]
        async fn test_remove(#[future(awt)] mock: Mock, #[values(true, false)] same_dir: bool) {
            let mut music_folder = mock.music_folder(0).await;
            music_folder
                .add_audio_filesystem::<&str>()
                .n_song(10)
                .depth(if same_dir { 0 } else { (1..3).fake() })
                .call()
                .await;
            music_folder.remove_audio_filesystem::<&str>().call().await;

            let database_audio = music_folder.query_filesystem().await;
            assert_eq!(database_audio, music_folder.filesystem);
        }

        #[rstest]
        #[tokio::test]
        async fn test_duplicate(
            #[future(awt)] mock: Mock,
            #[values(true, false)] same_dir: bool,
            #[values(true, false)] same_external_lyric: bool,
            #[values(true, false)] full: bool,
        ) {
            let mut music_folder = mock.music_folder(0).await;
            music_folder.add_audio_filesystem::<&str>().depth(0).call().await;
            let audio = music_folder.filesystem[0].clone();

            music_folder
                .add_audio_filesystem::<&str>()
                .metadata(audio.information.metadata.clone())
                .maybe_external_lyric(if same_external_lyric {
                    Some(audio.external_lyric.clone())
                } else {
                    None
                })
                .format(audio.information.file.format)
                .depth(if same_dir { 0 } else { (1..3).fake() })
                .full(scan::start::Full { file: full, ..Default::default() })
                .call()
                .await;

            let mut database_audio = music_folder.query_filesystem().await;
            assert_eq!(database_audio.len(), 1);
            let (database_path, database_audio) = database_audio.shift_remove_index(0).unwrap();

            let (path, audio) = music_folder
                .filesystem
                .shift_remove_index(usize::from(
                    audio.relative_path != database_audio.relative_path,
                ))
                .unwrap();
            assert_eq!(database_path, path);

            let (database_audio, audio) = if same_external_lyric {
                (database_audio, audio)
            } else {
                (database_audio.with_external_lyric(None), audio.with_external_lyric(None))
            };

            let (database_audio, audio) = if same_dir {
                (database_audio, audio)
            } else {
                (database_audio.with_dir_image(None), audio.with_dir_image(None))
            };
            assert_eq!(database_audio, audio);
        }

        #[rstest]
        #[tokio::test]
        async fn test_move(#[future(awt)] mock: Mock, #[values(true, false)] full: bool) {
            let mut music_folder = mock.music_folder(0).await;
            music_folder.add_audio_filesystem::<&str>().call().await;
            let audio = music_folder.filesystem[0].clone();
            music_folder.remove_audio_filesystem::<&str>().index(0).call().await;

            music_folder
                .add_audio_filesystem::<&str>()
                .metadata(audio.information.metadata.clone())
                .format(audio.information.file.format)
                .full(scan::start::Full { file: full, ..Default::default() })
                .call()
                .await;

            let database_audio = music_folder.query_filesystem().await;
            assert_eq!(database_audio, music_folder.filesystem);
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_scan_dir_image(#[future(awt)] mock: Mock) {
        let mut music_folder = mock.music_folder(0).await;
        music_folder
            .add_audio_filesystem::<&str>()
            .n_song(10)
            .depth(0)
            .recompute_dir_image(false)
            .call()
            .await;

        // All images are the same. However, the image will only be the same from the first
        // file that has a image so we have to filter out none before checking.
        let dir_images: Vec<_> = music_folder
            .filesystem
            .values()
            .filter_map(|information| information.dir_image.clone())
            .collect();
        assert!(dir_images.windows(2).all(|window| window[0] == window[1]));

        // On the other hand, data queried from database should have all the same image
        // regardless if the very first file have a image or not. So we use `map` instead of
        // `filter_map` here.
        let database_dir_images: Vec<_> = music_folder
            .query_filesystem()
            .await
            .values()
            .map(|information| information.dir_image.clone())
            .collect();
        assert!(database_dir_images.windows(2).all(|window| window[0] == window[1]));
    }
}
