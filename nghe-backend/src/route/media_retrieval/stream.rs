use axum_extra::headers::Range;
pub use nghe_api::media_retrieval::stream::{Format, Request};
use nghe_proc_macro::handler;
use uuid::Uuid;

use super::download;
use crate::cue;
use crate::database::Database;
use crate::file::audio::transcode;
use crate::filesystem::{Filesystem, Trait};
use crate::http::binary;
use crate::http::header::ToOffset;
#[cfg(test)]
use crate::test::binary::Status as BinaryStatus;
use crate::{Error, config, error};

#[handler]
pub async fn handler(
    database: &Database,
    filesystem: &Filesystem,
    #[handler(header)] range: Option<Range>,
    config: config::Transcode,
    user_id: Uuid,
    request: Request,
) -> Result<binary::Response, Error> {
    let (filesystem, source) =
        binary::Source::audio(database, filesystem, user_id, request.id).await?;

    let bitrate = request.max_bit_rate.unwrap_or(32);
    let time_offset = request.time_offset.unwrap_or(0);

    let source_path = source.path.to_path();
    let virtual_track = cue::VirtualCueTrack::parse_virtual_track_path(source_path);

    let format = match request.format.unwrap_or_default() {
        Format::Raw => {
            if let Some(virtual_track) = virtual_track {
                return cue::virtual_flac_track_response(
                    &filesystem,
                    &config,
                    &source.property,
                    &virtual_track,
                    range,
                )
                .await;
            }

            let size_offset =
                range.map(|range| range.to_offset(source.property.size.into())).transpose()?;
            return download::handler_impl(filesystem, source, size_offset).await;
        }
        Format::Transcode(format) => format,
    };
    let property = source.property.replace(format);
    let mut base_trim = transcode::Trim::default();
    let mut audio_source_path = source_path.to_path_buf();

    if let Some(virtual_track) = virtual_track {
        let cue_source_path = virtual_track.cue_path.to_path();
        let is_embedded =
            cue_source_path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("flac"));

        let (cue_sheet, audio_path) = if is_embedded {
            match &filesystem {
                crate::filesystem::Impl::Local(_) => {
                    let platform_path =
                        crate::filesystem::local::Filesystem::to_platform(cue_source_path)?;
                    let file = std::fs::File::open(platform_path.as_str())?;
                    let mut reader = std::io::BufReader::new(file);
                    let cue_str = crate::flac::extract_embedded_cuesheet_from_reader(&mut reader)?
                        .ok_or_else(|| error::Kind::NotFound)?;
                    let cue_sheet = cue::CueSheet::parse_str(&cue_str)?;
                    (cue_sheet, virtual_track.cue_path.clone())
                }
                crate::filesystem::Impl::S3(_) => {
                    // Download the object and parse the embedded cue sheet from bytes.
                    // TODO: This is less efficient than using a range request.
                    // But it's fine now since we don't even scan for embedded cue sheets on S3.
                    // Technically unreachable.
                    let bytes = filesystem.read(cue_source_path).await?;
                    let cue_str = crate::flac::extract_embedded_cuesheet_from_bytes(&bytes)?
                        .ok_or_else(|| error::Kind::NotFound)?;
                    let cue_sheet = cue::CueSheet::parse_str(&cue_str)?;
                    (cue_sheet, virtual_track.cue_path.clone())
                }
            }
        } else {
            let cue_bytes = filesystem.read(cue_source_path).await?;
            let cue_sheet = cue::CueSheet::parse(&cue_bytes)?;
            let audio_path = cue_sheet
                .resolve_audio_file_path(cue_source_path)
                .ok_or_else(|| error::Kind::NotFound)?;
            (cue_sheet, audio_path)
        };

        let span = cue_sheet
            .resolve_track_span(virtual_track.track_number)
            .ok_or_else(|| error::Kind::NotFound)?;

        base_trim = transcode::Trim { start: span.start_seconds, duration: span.duration_seconds };
        audio_source_path = audio_path;
    }

    let mut use_cached_output_as_input = false;
    let transcode_args = if let Some(ref cache_dir) = config.cache_dir {
        let output = property.path_create_dir(cache_dir, bitrate.to_string()).await?;
        let cache_exists = tokio::fs::try_exists(&output).await?;

        // If the cache exists, it means that the transcoding process is finish. Since we write the
        // transcoding cache atomically, we are guaranteed that that file is in a complete state and
        // is usable immediately. In that case, we have two cases:
        //  - If time offset is greater than 0, we can use the transcoded file as transcoder input
        //    so it only needs to activate `atrim` filter.
        //  - Otherwise, we only need to stream the transcoded file from local cache.
        if cache_exists {
            if time_offset > 0 {
                use_cached_output_as_input = true;
                (
                    transcode::Path { input: output.as_str().to_owned(), output: None },
                    #[cfg(test)]
                    BinaryStatus::UseCachedOutput,
                )
            } else {
                let size_offset =
                    range.map(|range| range.to_offset(property.size.into())).transpose()?;
                return binary::Response::from_path(
                    output,
                    format,
                    size_offset,
                    #[cfg(test)]
                    BinaryStatus::ServeCachedOutput,
                )
                .await;
            }
        } else {
            // If the file does not exist, we have two cases:
            //  - If time offset is greater than 0, we spawn a transcoding process without writing
            //    it back to the local cache.
            //  - Otherwise, we spawn a transcoding process and let the sink writes the transcoded
            //    chunk to the cache file.
            (
                transcode::Path {
                    input: filesystem.transcode_input(audio_source_path.to_path()).await?,
                    output: if time_offset > 0 { None } else { Some(output) },
                },
                #[cfg(test)]
                if time_offset > 0 { BinaryStatus::NoCache } else { BinaryStatus::WithCache },
            )
        }
    } else {
        (
            transcode::Path {
                input: filesystem.transcode_input(audio_source_path.to_path()).await?,
                output: None,
            },
            #[cfg(test)]
            BinaryStatus::NoCache,
        )
    };

    let offset = time_offset as f64;
    let trim = if use_cached_output_as_input {
        transcode::Trim::from_offset(time_offset)
    } else {
        transcode::Trim {
            start: base_trim.start + offset,
            duration: base_trim.duration.map(|d| (d - offset).max(0.0)),
        }
    };

    let (rx, _) = transcode::Transcoder::spawn(&config, transcode_args.0, format, bitrate, trim);

    binary::Response::from_rx(
        rx,
        format,
        #[cfg(test)]
        transcode_args.1,
    )
}

#[cfg(test)]
#[coverage(off)]
mod tests {
    use axum::http::StatusCode;
    use axum_extra::headers::HeaderMapExt;
    use diesel::{ExpressionMethods, QueryDsl};
    use diesel_async::RunQueryDsl;
    use itertools::Itertools;
    use nghe_api::common::{filesystem, format};
    use nghe_api::scan;
    use rstest::rstest;

    use super::*;
    use crate::file::audio;
    use crate::orm::{albums, songs};
    use crate::test::binary::Header as BinaryHeader;
    use crate::test::{Mock, mock};

    async fn spawn_stream(
        mock: &Mock,
        n_task: usize,
        user_id: Uuid,
        request: Request,
    ) -> (Vec<(StatusCode, Vec<u8>)>, Vec<BinaryStatus>) {
        let mut stream_set = tokio::task::JoinSet::new();
        for _ in 0..n_task {
            let database = mock.database().clone();
            let filesystem = mock.filesystem().clone();
            let config = mock.config.transcode.clone();
            stream_set.spawn(async move {
                handler(&database, &filesystem, None, config, user_id, request)
                    .await
                    .unwrap()
                    .extract()
                    .await
            });
        }
        let (responses, binary_status): (Vec<_>, Vec<_>) = stream_set
            .join_all()
            .await
            .into_iter()
            .map(|(status, headers, body)| {
                ((status, body), headers.typed_get::<BinaryHeader>().unwrap().0)
            })
            .unzip();
        (responses, binary_status.into_iter().sorted().collect())
    }

    #[rstest]
    #[tokio::test]
    async fn test_stream(
        #[future(awt)]
        #[with(1, 0)]
        mock: Mock,
        #[values(filesystem::Type::Local, filesystem::Type::S3)] ty: filesystem::Type,
    ) {
        mock.add_music_folder().ty(ty).call().await;
        let mut music_folder = mock.music_folder(0).await;
        music_folder.add_audio_filesystem::<&str>().format(audio::Format::Flac).call().await;

        let config = &mock.config.transcode;
        let user_id = mock.user_id(0).await;
        let song_id = music_folder.song_id_filesystem(0).await;
        let format = format::Transcode::Opus;
        let bitrate = 32;

        let transcoded = {
            let path = music_folder.absolute_path(0);
            let input = music_folder.to_impl().transcode_input(path.to_path()).await.unwrap();
            transcode::Transcoder::spawn_collect(config, &input, format, bitrate, 0).await
        };

        let request = Request {
            id: song_id,
            max_bit_rate: Some(bitrate),
            format: Some(format.into()),
            time_offset: None,
        };

        let (responses, binary_status) = spawn_stream(&mock, 2, user_id, request).await;
        for (status, body) in responses {
            assert_eq!(status, StatusCode::OK);
            assert_eq!(body, transcoded);
        }
        assert_eq!(binary_status, &[BinaryStatus::WithCache, BinaryStatus::WithCache]);

        let (responses, binary_status) = spawn_stream(&mock, 2, user_id, request).await;
        for (status, body) in responses {
            assert_eq!(status, StatusCode::OK);
            assert_eq!(body, transcoded);
        }
        assert_eq!(
            binary_status,
            &[BinaryStatus::ServeCachedOutput, BinaryStatus::ServeCachedOutput]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_stream_time_offset(
        #[future(awt)]
        #[with(1, 0)]
        mock: Mock,
        #[values(filesystem::Type::Local, filesystem::Type::S3)] ty: filesystem::Type,
    ) {
        mock.add_music_folder().ty(ty).call().await;
        let mut music_folder = mock.music_folder(0).await;
        music_folder.add_audio_filesystem::<&str>().format(audio::Format::Flac).call().await;

        let user_id = mock.user_id(0).await;
        let song_id = music_folder.song_id_filesystem(0).await;
        let config = &mock.config.transcode;
        let format = format::Transcode::Opus;
        let bitrate = 32;
        let time_offset = 10;

        let transcoded = {
            let path = music_folder.absolute_path(0);
            let input = music_folder.to_impl().transcode_input(path.to_path()).await.unwrap();
            transcode::Transcoder::spawn_collect(config, &input, format, bitrate, time_offset).await
        };

        let request = Request {
            id: song_id,
            max_bit_rate: Some(bitrate),
            format: Some(format.into()),
            time_offset: Some(time_offset),
        };

        let (responses, binary_status) = spawn_stream(&mock, 2, user_id, request).await;
        for (status, body) in responses {
            assert_eq!(status, StatusCode::OK);
            assert_eq!(transcoded, body);
        }
        assert_eq!(binary_status, &[BinaryStatus::NoCache, BinaryStatus::NoCache]);

        let binary_status =
            spawn_stream(&mock, 1, user_id, Request { time_offset: None, ..request }).await.1;
        assert_eq!(binary_status, &[BinaryStatus::WithCache]);

        let (responses, binary_status) = spawn_stream(&mock, 2, user_id, request).await;
        for (status, body) in responses {
            assert_eq!(status, StatusCode::OK);
            assert!(!body.is_empty());
        }
        assert_eq!(binary_status, &[BinaryStatus::UseCachedOutput, BinaryStatus::UseCachedOutput]);
    }

    #[rstest]
    #[tokio::test]
    async fn test_stream_virtual_cue_track(
        #[future(awt)]
        #[with(1, 0)]
        mock: Mock,
        #[values(filesystem::Type::Local, filesystem::Type::S3)] ty: filesystem::Type,
    ) {
        use crate::test::filesystem::Trait as _;
        use futures_lite::{StreamExt, stream};

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
FILE "Album.flac" WAVE
  TRACK 01 AUDIO
    TITLE "One"
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    TITLE "Two"
    INDEX 01 00:10:00
"#;
        let cue_abs = music_folder.absolutize("a/Album.cue");
        music_folder.to_impl().write(cue_abs.to_path(), cue_data.as_bytes()).await;
        music_folder.scan(scan::start::Full::default()).run().await.unwrap();

        let cue_rel = music_folder.path_str(&"a/Album.cue");
        let rel_track_1 = cue::build_virtual_track_relative_path(cue_rel, 1, audio::Format::Flac)
            .unwrap()
            .to_string();

        let song_id: Uuid = {
            let mut conn = mock.get().await;
            albums::table
                .inner_join(songs::table)
                .filter(albums::music_folder_id.eq(music_folder.id()))
                .filter(songs::relative_path.eq(&rel_track_1))
                .select(songs::id)
                .get_result(&mut conn)
                .await
                .unwrap()
        };

        let config = &mock.config.transcode;
        let user_id = mock.user_id(0).await;
        let format = format::Transcode::Opus;
        let bitrate = 32;

        let expected = {
            let cue_bytes = music_folder.to_impl().read(cue_abs.to_path()).await.unwrap();
            let cue_sheet = cue::CueSheet::parse(&cue_bytes).unwrap();
            let span = cue_sheet.resolve_track_span(1).unwrap();
            let audio_path = cue_sheet.resolve_audio_file_path(cue_abs.to_path()).unwrap();

            let input = music_folder.to_impl().transcode_input(audio_path.to_path()).await.unwrap();
            let trim =
                transcode::Trim { start: span.start_seconds, duration: span.duration_seconds };

            let (rx, handle) = transcode::Transcoder::spawn(
                config,
                transcode::Path { input, output: None },
                format,
                bitrate,
                trim,
            );
            let data: Vec<u8> = rx.into_stream().map(stream::iter).flatten().collect().await;
            handle.await.unwrap().unwrap();
            data
        };

        let request = Request {
            id: song_id,
            max_bit_rate: Some(bitrate),
            format: Some(format.into()),
            time_offset: None,
        };

        let (status, headers, body) =
            handler(mock.database(), mock.filesystem(), None, config.clone(), user_id, request)
                .await
                .unwrap()
                .extract()
                .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, expected);
        assert_eq!(headers.typed_get::<BinaryHeader>().unwrap().0, BinaryStatus::WithCache);

        let (status, headers, body) =
            handler(mock.database(), mock.filesystem(), None, config.clone(), user_id, request)
                .await
                .unwrap()
                .extract()
                .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, expected);
        assert_eq!(headers.typed_get::<BinaryHeader>().unwrap().0, BinaryStatus::ServeCachedOutput);
    }
}
