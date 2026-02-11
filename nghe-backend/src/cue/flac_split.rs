use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::num::NonZeroU64;

use atomic_write_file::AtomicWriteFile;
use num_rational::{Rational32, Rational64};
use typed_path::{Utf8PlatformPath, Utf8PlatformPathBuf};

use super::{CueSheet, VirtualCueTrack};
use crate::file;
use crate::file::audio;
use crate::filesystem;
use crate::filesystem::Trait as _;
use crate::http::header::ToOffset;
use crate::http::binary;
use crate::libflac_wrapper::{FlacDecoder, FlacEncoder, SeekableRead};
use crate::{Error, config, error, flac};

pub(crate) enum PreparedTrack {
    CachedPath {
        path: Utf8PlatformPathBuf,
        size: NonZeroU64,
    },
    InMemory {
        body: Vec<u8>,
        size: NonZeroU64,
    },
}

pub(crate) async fn prepare_virtual_flac_track(
    filesystem: &filesystem::Impl<'_>,
    config: &config::Transcode,
    cache_key: &file::Property<audio::Format>,
    virtual_track: &VirtualCueTrack,
) -> Result<PreparedTrack, Error> {
    let cue_bytes = filesystem.read(virtual_track.cue_path.to_path()).await?;
    let cue_sheet = CueSheet::parse(&cue_bytes)?;

    let audio_path = cue_sheet
        .resolve_audio_file_path(virtual_track.cue_path.to_path())
        .ok_or_else(|| error::Kind::NotFound)?;

    let cache_base_dir = config.cache_dir.as_ref().map(|dir| dir.join("cue_flac"));
    if let Some(cache_base_dir) = cache_base_dir {
        let output = cache_key.path_create_dir(cache_base_dir, "raw").await?;
        if tokio::fs::try_exists(&output).await? {
            let size = file_size_non_zero(&output).await?;
            return Ok(PreparedTrack::CachedPath { path: output, size });
        }

        // Build the raw split file once, atomically, then stream it with range support.
        let cue = cue_sheet.cue().clone();
        let cue_date = cue_sheet.date().map(str::to_owned);
        let track_number = virtual_track.track_number;

        let input = match filesystem {
            filesystem::Impl::Local(_) => Input::LocalPath(filesystem::local::Filesystem::to_platform(
                audio_path.to_path(),
            )?),
            filesystem::Impl::S3(_) => Input::Bytes(filesystem.read(audio_path.to_path()).await?),
        };

        let output_cloned = output.clone();
        tokio::task::spawn_blocking(move || {
            let atomic = AtomicWriteFile::open(output_cloned)?;
            {
                let mut writer = std::io::BufWriter::new(atomic.as_file());
                split_flac_track(input, &mut writer, cue, cue_date.as_deref(), track_number)?;
                writer.flush()?;
            }
            atomic.commit()?;
            Ok::<_, Error>(())
        })
        .await??;

        let size = file_size_non_zero(&output).await?;
        return Ok(PreparedTrack::CachedPath { path: output, size });
    }

    // No cache dir: generate the split file in-memory (still seekable for Range by slicing).
    let cue = cue_sheet.cue().clone();
    let cue_date = cue_sheet.date().map(str::to_owned);
    let track_number = virtual_track.track_number;

    let input = match filesystem {
        filesystem::Impl::Local(_) => Input::LocalPath(filesystem::local::Filesystem::to_platform(
            audio_path.to_path(),
        )?),
        filesystem::Impl::S3(_) => Input::Bytes(filesystem.read(audio_path.to_path()).await?),
    };

    let body = tokio::task::spawn_blocking(move || {
        let mut out = Cursor::new(Vec::<u8>::new());
        split_flac_track(input, &mut out, cue, cue_date.as_deref(), track_number)?;
        Ok::<_, Error>(out.into_inner())
    })
    .await??;

    let size =
        NonZeroU64::new(body.len().try_into()?).ok_or_else(|| error::Kind::EmptyFileEncountered)?;
    Ok(PreparedTrack::InMemory { body, size })
}

pub(crate) async fn response_from_prepared_track(
    prepared: PreparedTrack,
    range: Option<axum_extra::headers::Range>,
) -> Result<binary::Response, Error> {
    match prepared {
        PreparedTrack::CachedPath { path, size } => {
            let offset = range.map(|r| r.to_offset(size)).transpose()?;
            binary::Response::from_path_property(
                path,
                &file::PropertySize { size, format: audio::Format::Flac },
                offset,
                #[cfg(test)]
                None::<crate::test::binary::Status>,
            )
            .await
        }
        PreparedTrack::InMemory { body, size } => {
            let offset = range.map(|r| r.to_offset(size)).transpose()?;
            let offset_usize = offset.unwrap_or(0).try_into().unwrap_or(usize::MAX);
            let body = if offset_usize >= body.len() { vec![] } else { body[offset_usize..].to_vec() };
            binary::Response::from_body(
                axum::body::Body::from(body),
                &file::PropertySize { size, format: audio::Format::Flac },
                offset,
                #[cfg(test)]
                None::<crate::test::binary::Status>,
            )
        }
    }
}

async fn file_size_non_zero(path: impl AsRef<Utf8PlatformPath>) -> Result<NonZeroU64, Error> {
    let len = tokio::fs::metadata(path.as_ref()).await?.len();
    NonZeroU64::new(len).ok_or_else(|| error::Kind::EmptyFileEncountered.into())
}

#[derive(Clone)]
enum Input {
    LocalPath(Utf8PlatformPathBuf),
    Bytes(Vec<u8>),
}

impl Input {
    fn open(self) -> Result<(Box<dyn SeekableRead>, String), Error> {
        match self {
            Self::LocalPath(path) => {
                let file = std::fs::File::open(path.as_str())?;
                Ok((Box::new(std::io::BufReader::new(file)), path.to_string()))
            }
            Self::Bytes(bytes) => Ok((Box::new(Cursor::new(bytes)), "<memory>".to_owned())),
        }
    }
}

fn cue_ts_to_sample_pos(ts: cue_rw::CUETimeStamp, sample_rate: u32) -> u64 {
    let seconds = Rational32::from(ts);
    let seconds = Rational64::new((*seconds.numer()).into(), (*seconds.denom()).into());
    let samples = seconds * Rational64::from(sample_rate as i64);
    samples.to_integer() as u64
}

fn split_flac_track(
    input: Input,
    writer: &mut (impl Write + Seek),
    cue: cue_rw::CUEFile,
    cue_date: Option<&str>,
    track_number: u16,
) -> Result<(), Error> {
    let (mut reader, path_for_logging) = input.open()?;

    // Seek to the FLAC stream marker; some files may have leading tags (e.g. ID3).
    seek_to_flac_stream(&mut reader)?;

    // Parse metadata blocks (blocking).
    let metadata_blocks_all = get_metadata_blocks_sync(&mut reader)?;
    let stream_info =
        get_stream_info(&metadata_blocks_all).ok_or_else(|| error::Kind::NotFound)?;

    let sample_rate = stream_info.get_sample_rate();
    let file_id = cue_rw::FileID::from(0usize);

    // Map virtual `NN` to the Nth track (in-file order).
    let track_idx_in_file: usize = track_number.saturating_sub(1).into();
    let (track_id, track_total, track_sample_pos) =
        cue_track_sample_pos(&cue, file_id, sample_rate, track_idx_in_file)
            .ok_or_else(|| error::Kind::NotFound)?;

    let next_track_sample_pos = cue_next_track_sample_pos(&cue, file_id, sample_rate, track_idx_in_file);

    // Strip metadata blocks that reference original frame offsets.
    let mut metadata_blocks = metadata_blocks_all
        .iter()
        .filter(|block| {
            block.block_type != flac::FlacMetadataBlockType::SeekTable
                && block.block_type != flac::FlacMetadataBlockType::CUESheet
        })
        .cloned()
        .collect::<Vec<_>>();

    // Ensure VorbisComment exists and inject per-track metadata.
    let vorbis_comment_block = metadata_blocks
        .iter_mut()
        .find(|b| b.block_type == flac::FlacMetadataBlockType::VorbisComment);

    let vorbis_comment_block = match vorbis_comment_block {
        Some(block) => block,
        None => {
            let mut new_block = flac::FlacMetadataBlock {
                is_last:    false,
                block_type: flac::FlacMetadataBlockType::VorbisComment,
                content:    flac::FlacMetadataBlockContent::VorbisComment(flac::VorbisCommentBlock::new()),
            };
            if metadata_blocks.len() == 1 {
                // Ensure STREAMINFO stays first.
                metadata_blocks[0].is_last = false;
                new_block.is_last = true;
            }
            metadata_blocks.push(new_block);
            metadata_blocks.last_mut().unwrap()
        }
    };

    let flac::FlacMetadataBlock {
        content: flac::FlacMetadataBlockContent::VorbisComment(vorbis_comment),
        ..
    } = vorbis_comment_block
    else {
        unreachable!();
    };

    process_metadata(vorbis_comment, &cue, cue_date, track_id, track_total, track_number);
    metadata_blocks.sort_by_key(|b| b.is_last);

    // Scan frame sizes and sample positions.
    let mut decoder = FlacDecoder::new();
    let mut encoder = FlacEncoder::new();

    let ret = get_frame_sizes(&mut decoder, reader, &path_for_logging)?;
    let mut reader = ret.reader;
    let start_offset = ret.data.start_offset;
    let frame_sizes = ret.data.frame_sizes;

    reader.seek(SeekFrom::Start(start_offset))?;
    let frames_sample_pos =
        flac::FlacFrame::scan_frames(&mut reader, frame_sizes.iter().copied(), stream_info)?;

    // Encode sample-accurate head/tail frames (only these two are re-encoded).
    let ret = encode_track_head_tail_frames(
        &mut decoder,
        &mut encoder,
        stream_info,
        reader,
        track_sample_pos,
        next_track_sample_pos,
        &frames_sample_pos,
        &path_for_logging,
    )?;
    let mut reader = ret.reader;
    let head_frames = ret.data.head_frames;
    let tail_frames = ret.data.tail_frames;

    reader.seek(SeekFrom::Start(start_offset))?;
    write_track(
        &mut reader,
        writer,
        &metadata_blocks,
        track_sample_pos,
        next_track_sample_pos,
        &head_frames,
        tail_frames.as_ref(),
        &frame_sizes,
        &frames_sample_pos,
    )?;

    Ok(())
}

fn seek_to_flac_stream(reader: &mut (impl Read + Seek)) -> Result<(), Error> {
    reader.seek(SeekFrom::Start(0))?;
    let mut head = [0u8; 4];
    if reader.read_exact(&mut head).is_ok() && &head == b"fLaC" {
        return Ok(());
    }

    // Slow-path scan: search for "fLaC" marker and seek to it.
    reader.seek(SeekFrom::Start(0))?;
    let mut offset: u64 = 0;
    let mut buf = vec![0u8; 64 * 1024];
    let mut window = Vec::<u8>::with_capacity(3);

    loop {
        let read = reader.read(&mut buf)?;
        if read == 0 {
            break;
        }

        for &b in &buf[..read] {
            window.push(b);
            if window.len() > 4 {
                window.remove(0);
            }
            if window.len() == 4 && window.as_slice() == b"fLaC" {
                let pos = offset + 1; // current byte included
                reader.seek(SeekFrom::Start(pos - 4 + 4))?; // after marker
                return Ok(());
            }
            offset += 1;
        }
    }

    error::Kind::NotFound.into()
}

fn get_metadata_blocks_sync(
    mut reader: impl Read,
) -> Result<Vec<flac::FlacMetadataBlock>, Error> {
    let mut blocks = vec![];
    loop {
        let block = flac::FlacMetadataBlock::read_block_sync(&mut reader)?;
        let is_last = block.is_last;
        blocks.push(block);
        if is_last {
            break;
        }
    }
    Ok(blocks)
}

fn get_stream_info(blocks: &[flac::FlacMetadataBlock]) -> Option<&flac::StreamInfoBlock> {
    blocks
        .iter()
        .find(|b| b.block_type == flac::FlacMetadataBlockType::StreamInfo)
        .and_then(|b| {
            if let flac::FlacMetadataBlockContent::StreamInfo(ref content) = b.content {
                Some(content)
            } else {
                None
            }
        })
}

fn cue_track_sample_pos(
    cue: &cue_rw::CUEFile,
    file_id: cue_rw::FileID,
    sample_rate: u32,
    track_idx_in_file: usize,
) -> Option<(usize, usize, u64)> {
    let tracks = cue
        .tracks
        .iter()
        .enumerate()
        .filter(|(_, (fid, _))| *fid == file_id)
        .collect::<Vec<_>>();

    let (track_id, track_tuple) = *tracks.get(track_idx_in_file)?;
    let (_fid, track) = track_tuple;
    let total = tracks.len();

    let (_, ts) = track.indices.iter().find(|(id, _)| *id == 1)?;
    let sample_pos = cue_ts_to_sample_pos(*ts, sample_rate);

    Some((track_id, total, sample_pos))
}

fn cue_next_track_sample_pos(
    cue: &cue_rw::CUEFile,
    file_id: cue_rw::FileID,
    sample_rate: u32,
    track_idx_in_file: usize,
) -> Option<u64> {
    cue.tracks
        .iter()
        .filter(|(fid, _)| *fid == file_id)
        .nth(track_idx_in_file + 1)
        .and_then(|(_, t)| t.indices.iter().find(|(id, _)| *id == 1))
        .map(|(_, ts)| cue_ts_to_sample_pos(*ts, sample_rate))
}

fn process_metadata(
    vorbis_comment: &mut flac::VorbisCommentBlock,
    cue: &cue_rw::CUEFile,
    cue_date: Option<&str>,
    track_id: usize,
    track_total: usize,
    track_number: u16,
) {
    vorbis_comment.user_comments.remove("cuesheet");
    vorbis_comment.user_comments.remove("CUESHEET");

    let Some((_, cue_track)) = cue.tracks.get(track_id) else {
        return;
    };

    if !cue.performer.trim().is_empty() {
        vorbis_comment.add_vorbis_comment("ALBUMARTIST", cue.performer.clone());
    }
    if !cue.title.trim().is_empty() {
        vorbis_comment.add_vorbis_comment("ALBUM", cue.title.clone());
    }
    if let Some(ref catalog) = cue.catalog
        && !catalog.trim().is_empty()
    {
        vorbis_comment.add_vorbis_comment("CATALOG", catalog.clone());
    }

    if let Some(date) = cue_date.and_then(non_empty_str) {
        vorbis_comment.add_vorbis_comment("DATE", date);
    }

    if let Some(title) = non_empty_str(&cue_track.title) {
        vorbis_comment.add_vorbis_comment("TITLE", title);
    } else {
        vorbis_comment.add_vorbis_comment("TITLE", format!("Track {track_number:02}"));
    }

    let performer = cue_track
        .performer
        .as_ref()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(&cue.performer)
        .clone();
    if !performer.trim().is_empty() {
        vorbis_comment.add_vorbis_comment("ARTIST", performer);
    }

    if let Some(isrc) = cue_track.isrc.as_ref().and_then(|s| non_empty_str(s)) {
        vorbis_comment.add_vorbis_comment("ISRC", isrc);
    }

    vorbis_comment.add_vorbis_comment("TRACKNUMBER", track_number.to_string());
    vorbis_comment.add_vorbis_comment("TRACKTOTAL", track_total.to_string());
}

fn non_empty_str(s: &str) -> Option<&str> {
    let s = s.trim();
    if s.is_empty() { None } else { Some(s) }
}

struct RetWithReader<T> {
    reader: Box<dyn SeekableRead>,
    data:   T,
}

struct FileFrameSizes {
    start_offset: u64,
    frame_sizes:  Vec<u64>,
}

/// The start byte offset of flac frames, and frame sizes.
fn get_frame_sizes(
    decoder: &mut FlacDecoder,
    mut reader: Box<dyn SeekableRead>,
    path_for_logging: &str,
) -> Result<RetWithReader<FileFrameSizes>, Error> {
    reader.seek(SeekFrom::Start(0))?;
    decoder.init(reader, path_for_logging.to_string());

    let frame_byte_offsets = decoder.scan_frames();
    let frame_sizes = frame_byte_offsets
        .windows(2)
        .map(|w| w[1] - w[0])
        .collect::<Vec<_>>();

    let reader = decoder.finish();
    Ok(RetWithReader {
        reader,
        data: FileFrameSizes {
            start_offset: frame_byte_offsets[0],
            frame_sizes,
        },
    })
}

struct EncodedTrackFrames {
    head_frames: Vec<Vec<u8>>,
    tail_frames: Option<Vec<Vec<u8>>>,
}

fn encode_track_head_tail_frames(
    decoder: &mut FlacDecoder,
    encoder: &mut FlacEncoder,
    stream_info: &flac::StreamInfoBlock,
    mut reader: Box<dyn SeekableRead>,
    track_pos: u64,
    next_track_pos: Option<u64>,
    frames_sample_pos: &[u64],
    path_for_logging: &str,
) -> Result<RetWithReader<EncodedTrackFrames>, Error> {
    reader.seek(SeekFrom::Start(0))?;
    decoder.init(reader, path_for_logging.to_string());

    // `libFLAC` seek granularity depends on the underlying file and internal seek table usage.
    // To guarantee sample-accurate splits, decode the full frame that contains `track_pos` and
    // crop it to start exactly at `track_pos`.
    let head_frame_start = *frames_sample_pos
        .iter()
        .rfind(|&&pos| pos <= track_pos)
        .ok_or_else(|| error::Kind::NotFound)?;
    if !decoder.seek(head_frame_start) {
        return error::Kind::NotFound.into();
    }

    let Some(head_frame) = decoder.decode_frame() else {
        return error::Kind::NotFound.into();
    };
    let mut head = vec![];
    let crop = usize::try_from(track_pos.saturating_sub(head_frame_start)).unwrap_or(usize::MAX);
    for ch_data in head_frame {
        if crop > ch_data.len() {
            return error::Kind::NotFound.into();
        }
        let (_dropped, rest) = ch_data.split_at(crop);
        head.push(rest.to_vec());
    }

    let (tail, reader) = if let Some(next_track_pos) = next_track_pos {
        let tail_frame_start = *frames_sample_pos
            .iter()
            .rfind(|&&pos| pos <= next_track_pos)
            .ok_or_else(|| error::Kind::NotFound)?;

        if !decoder.seek(tail_frame_start) {
            return error::Kind::NotFound.into();
        }
        let Some(tail_frame) = decoder.decode_frame() else {
            return error::Kind::NotFound.into();
        };

        let mut tail = vec![];
        for ch_data in tail_frame {
            let split = usize::try_from(next_track_pos.saturating_sub(tail_frame_start))
                .unwrap_or(usize::MAX);
            if split > ch_data.len() {
                return error::Kind::NotFound.into();
            }
            let (tail_ch, _rest) = ch_data.split_at(split);
            tail.push(tail_ch.to_vec());
        }

        (Some(tail), decoder.finish())
    } else {
        (None, decoder.finish())
    };

    // Encode head/tail PCM into FLAC frames.
    let head_frames = {
        encoder.set_params(
            stream_info.get_channels(),
            stream_info.get_bits(),
            stream_info.get_sample_rate(),
            Some(head[0].len() as _),
        );
        encoder.init_stream();
        if !encoder.queue_encode(&head) {
            return error::Kind::NotFound.into();
        }
        let encoded_bytes = encoder.finish().ok_or_else(|| error::Kind::NotFound)?;
        extract_frames(decoder, &encoded_bytes)
    };

    let tail_frames = match tail {
        Some(tail) => {
            encoder.set_params(
                stream_info.get_channels(),
                stream_info.get_bits(),
                stream_info.get_sample_rate(),
                Some(tail[0].len() as _),
            );
            encoder.init_stream();
            if !encoder.queue_encode(&tail) {
                return error::Kind::NotFound.into();
            }
            let encoded_bytes = encoder.finish().ok_or_else(|| error::Kind::NotFound)?;
            Some(extract_frames(decoder, &encoded_bytes))
        }
        None => None,
    };

    Ok(RetWithReader {
        reader,
        data: EncodedTrackFrames {
            head_frames,
            tail_frames,
        },
    })
}

/// Unpacks frames from encoded flac bytes, omitting all metadata blocks.
fn extract_frames(decoder: &mut FlacDecoder, bytes: &[u8]) -> Vec<Vec<u8>> {
    let vec = bytes.to_vec();
    let cursor = Cursor::new(vec);
    decoder.init(Box::new(cursor), String::new());
    let frame_offsets = decoder.scan_frames();
    decoder.finish();

    let mut out = vec![];
    let mut last_offset = frame_offsets[0] as usize;
    let (_header, mut left) = bytes.split_at(last_offset);
    let mut frame;

    for offset in frame_offsets.into_iter().skip(1) {
        (frame, left) = left.split_at(offset as usize - last_offset);
        last_offset = offset as usize;
        out.push(frame.to_vec());
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn write_track(
    mut reader: impl Read + Seek,
    mut writer: impl Write + Seek,
    metadata_blocks: &[flac::FlacMetadataBlock],
    track_pos: u64,
    next_track_pos: Option<u64>,
    head_frames: &[Vec<u8>],
    tail_frames: Option<&Vec<Vec<u8>>>,
    frame_sizes: &[u64],
    frames_sample_pos: &[u64],
) -> Result<(), Error> {
    writer.write_all(b"fLaC")?;

    let stream_info = get_stream_info(metadata_blocks).ok_or_else(|| error::Kind::NotFound)?;
    for block in metadata_blocks.iter() {
        block.write_block(&mut writer)?;
    }

    let head_frames = head_frames
        .iter()
        .map(|bytes| flac::FlacFrame::read_frame(&**bytes, stream_info, bytes.len()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut head_frames = Some(head_frames); // make borrow checker happy
    let mut tail_frames = match tail_frames {
        Some(frames) => Some(
            frames
                .iter()
                .map(|bytes| flac::FlacFrame::read_frame(&**bytes, stream_info, bytes.len()))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        None => None,
    };

    let mut min_block_size = u16::MAX;
    let mut max_block_size = u16::MIN;

    let mut track_sample_pos = 0;
    let mut first_frame = true;
    let mut fixed_block_size_applied = false;

    // The original frame that contains `next_track_pos` must be replaced by `tail_frames`
    // (a re-encoded, sample-accurate prefix) so we stop before that frame.
    let tail_frame_start = next_track_pos.and_then(|next_track_pos| {
        frames_sample_pos
            .iter()
            .copied()
            .rfind(|&pos| pos <= next_track_pos)
    });

    for (size, frame_pos) in frame_sizes.iter().zip(frames_sample_pos.iter()) {
        // Always skip the frame that starts exactly at `track_pos` (in addition to those before),
        // because the `head_frames` already contain audio starting at `track_pos` and (when
        // aligned) cover the whole frame. Keeping the original frame would duplicate audio.
        if *frame_pos <= track_pos {
            reader.seek(SeekFrom::Current(*size as i64))?;
            continue;
        }

        let mut frame = flac::FlacFrame::read_frame(&mut reader, stream_info, *size as _)?;
        let block_size = frame.metadata.block_size.get_size();

        if first_frame {
            // This will only be executed once.
            let head_frames = head_frames.take().unwrap();
            for mut frame in head_frames {
                frame.metadata.blocking_strategy = flac::FlacBlockingStrategy::Variable;
                frame.metadata.position = flac::FlacFramePosition::SampleCount(track_sample_pos);
                let frame_samples = frame.metadata.block_size.get_size();
                track_sample_pos += frame_samples as u64;

                min_block_size = std::cmp::min(min_block_size, frame_samples);
                max_block_size = std::cmp::max(max_block_size, frame_samples);
                writer.write_all(&frame.into_bytes())?;
            }
            first_frame = false;
        }

        if !fixed_block_size_applied {
            min_block_size = std::cmp::min(min_block_size, block_size);
            max_block_size = std::cmp::max(max_block_size, block_size);
            if frame.metadata.blocking_strategy == flac::FlacBlockingStrategy::Fixed {
                fixed_block_size_applied = true;
            }
        }

        if tail_frame_start.is_some() && *frame_pos >= tail_frame_start.unwrap() {
            if let Some(tail_frames) = tail_frames.take() {
                for mut frame in tail_frames {
                    frame.metadata.blocking_strategy = flac::FlacBlockingStrategy::Variable;
                    frame.metadata.position = flac::FlacFramePosition::SampleCount(track_sample_pos);
                    let frame_samples = frame.metadata.block_size.get_size();
                    track_sample_pos += frame_samples as u64;

                    min_block_size = std::cmp::min(min_block_size, frame_samples);
                    max_block_size = std::cmp::max(max_block_size, frame_samples);
                    writer.write_all(&frame.into_bytes())?;
                }
            }
            break;
        }

        frame.metadata.blocking_strategy = flac::FlacBlockingStrategy::Variable;
        frame.metadata.position = flac::FlacFramePosition::SampleCount(track_sample_pos);
        writer.write_all(&frame.into_bytes())?;
        track_sample_pos += block_size as u64;
    }

    // Update STREAMINFO for the virtual track.
    let mut stream_info = stream_info.clone();
    stream_info.min_block_size = min_block_size;
    stream_info.max_block_size = max_block_size;
    stream_info.min_frame_size = 0;
    stream_info.max_frame_size = 0;
    stream_info.sample_count = track_sample_pos;
    stream_info.md5 = [0; 16];
    let block = flac::FlacMetadataBlock {
        is_last:    metadata_blocks.len() == 1,
        block_type: flac::FlacMetadataBlockType::StreamInfo,
        content:    flac::FlacMetadataBlockContent::StreamInfo(stream_info),
    };

    writer.seek(SeekFrom::Start(4))?;
    block.write_block(writer)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_stream_info(bytes: &[u8]) -> flac::StreamInfoBlock {
        let pos = bytes
            .windows(4)
            .position(|w| w == b"fLaC")
            .expect("missing fLaC marker");
        let mut cursor = Cursor::new(&bytes[pos + 4..]);

        let mut blocks = vec![];
        loop {
            let block = flac::FlacMetadataBlock::read_block_sync(&mut cursor).unwrap();
            let is_last = block.is_last;
            blocks.push(block);
            if is_last {
                break;
            }
        }

        blocks
            .into_iter()
            .find_map(|b| match b.content {
                flac::FlacMetadataBlockContent::StreamInfo(s) => Some(s),
                _ => None,
            })
            .expect("missing STREAMINFO")
    }

    #[test]
    fn test_split_flac_track_sample_count_matches_cue_span() {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let flac_path = manifest.join("../assets/test/sample.flac");
        let flac_bytes = std::fs::read(flac_path).unwrap();

        let original_info = read_stream_info(&flac_bytes);
        let sr = original_info.get_sample_rate() as u64;

        let cue_data = r#"
PERFORMER "Cue Artist"
TITLE "Cue Album"
FILE "sample.flac" WAVE
  TRACK 01 AUDIO
    TITLE "One"
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    TITLE "Two"
    INDEX 01 00:05:00
"#;
        let cue_sheet = CueSheet::parse_str(cue_data).unwrap();

        // Track 1: [0s, 5s)
        let mut out = Cursor::new(Vec::<u8>::new());
        split_flac_track(
            Input::Bytes(flac_bytes.clone()),
            &mut out,
            cue_sheet.cue().clone(),
            cue_sheet.date(),
            1,
        )
        .unwrap();
        let out_bytes = out.into_inner();
        let info = read_stream_info(&out_bytes);
        assert_eq!(info.get_sample_rate() as u64, sr);
        assert_eq!(info.sample_count, 5 * sr);

        // Track 2: [5s, EOF)
        let mut out = Cursor::new(Vec::<u8>::new());
        split_flac_track(
            Input::Bytes(flac_bytes.clone()),
            &mut out,
            cue_sheet.cue().clone(),
            cue_sheet.date(),
            2,
        )
        .unwrap();
        let out_bytes = out.into_inner();
        let info = read_stream_info(&out_bytes);
        assert_eq!(info.get_sample_rate() as u64, sr);
        assert_eq!(info.sample_count, original_info.sample_count - (5 * sr));
    }
}
