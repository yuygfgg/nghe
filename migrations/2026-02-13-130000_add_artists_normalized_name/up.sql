-- Your SQL goes here
alter table artists add column normalized_name text;

-- Backfill with the current name to keep existing rows valid. The application will
-- gradually update this to the real `normalize_name(name)` as items get rescanned.
update artists set normalized_name = name;

alter table artists alter column normalized_name set not null;

create index artists_normalized_name_idx on artists (normalized_name) where (mbz_id is null);

