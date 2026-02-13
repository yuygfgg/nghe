-- This file should undo anything in `up.sql`
drop index artists_normalized_name_idx;
alter table artists drop column normalized_name;

