use std::borrow::Cow;

use diesel::prelude::*;
use uuid::Uuid;

pub use crate::schema::artists::{self, *};

#[derive(Debug, Queryable, Selectable, Insertable, AsChangeset)]
#[diesel(table_name = artists, check_for_backend(crate::orm::Type))]
#[diesel(treat_none_as_null = true)]
pub struct Data<'a> {
    pub name: Cow<'a, str>,
    pub normalized_name: Cow<'a, str>,
    pub mbz_id: Option<Uuid>,
}

#[derive(Debug, Queryable, Selectable, Insertable, AsChangeset)]
#[diesel(table_name = artists, check_for_backend(crate::orm::Type))]
#[diesel(treat_none_as_null = true)]
pub struct Upsert<'a> {
    pub index: Cow<'a, str>,
    #[diesel(embed)]
    pub data: Data<'a>,
}

#[derive(Debug, Queryable, Selectable)]
#[diesel(table_name = artists, check_for_backend(crate::orm::Type))]
#[diesel(treat_none_as_null = true)]
pub struct Artist<'a> {
    pub id: Uuid,
    #[diesel(embed)]
    pub data: Data<'a>,
}

mod upsert {
    use diesel::result::Error as DieselError;
    use diesel::{DecoratableTarget, ExpressionMethods, QueryDsl};
    use diesel_async::RunQueryDsl;
    use uuid::Uuid;

    use super::{Upsert, artists};
    use crate::Error;
    use crate::database::Database;

    impl crate::orm::upsert::Insert for Upsert<'_> {
        async fn insert(&self, database: &Database) -> Result<Uuid, Error> {
            let now = crate::time::now().await;
            if self.data.mbz_id.is_some() {
                diesel::insert_into(artists::table)
                    .values(self)
                    .on_conflict(artists::mbz_id)
                    .do_update()
                    .set((self, artists::scanned_at.eq(now)))
                    .returning(artists::id)
                    .get_result(&mut database.get().await?)
                    .await
            } else {
                let mut conn = database.get().await?;

                // When linking relationships, treat artists with the same `normalized_name` as the
                // same artist, even if the raw tag strings differ.
                match artists::table
                    .filter(artists::mbz_id.is_null())
                    .filter(artists::normalized_name.eq(self.data.normalized_name.as_ref()))
                    .order_by(artists::created_at.asc())
                    .select(artists::id)
                    .first::<Uuid>(&mut conn)
                    .await
                {
                    Ok(id) => {
                        diesel::update(artists::table.filter(artists::id.eq(id)))
                            .set(artists::scanned_at.eq(now))
                            .returning(artists::id)
                            .get_result(&mut conn)
                            .await
                    }
                    Err(DieselError::NotFound) => {
                        diesel::insert_into(artists::table)
                            .values(self)
                            .on_conflict(artists::name)
                            .filter_target(artists::mbz_id.is_null())
                            .do_update()
                            .set((
                                artists::normalized_name.eq(self.data.normalized_name.as_ref()),
                                artists::scanned_at.eq(now),
                            ))
                            .returning(artists::id)
                            .get_result(&mut conn)
                            .await
                    }
                    Err(e) => Err(e),
                }
            }
            .map_err(Error::from)
        }
    }
}
