use std::collections::HashMap;
use std::sync::OnceLock;

use opencc_jieba_rs::OpenCC;

static OPENCC: OnceLock<OpenCC> = OnceLock::new();
static KATAKANA_TO_HIRAGANA: OnceLock<HashMap<char, char>> = OnceLock::new();

const KATAKANA: &str = "ァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンヴヵヶ";
const HIRAGANA: &str = "ぁぃぅぇぉゃゅょっーあいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんゔゕゖ";

fn opencc() -> &'static OpenCC {
    OPENCC.get_or_init(OpenCC::new)
}

fn kata_to_hira_map() -> &'static HashMap<char, char> {
    KATAKANA_TO_HIRAGANA.get_or_init(|| {
        debug_assert_eq!(KATAKANA.chars().count(), HIRAGANA.chars().count());
        KATAKANA.chars().zip(HIRAGANA.chars()).collect()
    })
}

pub fn normalize_name(name: &str, for_search: bool) -> String {
    let normalized_name = name.trim().to_lowercase();
    let normalized_name = opencc().convert(&normalized_name, "t2s", false);

    let map = kata_to_hira_map();
    let mut normalized_name: String =
        normalized_name.chars().map(|c| map.get(&c).copied().unwrap_or(c)).collect();

    if for_search {
        normalized_name.retain(|c| !c.is_whitespace());
    }

    normalized_name
}

#[cfg(test)]
#[coverage(off)]
mod test {
    use super::*;

    #[test]
    fn normalize_name_trims_and_lowercases() {
        assert_eq!(normalize_name("  AbC  ", false), "abc");
    }

    #[test]
    fn normalize_name_katakana_to_hiragana() {
        assert_eq!(normalize_name("アイウエオ", false), "あいうえお");
    }

    #[test]
    fn normalize_name_for_search_removes_whitespace() {
        assert_eq!(normalize_name("A  B\tC\n", true), "abc");
    }
}
