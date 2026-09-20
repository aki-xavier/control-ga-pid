// tests/comment_why.rs — the WHY-ONLY rule, held to this project: a comment here says something the
// code under it does not.
//
// The rules live in `../comment-why`, a sibling project that reads text and asks the compiler for
// nothing, so this runs on this project's own toolchain with the rest of the tests. What it decides is
// the three shapes that crate documents — process narration, a comment line whose content words are all
// in the code below it, and a short doc comment that re-says the item's own name. What it cannot decide
// goes to `make comments`, which prints that crate's local approximation beside this gate's verdicts,
// and to a reader.

use comment_why::{comments, line_of, scan, sources, words_of};
use std::path::{Path, PathBuf};

/// Every root this project reads prose from, and the same three the sibling crate's command line walks
/// by default.
const ROOTS: &[&str] = &["src", "tests", "examples"];

/// Comments this project keeps as they are, each with the reason. A record is keyed by the tail of the
/// file's path and the comment's words rather than by line, so an edit above one cannot drop it, and a
/// record that no longer matches a flagged comment fails the check below.
///
/// It is empty: every comment here says something the code under it does not, or is one the rules do
/// not reach. A comment worth keeping anyway goes here, with the reason next to it.
const TOLERATED: &[(&str, &str, &str)] = &[];

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn roots() -> Vec<String> {
    ROOTS.iter().map(|r| r.to_string()).collect()
}

fn read(file: &str) -> String {
    std::fs::read_to_string(manifest().join(file)).unwrap_or_else(|e| panic!("cannot read {file}: {e}"))
}

/// Whether a record keeps this comment: the file's tail and the comment's words, and nothing about
/// where it sits.
fn kept(file: &str, text: &str) -> bool {
    let said = words_of(text).join(" ");
    TOLERATED
        .iter()
        .any(|(f, kept, _)| file.ends_with(f) && words_of(kept).join(" ") == said)
}

#[test]
fn every_comment_says_why_rather_than_what() {
    let mut report = String::new();
    let mut flagged = 0usize;
    for file in sources(&manifest(), &roots()) {
        let src = read(&file);
        for finding in scan(&src) {
            if kept(&file, &finding.text) {
                continue;
            }
            flagged += 1;
            report.push_str(&format!(
                "{}:{}  {}\n      \"{}\"\n      {}: {}\n      under it: {}\n",
                file,
                line_of(&src, finding.start),
                finding.rule.name(),
                finding.text,
                finding.rule.name(),
                finding.rule.hint(),
                finding.under
            ));
        }
    }
    assert!(
        flagged == 0,
        "{flagged} comment(s) say what the code already says:\n\n{report}\n\
         Reword each one to state why the line is there, or delete it. A comment the rule should keep \
         as it is goes in TOLERATED in this file with the reason it stays."
    );
}

#[test]
fn a_tolerated_comment_still_exists() {
    for (file, text, reason) in TOLERATED {
        assert!(
            words_of(reason).len() >= 4,
            "{file}: \"{text}\" needs a reason a reader can weigh, not \"{reason}\""
        );
        let mut present = false;
        let mut flagged = false;
        for f in sources(&manifest(), &roots())
            .iter()
            .filter(|f| f.ends_with(file))
        {
            let src = read(f);
            present |= comments(&src).iter().any(|c| kept(file, &c.text));
            flagged |= scan(&src).iter().any(|finding| kept(file, &finding.text));
        }
        assert!(
            present,
            "{file}: the record \"{text}\" matches no comment any more — delete it"
        );
        assert!(
            flagged,
            "{file}: \"{text}\" escapes every rule now, so the record is dead weight — delete it"
        );
    }
}

#[test]
fn the_corpus_is_the_whole_project() {
    let files = sources(&manifest(), &roots());
    let mut blocks = 0usize;
    for file in &files {
        blocks += comments(&read(file)).len();
    }
    for root in ROOTS {
        let prefix = format!("{root}/");
        assert!(files.iter().any(|f| f.starts_with(&prefix)), "the walk lost {root}/ entirely");
    }
    assert!(files.len() >= 25, "the walk found only {} sources, so it is reading the wrong tree", files.len());
    assert!(blocks >= 200, "the walk found only {blocks} comment blocks, so it is reading nothing");
}

/// A path is spelled the same way by the walk and by a record — the record holds a tail, so this is
/// where that assumption is written down.
#[test]
fn a_record_is_keyed_by_a_path_tail() {
    assert!(Path::new("src/gains.rs").ends_with("src/gains.rs"));
}
