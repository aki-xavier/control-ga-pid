// tests/comment_why.rs — the WHY-ONLY rule, held to this project: a comment here says something the
// code under it does not.
//
// The rules, the walk over the tree and the report are `../comment-why`'s (`src/gate.rs`) — a sibling
// project that reads text and asks the compiler for nothing, so this runs on this project's own
// toolchain with the rest of the tests. What is stated here is what is THIS project's: the roots its
// prose lives in, the floor its corpus has to clear, and the comments it keeps anyway. The rules
// decide three shapes — process narration and filler, a comment line whose content words are all in
// the code below it, and a short doc comment that re-says the item's own name. What they cannot decide
// goes to `make comments`, which prints that crate's local approximation beside this gate's verdicts.

use comment_why::gate::Gate;
use std::path::PathBuf;

/// Every root this project reads prose from; `make comments` walks the same ones.
const ROOTS: &[&str] = &["src", "tests", "examples"];

/// Comments this project keeps as they are, each with the reason. A record is keyed by the tail of the
/// file's path and the comment's own words rather than by line, so an edit above one cannot drop it,
/// and a record that no longer matches a flagged comment fails the gate.
///
/// It is empty: every comment here says something the code under it does not, or is one the rules do
/// not reach. A comment worth keeping anyway goes here, with the reason next to it.
const TOLERATED: &[(&str, &str, &str)] = &[];

#[test]
fn every_comment_says_why_rather_than_what() {
    // A gate over this crate's manifest directory, opened with the walk's two floors: a walk that found
    // the wrong tree, or no comments at all, fails here rather than passing quietly.
    Gate::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")), ROOTS, TOLERATED)
        .at_least(25, 200)
        .check()
        .unwrap_or_else(|report| panic!("\n{report}"));
}
