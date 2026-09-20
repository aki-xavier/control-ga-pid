# control-ga-pid — the GA-PID control core: the law in plane space, the design surface, and the loop.
#
#   make test          # this crate's suite, then the comment rules over the tree
#   make comments      # the comment rules alone, with the local approximation for the rest
#
# Every Cargo command below is run as `mbx <subcommand>` (the build-cache wrapper). The rules live in
# ../comment-why, read text, and need no toolchain of their own: the gate inside `mbx test` is
# tests/comment_why.rs, and `make comments` is the same rules over the working tree, with that crate's
# local approximation for the comments the rules cannot decide.

COMMENT_WHY ?= ../comment-why

.PHONY: test comments

test:
	mbx test
	$(MAKE) comments

comments:
	mbx run --quiet --manifest-path $(COMMENT_WHY)/Cargo.toml --bin comment-why -- --review
