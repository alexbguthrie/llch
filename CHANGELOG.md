# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Created `CHANGELOG.md` to track project changes.
- Added unit tests for tokenizers in `tests/test_tokenizers.py`.

### Changed
- Refactored `WikiTextDataset` in `data/dataset.py` for improved clarity and maintainability.

### Fixed
- Corrected a bug in `BPETokenizer.train` where no merges were being learned due to an incorrect vocabulary processing strategy. The method now uses a word frequency map to correctly generate and apply merges.
- Fixed a serialization bug in `BPETokenizer.save` and `BPETokenizer.load` where tuple keys in the `merges` dictionary were not being correctly handled, causing errors when saving to and loading from JSON. 