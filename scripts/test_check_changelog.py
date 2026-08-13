import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from check_changelog import main, validate_changelog


VALID_CHANGELOG = """# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- An upcoming feature.

## [1.1.0] - 2024-02-02

### Fixed

- A released fix.

[Unreleased]: https://example.com/compare/v1.1.0...HEAD
[1.1.0]: https://example.com/releases/tag/v1.1.0
"""


class ValidateChangelogTests(unittest.TestCase):
    def validate(self, content: str) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "CHANGELOG.md")
            path.write_text(content, encoding="utf-8")
            return validate_changelog(path)

    def assert_invalid(self, content: str, expected: str) -> None:
        self.assertTrue(
            any(expected in error for error in self.validate(content)),
            f"expected an error containing {expected!r}",
        )

    def test_accepts_valid_changelog(self) -> None:
        self.assertEqual(self.validate(VALID_CHANGELOG), [])

    def test_rejects_missing_unreleased_section(self) -> None:
        content = VALID_CHANGELOG.replace("## [Unreleased]\n\n### Added\n\n- An upcoming feature.\n\n", "")
        content = content.replace(
            "[Unreleased]: https://example.com/compare/v1.1.0...HEAD\n", ""
        )
        self.assert_invalid(content, "Unreleased")

    def test_rejects_unknown_category(self) -> None:
        self.assert_invalid(
            VALID_CHANGELOG.replace("### Added", "### Improved"),
            "unsupported category",
        )

    def test_rejects_uncategorized_content(self) -> None:
        self.assert_invalid(
            VALID_CHANGELOG.replace(
                "## [Unreleased]\n", "## [Unreleased]\n\nUncategorized change."
            ),
            "uncategorized",
        )

    def test_rejects_non_semantic_version(self) -> None:
        self.assert_invalid(
            VALID_CHANGELOG.replace("1.1.0", "version-one"),
            "valid semantic version",
        )

    def test_rejects_invalid_release_date(self) -> None:
        self.assert_invalid(
            VALID_CHANGELOG.replace("2024-02-02", "02/02/2024"),
            "ISO 8601 date",
        )

    def test_rejects_missing_release_link(self) -> None:
        self.assert_invalid(
            VALID_CHANGELOG.replace(
                "[1.1.0]: https://example.com/releases/tag/v1.1.0\n", ""
            ),
            "reference link",
        )

    def test_rejects_duplicate_release(self) -> None:
        duplicate = "\n## [1.1.0] - 2024-02-02\n\n### Fixed\n\n- Duplicate.\n"
        self.assert_invalid(
            VALID_CHANGELOG.replace("\n[Unreleased]:", duplicate + "\n[Unreleased]:"),
            "duplicate release",
        )

    def test_rejects_release_dates_out_of_order(self) -> None:
        older_release = """
## [1.0.0] - 2024-03-03

### Added

- An older release.
"""
        content = VALID_CHANGELOG.replace(
            "\n[Unreleased]:",
            older_release + "\n[Unreleased]:",
        ).replace(
            "[1.1.0]: https://example.com/releases/tag/v1.1.0\n",
            "[1.1.0]: https://example.com/releases/tag/v1.1.0\n"
            "[1.0.0]: https://example.com/releases/tag/v1.0.0\n",
        )
        self.assert_invalid(content, "newest to oldest")

    def test_cli_reports_errors_and_returns_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "CHANGELOG.md")
            path.write_text("not a changelog\n", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = main([str(path)])

        self.assertEqual(result, 1)
        self.assertIn(f"{path}:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
